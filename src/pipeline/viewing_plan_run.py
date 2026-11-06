"""Daily card run for long-term viewing plans.

A second, non-streaming orchestration entry point parallel to
``orchestrator.run``. A viewing plan behaves like a scheduled intent package:

  S1 inject the plan's stage keywords as an intent package
  -> S2 ``execute_plan``
  -> explicitly subtract suppressed/disliked/consumed videos and the plan's own history
  -> S3 prefilter (intent branch: topic and duration window exempt, Shorts
     floor and blocks still apply)
  -> S4 triage -> S5 quality review
  -> S6 mode-fit judge with a plan ruling (profile / negative-preference
     sections not injected)
  -> S7 excluded types
  -> pick the top ``daily_count`` by ``W_FIT*fit + W_QUALITY*qnorm``.

No ranking pass, no card-pool writes, no prewarm trigger. Filter records for
S3/S5/S6/S7 are written as usual under a ``plan-<id>-<day>-<uuid8>`` refresh
id; YouTube quota is recorded inside ``execute_plan``.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Literal

import config
from agent import judges
from core import llmmeter
from core.models import Mode, RecommendationCard, RetrievalPlan, Video
from core.viewing_plan import Located, PlanRuling
from pipeline.orchestrator import (_channel_titles, _inject_intent_package,
                                   fit_effective, quality_norm_of)
from pipeline.prefilter import prefilter
from pipeline.prefs import load_excluded_types, partition_excluded
from pipeline.triage import quality_score, triage
from sources import youtube
from storage import db

logger = logging.getLogger(__name__)


class PlanRunError(Exception):
    """Plan card run failed; the API layer maps it to 503 with ``reason``.

    ``quota``: blocked by the budget gate or no upstream results;
    ``llm``: the judge failed as a whole.
    """

    def __init__(self, reason: Literal["quota", "llm"], detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


async def run_viewing_plan(user_id: int, mode: Mode, plan: dict, located: Located,
                           lang: str, taste: list[str],
                           ) -> tuple[list[RecommendationCard], dict]:
    # Usage metering: llmmeter is a ContextVar; start() installs a ledger in
    # this coroutine's context and judge calls accumulate into it.
    meter = llmmeter.start()
    ytu0 = await asyncio.to_thread(db.quota_used)
    refresh_id = f"plan-{plan['id']}-{db._pacific_today()}-{uuid.uuid4().hex[:8]}"
    queries = list(located.stage.keywords[: config.VIEWING_PLAN_SEARCH_MAX])
    daily_count = int(plan["daily_count"])

    # ---- S1: inject the plan package (no query expansion). Budget gate on
    # the search bucket: if fewer than MIN_QUERIES searches are left, give up
    # for today rather than run a partial package (a single query rarely
    # yields enough survivors).
    search_left = await asyncio.to_thread(db.quota_remaining, "search")
    if search_left < config.VIEWING_PLAN_MIN_QUERIES:
        raise PlanRunError("quota", f"search_left={search_left}")
    rplan = _inject_intent_package(
        RetrievalPlan(mode=mode.name, strategies=[], est_quota=0), queries,
        search_left, config.VIEWING_PLAN_SEARCH_MAX)
    n_queries = len(rplan.strategies[-1].params["queries"])

    # ---- S2: retrieval. execute_plan exempts intent-sourced videos from the
    # exclusion lists, and plans get no one-time pass, so suppression sets
    # and plan history are subtracted explicitly below.
    adapter = youtube.YouTubeAdapter(user_id=user_id)
    videos = await youtube.execute_plan(rplan, adapter)
    n_fetched = len(videos)
    # execute_plan swallows per-strategy exceptions (quota exhausted, 403,
    # network errors) and degrades them to empty lists. Without this check,
    # "everything upstream failed" would be cached as "found but all filtered"
    # and the user would get no cards for the rest of the day. Only an empty
    # result *after* filtering may be cached as empty.
    if n_fetched == 0:
        raise PlanRunError("quota", "S2 零召回(上游搜索失败或配额耗尽)")
    exempt = await asyncio.to_thread(db.accepted_appeal_video_ids, user_id)
    suppressed = await asyncio.to_thread(db.suppressed_video_ids, user_id)
    neg = await asyncio.to_thread(db.explicit_dislike_video_ids, user_id, mode.name)
    consumed, _pass_used = await asyncio.to_thread(db.consumed_suppress_sets, user_id)
    served = await asyncio.to_thread(db.plan_served_video_ids, plan["id"])
    banned = ((suppressed | neg | consumed) - exempt) | served
    videos = [v for v in videos if v.video_id not in banned]
    n_banned = n_fetched - len(videos)

    # ---- S3: rule prefilter (intent branch)
    kept = await asyncio.to_thread(prefilter, videos, mode, user_id, refresh_id)

    # ---- S4: triage (verdict cache + statistical direct pass).
    # Mirrors the orchestrator's S4/S5 logic; keep the two in sync.
    t = await asyncio.to_thread(
        triage, kept, direct_threshold=config.MODE_DIRECT_THRESHOLDS.get(mode.name))
    titles = await asyncio.to_thread(_channel_titles, kept)
    passed: list[Video] = []
    norm: dict[str, float] = {}
    for v in t.direct:
        passed.append(v)
        norm[v.video_id] = config.DIRECT_PASS_QUALITY_NORM
    for v, vd in t.cached:
        if vd.passed:
            passed.append(v)
            norm[v.video_id] = quality_norm_of(vd, quality_score(v))
        else:
            await asyncio.to_thread(db.add_filter_result, user_id, refresh_id,
                                    v.video_id, "quality", "cached", "曾被质量审核拒绝(7 天内)")

    # ---- S5: LLM quality review of the review bucket.
    # Mirrors the orchestrator's S5 pump; keep the two in sync.
    stat_scores = {v.video_id: quality_score(v) for v in t.review}
    review_by_id = {v.video_id: v for v in t.review}
    try:
        async for batch in judges.judge_quality(t.review, mode, stat_scores=stat_scores,
                                                channel_titles=titles):
            for vd in batch:
                if not vd.degraded:
                    await asyncio.to_thread(db.save_quality_verdict, vd)
                v = review_by_id[vd.video_id]
                if vd.passed:
                    passed.append(v)
                    norm[v.video_id] = quality_norm_of(vd, quality_score(v))
                else:
                    rule = (",".join(vd.categories) if vd.categories
                            else "degraded_reject" if vd.degraded else "unspecified")
                    await asyncio.to_thread(db.add_filter_result, user_id, refresh_id,
                                            vd.video_id, "quality", rule, vd.reason)
    except Exception as e:                       # noqa: BLE001 -- any judge failure maps to "llm"
        raise PlanRunError("llm", f"judge_quality: {e!r}") from e

    # ---- S6: plan ruling. User taste only breaks ties between candidates.
    # kind/theme select the ruling wording; plans without them default to
    # "progressive" with the plan name as theme.
    ruling = PlanRuling(plan_name=plan["name"], stage_title=located.stage.title,
                        stage_goal=located.stage.goal, level=plan.get("level", ""),
                        kind=plan.get("kind") or "progressive",
                        theme=plan.get("theme") or plan["name"],
                        taste_lines=list(taste[:2]))
    by_id = {v.video_id: v for v in passed}
    candidates: list[tuple[Video, float, float, str | None]] = []
    reasons: dict[str, str] = {}
    try:
        async for batch in judges.judge_mode_fit(passed, mode, channel_titles=titles,
                                                 lang=lang, plan_ruling=ruling):
            for vd in batch:
                v = by_id[vd.video_id]
                show, rule = fit_effective(vd)
                if show:
                    ctype = (config.normalize_content_type(vd.content_type)
                             if vd.content_type is not None else None)
                    reasons[v.video_id] = vd.reason
                    candidates.append((v, vd.fit_score, norm[v.video_id], ctype))
                else:
                    await asyncio.to_thread(db.add_filter_result, user_id, refresh_id,
                                            v.video_id, "mode", rule or "mode_unfit", vd.reason)
    except Exception as e:                       # noqa: BLE001
        raise PlanRunError("llm", f"judge_mode_fit: {e!r}") from e

    # ---- S7: excluded types (explicit user choice; plans get no exemption)
    excluded_types = await asyncio.to_thread(load_excluded_types, user_id, mode.name)
    if excluded_types:
        candidates, dropped, _ = partition_excluded(candidates, excluded_types)   # profile-entry exemption only applies to refreshes
        for v, _f, _q, ctype in dropped:
            await asyncio.to_thread(db.add_filter_result, user_id, refresh_id, v.video_id,
                                    "S7", "user_excluded_type", f"用户排除类型:{ctype}")

    # ---- Selection: same key as the lightweight snapshot (fit + quality)
    def _final(c):
        return config.W_FIT * c[1] + config.W_QUALITY * c[2]
    picked = sorted(candidates, key=lambda c: (-_final(c), c[0].video_id))[:daily_count]
    cards = [RecommendationCard(
        video=v, reason=reasons.get(v.video_id, ""), fit_score=fit, quality_norm=qn,
        pref_score=0.0, is_gem=False, final=_final((v, fit, qn, ct)),
        section="top", content_type=ct, overlay=False)
        for v, fit, qn, ct in picked]
    # Usage accounting: YouTube units spent are estimated from the change in
    # the global quota counter (there is no other way to attribute them);
    # max(0, ...) prevents a negative value under concurrency.
    yt_spent = max(0, await asyncio.to_thread(db.quota_used) - ytu0)
    await asyncio.to_thread(db.bump_user_usage, user_id, yt_spent,
                            meter.input_tokens, meter.output_tokens)
    stats = {"refresh_id": refresh_id, "n_fetched": n_fetched, "n_banned": n_banned,
             "n_kept": len(kept), "n_candidates": len(candidates), "n_picked": len(cards),
             "queries": n_queries, "yt_spent": yt_spent}
    logger.info("viewing_plan[%s/u%s/p%s]: 阶段「%s」词 %d → S2 %d/禁推减 %d/S3 %d/候选 %d/"
                "上屏 %d(N=%d),yt %du",
                mode.name, user_id, plan["id"], located.stage.title, n_queries, n_fetched,
                n_banned, len(kept), len(candidates), len(cards), daily_count, yt_spent)
    return cards, stats
