"""Single orchestrator for one recommendation refresh.

run() chains the pipeline: S0 state machine -> S1 planning -> S2 retrieval ->
S3 rule prefilter -> S4 triage -> S5 quality review -> S6 mode-fit judging
(progressive waves, yielded as they arrive) -> S7 exclusion filter -> one final
rank -> persistence and conservation check. All ledger writes for a refresh
converge here.

Dependency direction is orchestrator -> stages / agent / sources / storage;
stages never import each other. The orchestrator itself makes no LLM calls and
no YouTube API calls; those live behind the agent and sources layers.

Key design points:
- Progressive waves: the first wave (direct-pass + cached-pass candidates) is
  judged in parallel with S5 so cards reach the screen quickly; later waves
  wait for S5 results.
- Data conservation: every video entering S3 ends up exactly once in top, more
  or filter_results.
- Fail-soft bookkeeping: ledger and card-pool write failures are logged but can
  never drop the final snapshot.
"""
import asyncio
import json
import logging
import time
import traceback
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import replace as _dc_replace   # DiversityPlan is frozen; seed maps are attached via replace()

import config
from agent import intent_search, judges, planner
from agent.prompts import PROMPTS
from core.costing import estimate_plan_cost
from core import fit_cache
from core import llmmeter              # per-refresh LLM token metering
from core.logsafe import redact, redacted_traceback        # redact secrets from exception logs
from core.models import Mode, RecommendationCard, RetrievalPlan, StrategyCall, Video
from pipeline import stability
from pipeline.diversity import (DiversityPlan, draw_featured_subs, effective_excluded,
                                exempt_positions, manual_seat_plan, order_profile_queries,
                                weighted_profile_queries)
from pipeline.mix import apply_pool_ratio, enforce_source_cap, enforce_source_mix
from pipeline.prefilter import prefilter
from pipeline.prefs import effective_pool_ratio, effective_type_buckets, load_excluded_types, partition_excluded
from pipeline.ranker import RankedItem, rank
from pipeline.triage import quality_score, triage
from sources import youtube
from storage import db
from pipeline.subtopics import entry_rotation

logger = logging.getLogger(__name__)

# Valid content types for out-of-vocabulary observation (vocabulary is fixed at import time).
_KNOWN_TYPES = frozenset(config.CONTENT_TYPES) | {"other"}


def _flush_type_oov(mode: Mode, oov) -> None:
    """Report out-of-vocabulary content_type values as one aggregated warning per
    refresh (so a bad prompt version cannot flood the logs). Ranking is safe against
    OOV (unknown types fall back to weight 0.6) but observability is not: systematic
    OOV silently degrades the type axis. Zero OOV means zero log lines."""
    if not oov:
        return
    top = ", ".join(f"{t}×{n}" for t, n in oov.most_common(5))
    logger.warning("orchestrator[%s]: S6 词表外 content_type %d 条(prompt %s):%s",
                   mode.name, sum(oov.values()),
                   PROMPTS[f"fit_{mode.name}"].version, top)

def conservation_check(refresh_id: str, n_s3_input: int,
                       n_top: int, n_more: int) -> dict:
    """Data conservation check, shared by run() and the tests.

    Invariant: len(S3 input) == len(top) + len(more) + count(filter_results for
    this refresh). Every video entering triage must end up exactly once: shown
    (top/more) or filtered with a recorded reason. A mismatch means a video
    vanished (missing filter record) or was counted twice."""
    n_filtered = len(db.get_filter_results(refresh_id))
    rhs = n_top + n_more + n_filtered
    return {"ok": n_s3_input == rhs, "lhs": n_s3_input, "rhs": rhs,
            "n_top": n_top, "n_more": n_more, "n_filtered": n_filtered}


def quality_norm_of(vd, stat_score: float) -> float:
    """Quality norm for the non-direct paths: degraded_pass -> statistical score
    (fail-open); llm_grade / cached_grade -> grade/3 (ordering invariant
    0 < 1/3 < 0.6 direct-pass < 2/3 < 1); legacy cached rows with grade=NULL ->
    statistical score."""
    if vd.degraded:
        return stat_score
    if vd.quality_grade is not None:
        return vd.quality_grade / 3
    return stat_score


def fit_effective(vd) -> tuple[bool, str | None]:
    """Single consumer of S6 verdicts: returns (shown, filter_results rule).
    fit=False -> mode_degraded_reject (degraded) or mode_unfit (real verdict).
    fit=True and degraded bypasses the threshold (preserves fail-open); a real
    verdict below the threshold is recorded as fit_below_threshold."""
    if not vd.fit:
        return False, ("mode_degraded_reject" if vd.degraded else "mode_unfit")
    if vd.degraded:
        return True, None
    thr = config.FIT_SCORE_THRESHOLD
    if thr is not None and vd.fit_score < thr:
        return False, "fit_below_threshold"
    return True, None


def _reuse_allowed(state: str, intent_text: str) -> bool:
    """S0 reuse gate: a cached plan may be reused only in steady state with no
    intent. Intent is one-off input and must never be served from a cached plan;
    when present S1 always runs, otherwise the intent search package would have
    nowhere to be injected."""
    return state == "steady" and not intent_text


def _inject_intent_package(plan: RetrievalPlan, queries: list[str],
                           search_left: int, cap: int) -> RetrievalPlan:
    """S1.5 intent search package injection. Pure function, no IO.

    1. Hallucination reset: any intent_driven flag set by the planner is cleared;
       only this function may issue the pass (same "don't trust the LLM"
       discipline as validate_plan).
    2. Query count = min(search_left, cap, len(queries)). The caller computes cap
       (intent refresh: min(INTENT_SEARCH_MAX, user's remaining daily intent
       budget); viewing plan: VIEWING_PLAN_SEARCH_MAX). With no search quota left
       nothing is injected (the API would reject it anyway); the first query (the
       verbatim intent) goes out as long as one search is available.
    3. Package shape: a single strategy carries the whole package (stays within
       the 2-6 strategy validation), target = queries x INTENT_PER_QUERY_KEEP, so
       each query keeps exactly its top-KEEP results by relevance."""
    for s in plan.strategies:
        s.intent_driven = False
    n = min(search_left, cap, len(queries))
    if n < 1:
        logger.info("意图搜索包:搜索额度不足(search_left=%d cap=%d),本次不注入",
                    search_left, cap)
        return plan
    pkg_queries = queries[:n]
    plan.strategies.append(StrategyCall(
        tool="search_by_keyword",
        params={"queries": pkg_queries},
        target_count=len(pkg_queries) * config.INTENT_PER_QUERY_KEEP,
        rationale=f"意图搜索包({len(pkg_queries)} 词,首搜=声明原文;IP-2/3)",
        intent_driven=True))
    return plan


def _count_free_searches(plan: RetrievalPlan) -> int:
    """Number of search.list calls made by non-intent search strategies (batched
    queries count per query; a single query counts as 1)."""
    return sum(max(1, len(s.params.get("queries") or []))
               for s in plan.strategies
               if s.tool == "search_by_keyword" and not s.intent_driven)


def _strip_free_searches(plan: RetrievalPlan) -> tuple[RetrievalPlan, int]:
    """Remove non-intent search strategies (the first to yield when search quota is
    tight: profile rotation and planner free searches). Intent packages are left
    alone. Returns (plan, number of strategies removed)."""
    before = len(plan.strategies)
    plan.strategies = [s for s in plan.strategies
                       if s.tool != "search_by_keyword" or s.intent_driven]
    return plan, before - len(plan.strategies)


def _inject_profile_searches(plan: RetrievalPlan, queries: list[str],
                             cursor: int, search_left: int,
                             user_left: int) -> tuple[RetrievalPlan, int]:
    """Profile search rotation. Each refresh gives MAX_SEARCH_PER_REFRESH free
    search slots to the search_query of user-written profile entries, taken
    round-robin from a cursor, so preferences the channel pool does not cover still
    get supply via live search. Planner free searches yield to rotation queries;
    intent packages are left alone. Queries are not repeated when there are fewer
    queries than slots.

    Gate: when search_left < SEARCH_FLOOR_PROFILE this path yields entirely,
    injecting nothing and stripping planner free searches (the refresh still
    produces cards without search). Query count = min(MAX_SEARCH_PER_REFRESH,
    len(queries), user_left, search_left), where user_left is the user's remaining
    daily profile-search budget. Pure function, no IO.

    Returns (plan, queries injected). The caller advances the cursor when > 0; the
    per-user daily budget is charged by the unified fallback step in run()."""
    if search_left < config.SEARCH_FLOOR_PROFILE:
        plan, _ = _strip_free_searches(plan)
        return plan, 0
    if not queries:
        return plan, 0
    n = min(config.MAX_SEARCH_PER_REFRESH, len(queries), user_left, search_left)
    if n < 1:
        return plan, 0
    picked = [queries[(cursor + i) % len(queries)] for i in range(n)]
    plan.strategies = [s for s in plan.strategies
                       if s.tool != "search_by_keyword"
                       or getattr(s, "intent_driven", False)]
    plan.strategies.append(StrategyCall(
        tool="search_by_keyword", params={"queries": picked},
        target_count=n * config.INTENT_PER_QUERY_KEEP,
        rationale=f"画像搜索轮转({n} 词,游标 {cursor};PF 批 T5)"))
    return plan, n


def _resolve_profile_queries(profile_rows, subs_by_eid: dict, lang: str = "zh") -> list[str]:
    """Per profile entry, the query to issue this refresh, in position order (empty
    string when none; the caller filters). With lang="en", prefers the entry's
    search_query_en / subtopic query_en and falls back to the original query."""
    return [entry_rotation(r["search_query"], subs_by_eid.get(r["entry_id"]),
                           _row_get(r, "subtopic_cursor") or 0,
                           lang=lang, search_query_en=_row_get(r, "search_query_en")) or ""
            for r in profile_rows]


def _consumed_entry_ids(profile_rows, resolved: list[str], picked: list[str]) -> list[str]:
    """Entry ids whose resolved query was actually injected (all of them if several
    entries share a query, which is rare). Pure function."""
    pk = set(picked)
    return [r["entry_id"] for r, q in zip(profile_rows, resolved) if q and q in pk and r["entry_id"]]


def _featured_entry_ids(profile_rows, main_pos: list[int], excluded: set[int]) -> list[str]:
    """Entry ids of featured profile entries by seat allocation. main_pos is the
    featured position order (auto mode: [1] + drawn sub-entries; manual mode:
    entries with seat_pct > 0 by share, descending). Excluded positions,
    out-of-range positions and empty entry ids are skipped; deduplicated, order
    preserved. Feeds the candidate library's per-entry reranking. Seat allocation
    is used rather than injected queries because the library fills seats."""
    out: list[str] = []
    for p in main_pos:
        if p in excluded or p < 1 or p > len(profile_rows):
            continue
        eid = profile_rows[p - 1]["entry_id"]
        if eid and eid not in out:
            out.append(eid)
    return out


def _inject_library(plan: RetrievalPlan, mode_name: str, seed_ids: list[str],
                    top_ids: list[str], fingerprint: str,
                    featured_entry_ids: list[str] | None = None) -> tuple[RetrievalPlan, int]:
    """Inject the local high-quality candidate library strategy. The HQ set is all
    seed channels of the mode plus the user's top-K preferred channels (seeds first,
    deduplicated). Zero API cost, so it bypasses the budget gate. Previously
    injected use_library strategies are removed before appending (steady-state
    reuse persists the injected plan, so they would otherwise accumulate). Appended
    last, so under ratio truncation it only fills what subscribed/seed targets leave.

    featured_entry_ids: featured entry ids for per-entry reranking; [] means all
    entries by position. Returns (plan, HQ channel count); an empty HQ set means no
    injection. Pure function, no IO."""
    plan.strategies = [s for s in plan.strategies
                       if not (s.tool == "fetch_channel_videos" and s.params.get("use_library"))]
    hq: list[str] = []
    for cid in list(seed_ids) + list(top_ids):
        if cid and cid not in hq:
            hq.append(cid)
    if not hq or config.LIBRARY_PER_REFRESH <= 0:
        return plan, 0
    plan.strategies.append(StrategyCall(
        tool="fetch_channel_videos",
        params={"use_library": True, "mode": mode_name, "channel_ids": hq,
                "seed_channel_ids": list(seed_ids), "fit_fingerprint": fingerprint,
                "featured_entry_ids": list(featured_entry_ids or [])},
        target_count=config.LIBRARY_PER_REFRESH,
        rationale=f"高质量本地候选库({len(hq)} 频道;D5)"))
    return plan, len(hq)


async def _fit_with_cache(subwave: list[Video], itext: str, *, judge, user_id: int,
                          mode_name: str, fingerprint: str, stats: dict):
    """S6 judging through the fit verdict cache. For the regular pool (empty itext)
    with the cache enabled, hits are yielded first as one batch (no LLM); misses go
    to the judge batch by batch and non-degraded verdicts are written back. Intent
    pools and TTL=0 go straight to the judge. Cache errors fail open (read failure =
    all misses; write failure = warning only)."""
    ttl = int(config.FIT_CACHE_TTL_DAYS)
    if itext or ttl <= 0 or not subwave:
        async for b in judge(subwave):
            stats["llm"] += len(b)
            yield b
        return
    ids = [v.video_id for v in subwave]
    try:
        rows = await asyncio.to_thread(db.get_fit_verdicts, user_id, mode_name, ids, fingerprint, ttl)
    except Exception:                                   # noqa: BLE001
        logger.warning("S6 缓存读取失败(视为全 miss)\n%s", redacted_traceback())
        rows = {}
    hits = [fit_cache.verdict_from_row(rows[i]) for i in ids if i in rows]
    if hits:
        stats["cached"] += len(hits)
        yield hits
    misses = [v for v in subwave if v.video_id not in rows]
    if not misses:
        return
    async for b in judge(misses):
        stats["llm"] += len(b)
        keep = [vd for vd in b if not vd.degraded]
        if keep:
            try:
                await asyncio.to_thread(db.save_fit_verdicts, user_id, mode_name, fingerprint, keep)
            except Exception:                           # noqa: BLE001
                logger.warning("S6 缓存写入失败(忽略)\n%s", redacted_traceback())
        yield b


def _inject_seed_channels(plan: RetrievalPlan, channel_ids: list[str],
                          quota_left: int) -> tuple[RetrievalPlan, int]:
    """Inject targeted fetching of seed channels: channels hand-picked on featured
    profile entries join the candidate set and go through the normal S3-S6 path.
    The budget gate mirrors _inject_profile_searches: skip entirely if the budget
    cannot cover one channel, otherwise trim to what is affordable. Pure function.

    Steady-state reuse stores the plan *after* injection and runs it through this
    function again next refresh; appending alone would accumulate stale seed
    strategies (growing strategy count, old channel ids coming back). Previously
    injected use_seed_channels strategies are therefore removed first, so
    channel_ids are fresh each refresh and the strategy count stays constant.
    Returns (plan, channels injected)."""
    if not channel_ids:
        return plan, 0
    per_cost = config.QUOTA_COST["fetch_channel_videos"]
    avail = quota_left * config.PLAN_QUOTA_BUDGET_RATIO - estimate_plan_cost(plan).general
    n = min(len(channel_ids), int(avail // per_cost))
    if n < 1:
        return plan, 0
    picked = channel_ids[:n]
    plan.strategies = [s for s in plan.strategies
                       if not (s.tool == "fetch_channel_videos"
                               and s.params.get("use_seed_channels"))]
    plan.strategies.append(StrategyCall(
        tool="fetch_channel_videos",
        params={"use_seed_channels": True, "seed_channel_ids": picked,
                "n_per_channel": config.SEED_FETCH_PER_CHANNEL},
        target_count=len(picked) * config.SEED_FETCH_PER_CHANNEL,
        rationale=f"种子频道定向抓取({len(picked)} 频道;SC-4)"))
    return plan, len(picked)


def _mark_top_channels_featured(plan: RetrievalPlan, positions: list[int], mode: str) -> int:
    """Write this refresh's featured positions and mode into use_top_channels
    strategies (the sources layer weights channels by entry attribution history).
    Mutates params in place; returns the number of strategies updated."""
    n = 0
    for s in plan.strategies:
        if s.tool == "fetch_channel_videos" and s.params.get("use_top_channels"):
            s.params["featured_positions"] = list(positions)
            s.params["mode"] = mode
            n += 1
    return n


def _row_get(row, key: str):
    """Read a column from a sqlite3.Row; None if the column is missing (older schema)."""
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return None


def _inject_expanded_channels(plan: RetrievalPlan, channel_ids: list[str],
                              quota_left: int) -> tuple[RetrievalPlan, int]:
    """Inject targeted fetching of expanded channels (mirrors _inject_seed_channels).
    Channels the profile-expansion judge verified as a good fit
    (entry_expanded_channels, rotated by cursor across featured entries) join the
    candidate set tagged expanded_channel, pass S3 (topic gate exempt) through S6,
    and compete without reserved seats. Budget gate and remove-then-append
    idempotence match the seed path. Returns (plan, channels injected)."""
    if not channel_ids:
        return plan, 0
    per_cost = config.QUOTA_COST["fetch_channel_videos"]
    avail = quota_left * config.PLAN_QUOTA_BUDGET_RATIO - estimate_plan_cost(plan).general
    n = min(len(channel_ids), int(avail // per_cost))
    if n < 1:
        return plan, 0
    picked = channel_ids[:n]
    plan.strategies = [s for s in plan.strategies
                       if not (s.tool == "fetch_channel_videos"
                               and s.params.get("use_expanded_channels"))]
    plan.strategies.append(StrategyCall(
        tool="fetch_channel_videos",
        params={"use_expanded_channels": True, "expanded_channel_ids": picked,
                "n_per_channel": config.EXPAND_FETCH_PER_CHANNEL},
        target_count=len(picked) * config.EXPAND_FETCH_PER_CHANNEL,
        rationale=f"扩充频道定向抓取({len(picked)} 频道;XS-4)"))
    return plan, len(picked)


def _seed_plan_map(pairs: list[tuple[str, str]], pos_ids: dict[int, str],
                   ) -> dict[int, frozenset[str]]:
    """Map (entry_id, channel_id) pairs to {position: seed channel set} so the ranker
    can claim reserved seed seats by channel_id. Unknown or non-featured entry ids
    are ignored; a channel belongs to the first entry it appears under (the
    round-robin order already puts entry 1 first). Pure function."""
    eid_pos = {eid: pos for pos, eid in pos_ids.items()}
    out: dict[int, set[str]] = {}
    seen: set[str] = set()
    for eid, cid in pairs:
        pos = eid_pos.get(eid)
        if pos is None or cid in seen:
            continue
        seen.add(cid)
        out.setdefault(pos, set()).add(cid)
    return {pos: frozenset(cids) for pos, cids in out.items()}


def _lite_plan(queries: list[str], search_left: int, cap: int,
               mode_name: str, local_ids: list[str] | None = None) -> RetrievalPlan:
    """Lightweight intent refresh plan: *only* the intent search package, with no
    planner strategies and no regular retrieval, so an intent refresh does not rerun
    the full pipeline (quota drops from ~3000 units to a few searches). The query
    count is capped at INTENT_LITE_QUERIES, then by the search quota and cap (the
    user's remaining daily intent budget); the verbatim first query is preserved by
    reusing _inject_intent_package.

    local_ids: video ids from local FTS5 recall (the caller has applied the duration
    window and exclusion sets). When non-empty, a zero-API use_local_intent strategy
    is injected first, carrying the same intent pass as live search; if hits >=
    INTENT_LITE_TOP the search count drops to INTENT_LITE_QUERIES_MIN (the verbatim
    query still goes out, to catch new content not yet in the library)."""
    plan = RetrievalPlan(mode=mode_name, strategies=[], est_quota=0)
    n_q = config.INTENT_LITE_QUERIES
    if local_ids:
        plan.strategies.append(StrategyCall(
            tool="fetch_channel_videos",
            params={"use_local_intent": True, "video_ids": list(local_ids), "mode": mode_name},
            target_count=len(local_ids),
            rationale=f"意图本地召回({len(local_ids)} 条,FTS5;E2)",
            intent_driven=True))
        if len(local_ids) >= config.INTENT_LITE_TOP:
            n_q = max(0, int(config.INTENT_LITE_QUERIES_MIN))
    plan = _inject_intent_package(plan, queries[:n_q], search_left, cap)
    for s in plan.strategies:                # the injector clears intent_driven; restore it on our own local strategy
        if s.params.get("use_local_intent"):
            s.intent_driven = True
    return plan


def _lite_snapshot(candidates: list[tuple[Video, float, float, str | None]],
                   reasons: dict[str, str]) -> list[RecommendationCard]:
    """Final snapshot for a lightweight intent refresh: the top INTENT_LITE_TOP
    candidates by W_FIT*fit + W_QUALITY*qnorm, all section=top and overlay=True. The
    frontend overlays them at the head of the current screen (insert at head, shift
    existing cards down, trim the tail) instead of replacing it.
    Does not go through rank(): there is nothing to mix, and exploration slots,
    reserved seats and type weights are meaningless for a single intent pool.
    Channel preference is not mixed in either: intent recall is mostly unfamiliar
    channels and preference scores would only push it down."""
    seat = lambda t: (-(config.W_FIT * t[1] + config.W_QUALITY * t[2]),   # noqa: E731
                      t[0].video_id)
    picked = sorted(candidates, key=seat)[:config.INTENT_LITE_TOP]
    return [RecommendationCard(
        video=v, reason=reasons.get(v.video_id, ""), fit_score=fit,
        quality_norm=qnorm, pref_score=0.0, is_gem=False,
        final=config.W_FIT * fit + config.W_QUALITY * qnorm,
        section="top", content_type=ctype, overlay=True)
        for v, fit, qnorm, ctype in picked]


def _intent_fill_cards(user_id: int, mode: Mode, lang: str, need: int,
                       exclude_ids: set[str],
                       excluded_types: set[str]) -> list[RecommendationCard]:
    """Fill the rest of the Top area after an intent refresh. Seats not taken by
    intent cards are filled with regular supply from the chip shelf
    (chip_candidates), restricted to the user's top two type buckets; otherwise a
    pure intent snapshot would crowd regular recommendations off the Top area.

    Supply discipline (same as cold-start cards): zero quota, zero LLM; deduplicated
    against the user's full card_pool history; blacklisted channels, excluded types
    and the mode's duration window are filtered on read. These cards are *not* part
    of the conservation equation (they never entered S3).

    Type rotation: candidates are grouped by base type and taken round-robin across
    types, so coverage beats filling from one type (seats stay empty when supply
    runs out). Within a type, chips the user selected come before vocabulary
    fallback chips. The top two buckets already sit above the strict floor, so no
    extra strict exclusion is needed."""
    if need <= 0:
        return []
    buckets = effective_type_buckets(user_id, mode.name)
    top_types: list[str] = []                    # keep bucket order: tier-1 types come first
    for tier in buckets[:config.INTENT_FILL_BUCKET_TIERS]:
        for t in tier:
            if t not in excluded_types and t not in top_types:
                top_types.append(t)
    if not top_types:
        return []

    user_chips = db.get_user_chips(user_id)
    _user_set = set(user_chips)
    eligible = [s for s, (_d, base) in config.INTEREST_CHIPS.items()
                if base in top_types]
    # User-selected chips first, vocabulary fallback after (users with no selection still get fill).
    ordered_chips = ([c for c in user_chips if c in set(eligible)]
                     + [c for c in eligible if c not in _user_set])

    rows = db.get_chip_candidates(ordered_chips, lang,
                                  config.CHIP_CANDIDATE_TTL_DAYS)
    seen = db.user_card_pool_video_ids(user_id)
    black = db.blacklisted_channel_ids(user_id)

    by_chip: dict[str, list] = {}
    for r in rows:
        by_chip.setdefault(r["chip_slug"], []).append(r)
    by_type: dict[str, list] = {t: [] for t in top_types}
    for slug in ordered_chips:                   # group chips by type, preserving chip priority
        base = config.INTEREST_CHIPS.get(slug, ("", ""))[1]
        if base in by_type:
            by_type[base].extend(by_chip.get(slug, []))

    def _fits_window(dur: int | None) -> bool:
        # A missing duration passes: shelf rows were already filtered by
        # CHIP_CANDIDATE_MIN_DURATION_SEC, and a missing value is a data issue, not
        # a reason to silently drop the card.
        if dur is None:
            return True
        if dur < mode.duration_min_sec:
            return False
        return mode.duration_max_sec is None or dur <= mode.duration_max_sec

    picked: list[RecommendationCard] = []
    picked_ids: set[str] = set()
    cursor = {t: 0 for t in top_types}
    while len(picked) < need:
        advanced = False
        for t in top_types:                      # round-robin across types: coverage first
            lst = by_type[t]
            while cursor[t] < len(lst):
                r = lst[cursor[t]]
                cursor[t] += 1
                if (r["video_id"] in picked_ids or r["video_id"] in seen
                        or r["video_id"] in exclude_ids
                        or r["channel_id"] in black
                        or not _fits_window(r["duration_sec"])):
                    continue
                qnorm = (r["quality_score"] if r["quality_score"] is not None
                         else config.DIRECT_PASS_QUALITY_NORM)
                picked.append(RecommendationCard(
                    video=Video(video_id=r["video_id"],
                                channel_id=r["channel_id"],
                                title=r["title"],
                                duration_sec=r["duration_sec"],
                                source_strategy="channel_pool",
                                cover_url=r["cover_url"],
                                publish_time=(r["publish_time"] if "publish_time" in r.keys() else None),
                                view_count=r["view_count"] or 0,
                                like_count=r["like_count"] or 0),
                    reason=r["reason"] or "", fit_score=0.0,
                    quality_norm=qnorm, pref_score=0.0, is_gem=False,
                    final=config.W_QUALITY * qnorm,
                    section="top", content_type=t, overlay=True))
                picked_ids.add(r["video_id"])
                advanced = True
                break                            # one card per type per round
            if len(picked) >= need:
                break
        if not advanced:
            break                                # all types exhausted; leave seats empty
    return picked


def _truncate_first_wave(first_wave: list[Video],
                         stat_of: Callable[[Video], float],
                         limit: int | None, *,
                         intent_head: int = 0,
                         sims: dict[str, float] | None = None) -> tuple[list[Video], list[Video]]:
    """First-wave truncation gate: keep the top `limit` candidates by statistical
    quality score for the first wave; the rest are *deferred to later waves* (not
    dropped, no filter record). limit=None disables truncation. Ties keep insertion
    order for reproducibility.

    intent_head > 0 (intent fast lane): the top intent_head intent candidates by
    statistical score go at the *head* of the first wave and do not compete for
    `limit` slots. Search-sourced intent results usually have weaker engagement than
    subscribed channels; pure statistical truncation would push them all into later
    waves and the user would not see what they asked for on the first screen.
    Overflow on both sides is deferred. intent_head=0 or no intent candidates
    leaves behaviour unchanged.

    sims: {video_id: max cosine similarity to profile entries}. When given, regular
    candidates are ordered by RRF (statistical rank + similarity rank): similar
    candidates move forward, while candidates without vectors keep only their
    statistical rank instead of being pushed back (videos fetched by S2 today have
    no vectors yet, and the first screen must not be all old library content).
    Changes order only, never the set; None/empty leaves behaviour unchanged."""
    intent_pool: list[Video] = []
    normal_pool = first_wave
    if intent_head > 0:
        intent_pool = sorted((v for v in first_wave if v.intent_sourced),
                             key=stat_of, reverse=True)
        normal_pool = [v for v in first_wave if not v.intent_sourced]
        head, intent_overflow = intent_pool[:intent_head], intent_pool[intent_head:]
    else:
        head, intent_overflow = [], []

    if limit is None or len(normal_pool) <= limit:
        return head + list(normal_pool), list(intent_overflow)
    ordered = sorted(normal_pool, key=stat_of, reverse=True)
    if sims:
        K = float(config.RRF_K)
        fused = {v.video_id: 1.0 / (K + i) for i, v in enumerate(ordered, 1)}
        by_sim = sorted((v for v in ordered if v.video_id in sims),
                        key=lambda v: sims[v.video_id], reverse=True)
        for i, v in enumerate(by_sim, 1):
            fused[v.video_id] += 1.0 / (K + i)
        ordered = sorted(ordered, key=lambda v: fused[v.video_id], reverse=True)   # stable sort: ties keep statistical order
    return head + ordered[:limit], ordered[limit:] + intent_overflow


def _judge_pools(wave: list[Video], intent_text: str,
                 ) -> list[tuple[list[Video], str]]:
    """S6 pool order: the intent pool goes *before* the regular pool, so within a wave
    intent batches are judged and yielded first and the first incoming cards are
    what the user asked for. Without intent, a single unsplit pool (zero overhead)."""
    if not intent_text:
        return [(wave, "")]
    return [([v for v in wave if v.intent_sourced], intent_text),
            ([v for v in wave if not v.intent_sourced], "")]


def _review_intent_first(review: list[Video],
                         stat_of: Callable[[Video], float]) -> list[Video]:
    """Reorder the S5 review queue: intent candidates first by statistical score,
    regular ones in original order. judge_quality batches in order, so intent
    reviews finish first and reach the S6 intent batch sooner."""
    intent = sorted((v for v in review if v.intent_sourced),
                    key=stat_of, reverse=True)
    if not intent:
        return review
    return intent + [v for v in review if not v.intent_sourced]


def _drain_intent_head(buf: list[Video]) -> tuple[list[Video], list[Video]]:
    """Drain all intent candidates from buf (order preserved) for an early partial
    batch; regular candidates stay to fill full batches. No intent -> ([], buf)."""
    intent = [v for v in buf if v.intent_sourced]
    if not intent:
        return [], buf
    return intent, [v for v in buf if not v.intent_sourced]


def _apply_consumed_filter(videos: list[Video], consumed: set,
                           pass_used: set) -> list[Video]:
    """Consumed-video suppression: drop videos inside the suppression window. Intent
    search hits get a one-time pass: allowed while the pass is unused (not shown
    since being consumed), suppressed once it has been used."""
    return [v for v in videos if v.video_id not in consumed
            or (v.intent_sourced and v.video_id not in pass_used)]


def _drain_whole_batches(buf: list, batch_size: int) -> tuple[list, list]:
    """Split later-wave dispatch: take all *full* batches at once so the judge's
    inter-batch concurrency kicks in; the remainder keeps accumulating. "Full
    batches only" still holds: the first part is always a multiple of batch_size,
    and the caller flushes the tail batch after the loop."""
    n = (len(buf) // batch_size) * batch_size
    return buf[:n], buf[n:]


def _channel_titles(videos: list[Video]) -> dict[str, str]:
    """Channel titles for the judges' compress step. Judges never touch the DB, so
    the caller supplies titles, looked up once per distinct channel."""
    titles: dict[str, str] = {}
    for cid in {v.channel_id for v in videos}:
        row = db.get_channel(cid)
        if row is not None and row["channel_title"]:
            titles[cid] = row["channel_title"]
    return titles

def _clamp_entry(value, n_entries: int) -> int | None:
    """Clamp the profile-entry attribution returned by the judge. LLM output is
    untrusted: anything outside 1..n (including bools posing as ints) becomes None;
    with no profile (n=0) the result is always None."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if 1 <= value <= n_entries else None


def _final_card(item: RankedItem, section: str, reasons: dict[str, str]) -> RecommendationCard:
    """Translate a RankedItem (the ranker's intermediate state with per-signal
    scores) plus the fit reason kept from the waves into the external
    RecommendationCard contract. The two types are deliberately kept separate."""
    return RecommendationCard(
        video=item.video, reason=reasons.get(item.video.video_id, ""),
        fit_score=item.fit_score, quality_norm=item.quality_score_norm,
        pref_score=item.channel_pref_score, is_gem=item.is_gem,
        final=item.final, section=section,
        content_type=item.content_type,     # type passed through for card explainability
        entry_position=item.entry)          # entry attribution travels with the card for behaviour feedback


def _impression_tail_cards(user_id: int, mode_name: str, lang: str, seats: int,
                           exclude_ids: set[str],
                           excluded_types: set[str],
                           channel_counts: dict[str, int] | None = None) -> list[RecommendationCard]:
    """Build impression-penalty tail cards. Penalized cards (shown >= 2 times
    without being opened, within a 72h window) are not banned: they are pulled
    straight from card_pool and placed at the end of the shown area. They already
    passed S3-S7 when first shown, so quality, reason and type are in the pool row:
    no re-judging and no LLM (sending them back through the pipeline would multiply
    S6 LLM volume 5-8x).

    Hard suppression wins over tail seats: blacklisted channels, excluded types,
    explicit dislikes and consumed videos are all removed ("explicitly said no"
    beats "just didn't click"). Not part of the conservation equation (never
    entered S3). Showing them increments impressions via mark_card_pool_served, so
    frequently shown cards queue further back next time."""
    if seats <= 0:
        return []
    black = db.blacklisted_channel_ids(user_id)
    disliked = db.explicit_dislike_video_ids(user_id, mode_name)
    consumed, _ = db.consumed_suppress_sets(user_id)
    rows = db.impression_tail_rows(user_id, mode_name, lang, seats * 3,
                                   exclude_ids | disliked | consumed,
                                   min_quality=config.TAIL_HIGH_QUALITY_MIN,       # only high-quality or seed-channel rows qualify
                                   seed_channels=set(db.seed_channels_for_mode(user_id, mode_name)))
    cards: list[RecommendationCard] = []
    _ch = dict(channel_counts or {})
    _cap = int(config.RANK_CHANNEL_CAP or 0)
    for r in rows:
        if len(cards) >= seats:
            break
        if r["channel_id"] in black:
            continue
        if r["content_type"] and r["content_type"] in excluded_types:
            continue
        if _cap > 0 and _ch.get(r["channel_id"], 0) >= _cap:
            continue                             # tail seats also respect the per-channel cap
        _ch[r["channel_id"]] = _ch.get(r["channel_id"], 0) + 1
        v = Video(video_id=r["video_id"], channel_id=r["channel_id"],
                  title=r["title"] or "", description="",
                  duration_sec=r["duration_sec"] or 0,
                  cover_url=r["cover_url"], view_count=r["view_count"] or 0,
                  publish_time=(r["publish_time"] if "publish_time" in r.keys() else None),
                  like_count=r["like_count"] or 0,
                  source_strategy=r["source_strategy"] or "search")
        cards.append(RecommendationCard(
            video=v, reason=r["reason"] or "", fit_score=0.0,
            quality_norm=r["quality_score"] or 0.0, pref_score=0.0,
            is_gem=False, final=0.0, section="top",
            content_type=r["content_type"]))
    return cards


def _snapshot_to_pool_rows(snapshot: list[RecommendationCard]) -> list[dict]:
    """Final snapshot -> card_pool rows. Only pipeline verdicts are stored; video
    metadata is rejoined from videos/channels on read rather than duplicated."""
    return [{"video_id": c.video.video_id,
             "quality_score": c.quality_norm,
             "reason": c.reason,
             "content_type": c.content_type,
             "slot": "more" if c.section == "more" else "top",
             "entry_position": c.entry_position}
            for c in snapshot]


def _write_pool_safe(user_id: int, mode_name: str, lang: str, rows: list[dict]) -> None:
    """Fail-soft card-pool write: the pool is an accelerator, not the main path.
    Failures are logged and never raised, so a refresh never fails just because
    caching failed."""
    try:
        db.upsert_card_pool(user_id, mode_name, lang, rows)
    except Exception as exc:                       # noqa: BLE001 -- catch-all is intended
        logger.error("orchestrator[%s/u%s]: 写池失败(不影响本次刷新):%r",
                     mode_name, user_id, exc)


def _search_remaining(trial: bool) -> int:
    """Remaining search quota for today. For trial-account refreshes it reads as 0
    once the site-wide remainder drops below TRIAL_SEARCH_FLOOR: the intent path
    uses local recall only, profile rotation and planner free searches yield, and
    cards are still produced. This keeps anonymous trial traffic from exhausting
    the shared search bucket for regular users."""
    left = db.quota_remaining("search")
    if trial and left < config.TRIAL_SEARCH_FLOOR:
        return 0
    return left


async def run(user_id: int, mode: Mode, refresh_id: str, lang: str = "zh",
              consume_intent: bool = True, trial: bool = False,
              ) -> AsyncIterator[list[RecommendationCard]]:
    """Run one full refresh. refresh_id is generated and passed in by the caller.

    Yield contract: zero or more intermediate batches with section='incoming'
    (final=None, faded in by arrival order), and *the last yield is always the full
    final snapshot*; consumers treat the last frame as authoritative. lang is passed
    through to the S6 fit judge.

    consume_intent: background prewarm refreshes pass False. They only stock regular
    supply and must not consume a one-off session intent the user is waiting on in
    the foreground; otherwise the prewarm would run the intent refresh, its cards
    would be served out of order from the shelf (losing their head position), and
    intent search quota would be spent in the background."""
    try:
        _t0 = time.perf_counter()
        _tm: dict[str, float] = {}
        _last = _t0

        def _lap(key: str) -> None:
            nonlocal _last
            now = time.perf_counter()
            _tm[key] = _tm.get(key, 0.0) + (now - _last)
            _last = now

        meter = llmmeter.start()
        ytu0 = await asyncio.to_thread(db.quota_used)

        # S0: ask the state machine first
        state = await asyncio.to_thread(stability.mode_state, user_id, mode)

        # Read the active intent once before S1 and record it in plan_json under
        # "intent" (what the planner saw; same read point as the planner context, so
        # if the TTL expires mid-refresh the planning moment wins). Empty string = no
        # intent; the key is always present. Only intents declared for this mode are
        # used: other modes' intents read as empty, so the whole intent_sourced path
        # stays silent and nothing leaks across modes.
        intent_text = ""
        if config.ENABLE_SESSION_INTENT and consume_intent:
            irow = await asyncio.to_thread(db.get_active_intent, user_id, mode.name)
            if irow is not None:
                intent_text = irow["text"]
        # User-written profile entries: read once, used for rotation, seat quotas and the judge's direction section.
        profile_rows = await asyncio.to_thread(
            db.get_mode_profile_entries, user_id, mode.name)
        # Negative preferences: read once, used by the S6 judge.
        neg_rows = await asyncio.to_thread(db.get_negative_prefs, user_id, mode.name)
        # S6 fit-cache fingerprint: any change to prompt version, model, language,
        # profile entries or negative preferences invalidates it. One value per
        # refresh; the library strategy carries it too (unfit verdicts are not recycled).
        _fit_fp = fit_cache.fingerprint(
            PROMPTS[f"fit_{mode.name}"].version,
            config.MODE_LLM_OVERRIDES.get(mode.name, config.LLM_MODEL), lang,
            [r["text"] for r in profile_rows], [r["text"] for r in neg_rows])
        _fit_stats = {"cached": 0, "llm": 0}

        # ---- Profile diversity draw: entry 1 is always featured; sub-entries are drawn
        # independently with weights decreasing by position. Plays within the last 10
        # minutes double weight and seats; entries missed several refreshes in a row
        # are forced in; seeded by refresh_id for reproducibility. Intent refreshes skip
        # rank and the draw.
        _dv_featured: list[int] = []
        _dv_plan: DiversityPlan | None = None
        _entry_bases: dict[int, list[str]] = {}
        _seat_manual = False                    # manual seat-share mode
        _dv_exempt_s7: set[int] = set()         # positions exempt at S7 (exempt minus explicit 'keep excluded')
        _dv_main_pos: list[int] = []            # featured position order (auto: [1] + drawn; manual: by share desc)
        _pcts_pos: dict[int, int] = {}
        _dv_excluded: set[int] = set()          # read by _featured_entry_ids even when this block is skipped
        if profile_rows and not intent_text:
            _n_entries = len(profile_rows)
            for _i, _r in enumerate(profile_rows, 1):
                try:
                    _bases = (json.loads(_r["base_types_json"])
                              if _r["base_types_json"] else [])
                except (TypeError, ValueError):
                    _bases = []
                _entry_bases[_i] = [b for b in _bases if b in config.CONTENT_TYPES]
            _excl_types = await asyncio.to_thread(
                load_excluded_types, user_id, mode.name)
            _dv_excluded_raw = {i for i, bs in _entry_bases.items()
                                if bs and all(b in _excl_types for b in bs)}
            # Manual seat-share mode: seat_mode='manual' with any entry seat_pct > 0 ->
            # shares become reserved seats; the draw, play boost, miss streaks and
            # per-entry cap are off, and featured order is by share (descending) for
            # seed/expanded fetching. The auto-mode path is unchanged.
            _prow0 = await asyncio.to_thread(db.get_mode_prefs, user_id, mode.name)
            _pcts_pos = {i: (_row_get(_r, "seat_pct") or 0)
                         for i, _r in enumerate(profile_rows, 1)}
            _seat_manual = (_prow0 is not None and _row_get(_prow0, "seat_mode") == "manual"
                            and any(v > 0 for v in _pcts_pos.values()))
            # Exempt positions (auto: positions 1..N; manual: top-N by share) ignore
            # excluded types unless the user explicitly chose "keep excluded" in the
            # conflict dialog. The effective excluded set feeds both the seat side
            # (draw / manual_seat_plan / query order) and S7 (_dv_exempt_s7), in sync.
            _acks = {i: _row_get(_r, "conflict_ack") for i, _r in enumerate(profile_rows, 1)}
            _dv_exempt = exempt_positions(_n_entries, _pcts_pos, _seat_manual)
            _dv_excluded = effective_excluded(_dv_excluded_raw, _dv_exempt, _acks)
            _dv_exempt_s7 = {p for p in _dv_exempt if _acks.get(p) != "keep_excluded"}
            if _dv_excluded_raw - _dv_excluded:
                logger.info("orchestrator[%s/u%s]: CF-5 豁免档 %s 摘出排除集(原 %s → 有效 %s)",
                            mode.name, user_id, sorted(_dv_exempt),
                            sorted(_dv_excluded_raw), sorted(_dv_excluded))
            if _seat_manual:
                _dv_featured, _dv_plan = manual_seat_plan(
                    _pcts_pos, _dv_excluded, config.TOP_N, _entry_bases)
                _dv_main_pos = sorted((p for p, v in _pcts_pos.items()
                                       if v > 0 and p not in _dv_excluded),
                                      key=lambda p: (-_pcts_pos[p], p))
                logger.info("orchestrator[%s/u%s]: SM 手动席位 %s(档1 %d 席,副档 %s)",
                            mode.name, user_id, {p: _pcts_pos[p] for p in _dv_main_pos},
                            _dv_plan.tier1_seats, _dv_plan.sub_seats)
            else:
                _boosted = await asyncio.to_thread(
                    db.recent_played_entry_positions, user_id, mode.name,
                    config.DIVERSITY_PLAY_WINDOW_SEC)
                _streaks = {i: (r["miss_streak"] or 0)
                            for i, r in enumerate(profile_rows, 1)}
                _forced = {i for i, s in _streaks.items()
                           if i >= 2 and s >= config.DIVERSITY_MISS_FORCE}
                _dv_featured = draw_featured_subs(
                    _n_entries, boosted=_boosted - {1}, forced=_forced,
                    excluded=_dv_excluded, miss_streaks=_streaks, seed=refresh_id)
                _dv_plan = DiversityPlan(
                    tier1_seats=config.DIVERSITY_TIER1_SEATS,
                    sub_seats={p: config.DIVERSITY_SUB_SEATS
                               * (2 if p in _boosted else 1) for p in _dv_featured},
                    cap=config.DIVERSITY_ENTRY_CAP,
                    entry_bases=_entry_bases)
                _dv_main_pos = [1] + _dv_featured
                logger.info("orchestrator[%s/u%s]: DV 抽签 主推档 %s(boost %s,forced %s)",
                            mode.name, user_id, [1] + _dv_featured,
                            sorted(_boosted - {1}), sorted(_forced))
        _lap("s0")
        plan, source = None, ""
        if intent_text:
            # ---- Lightweight intent refresh: planner, ratio and regular retrieval are
            # skipped entirely. The plan is only the intent search package
            # (INTENT_LITE_QUERIES queries); the result is INTENT_LITE_TOP overlay cards
            # merged into the current screen rather than a full refresh. Steady-state
            # reuse never applies (intent is one-off input).
            _queries = await intent_search.intent_queries(intent_text, lang)
            _search_left = await asyncio.to_thread(_search_remaining, trial)
            _intent_used = await asyncio.to_thread(db.user_search_used, user_id, "intent")
            _cap = min(config.INTENT_SEARCH_MAX,
                       max(config.INTENT_SEARCH_DAILY_MAX - _intent_used, 0))
            # Query local videos_fts first (zero API). Exclusions = explicit dislikes ∪
            # consumed videos (the one-time intent pass does not apply to local recall:
            # an already-watched library video is not worth an intent seat); the mode's
            # duration window applies.
            _local_ids: list[str] = []
            if config.INTENT_LOCAL_RECALL:
                from core.embed_text import segment as _segment
                _lex = (await asyncio.to_thread(db.explicit_dislike_video_ids, user_id, mode.name)
                        | (await asyncio.to_thread(db.consumed_suppress_sets, user_id))[0])
                _lrows = await asyncio.to_thread(
                    db.intent_local_candidates, _segment(intent_text), config.INTENT_LOCAL_K,
                    _lex, mode.duration_min_sec, mode.duration_max_sec)
                _local_ids = [r["video_id"] for r in _lrows]
                logger.info("vec[E2]: 意图本地召回 u%s 命中 %d(阈 %d)", user_id,
                            len(_local_ids), config.INTENT_LITE_TOP)
            plan, source = _lite_plan(_queries, _search_left, _cap, mode.name, _local_ids), "intent_lite"
            _n_pkg = (len(plan.strategies[-1].params["queries"])
                      if plan.strategies and plan.strategies[-1].tool == "search_by_keyword" else 0)
            if _n_pkg:
                await asyncio.to_thread(db.bump_user_search, user_id, "intent", _n_pkg)
            logger.info("orchestrator[%s/u%s]: IP-11 轻量意图刷新,搜索包 %d 词 本地 %d 条"
                        "(search_left=%d 用户意图余额=%d)", mode.name, user_id,
                        _n_pkg, len(_local_ids), _search_left,
                        config.INTENT_SEARCH_DAILY_MAX - _intent_used)
        elif _reuse_allowed(state, intent_text):
            # Reuse checks (TTL / decode / revalidation) live in reusable_plan; validate_plan is injected so stability has no agent dependency.
            plan = await asyncio.to_thread(
                stability.reusable_plan, user_id, mode, planner.validate_plan)
            if plan is not None:
                source = "reused"
                logger.info("orchestrator[%s/u%s]: steady 态,复用上次检索计划(跳过 S1)",
                            mode.name, user_id)
            else:
                logger.info("orchestrator[%s/u%s]: steady 但没得复用,视同 active 走 S1",
                            mode.name, user_id)
        # Active state and failed reuse both go through S1 here (intent refresh excluded).
        if plan is None:
            plan, source = await planner.make_plan(user_id, mode)      # S1
        logger.info("orchestrator[%s/u%s]: S1 source=%s,策略 %d 个",
                    mode.name, user_id, source, len(plan.strategies))
        _lap("s1")

        if not intent_text:
            # Source ratio is enforced in code rather than suggested to the LLM, including
            # on reused steady-state plans (otherwise long-time users keep a stale ratio).
            # Injection adds at most ~10 units, negligible against the budget gate, so the
            # plan is not revalidated. Intent refreshes have no channel/pool strategies.
            ratio, ratio_source = await asyncio.to_thread(
                effective_pool_ratio, user_id, mode.name)
            has_pref = bool(await asyncio.to_thread(db.top_channels, user_id, 1))
            plan = apply_pool_ratio(plan, ratio, has_pref)
            logger.info("orchestrator[%s/u%s]: MR-1 配比 %.2f(%s) has_pref=%s",
                        mode.name, user_id, ratio, ratio_source, has_pref)
            # The "my channels" track draws toward this refresh's featured entries: write
            # positions and mode into use_top_channels params (overwritten every refresh,
            # so reused plans are never stale).
            _mark_top_channels_featured(plan, _dv_main_pos, mode.name)
            # Reset hallucinated flags: the planner's LLM output has an intent_driven
            # field, and a hallucinated True would let regular retrieval claim the S3
            # intent exemption (intent packages are only created in _lite_plan).
            for _s in plan.strategies:
                _s.intent_driven = False

            # Profile search rotation (also on reused plans, otherwise long-time users
            # keep stale queries). Query order: this refresh's drawn sub-entries ->
            # entry 1 -> the rest by cursor (order_profile_queries consumes the cursor
            # itself, so the injector gets cursor 0 and takes from the head).
            # Each entry resolves to this refresh's query: the representative query and
            # its subtopics rotate on a per-entry cursor; with subtopics disabled
            # globally or by the user's subtopic_search setting, the representative
            # query is used. order/weighted only see the query list.
            _subs_by_eid = (await asyncio.to_thread(db.entry_subtopics_for_mode, user_id, mode.name)
                            if (config.PROFILE_SUBTOPICS and config.PROFILE_SUBTOPIC_SEARCH
                                and await asyncio.to_thread(db.get_user_subtopic_search, user_id))
                            else {})
            _all_q = _resolve_profile_queries(profile_rows, _subs_by_eid, lang)   # en prefers English queries
            if any(_all_q):
                prow = await asyncio.to_thread(db.get_mode_prefs,
                                               user_id, mode.name)
                cursor = (prow["search_cursor"] if prow is not None else 0) or 0
                if _seat_manual:                 # manual mode: share-weighted draw, cursor does not advance
                    pqueries = [q for q in weighted_profile_queries(
                        _all_q, _pcts_pos, seed=refresh_id, excluded=_dv_excluded) if q]
                else:
                    pqueries = [q for q in order_profile_queries(
                        _all_q, _dv_featured, cursor, excluded=_dv_excluded) if q]
                # Check the search bucket and the user's remaining daily profile-search budget before injecting.
                _pf_search_left = await asyncio.to_thread(_search_remaining, trial)
                _pf_used = await asyncio.to_thread(db.user_search_used, user_id, "profile")
                plan, n_inj = _inject_profile_searches(
                    plan, pqueries, 0, _pf_search_left,
                    max(config.PROFILE_SEARCH_DAILY_MAX - _pf_used, 0))
                if n_inj and not _seat_manual:
                    await asyncio.to_thread(db.advance_search_cursor,
                                            user_id, mode.name, n_inj)
                if n_inj and _subs_by_eid:
                    # Advance subtopic cursors only for entries actually injected (also in
                    # manual mode: these are per-entry cursors, independent of the
                    # mode-level search_cursor).
                    _consumed = _consumed_entry_ids(profile_rows, _all_q, pqueries[:n_inj])
                    await asyncio.to_thread(db.advance_entry_subtopic_cursor, user_id, _consumed)
                    logger.info("es[PF] u%s/%s: 注入 %s(方向游标 %s)", user_id, mode.name,
                                pqueries[:n_inj],
                                {r["entry_id"][:6]: _row_get(r, "subtopic_cursor")
                                 for r in profile_rows if r["entry_id"] in _consumed})
                if n_inj:
                    # Log whether entry 1 got a live search this refresh (hit-rate tracking).
                    _t1_q = _all_q[0] if _all_q else ""
                    _t1_hit = bool(_t1_q) and _t1_q in pqueries[:n_inj]
                    logger.info("orchestrator[%s/u%s]: PF 轮转注入 %d 词(游标 %d;档1 %s;%s)",
                                mode.name, user_id, n_inj, cursor,
                                "✓" if _t1_hit else "✗",
                                "手动占比" if _seat_manual else f"副档 {_dv_featured}")

            # Seed channel fetching: hand-picked channels of featured entries (entry 1
            # first), total capped at SEED_FETCH_CHANNELS_MAX; seeds without an owning
            # entry are naturally excluded by the entry_id argument.
            if profile_rows:
                _pos_ids = {i: r["entry_id"]
                            for i, r in enumerate(profile_rows, 1)}
                _seed_eids = [_pos_ids[p] for p in _dv_main_pos      # manual mode: ordered by share
                              if _pos_ids.get(p)]
                _seed_pairs = await asyncio.to_thread(
                    db.seed_channels_by_entry, user_id, _seed_eids,
                    config.SEED_FETCH_CHANNELS_MAX,
                    config.SEED_TIER1_FETCH_ALL) if _seed_eids else []   # all of entry 1's seeds go first
                if _seed_pairs:
                    _seed_cids = [c for _, c in _seed_pairs]
                    _sc_left = await asyncio.to_thread(db.quota_remaining)
                    plan, _n_seed = _inject_seed_channels(plan, _seed_cids, _sc_left)
                    if _n_seed:
                        logger.info("orchestrator[%s/u%s]: SC-4 注入 %d 种子频道",
                                    mode.name, user_id, _n_seed)
                    # Attach the featured seed-channel map to the seat plan whether or not
                    # the fetch was skipped by the budget gate (seed seats are claimed by
                    # channel_id, so the same channel arriving via the subscribed track
                    # also counts).
                    if _dv_plan is not None:
                        _dv_plan = _dc_replace(
                            _dv_plan,
                            seed_channels=_seed_plan_map(_seed_pairs, _pos_ids),
                            seed_seats=config.SEED_ENTRY_SEATS,
                            seed_seats_tier1=config.SEED_TIER1_SEATS)   # entry 1 has its own seed seat count

                # Attach followed channels to the seat plan: within each entry's reserved
                # seats the ranker takes seeds first, then followed channels (attribution
                # must still match the entry).
                if _dv_plan is not None:
                    _dv_plan = _dc_replace(
                        _dv_plan,
                        followed_channels=frozenset(
                            await asyncio.to_thread(db.subscribed_channel_ids, user_id)))

                # Expanded-channel supply track: for the same featured entries, take
                # EXPAND_FETCH_PER_ENTRY channels each by cursor rotation, capped at
                # EXPAND_FETCH_CHANNELS_MAX per refresh; skip followed channels and this
                # refresh's seeds (they have their own tracks). Cursors advance only on
                # successful injection, so a budget-gated batch retries next refresh.
                if _seed_eids:
                    _exp_excl = set(await asyncio.to_thread(db.subscribed_channel_ids, user_id))
                    _exp_excl |= {c for _, c in _seed_pairs}
                    _exp_pairs = await asyncio.to_thread(
                        db.expanded_channels_by_entry, user_id, _seed_eids,
                        config.EXPAND_FETCH_PER_ENTRY, config.EXPAND_FETCH_CHANNELS_MAX,
                        _exp_excl)
                    if _exp_pairs:
                        _xs_left = await asyncio.to_thread(db.quota_remaining)
                        plan, _n_exp = _inject_expanded_channels(
                            plan, [c for _, c in _exp_pairs], _xs_left)
                        if _n_exp:
                            _cnt: dict[str, int] = {}
                            for eid, _ in _exp_pairs[:_n_exp]:
                                _cnt[eid] = _cnt.get(eid, 0) + 1
                            for eid, k in _cnt.items():
                                await asyncio.to_thread(db.advance_expand_cursor, user_id, eid, k)
                            logger.info("orchestrator[%s/u%s]: XS-4 注入 %d 扩充频道(%s)",
                                        mode.name, user_id, _n_exp,
                                        {e[:6]: k for e, k in _cnt.items()})

            # Final search fallback: whatever the plan's origin (planner / reuse /
            # static), remaining non-intent searches are checked against the search
            # bucket floor and the user's daily budget (PROFILE_SEARCH_DAILY_MAX, rotation
            # and planner searches combined); if they pass, the budget is charged.
            _n_free = _count_free_searches(plan)
            if _n_free:
                _sl = await asyncio.to_thread(_search_remaining, trial)
                _pu = await asyncio.to_thread(db.user_search_used, user_id, "profile")
                if _sl < config.SEARCH_FLOOR_PROFILE or _pu + _n_free > config.PROFILE_SEARCH_DAILY_MAX:
                    plan, _n_rm = _strip_free_searches(plan)
                    logger.info("orchestrator[%s/u%s]: 搜索让路——剔除 %d 个自由 search"
                                "(search_left=%d 用户今日 %d/%d)", mode.name, user_id,
                                _n_rm, _sl, _pu, config.PROFILE_SEARCH_DAILY_MAX)
                else:
                    await asyncio.to_thread(db.bump_user_search, user_id, "profile", _n_free)

            # High-quality local candidate library: all seed channels of the mode ∪
            # top-K preferred channels, zero API; remove-then-append like the other
            # injectors; nothing is injected when the HQ set is empty.
            if config.LIBRARY_PER_REFRESH > 0:
                _lib_seeds = await asyncio.to_thread(db.seed_channels_for_mode, user_id, mode.name)
                _lib_top = [r["channel_id"] for r in await asyncio.to_thread(
                    db.top_channels, user_id, config.LIBRARY_TOP_K)]
                _feat_ids = (_featured_entry_ids(profile_rows, list(_dv_main_pos), set(_dv_excluded))
                             if profile_rows else [])                       # featured entries by seat allocation
                plan, _n_lib = _inject_library(plan, mode.name, _lib_seeds, _lib_top, _fit_fp, _feat_ids)
                if _n_lib:
                    logger.info("orchestrator[%s/u%s]: D5 候选库注入(HQ 频道 %d 个,上限 %d 条;EI 主推条目 %s)",
                                mode.name, user_id, _n_lib, config.LIBRARY_PER_REFRESH,
                                [e[:6] for e in _feat_ids])

        # ---- Suppression sets computed before S2: appeal exemptions, impression
        # penalty, explicit dislikes and consumed videos are passed into execute_plan
        # so every video S2 delivers is showable. Otherwise suppressed cards would take
        # ratio slots in S2 only to be thrown away, truncating good fresh candidates;
        # enrichment is not wasted on them either (impression tail seats are their only
        # way back on screen). Intent candidates are exempt on the S2 side; the
        # one-time-pass logic stays in _apply_consumed_filter below. The sets are
        # computed once and reused by the fallback filters below.
        _exempt_ids = await asyncio.to_thread(db.accepted_appeal_video_ids, user_id)
        _suppressed = await asyncio.to_thread(db.suppressed_video_ids, user_id) - _exempt_ids
        _neg_disliked = await asyncio.to_thread(
            db.explicit_dislike_video_ids, user_id, mode.name) - _exempt_ids
        _consumed, _pass_used = await asyncio.to_thread(
            db.consumed_suppress_sets, user_id)
        _consumed -= _exempt_ids

        # ---- S2: retrieval (per-refresh adapter; a missing API key fails loudly at construction) ----
        adapter = youtube.YouTubeAdapter(user_id=user_id)                             # per-refresh adapter, not a global singleton
        videos = await youtube.execute_plan(
            plan, adapter,
            excluded_video_ids=_suppressed | _neg_disliked | _consumed)               # per-strategy failures are contained in execute_plan
        per_strategy = dict(Counter(v.source_strategy for v in videos))               # result count per retrieval strategy
        logger.info("orchestrator[%s/u%s]: S2 %d 条,来源分布 %s",
                    mode.name, user_id, len(videos), per_strategy)
        _lap("s2")

        # ---- Appeal exemptions: an accepted appeal is explicit user intent, so those
        # videos skip S3-S7 and gem filtering and go straight to the head of top. They
        # are split off before S3, so the conservation LHS naturally excludes them; no
        # filter_results or quality_verdicts are written. Immediate visibility is
        # handled by pushing the card into card_pool when the appeal is accepted; this
        # path only guarantees the video is not filtered when retrieved again.
        # Appeal ids are subtracted from the suppression sets above: an explicit appeal
        # beats an implicit impression penalty.
        # (The suppression sets were already applied before S2; the three filters below
        # are a fallback, because S2 exempts intent candidates and future retrieval
        # paths might bypass execute_plan. Normally a near no-op.)

        # Impression penalty: videos shown in top >= 2 times without being opened stay
        # out of the fresh pipeline during the penalty window (re-entering would
        # multiply S6 LLM volume 5-8x), but they are not banned: after the final
        # snapshot _impression_tail_cards brings them back from the pool at the end of
        # the shown area. Applied before S3, so it is outside the conservation LHS and
        # writes no filter_results.
        if _suppressed:
            _n0 = len(videos)
            videos = [v for v in videos if v.video_id not in _suppressed]
            if _n0 - len(videos):
                logger.info("orchestrator[%s/u%s]: FB-2 曝光惩罚移除 %d 条",
                            mode.name, user_id, _n0 - len(videos))

        # Video-level explicit dislike: a (user, video, mode) with a submitted dislike
        # reason is never recommended (no TTL). Same placement and rules as above.
        if _neg_disliked:
            _n0 = len(videos)
            videos = [v for v in videos if v.video_id not in _neg_disliked]
            if _n0 - len(videos):
                logger.info("orchestrator[%s/u%s]: NP 显式禁推移除 %d 条",
                            mode.name, user_id, _n0 - len(videos))

        # Consumed-video suppression: videos watched in the last 7 days, or liked /
        # favorited in the last 30 days, are not recommended until the window expires.
        # Same placement and rules as above. Intent search hits get a one-time pass: the
        # first hit after consumption is allowed and showing it burns the pass
        # (last_served_at later than consumption); later intent searches are suppressed.
        # S2 exempts all intent candidates, so the pass is decided only here.
        if _consumed:
            _n0 = len(videos)
            videos = _apply_consumed_filter(videos, _consumed, _pass_used)
            if _n0 - len(videos):
                logger.info("orchestrator[%s/u%s]: DR 已消费禁推移除 %d 条",
                            mode.name, user_id, _n0 - len(videos))

        _exempt_cards: list[RecommendationCard] = []
        if _exempt_ids:
            _hit = [v for v in videos if v.video_id in _exempt_ids]
            if _hit:
                videos = [v for v in videos if v.video_id not in _exempt_ids]
                _reason = config.APPEAL_EXEMPT_REASON.get(
                    lang, config.APPEAL_EXEMPT_REASON["zh"])
                _exempt_cards = [RecommendationCard(
                    video=v, reason=_reason, fit_score=1.0, quality_norm=0.0,
                    pref_score=0.0, is_gem=False, final=1.0, section="top",
                    content_type=None) for v in _hit]
                logger.info("orchestrator[%s/u%s]: 申诉豁免 %d 条,跳过 S3~S7 直达 top",
                            mode.name, user_id, len(_exempt_cards))
                yield [c.model_copy(update={"section": "incoming", "final": None})
                       for c in _exempt_cards]

        # ---- S3: rule prefilter (duration, blacklist, ...); prefilter writes its own filter_results.
        kept = await asyncio.to_thread(prefilter, videos, mode, user_id, refresh_id)
        _lap("s3")

        # ---- S4: triage + ledger for the cached bucket + quality norm assignment. An
        # empty `kept` leaves all three buckets empty, so the later generators run zero
        # batches and zero LLM calls and short-circuit to an empty final snapshot.
        # Shares S4 logic with pipeline/viewing_plan_run.py: change both together.

        # triage is the pipeline's LLM cost gate: quality_verdicts 7-day cache hits go to
        # t.cached (pass or fail); misses get a statistical quality score and, if
        # eligible, go to t.direct without LLM review; the rest go to t.review for S5.
        # The per-mode direct-pass threshold is passed in (None -> QUALITY_PASS_THRESHOLD).
        t = await asyncio.to_thread(     # triage batch-reads the quality_verdicts cache
            triage, kept, direct_threshold=config.MODE_DIRECT_THRESHOLDS.get(mode.name))
        titles = await asyncio.to_thread(_channel_titles, kept)   # {channel_id: title}, one lookup per channel
        quality_passed: list[Video] = []                    # candidates that passed the quality gate
        norm: dict[str, float] = {}                         # quality norm per video_id, 0..1
        first_wave: list[Video] = []                        # direct + cached passes skip S5 and go to S6 at once

        # Direct-pass bucket: join the qualified pool, record the constant norm, join the first wave.
        for v in t.direct:
            quality_passed.append(v)
            norm[v.video_id] = config.DIRECT_PASS_QUALITY_NORM          # no LLM grade; constant between grade 1/3 and 2/3
            first_wave.append(v)

        # Cached bucket: (Video, QualityVerdict) pairs decoded from LLM verdicts within
        # 7 days. Triage only routes; verdicts and ledger writes belong to the orchestrator.
        for v, vd in t.cached:
            if vd.passed:                                     # passed: same treatment as direct pass
                quality_passed.append(v)
                norm[v.video_id] = quality_norm_of(vd, quality_score(v))  # cached grade -> grade/3; legacy rows -> statistical score
                first_wave.append(v)
            else:                                             # previously rejected by the LLM: record it
                await asyncio.to_thread(db.add_filter_result, user_id, refresh_id,
                                        v.video_id, "quality", "cached",
                                        "曾被质量审核拒绝(7 天内)")

        # First-wave truncation: keep the top FIRST_WAVE_MAX by statistical quality
        # score; the rest are deferred into the later-wave buffer (not dropped, no
        # filter record). Without it total latency exceeds the 60s budget.
        # _deferred_to_buf seeds the buffer in _pump_s5.
        # Vector reordering: profile-entry vectors x candidate vectors move the most
        # similar candidates forward in the first wave via RRF.
        _e7_sims: dict[str, float] = {}
        if config.E7_VECTOR and db.vector_available() and not intent_text:
            _groups = await asyncio.to_thread(
                db.entry_vec_groups_for_mode, user_id, mode.name, bool(config.ES_VECTOR))   # grouped by entry
            if _groups:
                _e7_sims = await asyncio.to_thread(
                    db.video_sims_max, _groups, [v.video_id for v in first_wave],
                    float(config.VEC_SUBTOPIC_WEIGHT))
                logger.info("vec[E7]: 首波相似度重排 u%s/%s 有向量 %d/%d(条目 %d,方向向量 %d)",
                            user_id, mode.name, len(_e7_sims), len(first_wave),
                            len(_groups), sum(len(g.subs) for g in _groups))
        first_wave, _deferred_to_buf = _truncate_first_wave(
            first_wave, quality_score, config.FIRST_WAVE_MAX,
            intent_head=config.INTENT_FIRST_WAVE_MAX if intent_text else 0,
            sims=_e7_sims or None)

        _lap("s4")
        # ---- S6 wave machine (closures share candidates/reasons for the final rank and snapshot) ----
        candidates: list[tuple[Video, float, float, str | None]] = []       # (video, fit score, quality norm, content_type)
        entry_map: dict[str, int] = {}       # video_id -> profile entry attribution (clamped)
        reasons: dict[str, str] = {}                                        # {video_id: fit reason}
        type_oov: Counter = Counter()                                       # out-of-vocabulary types, aggregated per refresh

        # Nested in run() so it can close over run()'s local state (candidates, reasons, norm, titles, ...).
        async def _fit_wave(wave: list[Video], *, allow_escalation: bool = True,
                            ) -> AsyncIterator[list[RecommendationCard]]:
            """Run one wave of candidates through S6. Verdicts are consumed via
            fit_effective: shown candidates become incoming cards, yielded at once and
            added to the final candidates; rejected ones are recorded by rule.
            allow_escalation is passed to the judges (False for the first wave:
            first-screen latency matters more than that batch's verdict quality)."""
            by_id = {v.video_id: v for v in wave}                                           # verdicts carry only video_id; map back to Video
            # Pools split by criterion: intent recall is judged as its own batch with an
            # intent section prepended to the user prompt; the regular pool uses
            # fit_{mode.name}. The system prompt is unchanged. The intent pool goes first
            # so the first incoming cards are what the user asked for.
            _pools = _judge_pools(wave, intent_text)

            async def _fit_pools(pools):
                for subwave, itext in pools:
                    if not subwave:
                        continue
                    def _judge(sw, _itext=itext):
                        return judges.judge_mode_fit(
                            sw, mode, channel_titles=titles,
                            allow_escalation=allow_escalation,
                            lang=lang, intent_text=_itext,
                            profile_lines=(None if _itext or not profile_rows
                                          else [r["text"] for r in profile_rows]),
                            neg_pref_lines=(None if _itext or not neg_rows
                                           else [r["text"] for r in neg_rows]))
                    # Regular pool checks the fit cache first: hits are yielded first (no LLM), misses go to the judge and are written back.
                    async for b in _fit_with_cache(
                            subwave, itext, judge=_judge, user_id=user_id,
                            mode_name=mode.name, fingerprint=_fit_fp, stats=_fit_stats):
                        yield b

            # judge_mode_fit splits into LLM_BATCH_SIZE chunks (one LLM call each, prompt per mode) and yields verdict lists as they complete.
            async for batch in _fit_pools(_pools):
                cards: list[RecommendationCard] = []
                for vd in batch:
                    v = by_id[vd.video_id]
                    show, rule = fit_effective(vd)
                    if show:                                                           # shown
                        # Normalize the type spelling and count out-of-vocabulary types
                        # (the normalized value still goes on the card; type weighting
                        # falls back to 0.6 for unknown types).
                        ctype = vd.content_type
                        if ctype is not None:
                            ctype = config.normalize_content_type(ctype)
                            if ctype not in _KNOWN_TYPES:
                                type_oov[ctype] += 1
                        reasons[v.video_id] = vd.reason                                # kept for the final snapshot
                        candidates.append((v, vd.fit_score, norm[v.video_id], ctype))  # into the final candidate pool
                        # Entry attribution (intent batches get no profile section, so profile_entry is None there).
                        _entry = _clamp_entry(getattr(vd, "profile_entry", None),
                                              len(profile_rows))
                        if _entry is not None:
                            entry_map[v.video_id] = _entry
                        # Incoming card: pref_score is a placeholder (computed at final rank), final=None marks it unranked, the UI fades it in by arrival order.
                        cards.append(RecommendationCard(
                            video=v, reason=vd.reason, fit_score=vd.fit_score,
                            quality_norm=norm[v.video_id], pref_score=0.0,  # placeholder; computed at final rank
                            is_gem=v.source_strategy == "hidden_gem",
                            final=None, section="incoming",
                            content_type=ctype))                            # incoming cards already carry the type
                    # Rejected: record a filter result with stage='mode' and the rule from
                    # fit_effective (mode_unfit / mode_degraded_reject / fit_below_threshold).
                    # Threshold rejections do not reuse vd.reason: that is a *recommendation*
                    # reason generated with fit=True and would mislead later review.
                    else:
                        note = (f"fit 分 {vd.fit_score:.2f} 低于模式匹配阈值"
                                if rule == "fit_below_threshold"
                                else vd.reason or "不匹配当前模式")
                        if getattr(vd, "cached", False):
                            note = "[缓存] " + note          # same rule name; note marks cached verdicts
                        await asyncio.to_thread(db.add_filter_result, user_id,
                                                refresh_id, vd.video_id, "mode",
                                                rule, note)
                if cards:                    # an all-unfit batch yields no empty frame
                    yield cards

        # ---- The first wave runs in parallel with S5: they are independent (the first
        # wave judges direct + cached passes, S5 reviews the review bucket; disjoint sets,
        # no mutual inputs). Run serially they exceeded the 60s budget; in parallel the
        # cost is max(first wave, S5) + later waves. Later S6 waves still wait for S5
        # (a real dependency, kept inside _pump_s5).
        # Producer/consumer: two producer tasks put card batches on a queue and the
        # generator body consumes and yields them; each producer puts a sentinel when
        # done, and the consumer stops after counting all sentinels.
        # The S5 queue is intent-first: judge_quality batches in order, so intent
        # reviews finish first and reach the S6 intent batch sooner.
        _review = _review_intent_first(t.review, quality_score)
        stat_scores = {v.video_id: quality_score(v) for v in _review}              # statistical quality score per review candidate
        review_by_id = {v.video_id: v for v in _review}                            # {video_id: Video} lookup
        n_llm_judged = 0                                                            # real LLM verdict count, for cost observability
        _q: asyncio.Queue = asyncio.Queue()          # unbounded: batch count is finite and sentinels close it
        _SENTINEL = object()
        _tm["s6_first"] = 0.0
        _tm["s5"] = 0.0
        _tm["s6_waves"] = 0.0

        async def _pump_first_wave() -> None:
            """Producer 1: first wave = direct + cached passes, without waiting for S5.
            The only allowed partial batch (first screen first). Escalation is disabled
            for the first wave, bounding LLM calls for a bad batch at 2: first-screen
            latency beats that batch's verdict quality."""
            _t = time.perf_counter()
            try:
                async for cards in _fit_wave(first_wave, allow_escalation=False):
                    await _q.put(cards)
            finally:
                _tm["s6_first"] = time.perf_counter() - _t
                await _q.put(_SENTINEL)

        async def _pump_s5() -> None:
            """Producer 2: S5 quality review, consumed batch by batch with ledger
            writes; passes accumulate until LLM_BATCH_SIZE and are sent as later S6
            waves. buf starts with candidates deferred by first-wave truncation
            (deferred, not dropped; conservation still holds)."""
            # Shares S5 logic with pipeline/viewing_plan_run.py: change both together.
            nonlocal n_llm_judged
            _t = time.perf_counter()
            buf: list[Video] = list(_deferred_to_buf)
            _intent_flushed = False          # latch: at most one early intent batch per refresh
            try:
                # judge_quality splits the review bucket into batches, awaits one LLM call each and yields each batch as it completes.
                async for batch in judges.judge_quality(_review, mode,
                                                        stat_scores=stat_scores,
                                                        channel_titles=titles):
                    for vd in batch:
                        # degraded=True marks a fallback verdict (only the degrade helpers
                        # set it). Degraded verdicts are not cached, so a cache entry always
                        # means "the LLM actually judged this". confidence==0.0 is not used
                        # as the signal: the LLM may legitimately return confidence 0.
                        if not vd.degraded:                       # only real verdicts are cached and counted
                            await asyncio.to_thread(db.save_quality_verdict, vd)
                            n_llm_judged += 1
                        v = review_by_id[vd.video_id]             # verdict -> Video
                        # Passed: join the qualified pool, record the quality norm
                        # (llm_grade -> grade/3; degraded_pass -> statistical score), queue for S6.
                        if vd.passed:
                            quality_passed.append(v)
                            norm[v.video_id] = quality_norm_of(vd, quality_score(v))
                            buf.append(v)
                        # Filter rule name: LLM-provided low-quality categories joined by
                        # commas (e.g. "clickbait,empty"); else "degraded_reject" for a
                        # fallback rejection; else "unspecified" (rejected without category).
                        else:
                            rule = (",".join(vd.categories) if vd.categories
                                    else "degraded_reject" if vd.degraded
                                    else "unspecified")
                            await asyncio.to_thread(db.add_filter_result, user_id,
                                                    refresh_id, vd.video_id,
                                                    "quality", rule, vd.reason)

                    # Early intent partial batch (at most once per refresh): the review
                    # queue is intent-first, so the first S5 batch's passes are mostly
                    # intent candidates. Instead of waiting for a full batch, drain them
                    # from buf and send an S6 wave now. This is the second exception to
                    # "full batches only" (the first is the first wave), for the same
                    # reason: the first screen beats partial-batch cost.
                    if intent_text and not _intent_flushed:
                        _ib, buf = _drain_intent_head(buf)
                        if _ib:
                            _intent_flushed = True
                            _tw = time.perf_counter()
                            async for cards in _fit_wave(_ib):
                                await _q.put(cards)
                            _tm["s6_waves"] += time.perf_counter() - _tw

                    # After each S5 batch, send all full batches in buf at once so the
                    # judge's inter-batch concurrency actually kicks in (one batch per wave
                    # would serialize them). After draining, buf is shorter than one batch,
                    # so `if` is sufficient.
                    if len(buf) >= config.LLM_BATCH_SIZE:          # send only full batches to avoid partial-batch cost
                        _wave, buf = _drain_whole_batches(buf, config.LLM_BATCH_SIZE)
                        _tw = time.perf_counter()
                        async for cards in _fit_wave(_wave):
                            await _q.put(cards)
                        _tm["s6_waves"] += time.perf_counter() - _tw
                if buf:                                           # flush the tail batch
                    _tw = time.perf_counter()
                    async for cards in _fit_wave(buf):
                        await _q.put(cards)
                    _tm["s6_waves"] += time.perf_counter() - _tw
            finally:
                # Timing: s5 = this producer's total time minus the S6 waves it sent
                # (S5 and later waves are serial within _pump_s5). First-wave time is
                # recorded separately, so cross-producer overlap does not affect this.
                _tm["s5"] = (time.perf_counter() - _t) - _tm["s6_waves"]
                await _q.put(_SENTINEL)

        _producers = [asyncio.create_task(_pump_first_wave()),
                      asyncio.create_task(_pump_s5())]
        try:
            _finished = 0
            while _finished < len(_producers):
                item = await _q.get()
                if item is _SENTINEL:
                    _finished += 1
                else:
                    yield item
            for p in _producers:
                await p                       # surface producer exceptions instead of swallowing them
        finally:
            # Cancellation: this block always runs when the generator is closed or
            # cancelled and explicitly cancels unfinished producers; otherwise orphan
            # tasks would keep burning tokens. try/finally is used instead of TaskGroup
            # because we must yield *while* producers run, and yielding inside a
            # TaskGroup context breaks cancellation semantics (see PEP 789).
            # CancelledError is a BaseException, so the judges' `except Exception` does
            # not catch it; cancellation reaches the judge layer, whose own finally
            # cancels and gathers its batch tasks.
            for p in _producers:
                if not p.done():
                    p.cancel()
            await asyncio.gather(*_producers, return_exceptions=True)
        _last = time.perf_counter()          # reset the lap cursor for sequential timing
        logger.info("orchestrator[%s/u%s]: S4/S5 direct %d / review %d / cached %d "
                    "→ 质量通过 %d(LLM 实判 %d);S6 缓存命中 %d / 实判 %d", mode.name, user_id,
                    len(t.direct), len(t.review), len(t.cached), len(quality_passed), n_llm_judged,
                    _fit_stats["cached"], _fit_stats["llm"])

        # ---- Final: one rank pass + snapshot assembly
        _flush_type_oov(mode, type_oov)              # one aggregated out-of-vocabulary warning

        # Permanently excluded types are removed before rank: the semantics are "never
        # appear", not "sink to the bottom". They are recorded in the same ledger with
        # stage='S7' for transparency (visible in the filtered sidebar, appealable). An
        # empty exclusion set is the common case and takes the fast path.
        excluded_types = await asyncio.to_thread(load_excluded_types, user_id, mode.name)
        if excluded_types:
            # Cards attributed to an exempt entry pass through: the seat side exempts
            # them, so S7 must too, or those seats sit empty and searches are wasted.
            # Attribution comes from the S6 judge's entry_map; cards without an
            # attribution are not exempt. Passed cards still get an S7 ledger row
            # (exempt_top_entry) for transparency.
            _exempt_ids = ({vid for vid, e in entry_map.items() if e in _dv_exempt_s7}
                           if _dv_exempt_s7 else frozenset())
            candidates, dropped, exempted = partition_excluded(
                candidates, excluded_types, exempt_ids=_exempt_ids)
            for v, _fit_score, _quality_norm, ctype in dropped:
                await asyncio.to_thread(db.add_filter_result, user_id, refresh_id,
                                        v.video_id, "S7", "user_excluded_type",
                                        f"用户排除类型:{ctype}")
            for v, _fit_score, _quality_norm, ctype in exempted:
                await asyncio.to_thread(db.add_filter_result, user_id, refresh_id,
                                        v.video_id, "S7", "exempt_top_entry",
                                        f"豁免档放行(档{entry_map.get(v.video_id)},类型 {ctype})")
            logger.info(
                "orchestrator[%s/u%s]: S7 用户排除类型过滤 %d 条,豁免档放行 %d 条(排除集 %s,豁免档 %s)",
                mode.name, user_id, len(dropped), len(exempted), sorted(excluded_types),
                sorted(_dv_exempt_s7))

        if intent_text:
            # ---- Lightweight intent final snapshot: no rank, no mixing; take the top
            # INTENT_LITE_TOP overlay cards by fit + quality. The frontend overlays them
            # at the head of the current screen; regular content is not refreshed.
            # Excluded types were already filtered above (explicit user intent, like a
            # blacklist; intent gets no exemption). Unpicked candidates get an S7 filter
            # record so conservation holds and they stay visible and appealable.
            _lite_cards = _lite_snapshot(candidates, reasons)
            _picked_ids = {c.video.video_id for c in _lite_cards}
            for _v, _f, _q, _ct in candidates:
                if _v.video_id not in _picked_ids:
                    await asyncio.to_thread(
                        db.add_filter_result, user_id, refresh_id, _v.video_id,
                        "S7", "intent_lite_overflow",
                        f"意图轻快照仅取贴合+质量前 {config.INTENT_LITE_TOP} 张")
            # ---- Top-area fill: intent cards at the head, remaining seats filled from
            # the chip shelf of the user's top two type buckets (type rotation, zero
            # quota, zero LLM). Fill cards are outside conservation (never entered S3);
            # the top/more conservation numbers cover the intent cards only.
            _fill_cards = await asyncio.to_thread(
                _intent_fill_cards, user_id, mode, lang,
                config.TOP_N - len(_lite_cards),
                {c.video.video_id for c in _lite_cards}, excluded_types)
            snapshot = _exempt_cards + _lite_cards + _fill_cards
            top, more = _lite_cards, []          # ledger semantics: shown = len(top)
            logger.info("orchestrator[%s/u%s]: IP-11 轻快照 %d 张(候选 %d)"
                        "+ 前两档补位 %d 张(类型 %s)",
                        mode.name, user_id, len(_lite_cards), len(candidates),
                        len(_fill_cards),
                        sorted({c.content_type for c in _fill_cards}))
        else:
            # Type buckets are injected as a parameter: effective_type_buckets applies the
            # user's per-mode overrides on top of the default table (fail-soft to default).
            type_buckets = await asyncio.to_thread(effective_type_buckets, user_id, mode.name)
            top, more = await asyncio.to_thread(     # rank reads subscriber counts/prefs, writes gem-filter records
                rank, user_id, refresh_id, candidates,
                type_buckets=type_buckets, diversity_plan=_dv_plan,
                entry_map=entry_map)                  # seat plan and entry attribution injected
            if _dv_plan is not None and not _seat_manual:   # advance miss-streak ledger (prewarm too); not in manual mode
                await asyncio.to_thread(
                    db.bump_entry_feature_ledger, user_id, mode.name, _dv_featured)
            top, more = enforce_source_cap(top, more)     # single source <= 50% (swap only; cap first)
            top, more = enforce_source_mix(top, more)     # soft per-source minimums on screen (swap only)
            logger.info("orchestrator[%s/u%s]: 屏上来源构成 %s(频道 %d 个,类型 %d 种)",
                        mode.name, user_id, dict(Counter(it.video.source_strategy for it in top)),
                        len({it.video.channel_id for it in top}),
                        len({it.content_type for it in top if it.content_type}))
            # Impression tail seats: penalized cards are pulled from the pool into the
            # last IMPRESSION_TAIL_SEATS positions when fresh results fall short of
            # TOP_N. Prewarm (consume_intent=False) skips this: it only stocks fresh
            # supply, and tail seats are added at serve time (here and in the pool read
            # path) to avoid clashing with the shelf's suppression filter.
            _tail_cards: list[RecommendationCard] = []
            if consume_intent:
                # Fixed seats: reserve IMPRESSION_TAIL_FIXED_SEATS even when fresh results fill TOP_N.
                _seats = min(config.IMPRESSION_TAIL_SEATS,
                             max(config.IMPRESSION_TAIL_FIXED_SEATS, config.TOP_N - len(top)))
                if _seats > 0:
                    _shown = ({it.video.video_id for it in top}
                              | {it.video.video_id for it in more}
                              | {c.video.video_id for c in _exempt_cards})
                    _tail_cards = await asyncio.to_thread(
                        _impression_tail_cards, user_id, mode.name, lang,
                        _seats, _shown, excluded_types,
                        dict(Counter(it.video.channel_id for it in top)))
                    if _tail_cards:
                        _spill = len(top) + len(_tail_cards) - config.TOP_N   # fixed seats push the tail of top into the collapsed area
                        if _spill > 0:
                            more = top[-_spill:] + more
                            top = top[:-_spill]
                        logger.info("orchestrator[%s/u%s]: EP 尾席注入 %d 条",
                                    mode.name, user_id, len(_tail_cards))
            # Translate RankedItems into RecommendationCards (top / more): this is the final snapshot of the yield contract. Tail cards sit after top and before the collapsed area and are outside conservation and the shown count.
            snapshot = (_exempt_cards
                        + [_final_card(it, "top", reasons) for it in top]
                        + _tail_cards
                        + [_final_card(it, "more", reasons) for it in more])
        _lap("s7_rank")

        # ---- Plan log + conservation check. Bookkeeping happens before the final
        # yield: once consumers receive the last frame they may stop iterating, and
        # code after a yield only runs on the next anext.
        # The whole bookkeeping block is wrapped in try/except: any failure (e.g. disk
        # I/O) loses only the bookkeeping, never the final frame, which is yielded
        # outside this try. CancelledError is a BaseException and is not caught here,
        # so it still propagates to the outer cancellation handler.
        try:
            # Quality pass rate = quality passes / S3 survivors.
            pass_rate = len(quality_passed) / len(kept) if kept else 0.0
            await asyncio.to_thread(
                db.save_retrieval_plan,
                user_id, mode.name,
                json.dumps({"source": source, "plan": plan.model_dump(),
                            "shown": len(top),                         # shown = len(top)
                            "intent": intent_text},                    # intent text recorded in the plan JSON
                           ensure_ascii=False),
                per_strategy_counts=per_strategy, pass_rate=pass_rate,
                reused=(source == "reused"))                # set on steady-state reuse, for reuse-rate analysis

            # Conservation: S3 input == top + more + filter records (S3 prefilter, cached rejects, S5 rejects, S6 rejects, S7 exclusions, gem filter in rank).
            report = await asyncio.to_thread(     # reads filter_results
                conservation_check, refresh_id, len(videos), len(top), len(more))
            if report["ok"]:
                logger.info("orchestrator[%s/u%s]: 守恒成立 %d == top %d + more %d + 过滤 %d",
                            mode.name, user_id, report["lhs"], report["n_top"],
                            report["n_more"], report["n_filtered"])
            else:
                # A broken invariant is logged as ERROR for investigation without failing the refresh; assertions belong to tests.
                logger.error("orchestrator[%s/u%s]: 守恒破裂!S3 输入 %d != top+more+过滤 %d"
                             "(top %d / more %d / 过滤 %d),refresh_id=%s",
                             mode.name, user_id, report["lhs"], report["rhs"],
                             report["n_top"], report["n_more"], report["n_filtered"], refresh_id)

            # Usage accounting: yt_spent is the difference in quota_used() before and
            # after the refresh, i.e. YouTube units actually spent from the general pool
            # (search calls are counted per path in the user's daily usage). The quota
            # counter is global, so the difference is the only way to attribute spend to
            # one refresh; max(0, ...) guards against negatives from concurrent refreshes.
            # meter.input_tokens / output_tokens are exact sums over all judge and
            # planner LLM calls in this refresh.
            yt_spent = max(0, await asyncio.to_thread(db.quota_used) - ytu0)
            await asyncio.to_thread(db.bump_user_usage, user_id, yt_spent,
                                    meter.input_tokens, meter.output_tokens)

            _lap("ledger")
        except Exception as exc:
            # Exceptions here may carry a full request URL including the API key, or
            # SQL/column names from sqlite3. exc_info=True would bypass redaction, so the
            # traceback is formatted manually and redacted as a whole (frames are kept
            # intact for debugging).
            logger.error(
                "终局落账失败(refresh_id=%s)——账丢失但终局帧照发:%s",
                refresh_id, redact("".join(traceback.format_exception(exc))))

        # Instant first screen: store the final cards in the pool for fast serving next time. Fail-soft; never blocks the refresh.
        await asyncio.to_thread(
            _write_pool_safe, user_id, mode.name, lang, _snapshot_to_pool_rows(snapshot))

        logger.info(
            "orchestrator[%s/u%s]: P-1 耗时分解 total=%.1fs | S0=%.2f S1=%.2f S2=%.2f "
            "S3=%.2f S4=%.2f S6首波=%.2f S5质量判官=%.2f S6后续波=%.2f S7+rank=%.2f "
            "落账=%.2f(单位秒;首波与 S5 并行,各段之和 > 墙钟时长属正常,spec §3.3;"
            "含 SSE 消费方背压;LLM 实判 %d 条)",
            mode.name, user_id, time.perf_counter() - _t0,
            _tm.get("s0", 0.0), _tm.get("s1", 0.0), _tm.get("s2", 0.0),
            _tm.get("s3", 0.0), _tm.get("s4", 0.0), _tm.get("s6_first", 0.0),
            _tm.get("s5", 0.0), _tm.get("s6_waves", 0.0), _tm.get("s7_rank", 0.0),
            _tm.get("ledger", 0.0), n_llm_judged)

        yield snapshot                       # last frame: full final snapshot (top + more, with final scores)
    except asyncio.CancelledError:
        logger.info("刷新被取消(refresh_id=%s)", refresh_id)
        raise
