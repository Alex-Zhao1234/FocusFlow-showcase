"""Retrieval planning — pipeline stage S1.

The first LLM node of a refresh: the "plan" half of plan-then-execute.
``build_plan_context`` reads user state from the database, the LLM produces a
``RetrievalPlan``, ``validate_plan`` checks it in code, a failed plan is
re-planned once with the failure reason attached, and if it still fails (or
the LLM errors) a deep copy of ``DEFAULT_PLANS[mode]`` is used. ``make_plan``
returns ``(plan, source)``.

Boundaries:

- This module never imports ``sources/``: the LLM only produces a plan and
  never calls external APIs itself (planning is separated from execution).
- No raw SQL: all state is read through storage-layer functions. The
  agent-to-db edge is read-only and confined to this module (the judges are
  db-free); persisting the plan log is the orchestrator's job.

Degradation: every planner failure (missing API key, network, timeout,
exhausted schema retries, two failed validations) falls back to
``DEFAULT_PLANS``, so S1 always has a code-level fallback and the product never
shows an empty screen. Failures are logged at error level and
``source='default'`` makes the degradation rate measurable.
"""
import json
import logging

from pydantic_ai import Agent

from agent import _model_id, _model_settings
from agent.prompts import PROMPTS
import config                                       # toggles read via attribute access so they can change at runtime
from config import (
    DEFAULT_PLANS, LLM_MAX_SCHEMA_RETRIES, LLM_MODEL, LLM_TIMEOUT_SEC,
    DAILY_SEARCH_CALLS, SEARCH_FLOOR_PROFILE,
    MAX_SEARCH_PER_REFRESH, PLAN_MAX_STRATEGIES, PLAN_MIN_STRATEGIES,
    PLAN_QUOTA_BUDGET_RATIO, PLANNER_REPLAN_MAX,
)
from core import llmmeter                       # token-usage metering for the planning stage
from core.costing import estimate_plan_cost
from core.logsafe import safe_repr              # redact secrets from exceptions before logging
from core.models import Mode, PlanContext, RetrievalPlan
from storage import db

from pipeline.prefs import effective_pool_ratio, load_excluded_types

logger = logging.getLogger(__name__)


def allocate_channel_slots(user_rows: list[dict], pool_rows: list[dict],
                           total: int, pool_ratio: float) -> list[dict]:
    """Split ``total`` channel slots between the user side and the pool side.

    ``user_rows`` are the user's own preferred channels (real pref scores);
    ``pool_rows`` are global type-pool channels (neutral 0.5 score, already
    de-duplicated against the user side). Slots are split by ``pool_ratio``;
    if either side runs short, *all* of its unused slots go to the other side.
    A new user with almost no preferences therefore naturally gets a nearly
    all-pool list — matching "the system doesn't know you yet". If both sides
    run short the list is shorter than ``total``, which the planner accepts.

    ``pool_ratio`` spans [0.0, 1.0] inclusive: 0.0 means only my channels,
    1.0 means only the curated pool (pure exploration).
    The result lists user rows first, then pool rows (the order shown to the LLM).
    """
    pool_slots = round(total * pool_ratio)
    user_slots = total - pool_slots
    taken_user = user_rows[:user_slots]
    taken_pool = pool_rows[:pool_slots]

    if len(taken_user) < user_slots:
        taken_pool = pool_rows[:pool_slots + (user_slots - len(taken_user))]
    elif len(taken_pool) < pool_slots:
        taken_user = user_rows[:user_slots + (pool_slots - len(taken_pool))]
    return taken_user + taken_pool


def _pool_candidates(user_id: int, mode: Mode, exclude_ids: set) -> list[dict]:
    """Pool-side channel candidates for the current mode.

    Keeps type-pool channels whose content types fall in the mode's top two
    type tiers (minus types the user excluded), drops channels in
    ``exclude_ids``, blacklisted channels, and any channel the user already
    has a learned preference score for, then sorts by "matches a chip the user
    picked" first and subscriber count descending.

    Pool channels get a neutral ``pref_score`` of 0.5 *in memory only* — it is
    never written to the user's channel preferences.
    """
    tiers = config.MODE_TYPE_BUCKETS.get(mode.name, [])
    allowed = {b for tier in tiers[:2] for b in tier}   # base types admitted for this mode
    excluded = load_excluded_types(user_id, mode.name)
    allowed -= excluded
    prefer = set(db.get_user_chips(user_id))

    # Also exclude every channel with a learned score (not just the top 15):
    # re-entering it from the pool at 0.5 would silently erase what was learned.
    exclude_ids = (exclude_ids | db.blacklisted_channel_ids(user_id)
                  | {r["channel_id"] for r in db.list_channel_prefs(user_id)})
    rows = db.pool_channels_for_mode(user_id, allowed, exclude_ids, list(prefer), 0)  # limit 0 = unlimited; filtered below

    scored = []
    for r in rows:
        try:
            chips = json.loads(r["chip_tags"]) or []
        except (TypeError, ValueError):
            continue
        bases = {config.INTEREST_CHIPS[c][1] for c in chips if c in config.INTEREST_CHIPS}  # chip -> base content type
        if not (bases & allowed):
            continue                             # wrong mode, or excluded by the user

        hit_prefer = bool(prefer & set(chips))
        # Sort key: chip hits first (0 before 1), then subscribers descending
        # (negated so a single ascending sort works).
        scored.append((0 if hit_prefer else 1, -(r["channel_subs"] or 0),
                       {"channel_id": r["channel_id"],
                        "channel_title": r["channel_title"],
                        "pref_score": 0.5}))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in scored]


def _mode_summary(mode: Mode) -> dict:
    """Summarise a Mode for the planner prompt.

    Kept as a pure function in the agent layer rather than a Mode method:
    what the LLM sees is prompt-supply logic and evolves with prompt tuning.
    Blacklist keywords are deliberately omitted — blacklisting is S3's job,
    and feeding them to the planner wastes tokens and invites it to make
    filtering decisions it shouldn't.
    """
    return {
        "name": mode.name,
        "display_name": mode.display_name,
        "keywords": list(mode.keywords),
        "yt_category_ids": list(mode.yt_category_ids),
        "duration_window": [mode.duration_min_sec, mode.duration_max_sec],    # max=None means no upper bound
        "enable_discovery": mode.enable_discovery,
        "strict_mode": mode.strict_mode,
    }


def build_plan_context(user_id: int, mode: Mode) -> PlanContext:
    """Read everything the planner needs into a ``PlanContext`` (read-only).

    Covers the mode summary, candidate channels, recent feedback, quota, the
    last plan, the persistent user profile and the active session intent —
    all via storage-layer functions. Every ``sqlite3.Row`` is converted to
    plain types here, because the context is serialised into the prompt and
    any leftover Row would break ``model_dump_json``.
    """
    ratio, _ratio_source = effective_pool_ratio(user_id, mode.name)

    # User side: the user's 15 highest-scored channels.
    user_rows = [
        {"channel_id": r["channel_id"], "channel_title": r["channel_title"],
         "pref_score": r["pref_score"]}
        for r in db.top_channels(user_id, limit=15)
    ]
    pool_rows = _pool_candidates(user_id, mode, {c["channel_id"] for c in user_rows})
    channels = allocate_channel_slots(user_rows, pool_rows, 15, ratio)

    # Last plan for this user/mode (None on cold start). Treated as advisory:
    # malformed stored JSON is logged and nulled rather than failing the refresh.
    row = db.get_latest_retrieval_plan(user_id, mode.name)
    last_plan = None

    if row is not None:
        plan_payload = None
        if row["plan_json"]:
            try:
                parsed = json.loads(row["plan_json"])
                # Stored as a wrapper {"source", "plan", "shown"}; feed only the
                # plan itself. Bare legacy plans are accepted as-is.
                plan_payload = (parsed.get("plan", parsed)
                                if isinstance(parsed, dict) else parsed)

            except (TypeError, ValueError):
                logger.warning("build_plan_context[%s/u%s]: 上次 plan_json 反解失败"
                               "(坏 JSON),last_plan.plan 置 None", mode.name, user_id)

        counts = None
        if row["per_strategy_counts"]:
            # Videos actually retrieved per strategy last time, e.g.
            # {"search_by_keyword": 6, "fetch_channel_videos": 12}.
            try:
                counts = json.loads(row["per_strategy_counts"])
            except (TypeError, ValueError):
                logger.warning("build_plan_context[%s/u%s]: per_strategy_counts 反解"
                               "失败,置 None", mode.name, user_id)
        last_plan = {
            "plan": plan_payload,                    # may be None
            "per_strategy_counts": counts,           # may be None
            "pass_rate": row["pass_rate"],
            "reused": row["reused"],                 # 0/1: whether a cached plan was reused
        }

    # Persistent profile: {} when disabled or absent. Like last_plan it is
    # advisory, so every failure path is fail-soft.
    profile: dict = {}
    if config.ENABLE_USER_PROFILE:
        prow = db.get_user_profile(user_id)
        if prow is not None:
            if prow["profile_json"]:
                try:
                    parsed = json.loads(prow["profile_json"])
                    profile = parsed if isinstance(parsed, dict) else {}
                except (TypeError, ValueError):
                    logger.warning("build_plan_context[u%s]: profile_json 反解失败"
                                   "(坏 JSON),user_profile 置 {}", user_id)
            # profile_json is the LLM-structured form of raw_text; if
            # structuring failed, pass a truncated raw_text as a free note.
            elif prow["raw_text"]:
                profile = {"free_note": prow["raw_text"][:config.PROFILE_NOTE_MAX]}

    # Session intent: the natural-language request the user declared for this mode.
    intent = ""
    if config.ENABLE_SESSION_INTENT:
        irow = db.get_active_intent(user_id, mode.name)
        if irow is not None:
            intent = irow["text"]

    return PlanContext(
        mode_config=_mode_summary(mode),
        channels=channels,
        feedback=db.aggregate_recent_feedback(user_id),      # {channel_id: [likes, dislikes]}
        quota_left=db.quota_remaining(),                     # general API units left today
        search_left=db.quota_remaining("search"),            # search calls left today (separate bucket)
        last_plan=last_plan,
        time_stats=db.hourly_action_stats(user_id, mode.name),   # per-hour activity in this mode
        user_profile=profile,                                    # {} = none / disabled / bad JSON
        session_intent=intent,                                   # "" = none / disabled
    )


def validate_plan(plan: RetrievalPlan, quota_left: int,
                  search_left: int | None = None) -> tuple[bool, str]:
    """Validate an LLM-produced plan in code; never trust the LLM to self-police.

    Returns ``(ok, reason)``; the reason is fed back for re-planning.
    Four gates:

    1. Free-form search strategies <= ``MAX_SEARCH_PER_REFRESH``, and zero
       when the search bucket is below ``SEARCH_FLOOR_PROFILE``.
    2. Cost: general cost <= remaining quota x ``PLAN_QUOTA_BUDGET_RATIO`` and
       search calls <= remaining search bucket. Cost is computed by
       ``core.costing``; the LLM's self-reported ``est_quota`` is ignored.
    3. Strategy count window: the minimum counts all strategies; the maximum
       counts only LLM-authored ones (code-injected seed/library/expanded/
       local-intent strategies are excluded).
    4. Required params present for each tool (a missing key would otherwise
       void the strategy with a KeyError at execution time).

    ``search_left=None`` is treated as a full bucket.
    """
    if search_left is None:
        search_left = DAILY_SEARCH_CALLS

    # (1) Intent-driven searches are injected deterministically by the
    # orchestrator, which budgets them itself; only free planner searches count.
    n_search = sum(1 for s in plan.strategies
                   if s.tool == "search_by_keyword" and not s.intent_driven)
    if n_search > MAX_SEARCH_PER_REFRESH:
        return False, (f"search_by_keyword 策略 {n_search} 个,超过上限 "
                       f"{MAX_SEARCH_PER_REFRESH}(搜索桶全站每日仅 100 次)")
    # Below the floor, refreshes still work via channel/trending/pool strategies.
    if n_search > 0 and search_left < SEARCH_FLOOR_PROFILE:
        return False, (f"搜索桶剩余 {search_left} 低于门槛 {SEARCH_FLOOR_PROFILE},"
                       f"本次不得使用 search_by_keyword(改用频道/热门/池策略)")

    # (2) Budget
    cost = estimate_plan_cost(plan)
    budget = quota_left * PLAN_QUOTA_BUDGET_RATIO

    if cost.general > budget:
        return False, (f"计划预估成本 {cost.general} 超过预算上限 {budget:.0f}"
                       f"(今日剩余 {quota_left} × {PLAN_QUOTA_BUDGET_RATIO});"
                       f"请降低 n_channels 或 target_count")
    if cost.search > search_left:
        return False, (f"计划搜索次数 {cost.search} 超过今日搜索桶剩余 {search_left};"
                       f"请减少 search 策略")

    # (3) Strategy count. Strategies flagged with these params are appended by
    # the orchestrator after planning, so they don't count toward the maximum.
    n_all = len(plan.strategies)
    n = sum(1 for s in plan.strategies
            if not (s.params.get("use_seed_channels") or s.params.get("use_library")
                    or s.params.get("use_expanded_channels") or s.params.get("use_local_intent")))
    if n_all < PLAN_MIN_STRATEGIES or n > PLAN_MAX_STRATEGIES:
        return False, (f"策略数 {n_all}(LLM 自产 {n})不在 [{PLAN_MIN_STRATEGIES}, "
                       f"{PLAN_MAX_STRATEGIES}] 内")

    # (4) Required params; strategies are numbered from 1 in messages to the LLM.
    for i, s in enumerate(plan.strategies, start=1):
        p = s.params
        if (s.tool == "search_by_keyword" and not p.get("query")
                and not p.get("queries")):   # batched "queries" form also satisfies this
            return False, f"策略{i}(search_by_keyword)缺必需参数 query"
        if s.tool == "fetch_trending_by_category" and p.get("category_id") is None:
            return False, f"策略{i}(fetch_trending_by_category)缺必需参数 category_id"
        if (s.tool == "fetch_channel_videos"
                and not p.get("use_top_channels") and not p.get("channel_id")
                and not p.get("use_type_pool")
                and not p.get("use_seed_channels")
                and not p.get("use_library")
                and not p.get("use_expanded_channels")
                and not p.get("use_local_intent")):
            return False, (f"策略{i}(fetch_channel_videos)须有 use_top_channels、"
                           f"channel_id、use_type_pool、use_seed_channels、use_library、"
                           f"use_expanded_channels 或 use_local_intent 之一")

    return True, ""


_agent: Agent | None = None


def _get_agent() -> Agent:
    """Lazily build the planner Agent.

    Not built at import time, so offline tests run without an API key. A
    missing key raises on first construction; ``make_plan`` catches it and
    falls back to ``DEFAULT_PLANS`` with an error log.
    """
    global _agent
    if _agent is None:
        _agent = Agent(
            _model_id(LLM_MODEL),
            output_type=RetrievalPlan,
            system_prompt=PROMPTS["planner"].system,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _agent


async def _run_planner(ctx: PlanContext, mode: Mode, fail_reason: str | None = None) -> RetrievalPlan:
    """One LLM planning call.

    A non-empty ``fail_reason`` means re-planning: the previous validation
    failure is placed at the top of the user prompt. The system prompt stays
    static; failure context belongs to this turn's input.
    """
    parts = []

    if fail_reason:
        parts.append(f"你上一份计划未通过代码校验,原因:{fail_reason}。"
                     f"请修正该问题后重新产出完整计划。")
    parts.append(f"当前模式:{mode.name}({mode.display_name})。"
                 f"以下是规划输入 PlanContext(JSON):")

    parts.append(ctx.model_dump_json())

    result = await _get_agent().run("\n".join(parts))

    # Record token usage; metering errors must never interrupt planning.
    try:
        llmmeter.add(result.usage())
    except Exception:
        pass
    return result.output           # schema-validated RetrievalPlan


# ============================================================================
# S1 entry point
# ============================================================================
async def make_plan(user_id: int, mode: Mode) -> tuple[RetrievalPlan, str]:
    """S1 entry point. Returns ``(plan, source)`` with source in
    ``{agent, agent_replan, default}``.

    Flow: build context → LLM plan → validate → on failure re-plan
    ``PLANNER_REPLAN_MAX`` times with the reason attached → if still invalid,
    or on any LLM error (construction, schema retries exhausted, timeout,
    network), return a deep copy of ``DEFAULT_PLANS[mode]`` with
    ``source='default'``.
    """
    ctx = build_plan_context(user_id, mode)

    def _check(plan: RetrievalPlan) -> tuple[bool, str]:
        if plan.mode != mode.name:
            return False, f"plan.mode={plan.mode!r} 与请求模式 {mode.name!r} 不符"
        return validate_plan(plan, ctx.quota_left, ctx.search_left)

    try:
        plan = await _run_planner(ctx, mode)
        ok, why = _check(plan)
        if ok:
            return plan, "agent"

        for _ in range(PLANNER_REPLAN_MAX):
            logger.info("planner[%s]: 计划未过校验(%s),附原因重规划", mode.name, why)
            plan = await _run_planner(ctx, mode, fail_reason=why)
            ok, why = _check(plan)
            if ok:
                return plan, "agent_replan"
        logger.warning("planner[%s]: 重规划后仍未过校验(%s),降级 DEFAULT_PLANS", mode.name, why)
    except Exception as e:                      # catch every LLM error, including a missing key
        logger.error("planner[%s]: LLM 异常(%s),降级 DEFAULT_PLANS", mode.name, safe_repr(e))

    # Both failure paths converge here: static fallback plan from config.
    return DEFAULT_PLANS[mode.name].model_copy(deep=True), "default"
