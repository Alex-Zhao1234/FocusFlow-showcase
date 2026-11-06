"""S0 state decision: whether a (user, mode) is ``active`` or ``steady``.

Each ``(user_id, mode)`` pair maintains its own two-state machine:

- ``mode_state`` runs decision checks 0-4; any hit means ``active``. The
  direction is fail-safe: any lack of data (new user with no actions, fewer
  refreshes than ``K_WINDOW``, NULL pass rates) falls to ``active``, because
  spending a few more tokens is always safer than wrongly declaring
  convergence and serving stale content. Zero LLM tokens -- pure SQL
  aggregates.
- ``reusable_plan`` gates plan reuse in the steady state behind four checks.
  The plan validator is injected by the orchestrator (it lives in the agent
  layer, which this module must not import) -- the same pattern as the
  injected adapter for plan execution.

Reuse is not replay: steady state only skips S1 planning; S2 still runs in
full, and channel uploads are fetched newest-first, so new videos keep
arriving. The other anti-stagnation safeguards are the ranker's explore floor
and this module's play-rate-drop trigger.

Dependencies: called only by the orchestrator; does not import other stages,
the agent layer, or ``sources/``; all data access goes through read-only
``storage.db`` functions.
"""
import json
import logging
import time
from typing import Callable, Literal

import config
from core.models import Mode, RetrievalPlan
from storage import db

logger = logging.getLogger(__name__)

# Explicit user actions that immediately revert to active:
# "explore more", an active search, or a mode configuration change.
_TRIGGER_ACTIONS = frozenset({"explore_more", "search", "mode_config_change"})


def mode_state(user_id: int, mode: Mode) -> Literal["active", "steady"]:
    """S0 state decision: checks 0-4 short-circuit; any hit -> active, else steady.

    Config values are read at call time (``import config`` rather than
    from-imports) so runtime toggles and tests can flip
    ``ENABLE_ACTION_SIGNALS`` and threshold changes take effect immediately.
    """
    # Check 0: master switch. Off -> always active (pure re-planning every time).
    if not config.ENABLE_ACTION_SIGNALS:
        return "active"

    since = int(time.time()) - config.SIGNAL_WINDOW_DAYS * 86400

    # Check 1a: explicit trigger actions. The user is signalling that the feed
    # feels off; revert regardless of other statistics.
    for row in db.get_actions(user_id, mode.name, since=since):
        if row["action"] in _TRIGGER_ACTIONS:
            logger.info("stability[%s/u%s]: 触发器动作 %s 命中,回退 active",
                        mode.name, user_id, row["action"])
            return "active"

    # Check 1b: dislike-rate spike. Zero feedback returns 0.0 and does not
    # trigger here -- insufficient samples are handled by check 2.
    dislike = db.recent_dislike_rate(user_id, mode.name, config.SIGNAL_WINDOW_DAYS)
    if dislike > config.DISLIKE_CEIL:
        logger.info("stability[%s/u%s]: dislike 率 %.2f 超上限 %.2f,回退 active",
                    mode.name, user_id, dislike, config.DISLIKE_CEIL)
        return "active"

    # Check 1c: play-rate drop. Only fires when the last PLAY_DROP_K refreshes
    # are *all* below the floor; a single good refresh breaks the streak. A
    # short series cannot fire this check -- the data-insufficiency fail-safe
    # belongs to checks 2 and 4.
    rates = db.play_rate_series(user_id, mode.name, config.PLAY_DROP_K)
    if len(rates) >= config.PLAY_DROP_K and all(r < config.PLAY_RATE_FLOOR for r in rates):
        logger.info("stability[%s/u%s]: 播放率连续 %d 次低于 %.2f(%s),回退 active",
                    mode.name, user_id, config.PLAY_DROP_K, config.PLAY_RATE_FLOOR, rates)
        return "active"

    # Check 2: too few recent signals (feedback + plays) to judge convergence.
    if db.count_signals(user_id, mode.name, config.SIGNAL_WINDOW_DAYS) < config.STABLE_MIN_SIGNALS:
        return "active"

    # Check 4: pass rate over the last K_WINDOW refreshes has not converged
    # (std too high) -> planning is still oscillating. None (too few rows or
    # NULLs) is treated as active (fail-safe).
    std = db.recent_pass_rate_std(user_id, mode.name, config.K_WINDOW)
    if std is None or std > config.PASS_RATE_STD_MAX:
        return "active"

    return "steady"


def reusable_plan(
    user_id: int,
    mode: Mode,
    validate: Callable[[RetrievalPlan, int], tuple[bool, str]],
) -> RetrievalPlan | None:
    """Return the latest plan for reuse in steady state, or None.

    All four checks must pass; otherwise None, and the orchestrator runs S1
    as if active (the steady verdict stands, there is just nothing to reuse
    this time).

    ``validate`` is the planner's plan validator, injected by the
    orchestrator so this module has no agent-layer dependency; offline tests
    pass a fake.
    """
    # 1. A previous plan exists (otherwise cold start).
    row = db.get_latest_retrieval_plan(user_id, mode.name)
    if row is None:
        return None

    # 2. The plan is younger than PLAN_REUSE_TTL_HOURS (content freshness).
    if int(time.time()) - row["executed_at"] > config.PLAN_REUSE_TTL_HOURS * 3600:
        return None

    # 3. plan_json deserializes to a valid plan for this mode. Legacy rows,
    # bad JSON and invalid fields all stop here; the mode check is defensive.
    try:
        plan = RetrievalPlan(**json.loads(row["plan_json"])["plan"])
    except Exception:  # noqa: BLE001 -- expected for legacy rows; silently fall back to S1
        logger.info("stability[%s/u%s]: 最近计划反解失败(历史行/坏 JSON),本次走 S1",
                    mode.name, user_id)
        return None
    if plan.mode != mode.name:
        return None

    # 4. Re-validate against *today's* quota, including the separate search
    # bucket. A plan that was valid when stored may not be today; reuse saves
    # the LLM planning call but is not exempt from budget rules.
    ok, why = validate(plan, db.quota_remaining(), db.quota_remaining("search"))
    if not ok:
        logger.info("stability[%s/u%s]: 复用计划未过重校验(%s),本次走 S1",
                    mode.name, user_id, why)
        return None

    return plan
