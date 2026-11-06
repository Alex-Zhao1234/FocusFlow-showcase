"""Tests for injecting seed channels of the primary profile tier into the retrieval plan."""
import config
from agent.planner import validate_plan
from core.models import RetrievalPlan, StrategyCall
from pipeline.orchestrator import _inject_seed_channels


def _plan():
    return RetrievalPlan(mode="focus", strategies=[StrategyCall(
        tool="search_by_keyword", params={"query": "x"},
        target_count=10, rationale="t")], est_quota=0)


def test_inject_appends_seed_strategy():
    plan, n = _inject_seed_channels(_plan(), ["UCa", "UCb"], quota_left=10000)
    assert n == 2
    s = plan.strategies[-1]
    assert s.params["use_seed_channels"] is True
    assert s.params["seed_channel_ids"] == ["UCa", "UCb"]
    assert s.params["n_per_channel"] == config.SEED_FETCH_PER_CHANNEL
    assert s.target_count == 2 * config.SEED_FETCH_PER_CHANNEL


def test_inject_budget_gate_trims_and_skips():
    plan, n = _inject_seed_channels(_plan(), ["UCa", "UCb", "UCc"], quota_left=0)
    assert n == 0 and len(plan.strategies) == 1      # cannot afford a single channel: skip entirely
    # Trim to 1 when quota_left affords exactly one channel. The gate computes
    # avail = left * 0.5 - general cost of the current plan. Searches are billed
    # to a separate search bucket, so the general cost here is enrich = 1,
    # giving avail = 1 at left = 4.
    plan, n = _inject_seed_channels(_plan(), ["UCa", "UCb"], quota_left=4)
    assert n == 1


def test_inject_empty_noop():
    plan, n = _inject_seed_channels(_plan(), [], quota_left=10000)
    assert n == 0 and len(plan.strategies) == 1


def test_injected_plan_passes_validate_plan():
    """validate_plan must accept use_seed_channels as a valid strategy shape;
    otherwise re-validating a reused plan in steady state would always fail."""
    plan, n = _inject_seed_channels(_plan(), ["UCa", "UCb"], quota_left=10000)
    assert n == 2
    ok, why = validate_plan(plan, quota_left=10000)
    assert ok, why


def test_reinject_on_reused_plan_is_idempotent():
    """Re-injecting into a plan that already has a seed strategy (as happens on
    every steady-state refresh) replaces the old one: the strategy count stays
    the same and the new channel ids win, rather than accumulating."""
    plan, n1 = _inject_seed_channels(_plan(), ["UCa", "UCb"], quota_left=10000)
    assert n1 == 2
    n_strategies_before = len(plan.strategies)
    plan, n2 = _inject_seed_channels(plan, ["UCc", "UCd", "UCe"], quota_left=10000)
    assert n2 == 3
    assert len(plan.strategies) == n_strategies_before   # strategy count unchanged
    seed_strategies = [s for s in plan.strategies
                       if s.params.get("use_seed_channels")]
    assert len(seed_strategies) == 1                     # exactly one seed strategy
    assert seed_strategies[0].params["seed_channel_ids"] == ["UCc", "UCd", "UCe"]
