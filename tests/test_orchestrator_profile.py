"""Offline tests for rotating profile-derived search queries into the retrieval plan."""
import config
from core.models import RetrievalPlan, StrategyCall
from pipeline import orchestrator


def _plan_with_free_search():
    return RetrievalPlan(mode="focus", strategies=[
        StrategyCall(tool="fetch_channel_videos", params={"channel_ids": ["c1"]},
                     target_count=10, rationale="频道"),
        StrategyCall(tool="search_by_keyword", params={"queries": ["planner 自选词"]},
                     target_count=10, rationale="自由搜索"),
    ], est_quota=0)


_ABUNDANT_QUOTA = 100000  # ample budget: behaves as if no budget gate were applied


def test_inject_replaces_free_search_rotates_by_cursor():
    plan, n = orchestrator._inject_profile_searches(
        _plan_with_free_search(), ["q0", "q1", "q2"], cursor=2,
        search_left=100, user_left=40)
    assert n == 2
    searches = [s for s in plan.strategies if s.tool == "search_by_keyword"]
    assert len(searches) == 1                       # planner's free-form search is replaced
    assert searches[0].params["queries"] == ["q2", "q0"]  # wraps around starting at cursor 2


def test_inject_noop_without_queries():
    plan = _plan_with_free_search()
    out, n = orchestrator._inject_profile_searches(
        plan, [], cursor=0, search_left=100, user_left=40)
    assert n == 0
    assert [s.tool for s in out.strategies] == [
        "fetch_channel_videos", "search_by_keyword"]  # no profile queries: plan untouched


def test_inject_keeps_intent_package():
    plan = _plan_with_free_search()
    plan.strategies.append(StrategyCall(
        tool="search_by_keyword", params={"queries": ["意图词"]},
        target_count=10, rationale="意图包", intent_driven=True))
    out, _ = orchestrator._inject_profile_searches(
        plan, ["q0"], cursor=0, search_left=100, user_left=40)
    intents = [s for s in out.strategies if getattr(s, "intent_driven", False)]
    assert len(intents) == 1                        # intent package is left alone


def test_single_entry_not_duplicated():
    plan, n = orchestrator._inject_profile_searches(
        _plan_with_free_search(), ["only"], cursor=5, search_left=100, user_left=40)
    assert n == 1
    q = [s for s in plan.strategies if s.tool == "search_by_keyword"][0]
    assert q.params["queries"] == ["only"]          # fewer queries than slots: no duplicates


# ---- Quota budget gate ----

def test_inject_affordable_unchanged_behavior():
    """Ample budget: take MAX_SEARCH_PER_REFRESH queries, wrapping from the cursor."""
    plan, n = orchestrator._inject_profile_searches(
        _plan_with_free_search(), ["q0", "q1", "q2"], cursor=2,
        search_left=100, user_left=40)
    assert n == 2
    searches = [s for s in plan.strategies if s.tool == "search_by_keyword"]
    assert searches[0].params["queries"] == ["q2", "q0"]


def test_inject_only_one_affordable_trims_to_one_word():
    """User has one profile search left today: trim to one query instead of skipping
    (trimming is by daily search count, not quota units)."""
    out, n = orchestrator._inject_profile_searches(
        _plan_with_free_search(), ["q0", "q1", "q2"], cursor=0,
        search_left=100, user_left=1)
    assert n == 1
    q = [s for s in out.strategies if s.tool == "search_by_keyword"][0]
    assert q.params["queries"] == ["q0"]


def test_inject_below_floor_strips_free_search():
    """Search bucket below the floor of 20: inject nothing and drop the planner's
    free-form search (it is the first consumer to yield)."""
    plan = _plan_with_free_search()
    out, n = orchestrator._inject_profile_searches(
        plan, ["q0", "q1"], cursor=0, search_left=19, user_left=40)
    assert n == 0
    assert [s.tool for s in out.strategies] == ["fetch_channel_videos"]


def test_inject_zero_user_left_is_noop():
    """User daily allowance is 0 (bucket still ample): no injection, plan unchanged."""
    plan = _plan_with_free_search()
    out, n = orchestrator._inject_profile_searches(
        plan, ["q0", "q1"], cursor=0, search_left=100, user_left=0)
    assert n == 0
    assert [s.tool for s in out.strategies] == [
        "fetch_channel_videos", "search_by_keyword"]
