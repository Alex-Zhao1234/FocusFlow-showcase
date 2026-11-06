"""Source-mix pure functions used by the orchestrator.

- ``apply_pool_ratio``: between S1 (planning) and S2 (retrieval), re-split the
  channel-based supply between the user's preferred channels and the shared
  type pool according to a target ratio.
- ``enforce_source_mix`` / ``enforce_source_cap``: after ranking, swap items
  between Top and the fold so that (a) neither channel-based side drops below
  a soft floor and (b) no single source takes more than a given share of Top.

No DB access. All swaps are pure: nothing is added, dropped or recorded as
filtered.
"""
from __future__ import annotations

import logging
import math
from collections import Counter

import config
from core.models import RetrievalPlan, StrategyCall

logger = logging.getLogger(__name__)

_SIDES = ("subscribed", "channel_pool")   # the two channel-based sides the soft floor balances


def apply_pool_ratio(plan: RetrievalPlan, ratio: float,
                     has_pref_channels: bool) -> RetrievalPlan:
    """Re-split channel-based supply between preferred channels and the type pool.

    The two ``fetch_channel_videos`` strategies (identified by
    ``use_top_channels`` / ``use_type_pool`` params) have their
    ``target_count`` summed and redistributed by ``ratio`` (the pool share).
    The ratio comes from user settings or slides from a high pool share for new
    users toward an even split as preference signals accumulate. Applied to
    LLM-generated plans, default plans and reused plans alike; skipped on
    intent refreshes.

    Returns a deep copy and never mutates in place (default plans are shared
    config singletons). If the user has no preferred channels, the whole share
    goes to the pool and the preference strategy is removed -- otherwise that
    share would come back empty and new users would silently get less supply.
    A plan without any channel-based strategy is returned unchanged; this
    function re-splits existing supply, it never adds it from nothing.
    """
    plan = plan.model_copy(deep=True)

    # Same tool, two identities distinguished by a boolean param.
    top_s = next((s for s in plan.strategies
                  if s.tool == "fetch_channel_videos"
                  and s.params.get("use_top_channels")), None)
    pool_s = next((s for s in plan.strategies
                   if s.tool == "fetch_channel_videos"
                   and s.params.get("use_type_pool")), None)

    if top_s is None and pool_s is None:
        return plan

    total = ((top_s.target_count if top_s else 0)
             + (pool_s.target_count if pool_s else 0))
    if total <= 0:
        return plan

    if not has_pref_channels:
        pool_share, top_share = total, 0
    else:
        pool_share = round(total * ratio)
        top_share = total - pool_share

    # Inject a pool strategy if the plan lacks one but the share calls for it.
    if pool_s is None and pool_share > 0:
        pool_s = StrategyCall(tool="fetch_channel_videos",
                              params={"use_type_pool": True, "n_channels": 8},
                              target_count=pool_share,
                              rationale="配比注入:类型池直抓(MR-1)")
        plan.strategies.append(pool_s)
    # Existing strategy: only the count changes; other params are respected.
    elif pool_s is not None:
        pool_s.target_count = pool_share

    # Symmetric for the preference side, but never injected without preferred channels.
    if top_s is None and top_share > 0 and has_pref_channels:
        top_s = StrategyCall(tool="fetch_channel_videos",
                             params={"use_top_channels": True, "n_channels": 10},
                             target_count=top_share,
                             rationale="配比注入:偏好频道直抓(MR-1)")
        plan.strategies.append(top_s)
    elif top_s is not None:
        top_s.target_count = top_share

    # Drop channel-based strategies whose share ended up at zero.
    plan.strategies = [s for s in plan.strategies
                       if not (s.tool == "fetch_channel_videos"
                               and (s.params.get("use_top_channels")
                                    or s.params.get("use_type_pool"))
                               and s.target_count <= 0)]
    return plan


def _channel_full(top: list, it, skip_idx: int | None = None) -> bool:
    """Would swapping ``it`` into Top exceed the per-channel cap?

    Counts Top items from the same channel, excluding ``skip_idx`` (the item
    about to be swapped out). Returns True when the count is already
    >= ``RANK_CHANNEL_CAP``; a cap <= 0 disables the check. Without this, a
    source-mix swap could bring back a channel's item the ranker had just
    capped.
    """
    cap = int(getattr(config, "RANK_CHANNEL_CAP", 0) or 0)

    if cap <= 0:
        return False
    n = sum(1 for i, x in enumerate(top)
            if i != skip_idx and x.video.channel_id == it.video.channel_id)
    return n >= cap


def _side(item) -> str | None:
    """Map an item to its channel-based side, or None for any other source.

    ``subscribed`` covers the user's own channels (followed or learned from
    feedback); ``channel_pool`` is the shared type pool.
    """
    s = item.video.source_strategy
    return s if s in _SIDES else None


def enforce_source_mix(top: list, more: list,
                       floor: float | None = None) -> tuple[list, list]:
    """On-screen soft floor for the two channel-based sides.

    If one side's share among Top's channel-based items is below ``floor``
    (default ``SOURCE_MIX_FLOOR``) and the fold has candidates from that side,
    swap the best such candidate with the lowest-ranked item from the other
    side; repeat until satisfied or out of candidates.

    Only channel-based items count toward the denominator, so search,
    trending, gem, seed and intent items neither dilute the ratio nor get
    displaced. When the denominator is too small for both sides to reach the
    floor, nothing is done (swapping would just oscillate). Seated items
    (reserved seats from the ranker) are never swapped out, and incoming items
    must not break the per-channel cap. If any swap happened, both lists are
    re-sorted by ``(-final, video_id)``; otherwise the input order is kept.
    """
    floor = config.SOURCE_MIX_FLOOR if floor is None else floor
    top, more = list(top), list(more)
    swapped = False
    for _ in range(len(top) + 1):                     # hard bound against infinite loops
        ch_idx = [i for i, it in enumerate(top) if _side(it)]
        n = len(ch_idx)
        if n == 0:
            break

        need = math.ceil(floor * n)
        # Both sides cannot reach the floor simultaneously; don't force it.
        if 2 * need > n:
            break

        deficit = next((s for s in _SIDES
                        if sum(1 for i in ch_idx if _side(top[i]) == s) / n < floor),
                       None)
        if deficit is None:
            break

        other = _SIDES[1] if deficit == _SIDES[0] else _SIDES[0]
        # Swap out the weakest non-seated item of the surplus side and swap in
        # the best fold candidate of the deficit side.
        i = next((i for i in reversed(ch_idx)
                  if _side(top[i]) == other and not getattr(top[i], "seated", False)), None)
        j = next((j for j, it in enumerate(more)
                  if _side(it) == deficit and not _channel_full(top, it, i)), None)
        if j is None or i is None:
            break

        top[i], more[j] = more[j], top[i]
        swapped = True
    # The swapped-in item lands at index i and may outscore the tail (e.g. an
    # item demoted by a cap pass), so restore score order.
    if swapped:
        top.sort(key=lambda it: (-it.final, it.video.video_id))
        more.sort(key=lambda it: (-it.final, it.video.video_id))
    return top, more


def enforce_source_cap(top: list, more: list, share: float | None = None) -> tuple[list, list]:
    """On-screen per-source ceiling.

    While any ``source_strategy`` has more than ``ceil(share * len(top))``
    items in Top (default share ``SOURCE_MAX_SHARE``), swap its lowest-scoring
    non-seated item with the highest-scoring fold candidate from a source that
    is not over the limit (and whose channel is not at cap). Intent cards are
    neither counted nor swapped. Pure swap; both lists are re-sorted by score.
    ``share <= 0`` disables.

    Run this before ``enforce_source_mix``: the soft floor only swaps items
    into the deficit side, so it cannot undo this ceiling. Without the ceiling,
    a single channel-based source could fill most of the screen and leave
    nothing to replace a channel the user dislikes.
    """
    share = config.SOURCE_MAX_SHARE if share is None else share
    if share <= 0 or not top or not more:
        return top, more
    top, more = list(top), list(more)
    limit = math.ceil(share * len(top))
    for _ in range(len(top) + 1):                                     # one swap per pass; hard bound
        counts = Counter(it.video.source_strategy for it in top if not it.video.intent_sourced)
        over = [s for s, n in counts.items() if n > limit]
        if not over:
            break
        s = max(over, key=lambda k: counts[k])                  # worst offender first
        outs = [i for i, it in enumerate(top)
                if it.video.source_strategy == s and not it.video.intent_sourced
                and not getattr(it, "seated", False)]
        if not outs:
            break                                     # over-limit source is all seated items
        i = min(outs, key=lambda i: (top[i].final, top[i].video.video_id))

        cands = [j for j, it in enumerate(more)
                 if counts.get(it.video.source_strategy, 0) < limit
                 and not _channel_full(top, it, i)]
        if not cands:
            break
        j = min(cands, key=lambda j: (-more[j].final, more[j].video.video_id))
        top[i], more[j] = more[j], top[i]
    top.sort(key=lambda it: (-it.final, it.video.video_id))
    more.sort(key=lambda it: (-it.final, it.video.video_id))
    return top, more
