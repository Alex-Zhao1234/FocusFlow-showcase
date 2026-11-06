"""Final weighted ranking and Top/fold split (pipeline stage S7, code half of S6).

Takes the candidates that survived the S6 judge, computes a weighted ``final``
score per item, sorts, splits into an on-screen Top N and a folded "more"
area, then runs a chain of seat-allocation passes (explore floor, profile
diversity seats, per-channel cap, per-type cap, intent floor).

Boundaries:
  - ``fit_score`` and ``quality_score_norm`` are computed by the orchestrator
    and passed in; this module only consumes numbers. ``candidates`` contains
    only items the judge marked ``fit=True``.
  - ``RankedItem`` is a lightweight internal payload, not the
    ``RecommendationCard`` model (reason text and badges belong to the card
    layer). It is a dataclass rather than pydantic: purely internal, all fields
    are numbers this module just computed, no validation needed.
  - Freshness soft constraints and the stability state machine live elsewhere.
  - This stage does not import other stages or ``sources/``, issues no SQL
    directly (all reads/writes go through ``storage.db``), and keeps every
    weight/threshold/seat count in ``config``.

Side effects: only the gem median filter writes ``filter_results``
(stage='mode', rule='gem_below_median'). Items moved to the fold or swapped out
by seat passes are *not* filtered -- they still render in folded form -- so
they are not recorded. Every DB access is scoped by ``user_id``.
"""
import logging
import math
import statistics
from collections import Counter
from dataclasses import dataclass

import config as _cfg                               # read dynamically so tests can monkeypatch
from config import (
    W_FIT, W_TYPE, W_QUALITY, W_PREF, GEM_BONUS, TOP_N, EXPLORE_SLOT_FLOOR,
    INTENT_RESERVED_SLOTS,
    GEM_VIEW_SWEET_MIN, GEM_VIEW_SWEET_MAX,
    GEM_PLAY_FACTOR_SWEET, GEM_PLAY_FACTOR_LOW, GEM_PLAY_FACTOR_HIGH,
    GEM_FAN_BONUS, GEM_MEDIAN_MIN_N, SMALL_CHANNEL_SUBS_MAX,
    type_weight_of,
)
from core.models import Video
from pipeline.diversity import DiversityPlan
from storage import db

logger = logging.getLogger(__name__)

# Source strategies that count toward the explore-slot floor.
_EXPLORE_SOURCES = frozenset({"hidden_gem", "trending"})


@dataclass
class RankedItem:
    """Internal ranking payload: the video, every score component, and ``final``.

    All components are kept (not just ``final``) so ranking stays explainable
    and tunable: when calibrating weights you can see which term contributed
    what without recomputing.
    """
    video: Video                      # survivor of S4 triage + S5 quality review
    fit_score: float                  # mode-fit score from the S6 LLM judge
    quality_score_norm: float         # normalized quality score, 0..1
    channel_pref_score: float         # user's preference for the channel (read from DB, not recomputed)
    is_gem: bool                      # retrieved by the hidden_gem strategy
    final: float                      # weighted total; drives sorting and the Top/fold split
    content_type: str | None = None   # global content type judged in S6 (one of 17 base types; None = missing)
    type_w: float = 0.0               # bucket weight of content_type under the current mode
    entry: int | None = None          # profile entry the S6 judge assigned (1 = primary); None = unassigned
    seated: bool = False              # claimed as a reserved seat by _apply_profile_diversity; later
                                      # mix/cap passes must not swap it out (``entry`` only means "eligible")


def hidden_gem_score(v: Video, channel_subs: int | None) -> float:
    """Hidden-gem score = engagement_rate x play_factor x fan_factor.

    - engagement reuses the model's computed field ``(like + 5*comment) / max(view, 1)``;
      it is a different formula with a different purpose from ``quality_score``.
    - play_factor: views in the 2,000-30,000 sweet spot get 1.0; below that 0.7
      (too few views, engagement is noisy); above that 0.5 (already promoted by
      the platform algorithm, so not "buried").
    - fan_factor: small channels (< 200k subscribers) get a 1.2 bonus. Unknown
      subscriber counts are treated as 0 and receive the bonus, so missing data
      is never penalized (consistent with cold-start protection).
    """
    # Sweet spot: enough views for stable stats, not yet picked up by the mainstream algorithm.
    if GEM_VIEW_SWEET_MIN <= v.view_count <= GEM_VIEW_SWEET_MAX:
        play_factor = GEM_PLAY_FACTOR_SWEET
    # Low end: penalizes statistical noise, not obscurity.
    elif v.view_count < GEM_VIEW_SWEET_MIN:
        play_factor = GEM_PLAY_FACTOR_LOW
    # High end: penalized harder -- already surfaced by the platform.
    else:
        play_factor = GEM_PLAY_FACTOR_HIGH
    fan_factor = GEM_FAN_BONUS if (channel_subs or 0) < SMALL_CHANNEL_SUBS_MAX else 1.0
    return v.engagement_rate * play_factor * fan_factor


# ---------------------------------------------------------------------------
# Internal passes, in the order rank() calls them
# ---------------------------------------------------------------------------

def _filter_gems_below_median(user_id: int, refresh_id: str,
                              candidates: list[tuple[Video, float, float, str | None]],
                              ) -> list[tuple[Video, float, float, str | None]]:
    """Relative quality check over the hidden_gem pool.

    When there are at least ``GEM_MEDIAN_MIN_N`` (4) gem candidates, those whose
    gem score is *strictly below* the pool median are dropped and recorded in
    ``filter_results``; smaller pools are left alone (sample too small).
    Items exactly at the median are kept: the gem pool is scarce (~6 items),
    and a literal "above the median" rule would always cut the middle member
    of an odd-sized pool. Non-gem candidates pass through untouched, order
    preserved.

    Subscriber counts are looked up once per channel (a refresh has ~10 gem
    channels, primary-key lookups). A missing channel row is treated as 0
    subscribers and therefore gets the small-channel bonus.
    """
    gems = [c for c in candidates if c[0].source_strategy == "hidden_gem"]
    if len(gems) < GEM_MEDIAN_MIN_N:
        return candidates                                 # sample too small for a meaningful median

    # channel_id -> subscriber count (or None); one lookup per channel
    subs_cache: dict[str, int | None] = {}
    for v, _, _, _ in gems:
        if v.channel_id not in subs_cache:
            row = db.get_channel(v.channel_id)
            subs_cache[v.channel_id] = row["channel_subs"] if row is not None else None

    scores = {v.video_id: hidden_gem_score(v, subs_cache[v.channel_id]) for v, _, _, _ in gems}
    median = statistics.median(scores.values())
    dropped: set[str] = set()
    for v, _, _, _ in gems:
        s = scores[v.video_id]
        if s < median:
            dropped.add(v.video_id)
            # Numbers in the reason text help later calibration.
            db.add_filter_result(
                user_id, refresh_id, v.video_id, "mode", "gem_below_median",
                f"小众分低于本次挖掘池中位数（{s:.4f} < {median:.4f},池 {len(gems)} 条）")
    if dropped:
        logger.info("ranker/gem 中位数过滤: 池 %d 条,中位数 %.4f,丢弃 %d 条",
                    len(gems), median, len(dropped))
    return [c for c in candidates if c[0].video_id not in dropped]


def _pref_score(user_id: int, channel_id: str,
                cache: dict[str, float]) -> float:
    """Return the user's channel preference score, cached per channel.

    The score is read from the DB, never recomputed here (aggregation is the
    feedback updater's job). It already includes any active soft cooldown
    factor. Missing rows / NULL scores fall back to a neutral 0.5 so channels
    without feedback history have no effect on ``final``.
    """
    if channel_id not in cache:
        eff = db.effective_channel_pref(user_id, channel_id)
        cache[channel_id] = 0.5 if eff is None else eff
    return cache[channel_id]


def _apply_profile_diversity(top: list[RankedItem], more: list[RankedItem],
                             plan: DiversityPlan | None,
                             seeded_out: set[str] | None = None,
                             ) -> tuple[list[RankedItem], list[RankedItem]]:
    """Reserve per-profile-entry seats in the Top area.

    1. Claim reserved seats: entry 1 gets up to ``tier1_seats``, each drawn
       secondary entry up to its ``sub_seats``. Within each entry, seed-channel
       videos claim "pinned" seats first (up to ``seed_seats``; their entry is
       rewritten to this one), then items the judge assigned to this entry fill
       the rest -- followed channels above a preference threshold ahead of
       others, each group in ``final`` order. Seats are never filled with
       placeholders. If no item in the pool has an entry assignment (judge
       degraded), fall back to matching ``content_type`` against the entry's
       base types.
    2. Per-entry cap: if any entry holds more than ``cap`` items in Top, keep
       the best ``cap`` (pinned seats first) and swap the excess with
       non-capped items from the fold.

    Pure swaps: Top + fold keep the same total set, nothing is written to
    ``filter_results``. Both areas are re-sorted by ``(-final, video_id)``.
    Seated items get ``seated=True``; pinned-seat ids are added to
    ``seeded_out`` so later cap passes can protect them.
    """
    if plan is None or (plan.tier1_seats <= 0 and not plan.sub_seats):
        return top, more
    pool = top + more
    n_top = len(top)
    use_fallback = not any(it.entry for it in pool)   # judge assigned nothing -> fall back to base types

    def _belongs(it: RankedItem, pos: int) -> bool:
        if not use_fallback:
            return it.entry == pos
        bases = plan.entry_bases.get(pos) or []
        return it.content_type is not None and it.content_type in bases

    seats = {1: plan.tier1_seats, **plan.sub_seats}        # {entry position: seat count}
    claimed: set[str] = set()
    seeded: set[str] = set()                               # pinned (seed-channel) seats only
    seed_log: dict[int, int] = {}
    followed_log: dict[int, int] = {}
    seated: list[RankedItem] = []                          # claimed items, in claim order

    for pos in sorted(seats):
        if len(seated) >= n_top:
            break
        quota = min(seats[pos], n_top - len(seated))

        # (1) Pinned seats: this entry's seed-channel videos go first.
        seed_set = plan.seed_channels.get(pos)
        picked: list[RankedItem] = []
        _seed_cap = (plan.seed_seats_tier1 if pos == 1 and plan.seed_seats_tier1 > 0
                     else plan.seed_seats)
        if seed_set and _seed_cap > 0:
            n_seed = min(_seed_cap, quota)
            # pool is sorted by final descending, so the first matches are the best
            for it in pool:
                if len(picked) >= n_seed:
                    break
                if (it.video.video_id not in claimed
                        and it.video.channel_id in seed_set):
                    it.entry = pos                 # pinning overrides the judge's assignment
                    picked.append(it)
                    seeded.add(it.video.video_id)
            if picked:
                seed_log[pos] = len(picked)

        # (2) Remaining seats: items assigned to this entry (or matching base
        # types in fallback mode). Followed channels go first; resulting order
        # within an entry is seed > followed > others. No cap on followed seats.
        picked_ids = {it.video.video_id for it in picked}
        cand = [it for it in pool
                if it.video.video_id not in claimed
                and it.video.video_id not in picked_ids and _belongs(it, pos)]
        if plan.followed_channels and _cfg.FOLLOW_ENTRY_SEATS_ON:
            # The preference threshold keeps followed-but-never-liked channels
            # (initial score) competing on score instead of jumping the queue.
            def _fol_first(it: RankedItem) -> bool:
                return not (it.video.channel_id in plan.followed_channels
                            and it.channel_pref_score > _cfg.FOLLOW_SEAT_MIN_PREF)
            cand.sort(key=_fol_first)                     # stable: followed first, final order kept within groups

            n_fol = sum(1 for it in cand[: quota - len(picked)] if not _fol_first(it))
            if n_fol:
                followed_log[pos] = n_fol
        for it in picked + cand[: quota - len(picked)]:
            claimed.add(it.video.video_id)
            seated.append(it)
    if seeded_out is not None:
        seeded_out |= seeded
    if claimed:
        rest = [it for it in pool if it.video.video_id not in claimed]
        # max(0, ...) guards against a negative slice if seats exceed Top capacity
        top = seated + rest[: max(0, n_top - len(seated))]
        more = rest[max(0, n_top - len(seated)):]
        logger.info("ranker/画像多样性: 档位 %s 认领 %d 席(钦点席 %s,关注席 %s)",
                    sorted(seats), len(claimed), seed_log or "无", followed_log or "无")

    # Per-entry cap. Only applies with real judge assignments (fallback mode
    # has no entry to count).
    if not use_fallback and plan.cap > 0:
        from collections import Counter
        counts = Counter(it.entry for it in top if it.entry is not None)
        excess: list[RankedItem] = []
        for pos, n in counts.items():
            if n > plan.cap:
                # pinned seats first, then higher score, video_id as tie-break
                mine = sorted((it for it in top if it.entry == pos),
                              key=lambda it: (it.video.video_id not in seeded,
                                              -it.final, it.video.video_id))
                excess.extend(mine[plan.cap:])
        if excess:
            excess_ids = {it.video.video_id for it in excess}
            kept = [it for it in top if it.video.video_id not in excess_ids]
            counts = Counter(it.entry for it in kept if it.entry is not None)
            # Refill from the fold by score, skipping entries already at cap.
            refill: list[RankedItem] = []
            for it in sorted(more, key=lambda x: (-x.final, x.video.video_id)):
                if len(kept) + len(refill) >= n_top:
                    break
                if it.entry is None or counts[it.entry] < plan.cap:
                    refill.append(it)
                    if it.entry is not None:
                        counts[it.entry] += 1

            refill_ids = {it.video.video_id for it in refill}
            top = kept + refill
            more = [it for it in more if it.video.video_id not in refill_ids] + excess
            logger.info("ranker/单档上限: 超限降席 %d 条(cap=%d)",
                        len(excess), plan.cap)

    # Mark only claimed items still in Top: items demoted by the cap lost their seat.
    for it in top:
        if it.video.video_id in claimed:
            it.seated = True
    top.sort(key=lambda it: (-it.final, it.video.video_id))
    more.sort(key=lambda it: (-it.final, it.video.video_id))
    return top, more


def _apply_cap(top: list[RankedItem], more: list[RankedItem], *, key, cap: int,
               protected: frozenset | set = frozenset(), skip=None, channel_cap: int = 0,
               ) -> tuple[list[RankedItem], list[RankedItem], int]:
    """Generic "at most ``cap`` Top items per key" pass.

    - ``key(it)`` returning None means the item is not counted.
    - ``skip(it)`` true (e.g. intent cards) means the item is neither counted
      nor swapped.
    - For each over-cap key, keep ``cap`` items ordered by
      ``(not protected, -final, video_id)``; the excess moves to the fold.
    - Freed seats are refilled from the fold by ``(-final, video_id)``,
      skipping keys already at cap. With ``channel_cap > 0`` the refill also
      skips channels that already have ``channel_cap`` items in Top, so a later
      cap pass cannot undo an earlier per-channel cap.

    Pure swap: the combined set is unchanged and nothing is written to
    ``filter_results``. Returns ``(top, more, excess_count)``.
    """
    if cap <= 0 or not top:
        return top, more, 0

    # Fold skip into key so skipped items are invisible to counting, trimming and refill.
    def _k(it: RankedItem):
        return None if (skip is not None and skip(it)) else key(it)

    counts = Counter(k for it in top if (k := _k(it)) is not None)
    over = {k for k, n in counts.items() if n > cap}
    if not over:
        return top, more, 0
    excess: list[RankedItem] = []

    for k in over:
        mine = sorted((it for it in top if _k(it) == k),
                      key=lambda it: (it.video.video_id not in protected, -it.final, it.video.video_id))
        excess.extend(mine[cap:])

    excess_ids = {it.video.video_id for it in excess}
    kept = [it for it in top if it.video.video_id not in excess_ids]
    counts = Counter(k for it in kept if (k := _k(it)) is not None)
    ch_counts = Counter(it.video.channel_id for it in kept)

    refill: list[RankedItem] = []
    for it in sorted(more, key=lambda x: (-x.final, x.video.video_id)):
        if len(kept) + len(refill) >= len(top):
            break                                                          # Top size is conserved
        k = _k(it)
        if channel_cap > 0 and ch_counts[it.video.channel_id] >= channel_cap:
            continue                          # refill must not break the per-channel cap
        if k is None or counts[k] < cap:
            refill.append(it)
            ch_counts[it.video.channel_id] += 1
            if k is not None:
                counts[k] += 1

    refill_ids = {it.video.video_id for it in refill}
    new_top = kept + refill
    new_more = [it for it in more if it.video.video_id not in refill_ids] + excess
    new_top.sort(key=lambda it: (-it.final, it.video.video_id))
    new_more.sort(key=lambda it: (-it.final, it.video.video_id))
    return new_top, new_more, len(excess)


def _apply_channel_cap(top: list[RankedItem], more: list[RankedItem],
                       seeded: set[str]) -> tuple[list[RankedItem], list[RankedItem]]:
    """At most ``RANK_CHANNEL_CAP`` Top items per channel (the fold is unrestricted).

    Pinned seed-channel items are kept preferentially; intent cards are exempt.
    Without this, a channel-based source that fetches ~30 videos per channel
    could take 20+ seats in a single refresh.
    """
    cap = int(_cfg.RANK_CHANNEL_CAP or 0)
    top, more, n = _apply_cap(top, more, key=lambda it: it.video.channel_id, cap=cap,
                              protected=seeded, skip=lambda it: it.video.intent_sourced)
    if n:
        logger.info("ranker/单频道上限: 超限降席 %d 条(cap=%d)", n, cap)
    return top, more


def _apply_type_cap(top: list[RankedItem], more: list[RankedItem],
                    seeded: set[str] | frozenset = frozenset(),
                    ) -> tuple[list[RankedItem], list[RankedItem]]:
    """At most ``RANK_TYPE_CAP`` Top items per content type.

    Items without a content type are not counted; intent cards are exempt;
    pinned seats are kept preferentially. Manual seat ratios allocate by
    profile entry rather than type, so type diversity is enforced here. The
    refill also honours the per-channel cap.
    """
    cap = int(_cfg.RANK_TYPE_CAP or 0)
    top, more, n = _apply_cap(top, more, key=lambda it: it.content_type, cap=cap,
                              protected=seeded, skip=lambda it: it.video.intent_sourced,
                              channel_cap=int(_cfg.RANK_CHANNEL_CAP or 0))
    if n:
        logger.info("ranker/单类型上限: 超限降席 %d 条(cap=%d)", n, cap)
    return top, more


def _apply_intent_floor(top: list[RankedItem], more: list[RankedItem],
                        ) -> tuple[list[RankedItem], list[RankedItem]]:
    """Reserve the head of Top for intent cards on an intent refresh.

    Up to ``INTENT_RESERVED_SLOTS`` ``intent_sourced`` items are placed at the
    very top: when the user states an intent, it should be on the first screen.

    Seat selection key is ``W_FIT*fit + W_QUALITY*qnorm`` -- balancing intent
    fit and quality. In the intent pool ``fit_score`` comes from the S6 intent
    adjudication and already means "matches the intent". Channel preference is
    deliberately excluded: intent recall mostly surfaces unfamiliar channels,
    and preference would only push them down.

    Fills as many seats as there are intent items, never placeholders; a no-op
    for normal refreshes. Runs after the explore floor, so intent wins when the
    two conflict (explicit user requests beat default tendencies). Displaced
    items go to the fold and are not recorded as filtered.
    """
    pool = top + more
    intent_all = [it for it in pool if it.video.intent_sourced]
    if not intent_all:
        return top, more

    intent_all.sort(key=lambda it: (
        -(W_FIT * it.fit_score + W_QUALITY * it.quality_score_norm),
        it.video.video_id))
    seats = intent_all[:INTENT_RESERVED_SLOTS]
    seat_ids = {it.video.video_id for it in seats}
    rest = [it for it in pool if it.video.video_id not in seat_ids]

    n_top = len(top)
    new_top = seats + rest[: max(0, n_top - len(seats))]
    new_more = rest[max(0, n_top - len(seats)):]
    return new_top, new_more


def _apply_explore_floor(top: list[RankedItem], more: list[RankedItem],
                         ) -> tuple[list[RankedItem], list[RankedItem]]:
    """Guarantee a minimum number of exploration items in Top.

    Top must contain at least ``ceil(TOP_N * EXPLORE_SLOT_FLOOR)``
    hidden_gem/trending items (30 x 0.15 = 4.5 -> 5 seats). This keeps the
    feed from stagnating into only familiar content.

    When short, the best exploration items in the fold are promoted and the
    lowest-scoring non-exploration items in Top are demoted (not recorded as
    filtered). If the fold has too few exploration items, promote what exists;
    never force it. Both areas are re-sorted by ``final`` afterwards: the floor
    changes *who* is on screen, not the score-based display order.

    The seat count is based on the ``TOP_N`` constant, not ``len(top)``; when
    there are fewer than TOP_N candidates the fold is empty and this is a no-op.
    """

    required = math.ceil(TOP_N * EXPLORE_SLOT_FLOOR)
    have = sum(1 for it in top if it.video.source_strategy in _EXPLORE_SOURCES)
    if have >= required:
        return top, more                 # common case: floor already satisfied

    # Both pools inherit the descending order of their source lists.
    promo_pool = [it for it in more if it.video.source_strategy in _EXPLORE_SOURCES]
    demote_pool = [it for it in top if it.video.source_strategy not in _EXPLORE_SOURCES]

    n_swap = min(required - have, len(promo_pool), len(demote_pool))
    if n_swap == 0:
        return top, more

    # Promote the best exploration items, demote the weakest non-exploration
    # ones -- minimal disturbance to overall ranking quality.
    promoted = promo_pool[:n_swap]
    demoted = demote_pool[-n_swap:]
    promoted_ids = {it.video.video_id for it in promoted}
    demoted_ids = {it.video.video_id for it in demoted}

    new_top = [it for it in top if it.video.video_id not in demoted_ids] + promoted
    new_more = [it for it in more if it.video.video_id not in promoted_ids] + demoted

    new_top.sort(key=lambda it: (-it.final, it.video.video_id))
    new_more.sort(key=lambda it: (-it.final, it.video.video_id))
    logger.info("ranker/探索位下限: Top 探索 %d < %d 席,从折叠区提升 %d 条",
                have, required, n_swap)
    return new_top, new_more


def _apply_recency_tiebreak(items: list[RankedItem], delta: float) -> int:
    """Within a channel, put newer uploads first among near-tied items.

    Deep channel fetches surface a channel's older videos, which often score
    almost identically to its new ones. Per channel, items are grouped (in
    ``final`` descending order) into clusters whose score is within ``delta``
    of the cluster head; each cluster is reordered by ``publish_time``
    descending (unknown last), and ``final`` is rewritten to
    ``head - index * 1e-6``.

    The order is encoded into ``final`` rather than list position because every
    later pass (Top/fold split, cap and seat passes) re-sorts by ``final``;
    1e-6 is invisible to display and thresholds. Cross-channel order is
    untouched, so diversity and seat logic are unaffected. ``delta <= 0``
    disables the pass. Mutates ``items`` in place; returns the number of
    reordered clusters.
    """
    if delta <= 0 or len(items) < 2:
        return 0
    by_ch: dict[str, list[RankedItem]] = {}
    for it in items:
        by_ch.setdefault(it.video.channel_id, []).append(it)
    changed = 0
    for group in by_ch.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda it: (-it.final, it.video.video_id))
        i = 0
        while i < len(group):
            head = group[i]
            j = i + 1
            while j < len(group) and head.final - group[j].final <= delta:
                j += 1
            cluster = group[i:j]
            if len(cluster) > 1:
                ordered = sorted(cluster, key=lambda it: (-(it.video.publish_time or 0), it.video.video_id))
                # Only rescore when the order actually changes; otherwise scores stay exactly equal.
                if [it.video.video_id for it in ordered] != [it.video.video_id for it in cluster]:
                    changed += 1
                    for k, it in enumerate(ordered):
                        it.final = head.final - k * 1e-6
            i = j
    return changed


def rank(user_id: int, refresh_id: str,
         candidates: list[tuple[Video, float, float, str | None]],
         *, type_buckets: list[list[str]],
         diversity_plan: DiversityPlan | None = None,
         entry_map: dict[str, int] | None = None,
         ) -> tuple[list[RankedItem], list[RankedItem]]:
    """Final ranking and split.

    Args:
        user_id: scopes channel-preference and impression lookups.
        refresh_id: used for filter records and logs.
        candidates: ``(video, fit_score, quality_score_norm, content_type)``
            tuples, judge-approved only; ``content_type`` is one of the 17 base
            types or None.
        type_buckets: the current mode's tiered content-type buckets.
        diversity_plan: per-profile-entry seat plan.
        entry_map: ``video_id -> profile entry`` as assigned by the S6 judge.

    Computes ``final`` for each surviving candidate, sorts, cuts Top N and the
    fold, then runs the seat passes (explore floor -> profile seats ->
    per-channel cap -> per-type cap -> intent floor). Returns ``(top, more)``.
    The fold is not "filtered"; conservation holds:
    ``len(candidates) == len(top) + len(more) + gem_filtered``.
    """
    # ---- 1. Gem median filter (the only pass that drops items) ----
    survivors = _filter_gems_below_median(user_id, refresh_id, candidates)

    # ---- 2. Weighted scoring ----
    # Impression decay: items already shown in the penalty window without being
    # opened get final *= decay ** n, so they stop occupying seats on
    # consecutive refreshes. One batched query, no LLM. Intent cards are exempt.
    _imp = db.window_impressions(
        user_id, [v.video_id for v, _f, _q, _c in survivors])
    pref_cache: dict[str, float] = {}
    items: list[RankedItem] = []
    for v, fit, qnorm, ctype in survivors:
        pref = _pref_score(user_id, v.channel_id, pref_cache)
        is_gem = v.source_strategy == "hidden_gem"
        type_w = type_weight_of(ctype, type_buckets)
        # Mode match (fit + type) is the largest axis -- the product recommends
        # by current mode; quality second; preference is capped to avoid
        # overfitting to old taste; gem is a small constant nudge.
        final = (W_FIT * fit
                 + W_TYPE * type_w
                 + W_QUALITY * qnorm
                 + W_PREF * pref
                 + GEM_BONUS * is_gem)
        n_seen = 0 if v.intent_sourced else _imp.get(v.video_id, 0)
        if n_seen:
            final *= _cfg.IMPRESSION_RANK_DECAY ** n_seen
        items.append(RankedItem(video=v, fit_score=fit, quality_score_norm=qnorm,
                                channel_pref_score=pref, is_gem=is_gem, final=final,
                                content_type=ctype, type_w=type_w,
                                entry=(entry_map or {}).get(v.video_id)))

    # ---- 3. Same-channel recency tie-break, then global sort ----
    _apply_recency_tiebreak(items, _cfg.RANK_RECENCY_TIE_DELTA)
    items.sort(key=lambda it: (-it.final, it.video.video_id))

    # ---- 4. Split ----
    top, more = items[:TOP_N], items[TOP_N:]

    # ---- 5. Seat passes (pure swaps; later passes override earlier ones) ----
    _seeded: set[str] = set()                            # pinned-seat ids: filled by profile pass, protected by caps
    top, more = _apply_explore_floor(top, more)
    top, more = _apply_profile_diversity(top, more, diversity_plan, seeded_out=_seeded)
    top, more = _apply_channel_cap(top, more, _seeded)
    top, more = _apply_type_cap(top, more, _seeded)
    top, more = _apply_intent_floor(top, more)           # intent wins conflicts with the explore floor

    logger.info("rank[%s]: 候选 %d → top %d / more %d(gem 过滤 %d)",
                refresh_id, len(candidates), len(top), len(more),
                len(candidates) - len(survivors))
    return top, more
