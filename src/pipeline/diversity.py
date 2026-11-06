"""Pure-function half of the profile-diversity seat system.

Decides, per refresh, which profile entries get reserved seats on screen and
in which order their search queries are spent:

- automatic mode: entry 1 always gets seats; secondary entries are drawn
  independently with position-decaying weights, a behavioural boost for
  recently watched entries, and starvation protection for entries that keep
  losing the draw;
- manual mode: seats are derived directly from user-set percentages;
- query ordering is coupled to the seat plan, so entries that were given seats
  are also the ones that receive search supply.

No DB access: boosted/forced/excluded/miss-streak inputs are assembled by the
orchestrator. Randomness is seeded from ``refresh_id`` so a refresh is
reproducible. The seat *execution* half lives in
``pipeline.ranker._apply_profile_diversity``; the plan is passed in as a
parameter so the ranker stays SQL-free.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import config


@dataclass(frozen=True)
class DiversityPlan:
    """Read-only seat plan handed from the orchestrator to the ranker.

    ``sub_seats`` already includes the behavioural boost in automatic mode.
    ``entry_bases`` is only used as a fallback when the judge assigned no
    profile entry to any item in the pool.
    """
    tier1_seats: int                     # reserved seats for entry 1 (the primary profile entry)
    sub_seats: dict[int, int]            # {secondary entry position: seats}
    cap: int                             # max Top items per entry; <= 0 disables (manual mode uses 0)
    entry_bases: dict[int, list[str]] = field(default_factory=dict)  # {position: base content types}; fallback only

    seed_channels: dict[int, frozenset[str]] = field(default_factory=dict)  # {position: seed channel ids}; matched by channel, not source tag
    seed_seats: int = 0                                                     # pinned seed-channel seats per entry, counted within that entry's seats
    seed_seats_tier1: int = 0                                               # pinned-seat cap for entry 1 (0 = same as seed_seats); fewer if supply is short
    followed_channels: frozenset[str] = frozenset()                         # followed channel ids; seated after seeds, before ordinary candidates


def _effective_weights(n_entries: int, *, boosted: set[int],
                       excluded: set[int]) -> dict[int, float]:
    """Draw probability for each secondary entry (positions 2..n).

    Base weight decays with position (entries past the end of the weight table
    reuse its last value); recently watched entries are multiplied by
    ``DIVERSITY_PLAY_WEIGHT_MULT`` but capped at ``DIVERSITY_WEIGHT_CAP`` so
    they are never guaranteed; excluded entries get 0.
    """
    out: dict[int, float] = {}
    for pos in range(2, n_entries + 1):
        i = pos - 2
        w = (config.DIVERSITY_SUB_BASE_WEIGHTS[i]
             if i < len(config.DIVERSITY_SUB_BASE_WEIGHTS)
             else config.DIVERSITY_SUB_BASE_WEIGHTS[-1])
        if pos in boosted:
            w = min(w * config.DIVERSITY_PLAY_WEIGHT_MULT,
                    config.DIVERSITY_WEIGHT_CAP)
        if pos in excluded:
            w = 0.0
        out[pos] = w
    return out


def draw_featured_subs(n_entries: int, *, boosted: set[int], forced: set[int],
                       excluded: set[int], miss_streaks: dict[int, int],
                       seed: str) -> list[int]:
    """Draw which secondary entries are featured this refresh.

    Args:
        n_entries: number of profile entries for this user/mode.
        boosted: entries watched recently (behavioural boost).
        forced: entries that lost the draw too many times in a row
            (starvation protection).
        excluded: entries whose content types are all excluded by the user.
        miss_streaks: consecutive-miss counter per entry.
        seed: the refresh id; hashed to seed the RNG for reproducibility.

    Returns ascending positions, length in ``[1, DIVERSITY_MAX_SUBS]`` (empty
    when only entry 1 exists). Priority when truncating: forced entries are
    always kept and cut last, then boosted, then the rest by position. If the
    draw comes up empty, the non-excluded entry with the longest miss streak is
    added -- starvation protection relies on at least one secondary entry per
    refresh.
    """
    if n_entries < 2:
        return []
    # Forced entries must be valid secondary positions and not excluded.
    forced = {p for p in forced if 2 <= p <= n_entries and p not in excluded}
    weights = _effective_weights(n_entries, boosted=boosted, excluded=excluded)

    # Deterministic RNG per refresh id.
    rng = random.Random(int.from_bytes(
        hashlib.sha256(seed.encode()).digest()[:8], "big"))

    # One independent Bernoulli trial per non-forced secondary entry.
    drawn = [pos for pos in range(2, n_entries + 1)
             if pos not in forced and rng.random() < weights[pos]]

    # Priority queue: forced, then boosted winners, then other winners.
    ordered = (sorted(forced)
               + [p for p in drawn if p in boosted]
               + [p for p in drawn if p not in boosted])
    picked: list[int] = []
    for p in ordered:
        if p not in picked:
            picked.append(p)
        if len(picked) >= config.DIVERSITY_MAX_SUBS:
            break
    # An all-miss draw is quite likely with several low weights; always feature one.
    if not picked:
        candidates = [p for p in range(2, n_entries + 1) if p not in excluded]
        if candidates:
            # Longest miss streak wins; ties go to the earlier position.
            picked = [max(candidates, key=lambda p: (miss_streaks.get(p, 0), -p))]
    return sorted(picked)


def order_profile_queries(queries: list[str], featured: list[int],
                          cursor: int, excluded: set[int] | frozenset[int] = frozenset(),
                          tier1_reserved: int | None = None) -> list[str]:
    """Order profile search queries so search supply follows the seat plan.

    Order: entry 1 -> featured secondary entries (rotated by ``cursor``) ->
    remaining entries (rotated by ``cursor``). ``queries`` is indexed by entry
    position (index 0 = entry 1); the caller takes the first
    ``MAX_SEARCH_PER_REFRESH`` non-empty queries. When
    ``PROFILE_SEARCH_TIER1_RESERVED`` is 0, entry 1 goes after the featured
    entries instead of first. An empty query for entry 1 is filtered by the
    caller, so its slot naturally passes to the next entry.

    Rotating the featured segment prevents entry 2 from monopolizing the
    budget when only one slot remains. Excluded entries are removed from every
    segment: their videos would be dropped later anyway, so no search budget
    is spent on them.
    """
    n = len(queries)
    if n == 0:
        return []
    reserved = (config.PROFILE_SEARCH_TIER1_RESERVED if tier1_reserved is None
                else tier1_reserved)
    feat = [p for p in featured if 1 <= p <= n and p != 1 and p not in excluded]

    def _rot(seq: list[int]) -> list[int]:
        return [seq[(cursor + i) % len(seq)] for i in range(len(seq))] if seq else []

    if reserved > 0:
        head_pos = ([1] if 1 not in excluded else []) + _rot(feat)
    else:
        head_pos = feat + ([1] if 1 not in excluded else [])
    head, seen = [], set()
    for p in head_pos:
        if p not in seen:
            seen.add(p)
            head.append(queries[p - 1])
    rest_pos = [p for p in range(1, n + 1) if p not in seen and p not in excluded]
    rest = [queries[p - 1] for p in _rot(rest_pos)]
    return head + rest


# ---- Manual seat ratios ----

def manual_seat_plan(pcts_by_pos: dict[int, int], excluded: set[int], top_n: int,
                     entry_bases: dict[int, list[str]]) -> tuple[list[int], DiversityPlan]:
    """Build a seat plan from user-set percentages (manual mode).

    Seats per entry = ``round(pct/100 * share * top_n)`` with
    ``share = SEAT_PROFILE_SHARE_PCT / 100`` (0.85): the user's percentages
    (summing to 100, enforced by the API/UI) describe the profile's internal
    split, while the profile as a whole only gets 85% of the screen; the rest
    is silently left to exploration seats.

    Every non-excluded secondary entry with pct > 0 is featured (no draw, no
    boost, no starvation logic). The per-entry cap is disabled. Entries at 0%
    get no reserved seats but can still win seats on score. Returns
    ``(featured secondary positions ascending, plan)``.
    """
    share = config.SEAT_PROFILE_SHARE_PCT / 100
    seats = {pos: int(round((pct or 0) / 100 * share * top_n))
             for pos, pct in pcts_by_pos.items() if pos not in excluded}
    featured = sorted(p for p, n in seats.items() if p >= 2 and n > 0)
    return featured, DiversityPlan(
        tier1_seats=seats.get(1, 0),
        sub_seats={p: seats[p] for p in featured},
        cap=0,                                     # users set per-entry seats explicitly
        entry_bases=entry_bases)


def weighted_profile_queries(queries: list[str], pcts_by_pos: dict[int, int],
                             seed: str, excluded: set[int] | frozenset[int] = frozenset(),
                             tier1_reserved: int | None = None) -> list[str]:
    """Query order for manual mode; counterpart of ``order_profile_queries``.

    If entry 1 has pct > 0 it always goes first (a 0% entry 1 is respected and
    trails with the other zeros). Other entries with pct > 0 are drawn in a
    weighted order *without replacement*, so higher shares lead more often but
    low shares are never starved. Entries at 0% follow in position order;
    excluded entries are omitted. Seeded by the refresh id; the rotation cursor
    is not advanced.
    """
    n = len(queries)
    if n == 0:
        return []
    reserved = (config.PROFILE_SEARCH_TIER1_RESERVED if tier1_reserved is None
                else tier1_reserved)
    rng = random.Random(int.from_bytes(
        hashlib.sha256(f"seatplan:{seed}".encode()).digest()[:8], "big"))
    pool = [(pos, float(pcts_by_pos.get(pos) or 0))
            for pos in range(1, n + 1) if pos not in excluded]
    weighted = [(p, w) for p, w in pool if w > 0]
    zeros = [p for p, w in pool if w <= 0]
    head: list[int] = []
    if reserved > 0:
        # Entry 1 is only in `weighted` when pct > 0 and not excluded.
        for i, (p, _w) in enumerate(weighted):
            if p == 1:
                head = [weighted.pop(i)[0]]
                break
    order: list[int] = []
    while weighted:
        total = sum(w for _, w in weighted)
        r = rng.random() * total
        acc = 0.0
        for i, (p, w) in enumerate(weighted):
            acc += w
            if r <= acc:
                order.append(p)
                weighted.pop(i)
                break
        else:
            order.append(weighted.pop()[0])
    return [queries[p - 1] for p in head + order + zeros]


# ---- Exempt entries and the effective exclusion set ----

def exempt_positions(n_entries: int, pcts_by_pos: dict[int, int], manual: bool,
                     top_n: int | None = None) -> set[int]:
    """Entry positions exempt from the user's excluded content types.

    Automatic mode: positions ``1..N``. Manual mode: the top N by percentage
    (ties go to the lower position; all-zero falls back to ``1..N``).
    ``N = EXCLUDE_EXEMPT_TOP_N``; 0 disables. Rationale: profile entries are
    written deliberately while exclusions are ticked casually, so on conflict
    the profile wins -- but only for the most important entries, so the
    exemption does not spread.
    """
    top_n = config.EXCLUDE_EXEMPT_TOP_N if top_n is None else top_n
    if top_n <= 0 or n_entries <= 0:
        return set()
    positions = range(1, n_entries + 1)
    if manual and any((pcts_by_pos.get(p) or 0) > 0 for p in positions):
        ranked = sorted(positions, key=lambda p: (-(pcts_by_pos.get(p) or 0), p))
        return set(ranked[:top_n])
    return set(positions[:top_n])


def effective_excluded(raw_excluded: set[int], exempt: set[int],
                       acks: dict[int, str | None]) -> set[int]:
    """Effective exclusion set: exempt entries are removed from the exclusion set.

    Exception: an entry whose ``conflict_ack == 'keep_excluded'`` stays
    excluded -- the user explicitly chose to keep the exclusion in the
    conflict dialog, which overrides the exemption. Both the seat side (draw,
    manual plan, query order) and the S7 filter consume this same set.
    """
    return {p for p in raw_excluded
            if not (p in exempt and acks.get(p) != "keep_excluded")}
