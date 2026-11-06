"""Single read path for per-user personalization parameters.

The orchestrator uses these helpers to fetch a user's effective content-type
buckets, excluded types and pool ratio for a mode, then injects them into the
ranker -- so the ranker and ``config`` never need to know about users.

All functions are read-only and fail-soft: bad stored data degrades to the
defaults and is logged, never surfacing as a 500. Returned buckets are always
independent deep copies, so callers may mutate them without corrupting the
shared defaults.
"""
from __future__ import annotations

import copy
import json
import logging

import config
from config import CONTENT_TYPES, MODE_TYPE_BUCKETS, normalize_content_type
from core.models import Video
from storage import db

logger = logging.getLogger(__name__)

_CONTENT_TYPE_SET = set(CONTENT_TYPES)

# Final-stage candidate tuple: (video, fit score, normalized quality, content_type),
# same shape as the orchestrator's candidate list.
Candidate = tuple[Video, float, float, "str | None"]


def _valid(buckets: object) -> bool:
    """Validate a stored bucket override.

    Must be a list of exactly four lists of strings, and the four tiers
    together must contain every type in ``CONTENT_TYPES`` exactly once -- no
    missing, extra or duplicate entries (within or across tiers). The
    vocabulary always comes from ``CONTENT_TYPES``; never hard-code its size.
    """
    if not isinstance(buckets, list) or len(buckets) != 4:
        return False
    flat: list[str] = []
    for tier in buckets:
        if not isinstance(tier, list) or not all(isinstance(s, str) for s in tier):
            return False
        flat.extend(tier)
    return (len(flat) == len(CONTENT_TYPES)               # total count preserved
            and len(set(flat)) == len(CONTENT_TYPES)      # no duplicates
            and set(flat) == _CONTENT_TYPE_SET)           # exactly the vocabulary


def effective_type_buckets(user_id: int, mode_key: str) -> list[list[str]]:
    """Return the user's effective four-tier content-type buckets for a mode.

    No override, or an invalid one, falls back to the mode default with a
    warning (fail-soft). Always returns a deep copy. An unknown ``mode_key``
    deliberately raises ``KeyError``: that is a programming error, not bad
    data, and should fail loudly.
    """
    default = MODE_TYPE_BUCKETS[mode_key]
    try:
        row = db.get_mode_prefs(user_id, mode_key)
        if row is None or row["type_buckets_json"] is None:
            # Buckets are nested lists, so a shallow copy would still share tiers.
            return copy.deepcopy(default)
        buckets = json.loads(row["type_buckets_json"])
        if not _valid(buckets):
            raise ValueError("覆盖桶结构非法或与全局词表(CONTENT_TYPES)不守恒")
        return copy.deepcopy(buckets)
    except Exception:
        # Personalized buckets are a nice-to-have: bad data degrades to the
        # defaults instead of breaking the whole recommendation pipeline.
        logger.warning(
            "effective_type_buckets: 坏覆盖桶,回退默认 user=%s mode=%s",
            user_id, mode_key,
        )
        return copy.deepcopy(default)


def promote_types(default_buckets: list[list[str]], preferred: list[str],) -> list[list[str]]:
    """Move each preferred content type up one tier.

    Types already in the first tier stay put; unselected types are unchanged.
    The vocabulary is preserved by construction (items are only moved).
    ``preferred`` is de-duplicated in order so each type moves exactly one
    tier. The input is never mutated (it is often a shared default).
    """
    new = copy.deepcopy(default_buckets)

    for ct in dict.fromkeys(preferred):        # ordered de-duplication
        for i, tier in enumerate(new):
            if ct in tier:
                if i > 0:
                    tier.remove(ct)
                    new[i - 1].append(ct)
                break
    return new


def load_excluded_types(user_id: int, mode_key: str) -> set[str]:
    """Return the user's permanently excluded content types for a mode (normalized).

    Fail-soft granularity:
      - no row / NULL -> empty set, no warning (normal state);
      - DB error or malformed JSON -> empty set with a warning;
      - out-of-vocabulary slugs -> only that item is dropped (with a warning);
        the remaining valid items still apply.
    """
    try:
        row = db.get_mode_prefs(user_id, mode_key)
    except Exception:
        logger.warning(
            "load_excluded_types: db 查询异常,回退空集 user=%s mode=%s",
            user_id, mode_key,
        )
        return set()

    if row is None or row["excluded_types_json"] is None:
        return set()

    try:
        raw = json.loads(row["excluded_types_json"])
        # Reject valid-but-wrong JSON shapes (dict/int/null, or non-string items).
        if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
            raise ValueError("excluded_types_json 不是字符串 list")
    except Exception:
        logger.warning(
            "load_excluded_types: 坏 JSON/结构非法,回退空集 user=%s mode=%s",
            user_id, mode_key,
        )
        return set()

    result: set[str] = set()
    for slug in raw:
        norm = normalize_content_type(slug)
        if norm not in _CONTENT_TYPE_SET:
            logger.warning(
                "load_excluded_types: 排除清单含词表外 slug %r,已丢弃 "
                "user=%s mode=%s", slug, user_id, mode_key,
            )
            continue
        result.add(norm)
    return result


def partition_excluded(candidates: list[Candidate], excluded: set[str],
                       exempt_ids: set[str] | frozenset[str] = frozenset(),
                       ) -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """Split candidates by the user's excluded types into ``(kept, dropped, exempted)``.

    Used by the S7 exclusion filter; the caller records ``dropped`` (and
    ``exempted``) in ``filter_results`` so the user can see and appeal them.

    - A ``content_type`` of None (judge degraded) is never excluded: unknown
      means not guilty.
    - Non-None types are normalized before the membership test (idempotent).
    - ``exempt_ids`` are video ids belonging to exempt profile entries; an
      item that would be dropped but is exempt goes to ``kept`` and is also
      listed in ``exempted`` for transparency. Items without an entry
      assignment are not in that set and so are not exempt.

    Works with an empty ``excluded`` set, but skipping the call entirely in
    that case is the caller's fast path.
    """
    kept: list[Candidate] = []
    dropped: list[Candidate] = []
    exempted: list[Candidate] = []
    for c in candidates:
        ctype = c[3]
        if ctype is not None and normalize_content_type(ctype) in excluded:
            if c[0].video_id in exempt_ids:
                kept.append(c)
                exempted.append(c)
            else:
                dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped, exempted


def effective_pool_ratio(user_id: int, mode: str) -> tuple[float, str]:
    """Share of channel-based supply that goes to the shared type pool.

    Returns ``(ratio, source)`` with three levels of precedence:
      1. the user's slider (``pool_ratio`` not NULL) -> ``"manual"``;
         0.0 means only the user's own channels, 1.0 only the type pool;
      2. otherwise slide linearly from ``POOL_RATIO_NEW`` to
         ``POOL_RATIO_MATURE`` with signal maturity -> ``"auto"``;
      3. on any error, ``POOL_RATIO_NEW``.

    New users have little preference data, so they get more of the shared
    pool; as signals accumulate their own channels take over.
    """
    try:
        row = db.get_mode_prefs(user_id, mode)

        if row is not None and row["pool_ratio"] is not None:
            return float(row["pool_ratio"]), "manual"

        # Maturity in [0, 1]: recent signals / STABLE_MIN_SIGNALS, capped.
        signals = db.count_signals(user_id, mode, config.SIGNAL_WINDOW_DAYS)
        maturity = min(1.0, signals / config.STABLE_MIN_SIGNALS) if config.STABLE_MIN_SIGNALS else 1.0

        # Linear rather than stepped, so the screen mix never jumps at a threshold.
        ratio = config.POOL_RATIO_NEW - (config.POOL_RATIO_NEW - config.POOL_RATIO_MATURE) * maturity
        return ratio, "auto"

    # Fail-safe direction: without a reliable preference estimate, lean on the pool.
    except Exception as exc:                       # noqa: BLE001 -- fail-soft fallback
        logger.warning("effective_pool_ratio 降级为新用户档(u%s/%s):%r", user_id, mode, exc)
        return config.POOL_RATIO_NEW, "auto"
