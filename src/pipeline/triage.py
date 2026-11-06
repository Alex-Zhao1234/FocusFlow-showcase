"""S4 quality triage -- the cost gate in front of the LLM quality review.

Each candidate goes into one of three buckets:

- ``cached``: a quality verdict from the last ``QUALITY_CACHE_TTL_DAYS`` days
  exists. Cached items go here regardless of pass/fail; recording failed ones
  in ``filter_results`` is the caller's job -- triage only routes, it never
  rules.
- ``direct``: statistical score >= ``QUALITY_PASS_THRESHOLD`` and eligible
  for direct pass; no LLM call.
- ``review``: everything else, sent to the S5 LLM review. Low scores also go
  to review rather than being rejected: statistics lie for new videos and
  small channels, so there is no statistical veto (cold-start protection; no
  hard absolute view-count thresholds).

This stage costs nothing itself, but its direct-pass rate (target 30-40%)
directly determines S5 LLM volume.

Two invariants:
  - No ``user_id``: quality is a property of the content, so no per-user
    tables are read or written.
  - S4 never writes. The verdict cache only stores S5 LLM decisions.
    Statistical scores are fresh on every refresh and free to recompute, so
    caching them saves nothing and would freeze a free signal for days. S4
    reads the cache, S5 writes it -- a single writer, no races.

Stage rules: no imports of other stages or ``sources/``, no direct SQL (batch
cache lookup goes through ``storage.db``), all thresholds in ``config``.
"""


import json
import logging
import sqlite3

from config import (
    ALLOW_LIKE_HIDDEN_DIRECT, DIRECT_LIKE_TERM_FLOOR,
    QUALITY_PASS_THRESHOLD, QUALITY_W_LIKE, QUALITY_W_COMMENT,
    QUALITY_LIKE_RATE_SCALE, QUALITY_COMMENT_RATE_SCALE, QUALITY_CACHE_TTL_DAYS,
)
from core.models import QualityVerdict, Triage, Video
from storage import db

logger = logging.getLogger(__name__)


def _like_term(v: Video) -> float:
    """Like component in [0, 1]: like rate x ``QUALITY_LIKE_RATE_SCALE``, capped at 1.

    The single definition shared by ``quality_score`` and ``direct_eligible``,
    so swapping in e.g. a Wilson lower bound only needs to change this
    function. Only called when likes are visible.
    """
    return min(v.like_count / max(v.view_count, 1) * QUALITY_LIKE_RATE_SCALE, 1.0)


def quality_score(v: Video) -> float:
    """Statistical composite quality score in [0, 1].

    ``like term * QUALITY_W_LIKE + comment term * QUALITY_W_COMMENT``; each
    rate is scaled and capped at 1.0, with ``max(view, 1)`` against division
    by zero. Only views, likes and comments are available to a third-party
    app (watch time, retention and CTR are first-party data), hence the thin
    signal.

    If the channel hides likes, only the comment branch is used: a like count
    of 0 there means "unknown", not "nobody liked it", and mixing it in would
    penalize whole channels.

    Not to be confused with ``Video.engagement_rate``, which is a different
    formula for a different purpose (hidden-gem scoring in the ranker).
    """
    if v.like_hidden:
        return min(v.comment_count / max(v.view_count, 1) * QUALITY_COMMENT_RATE_SCALE, 1.0)

    comment_rate = v.comment_count / max(v.view_count, 1)
    return (_like_term(v) * QUALITY_W_LIKE
            + min(comment_rate * QUALITY_COMMENT_RATE_SCALE, 1.0) * QUALITY_W_COMMENT)


def direct_eligible(v: Video, score: float, direct_threshold: float | None = None) -> bool:
    """Eligibility gate for direct pass (skipping the LLM review).

    Passing the score threshold is necessary but not sufficient:
      - with likes hidden, a single signal cannot buy an exemption unless
        ``ALLOW_LIKE_HIDDEN_DIRECT`` is set (default False);
      - otherwise the like term must also be >= ``DIRECT_LIKE_TERM_FLOOR``.

    Ineligible items go to review and are *not* written to
    ``filter_results``: routing is not filtering. This is a robustness guard
    against parameter changes rather than an active filter under current
    settings.
    """
    thr = QUALITY_PASS_THRESHOLD if direct_threshold is None else direct_threshold
    if score < thr:
        return False
    if v.like_hidden:
        return ALLOW_LIKE_HIDDEN_DIRECT
    return _like_term(v) >= DIRECT_LIKE_TERM_FLOOR


def _verdict_from_row(row: sqlite3.Row) -> QualityVerdict:
    """Convert a ``quality_verdicts`` row into a ``QualityVerdict``.

    ``passed`` 0/1 -> bool, ``categories`` JSON -> list, NULLs -> model
    defaults. A NULL ``quality_grade`` (older rows) yields None, which the
    orchestrator handles via the legacy statistical-score path.
    """
    keys = row.keys()
    return QualityVerdict(
        video_id=row["video_id"],
        passed=bool(row["passed"]),
        quality_grade=row["quality_grade"] if "quality_grade" in keys else None,
        categories=json.loads(row["categories"]) if row["categories"] else [],
        confidence=row["confidence"] if row["confidence"] is not None else 0.0,
        reason=row["reason"] or "",
    )


def triage(videos: list[Video], direct_threshold: float | None = None) -> Triage:
    """S4 quality triage: route each video to cached / direct / review.

    Order is fixed: a cache hit takes precedence over the statistical score.
    Writes nothing (the verdict table is unchanged no matter how often this
    runs) and takes no ``user_id``. ``direct_threshold`` may be passed per
    mode; None means ``QUALITY_PASS_THRESHOLD``.
    """

    cache = db.get_quality_verdicts([v.video_id for v in videos],
                                  ttl_days=QUALITY_CACHE_TTL_DAYS)
    direct: list[Video] = []
    review: list[Video] = []
    # Videos already reviewed by the LLM recently; no need to pay again.
    cached: list[tuple[Video, QualityVerdict]] = []

    for v in videos:
        row = cache.get(v.video_id)
        if row is not None:
            cached.append((v, _verdict_from_row(row)))
        elif direct_eligible(v, quality_score(v), direct_threshold):
            direct.append(v)
        else:
            review.append(v)

    total = len(videos)
    logger.info("triage: 输入 %d → direct %d / review %d / cached %d(直通率 %.0f%%)",
                total, len(direct), len(review), len(cached),
                (len(direct) / total * 100) if total else 0.0)
    return Triage(direct=direct, review=review, cached=cached)
