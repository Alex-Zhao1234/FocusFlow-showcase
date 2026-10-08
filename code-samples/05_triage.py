"""
S4 · triage. The LLM cost gate, complete.

Position in the pipeline: after S3 (free rules), before S5 (the quality judge,
which costs money per batch). It splits candidates into three buckets and
makes NO decisions of its own:

  cached  an LLM verdict from the last N days exists. Good or bad, it goes in
          this bucket; the orchestrator applies the stored verdict. Free.
  direct  statistical quality is above the bar AND the eligibility gate
          passes. Skips the judge. Free.
  review  everything else, including low scores. Low statistical signal is
          sent to the judge rather than rejected outright, because new videos
          and small channels have sparse numbers and a statistical veto would
          punish them for being new. Costs a model call.

Disciplines this module keeps:
  * does not import any other stage, and does not touch the YouTube adapter;
  * no SQL of its own (the cache lookup is one storage call);
  * no user identity: quality is a property of the content, not of the viewer;
  * zero writes: only the judge writes verdicts. Running triage a thousand
    times leaves the verdict table unchanged;
  * zero magic numbers: every weight, scale and threshold is a named config
    value, and a per-mode override can be passed in by the caller.
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
    """Like component in [0, 1]. The single definition used by both the score
    and the eligibility gate, so changing it (e.g. to a Wilson lower bound)
    changes both in lockstep. Only called when likes are visible."""
    return min(v.like_count / max(v.view_count, 1) * QUALITY_LIKE_RATE_SCALE, 1.0)


def quality_score(v: Video) -> float:
    """Composite statistical quality in [0, 1] from the only three public
    signals a third party gets: views, likes, comments.

    Channels that hide their like count get the comment branch only. A hidden
    like count is "unknown", not "zero people liked this", and folding it into
    the weighted sum would penalise an entire class of channels."""
    if v.like_hidden:
        return min(v.comment_count / max(v.view_count, 1) * QUALITY_COMMENT_RATE_SCALE, 1.0)
    comment_rate = v.comment_count / max(v.view_count, 1)
    return (_like_term(v) * QUALITY_W_LIKE
            + min(comment_rate * QUALITY_COMMENT_RATE_SCALE, 1.0) * QUALITY_W_COMMENT)


def direct_eligible(v: Video, score: float, direct_threshold: float | None = None) -> bool:
    """Gate for skipping the judge. A score above the bar is necessary, not
    sufficient: a single hidden-likes signal cannot buy a pass unless the
    policy switch allows it, and visible likes must clear their own floor.
    A failed gate sends the video to review; it does NOT write a ledger row,
    because triage routes, it never rejects."""
    thr = QUALITY_PASS_THRESHOLD if direct_threshold is None else direct_threshold
    if score < thr:
        return False
    if v.like_hidden:
        return ALLOW_LIKE_HIDDEN_DIRECT
    return _like_term(v) >= DIRECT_LIKE_TERM_FLOOR


def _verdict_from_row(row: sqlite3.Row) -> QualityVerdict:
    """Rehydrate a stored verdict. SQLite only holds 0/1, JSON strings and
    NULL; this restores bools, lists and model defaults. A NULL grade marks a
    row written before grades existed; the orchestrator handles that path."""
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
    """Route each candidate. Cache beats statistics: a stored verdict wins
    even if today's numbers would have passed directly."""
    cache = db.get_quality_verdicts([v.video_id for v in videos],
                                    ttl_days=QUALITY_CACHE_TTL_DAYS)
    direct: list[Video] = []
    review: list[Video] = []
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
    logger.info("triage: %d in -> direct %d / review %d / cached %d (direct rate %.0f%%)",
                total, len(direct), len(review), len(cached),
                (len(direct) / total * 100) if total else 0.0)
    return Triage(direct=direct, review=review, cached=cached)
