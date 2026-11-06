"""Pure helpers for the S6 mode-fit verdict cache.

Two pure functions (no I/O, no agent/db imports — the prompt version and model
name are passed in by the caller): computing the cache fingerprint, and turning
a cached row back into a ``ModeFitVerdict``. The fingerprint covers the whole
context that determines the judge's answer (prompt version, model, language,
the mode's profile entries, negative preferences); any change to it invalidates
the cached verdicts and forces a re-judge.
"""
from __future__ import annotations

import hashlib

from core.models import ModeFitVerdict


def fingerprint(prompt_version: str, model: str, lang: str,
                profile_lines: list[str] | None, neg_pref_lines: list[str] | None) -> str:
    """sha1(prompt version | model | language | profile entries | negative prefs)[:16].

    These five inputs are exactly the variables that can change the judge's
    verdict. None and [] are treated the same. The hash is only compared for
    equality (not a security use), so sha1 is fine; truncation just keeps the
    column short."""
    raw = "|".join([
        prompt_version or "", model or "", lang or "",
        "\n".join(profile_lines or []),
        "\n".join(neg_pref_lines or []),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16] 

def verdict_from_row(row) -> ModeFitVerdict:
    """Rebuild a verdict from a cache row (degraded=False, cached=True) so cache
    hits and freshly judged videos flow through the same downstream path. The
    fit threshold is still applied on the consumer side."""
    return ModeFitVerdict(
        video_id=row["video_id"], fit=bool(row["fit"]),
        fit_score=float(row["fit_score"] if row["fit_score"] is not None else 0.5),
        content_type=row["content_type"], degraded=False,
        reason=row["reason"] or "", profile_entry=row["profile_entry"], cached=True)
