"""Sub-direction rotation for profile-entry searches.

Pure function, no I/O: given an entry's representative query, its derived
sub-directions and a per-entry cursor, return the query this entry should
search this time.

The sequence is ``[representative query] + [each direction's query]``. Cursor
0 is always the representative query, so the first refresh after saving an
entry behaves exactly like a plain single-query entry; each later turn moves to
the next direction. Feature flags (``PROFILE_SUBTOPICS``,
``PROFILE_SUBTOPIC_SEARCH``), the user's opt-out, or missing sub-directions all
fall back to the representative query.

With ``en_every = N > 0``, a direction slot whose ``cursor % N == 0`` uses the
direction's English query instead (falling back to Chinese); the
representative slot never switches language. The caller advances the cursor
per entry; this module keeps no state.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import config


def entry_rotation(search_query: str | None, subtopics: Sequence[Mapping[str, Any]] | None,
                   cursor: int = 0, en_every: int | None = None,
                   enabled: bool = True, lang: str = "zh",
                   search_query_en: str | None = None) -> str | None:
    """Return the search query for this entry at ``cursor``.

    ``enabled`` is the user's "divergent search" setting; False always returns
    the representative query. With ``lang="en"`` every slot prefers its
    English text (``search_query_en`` for the representative slot,
    ``query_en`` for directions), falling back to the original; the Chinese
    path is unaffected and ``en_every`` only applies there.
    """
    head = (search_query or "").strip() or None
    en_mode = lang == "en"
    if en_mode:
        head = (search_query_en or "").strip() or head
    if not (config.PROFILE_SUBTOPICS and config.PROFILE_SUBTOPIC_SEARCH and enabled) or not subtopics:
        return head
    seq: list[tuple[str, str]] = ([(head, "")] if head else [])       # (primary query, English query)
    for s in subtopics:
        zh = (s["query_zh"] or "").strip()
        if zh:
            seq.append((zh, (s["query_en"] or "").strip()))
    if not seq:
        return head
    i = cursor % len(seq)
    zh, en = seq[i]
    if en_mode:
        return en or zh                      # seq[0] already holds the English representative query
    n = config.PROFILE_SUBTOPIC_EN_EVERY if en_every is None else en_every
    is_direction = not (head and i == 0)
    if n and n > 0 and is_direction and cursor % n == 0 and en:
        return en
    return zh
