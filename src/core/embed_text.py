"""Text preparation for vector embedding and full-text search.

Pure functions (no I/O, no model) that turn stored raw text into the text that
gets embedded, the segmented text that goes into the FTS5 index, and a
``text_hash`` used to decide which rows need re-embedding.

- ``clean_text``: strips URLs, hashtags, e-mail addresses and runs of
  whitespace. Sponsor links and social handles in descriptions are noise
  shared by nearly every video and contribute nothing semantically.
- ``video_doc`` / ``channel_doc`` / ``entry_doc``: the three embedding
  templates. Changing a template changes every hash and triggers a full
  re-embed, so edit them with care.
- ``text_hash``: sha1 of the template text; the embedding job skips rows
  whose hash is unchanged.
- ``segment``: jieba word segmentation joined by spaces. FTS5's unicode61
  tokenizer treats an entire run of CJK characters as one token, so Chinese
  must be pre-segmented before indexing; English/numeric tokens pass through.

This module never imports the embedding model or sqlite; jieba is imported
lazily so only processes that actually segment text pay for loading its
dictionary.
"""
from __future__ import annotations

import hashlib
import logging
import re

import config

_URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
_HASHTAG_RE = re.compile(r"(?<!\w)#\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def clean_text(s: str | None) -> str:
    """Remove URLs, #hashtags and e-mail addresses, collapse whitespace, strip.
    None → ''."""
    if not s:
        return ""
    s = _URL_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def video_doc(title: str | None, description: str | None) -> str:
    """Video embedding text: ``clean(title) | clean(desc)[:EMBED_DESC_CHARS]``;
    title only when the description is empty (no dangling separator)."""
    t = clean_text(title)
    d = clean_text(description)[: config.EMBED_DESC_CHARS]
    return f"{t} | {d}" if d else t


def channel_doc(title: str | None, chip_names: list[str], upload_titles: list[str]) -> str:
    """Channel embedding text: ``name | chip names, … | title1; title2; …``,
    using at most EMBED_CHANNEL_TITLES_MAX recent upload titles."""
    parts = [clean_text(title)]
    if chip_names:
        parts.append(", ".join(chip_names))
    titles = [clean_text(x) for x in upload_titles[: config.EMBED_CHANNEL_TITLES_MAX]]
    titles = [x for x in titles if x]
    if titles:
        parts.append("; ".join(titles))
    return " | ".join(p for p in parts if p)

def entry_doc(text: str | None, search_query: str | None) -> str:
    """Profile-entry / negative-preference embedding text: ``text | search_query``
    (the user's wording plus the derived search query); text only when there is
    no query."""
    t = clean_text(text)
    q = clean_text(search_query)
    return f"{t} | {q}" if q else t

def text_hash(doc: str) -> str:
    """Incremental-embedding key: sha1 hex digest (40 chars). Used only for
    equality checks, not for security."""
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()


def has_cjk(s: str | None) -> bool:
    """True if the text contains any CJK ideograph."""
    return bool(s) and _CJK_RE.search(s) is not None


def segment(s: str | None) -> str:
    """Segment text for FTS5 with jieba (accurate mode), joined by single spaces;
    whitespace-only tokens are dropped. Empty input returns ''. Loading the
    dictionary takes about a second and is deferred to the first call."""
    s = clean_text(s)
    if not s:
        return ""
    import jieba  # lazy import: only processes that segment text load the dictionary
    jieba.setLogLevel(logging.WARNING)

    return " ".join(tok for tok in jieba.lcut(s) if tok.strip())
