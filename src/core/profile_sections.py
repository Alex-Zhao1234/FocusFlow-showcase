"""Three-section layout of a user's free-text profile.

Pure functions (no I/O, only config constants) over the profile's raw text,
which is split into three sections that never overwrite each other:

- the user's own section — the only part the user writes directly;
- chat-derived short-term signals — dated, FIFO-capped and expiring;
- chat-derived long-term signals — a frequency ledger with one line per
  statement, ``[date|xN|mode] text``. Entries are grouped by mode and each mode
  keeps at most CHAT_LONG_MAX entries independently. An exact repeat in the same
  mode increments its count; when a mode is full, the lowest count is evicted
  first and ties go to the entry not restated for longest; entries touched in
  the current turn are protected. Legacy entries with no detectable mode go to
  the ``general`` bucket.

merge/replace/consolidate are deterministic text transforms and therefore
testable. Deciding that statements are semantically equivalent is done by an LLM
in the agent layer; applying that decision is ``consolidate_long`` here.
"""
import re
import time

SHORT_HEADER = "【聊天提炼·短期】(系统维护区,高度可变)"
LONG_HEADER = "【聊天提炼·长期】(系统维护区,较为固定)"

_DATE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\] ")

# Long-term entry metadata: `[YYYY-MM-DD|xN|mode] text` — date of the last
# restatement, number of times stated, and owning mode (a config.MODES key, or
# general). The mode part is optional for entries written before grouping.
_LONG_META_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\|x(\d+)(?:\|([a-z_]+))?\] (.*)$")

GENERAL_BUCKET = "general"   # fallback bucket for entries with no detectable mode; same cap as each mode

def parse_long_line(line: str) -> tuple[str, int, str | None, str]:
    """Long-term line → (text, count, last-restated date | None, bucket).
    Unmarked legacy lines → (line, 1, None, general); `[date|xN]` lines without a
    mode → (text, N, date, general)."""
    m = _LONG_META_RE.match(line)
    if not m:
        return line, 1, None, GENERAL_BUCKET
    return m.group(4), int(m.group(2)), m.group(1), m.group(3) or GENERAL_BUCKET

def format_long_line(text: str, count: int, date: str | None, bucket: str) -> str:
    """Inverse of parse_long_line. Legacy entries with date=None stay unmarked
    (lazy upgrade: metadata is only added once an entry is hit or merged, so a
    single merge does not rewrite the section with fabricated dates)."""
    if date is None:
        return text
    return f"[{date}|x{count}|{bucket}] {text}"


def long_texts(raw_text: str, bucket: str) -> list[str]:
    """Plain texts (metadata stripped) of the long-term entries in one bucket,
    used for exact matching and as input to the LLM duplicate check."""
    return [p[0] for p in map(parse_long_line, split_sections(raw_text)[2])
            if p[3] == bucket]

def _evict_long(parsed: list[tuple[str, int, str | None, str]],
                protected: set[tuple[str, str]],
                cap: int) -> list[tuple[str, int, str | None, str]]:
    """Evict over-cap long-term entries, independently per mode.

    Within a mode the lowest count goes first; ties go to the entry not restated
    for longest (None counts as oldest). ``protected`` (text, mode) pairs touched
    this turn are exempt, so a new statement is never silently dropped. If the
    protected entries alone exceed the cap, the same rule applies within them.
    Entries are removed one at a time because the candidate set changes after
    each removal. Mutates and returns ``parsed``."""
    from collections import Counter
    sizes = Counter(p[3] for p in parsed)
    for bucket in [b for b, n in sizes.items() if n > cap]:
        while sizes[bucket] > cap:
            members = [p for p in parsed if p[3] == bucket]
            # Fall back to all members if every entry in the bucket is protected.
            pool = [p for p in members if (p[0], p[3]) not in protected] or members
            # Order by (count, date); a None date sorts as "" i.e. oldest.
            parsed.remove(min(pool, key=lambda p: (p[1], p[2] or "")))
            sizes[bucket] -= 1
    return parsed


def split_sections(raw_text: str) -> tuple[str, list[str], list[str]]:
    """Split into (user section text, short-term entries, long-term entries).
    Legacy text with no headers is entirely the user section. Entries are stored
    one per line as `- <text>`."""
    def _cut(text: str, header: str) -> tuple[str, str]:
        idx = text.find(header)
        if idx < 0:
            return text, ""
        return text[:idx], text[idx + len(header):]

    rest, short_and_after = _cut(raw_text, SHORT_HEADER)
    if short_and_after:
        short_part, long_part = _cut(short_and_after, LONG_HEADER)

    # No short-term header: there may still be a long-term section on its own.
    else:
        rest, long_part = _cut(rest, LONG_HEADER)
        short_part = ""

    def _lines(block: str) -> list[str]:
        return [ln[2:].strip() for ln in block.splitlines() if ln.startswith("- ")]

    return rest.rstrip(), _lines(short_part), _lines(long_part)


def compose(user_part: str, short_lines: list[str], long_lines: list[str]) -> str:
    """Inverse of split_sections: reassemble the full profile text. A section
    header is written only when the section has entries; sections are separated
    by a blank line."""
    parts = [user_part.rstrip()]
    if short_lines:
        parts.append(SHORT_HEADER + "\n" + "\n".join(f"- {l}" for l in short_lines))
    if long_lines:
        parts.append(LONG_HEADER + "\n" + "\n".join(f"- {l}" for l in long_lines))
    return "\n\n".join(p for p in parts if p)


def merge_signals(raw_text: str, short_signals: list[str],
                  long_signals: list[tuple[str, str]],
                  *, today: str | None = None) -> str:
    """The single write path for chat-derived signals; the user section is untouched.

    Short-term: drop expired entries → append new entries stamped with today's
    date → keep the newest CHAT_SHORT_MAX (FIFO). Long-term, given (text, mode)
    pairs: an exact match in the same mode increments the count and refreshes the
    date, otherwise the entry is appended as x1; over-cap modes are evicted
    independently (see ``_evict_long``), with entries touched this turn exempt.

    Signals are flattened to a single line: the section format is one entry per
    line, and an embedded newline (common in LLM output) would silently truncate
    the entry on the next split. This is enforced here because this is the only
    write path."""

    import config
    today = today or time.strftime("%Y-%m-%d")
    short_signals = [" ".join(s.split()) for s in short_signals]
    long_signals = [(" ".join(s.split()), bucket) for s, bucket in long_signals]
    user_part, short_lines, long_lines = split_sections(raw_text)

    def _fresh(line: str) -> bool:
        m = _DATE_RE.match(line)
        if not m:
            return True                      # an undated entry is kept rather than dropped on a parse failure
        age = (time.mktime(time.strptime(today, "%Y-%m-%d"))
               - time.mktime(time.strptime(m.group(1), "%Y-%m-%d"))) / 86400
        return age <= config.CHAT_SHORT_TTL_DAYS

    short_lines = [l for l in short_lines if _fresh(l)]
    short_lines += [f"[{today}] {s}" for s in short_signals]
    short_lines = short_lines[-config.CHAT_SHORT_MAX:]           # oldest first, so overflow drops the oldest

    # Long-term: the same text under different modes is an independent entry.
    parsed = [parse_long_line(l) for l in long_lines]
    touched: set[tuple[str, str]] = set()
    for s, bucket in long_signals:
        for i, (text, count, _, b) in enumerate(parsed):
            if text == s and b == bucket:
                parsed[i] = (text, count + 1, today, bucket)
                break
        else:
            parsed.append((s, 1, today, bucket))
        touched.add((s, bucket))
    parsed = _evict_long(parsed, touched, config.CHAT_LONG_MAX)
    long_lines = [format_long_line(*p) for p in parsed]

    return compose(user_part, short_lines, long_lines)


def consolidate_long(raw_text: str, matched_texts: list[str], merged_text: str,
                     bucket: str, *, today: str | None = None) -> str:
    """Apply a semantic merge decided by the LLM.

    Removes the ``matched_texts`` entries in ``bucket`` and adds one umbrella
    entry ``merged_text`` whose count is the sum of the merged counts plus one
    (the current statement) and whose date is today. Identical texts in other
    modes are unaffected. If the mode is still over its cap, ``_evict_long``
    applies with the umbrella entry protected. If none of the matched texts are
    found, this degrades to appending the umbrella entry as x1 — the caller has
    already validated the match, so this is a fallback rather than an error."""
    import config
    today = today or time.strftime("%Y-%m-%d")
    merged_text = " ".join(merged_text.split())                        # LLM output: flatten newlines too
    user_part, short_lines, long_lines = split_sections(raw_text)
    parsed = [parse_long_line(l) for l in long_lines]
    matched_set = set(matched_texts)

    hit = [p for p in parsed if p[0] in matched_set and p[3] == bucket]
    parsed = [p for p in parsed if not (p[0] in matched_set and p[3] == bucket)]

    parsed.append((merged_text, sum(c for _, c, _, _ in hit) + 1, today, bucket))
    parsed = _evict_long(parsed, {(merged_text, bucket)}, config.CHAT_LONG_MAX)
    return compose(user_part, short_lines, [format_long_line(*p) for p in parsed])


def replace_user_part(raw_text: str, new_user_text: str) -> str:
    """Replace the user section, leaving the chat sections untouched. If the new
    text itself contains section headers, only the part before them is used —
    this prevents a user who edits and resubmits the full three-section text from
    injecting or duplicating the chat sections."""
    _, short_lines, long_lines = split_sections(raw_text)
    new_user_part, _, _ = split_sections(new_user_text)
    return compose(new_user_part, short_lines, long_lines)
