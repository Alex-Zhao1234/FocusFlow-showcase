"""S3 rule-based prefilter.

Cheap rule filtering (~150 -> ~80 candidates) with zero LLM cost; it directly
determines how many items reach the S5 LLM review.

Rules are ordered by signal strength and check cost. The first hit filters the
video and short-circuits the rest, so each video records exactly one reason in
``filter_results``:

  1. Blocked channels. Search/trending strategies do not go through the
     channel-level block exclusion, so they must be caught here. An explicit
     user block is the strongest signal, so it runs first. Channels in a
     cooldown period are treated the same way.
  2. Blacklisted words, matched against title + first 200 characters of the
     description + tags. Two matching tracks: ASCII terms as whole words
     (lookaround), CJK terms as substrings; on multiple hits the longest term
     is reported.
  3. Duration window from the mode. ``duration_sec == 0`` (metadata enrichment
     failed) is always below any mode minimum and so is naturally filtered.
  4. Category whitelist. Channel-based sources (subscribed, type pool, seed
     and expanded channels) and user-trusted channels are exempt -- the user or
     the pool curation already vouches for the topic. Only this rule is
     waived; later quality stages are unchanged. A None category counts as
     "not in the whitelist".

Intent-sourced videos skip rules 2-4 (whether they match the stated intent is
decided by the S6 judge, not static rules) but must still pass the block check
and an anti-Shorts minimum duration.

Precondition: videos must already be persisted, because
``filter_results.video_id`` is a foreign key to ``videos``.

Stage rules: no imports of other stages or ``sources/``; all DB access goes
through ``storage.db``. Helper functions are pure and testable offline.
"""
import logging
import re
import time
from functools import lru_cache

import config
from core.models import Mode, Video
from storage import db

logger = logging.getLogger(__name__)

# Only the first 200 description characters are matched: clickbait / anxiety
# marketing terms almost always appear near the start; matching the full text
# only increases false positives.
_DESC_SLICE = 200


def _haystack(v: Video) -> str:
    """Text matched against the blacklist: title + description head + tags, lowercased."""
    return f"{v.title} {v.description[:_DESC_SLICE]} {' '.join(v.tags)}".lower()

# Two-track word-boundary matching:
_ASCII_TERM = re.compile(r"^[\x20-\x7e]+$")    # printable ASCII -> whole-word track
_GUARD = "[a-z0-9]"                            # word characters; _, -, and CJK all count as boundaries


@lru_cache(maxsize=16)
def _compile_blacklist(terms: tuple[str, ...]):
    """Precompile a blacklist into matcher functions, cached by content.

    ``Mode`` is a non-frozen Pydantic model and not hashable, so the term tuple
    is the cache key; each mode compiles once and then always hits the cache.
    Returns ``[(term, matcher)]`` sorted by term length descending, which is
    the attribution priority: when several terms match, the most specific
    (longest) one is reported, independent of config order.
    """
    compiled = []
    for t in terms:
        low = t.lower()
        if _ASCII_TERM.match(t):
            pat = re.compile(rf"(?<!{_GUARD}){re.escape(low)}(?!{_GUARD})")
            compiled.append((t, pat.search))
        else:
            compiled.append((t, lambda hay, _w=low: _w in hay))
    compiled.sort(key=lambda p: len(p[0]), reverse=True)
    return tuple(compiled)


def _hit_blacklist_word(v: Video, blacklist: list[str]) -> str | None:
    """Return the matching blacklist term (original spelling), or None.

    Case-insensitive; ASCII terms match whole words, CJK terms match
    substrings; attribution is by term length descending.
    """
    hay = _haystack(v)
    # Lists are unhashable, so convert to a tuple for the cache key.
    for word, hit in _compile_blacklist(tuple(blacklist)):
        if hit(hay):
            return word
    return None


def _in_duration_window(duration_sec: int, min_sec: int, max_sec: int | None) -> bool:
    """Whether a duration falls inside a window; ``max_sec=None`` means no upper bound.

    Mode-bound checks use ``Mode.fits_duration`` (shared by this rule and the
    S2 early duration filter). This function is kept for callers that pass an
    explicit window (offline evaluation scripts and tests); its logic must stay
    identical to that predicate.
    """
    if duration_sec < min_sec:
        return False
    if max_sec is not None and duration_sec > max_sec:
        return False
    return True


def _category_ok(v: Video, mode: Mode, trusted: set[str]) -> bool:
    """Category whitelist check, with exemptions for channel-based sources and trusted channels.

    Exemptions waive only the topic check, not the S4-S6 quality checks.
    A None category is treated as not whitelisted.
    """
    if v.source_strategy in ("subscribed", "channel_pool", "seed_channel", "expanded_channel"):
        # - subscribed: the user's own channels express trust.
        # - channel_pool: curated pool channels already passed subscriber /
        #   video-count thresholds and are topic-matched via chips; YouTube
        #   categories do not align with mode whitelists and would wrongly
        #   reject large parts of the pool.
        # - seed_channel: hand-picked by the user for a profile entry; most
        #   seed videos would otherwise be lost at this rule.
        # - expanded_channel: verified by the profile-expansion judge against
        #   the channel's portfolio.
        # Duration, blacklist, block checks and S4-S6 quality still apply.
        return True
    if v.channel_id in trusted:
        return True
    return v.category_id in mode.yt_category_ids


def prefilter(videos: list[Video], mode: Mode, user_id: int, refresh_id: str) -> list[Video]:
    """Rule-based prefilter; each filtered video is recorded with its reason (stage='prefilter').

    Local conservation: ``len(input) == len(output) + number of new
    filter_results rows``. ``user_id`` / ``refresh_id`` are passed in rather
    than read from global state: prefilter is stateless and per-user isolation
    is the caller's responsibility.
    """

    black_channels = db.blacklisted_channel_ids(user_id)
    cooled_channels = db.cooled_channel_ids(user_id, int(time.time()))   # cooldown blocks like a block list; no exemptions
    trusted_channels = db.trusted_channel_ids(user_id)

    kept: list[Video] = []
    for v in videos:
        # Rule 1: blocked / cooling-down channels (strongest signal, checked first).
        if v.channel_id in black_channels:
            db.add_filter_result(user_id, refresh_id, v.video_id, "prefilter",
                                 "channel_blacklisted", "来自你拉黑的频道")
            continue
        if v.channel_id in cooled_channels:
            db.add_filter_result(user_id, refresh_id, v.video_id, "prefilter",
                                 "channel_cooldown", "该频道冷却中(你说过它推太多)")
            continue

        # Intent pass: exempt from blacklist words, duration window and
        # category whitelist -- intent fit is judged in S6, not vetoed by
        # static rules. Only a minimum duration is kept to keep out Shorts,
        # which otherwise dominate the losses in search results. The block
        # check above still applies: a persistent explicit choice outranks a
        # temporary broad one.
        if v.intent_sourced:
            if v.duration_sec < config.INTENT_MIN_DURATION_SEC:
                db.add_filter_result(
                    user_id, refresh_id, v.video_id, "prefilter",
                    "intent_shorts_floor",
                    f"意图召回但时长 {v.duration_sec} 秒低于反 Shorts 地板 "
                    f"{config.INTENT_MIN_DURATION_SEC} 秒")
                continue
            kept.append(v)
            continue

        # Rule 2: blacklisted words. The matched term goes into `rule`, so
        # reviews can group by rule to see which term causes the most false
        # positives.
        word = _hit_blacklist_word(v, mode.blacklist_keywords)
        if word is not None:
            db.add_filter_result(user_id, refresh_id, v.video_id, "prefilter",
                                 f"blacklist:{word}", f"命中黑名单词「{word}」")
            continue

        # Rule 3: duration window. Shares Mode.fits_duration with the S2 early
        # filter, so this is mostly a no-op kept for defence in depth.
        if not mode.fits_duration(v.duration_sec):
            max_disp = mode.duration_max_sec if mode.duration_max_sec is not None else "∞"
            db.add_filter_result(
                user_id, refresh_id, v.video_id, "prefilter", "duration",
                f"时长 {v.duration_sec} 秒不在模式窗口 [{mode.duration_min_sec}, {max_disp}]")
            continue

        # Rule 4: category whitelist (channel-based and trusted sources exempt).
        if not _category_ok(v, mode, trusted_channels):
            db.add_filter_result(
                user_id, refresh_id, v.video_id, "prefilter", "category",
                f"分类 {v.category_id} 不在模式白名单（订阅/信任来源豁免）")
            continue

        kept.append(v)

    logger.info("prefilter[%s/%s]: 输入 %d → 保留 %d(滤 %d)",
                mode.name, refresh_id, len(videos), len(kept), len(videos) - len(kept))
    return kept
