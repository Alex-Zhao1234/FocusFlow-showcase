"""YouTube Data API v3 adapter and S2 plan executor.

This is the only module in the project that talks to the YouTube API. It depends
downward on storage/db, core/models and config, and is never imported by the agent
layer (planning and execution are kept strictly separate).

Contents:
  - ``YouTubeAdapter``: the ``SourceAdapter`` implementation. Its four retrieval
    methods are the "tools" the planner chooses from; they return *bare* videos
    (id / channel / source tag only), and ``enrich_stats`` fills in stats, duration
    and cover in batches.
  - ``QuotaCounter``: quota accounting, a thin wrapper over the storage layer.
  - ``dedup`` / ``persist_videos``: source-priority dedup and FK-safe persistence.
  - ``execute_plan`` / ``_run_strategy``: the executor of the plan-then-execute
    agent. It runs a ``RetrievalPlan``'s strategies concurrently, then dedups,
    enriches, persists and trims the result back to the plan's per-source ratios.

Key design points:
  - Async bridge: google-api-python-client is a blocking library. Every call goes
    through ``asyncio.to_thread`` so ``gather`` gives real concurrency (the GIL is
    released while threads wait on network IO).
  - API key: read from the environment variable ``FOCUSFLOW_YT_API_KEY`` (never in
    config or code); a missing key fails loudly.
  - Region / language: ``regionCode`` and ``relevanceLanguage`` come from the Mode
    and are injected into the adapter by ``execute_plan``; ``DEFAULT_REGION`` is only
    the fallback when a method is called directly.
  - Channel writes always go through ``db.merge_channel`` (preserving merge): ``None``
    fields keep the stored value and ``is_small_pool`` is only ever set, never cleared,
    so incidental writes cannot clobber authoritative channel data.
  - After a successful channel fetch, ``db.touch_channel_fetched`` is called so the
    small-pool cooldown window actually works and the same channel is not re-sampled
    and re-billed over and over.
  - Pool channels with no cached uploads playlist id are resolved on the spot
    (1 quota, once, then cached), so the hidden-gem path self-heals instead of
    idling forever.
  - ``execute_plan`` keeps its own hard budget guard even though the planner's
    ``validate_plan`` also checks budgets: callers that bypass the planner
    (calibration scripts, ablations) still exist, and both guards share the
    ``core.costing`` cost model so they never drift.

Quota accounting (two independent buckets, matching the two Google Console counters):
  - General pool (10,000 units/day): channels.list = 1/batch, playlistItems.list =
    1/channel page, videos.list (trending/enrich) = 1/batch, channelSections.list = 1.
  - Search bucket (100 calls/day): search.list = 1 per call (``search_by_keyword``,
    ``search_channels``, ``search_video_channels``).
  Every real call is charged immediately *after it succeeds*, routed to its bucket via
  ``config.quota_bucket``. Known caveat: YouTube still bills some failed requests
  (e.g. 400s) while this layer skips the charge on failure, so if the local ledger
  undercounts, look for calls that raised that day.
"""
import asyncio
import time
import logging
import math
import os
import random

import isodate                                                  # ISO 8601 duration parsing
import config as _cfg                                           # read tunables dynamically so tests can monkeypatch them
from config import (
    QUOTA_COST, SOURCE_PRIORITY, DEFAULT_USER_ID, SMALL_POOL_SAMPLE_K, MODES,
    SEED_FETCH_PER_CHANNEL, quota_bucket,
)                                                               # single source for costs / priorities / defaults

from core.costing import ENRICH_BATCH_SIZE, estimate_plan_cost  # the one place costs are defined
from core.logsafe import safe_repr                              # redact exceptions before logging so `key=<developerKey>` never reaches log files
from core.models import Channel, RetrievalPlan, StrategyCall, Video
from sources.base import SourceAdapter
from storage import db

logger = logging.getLogger(__name__)

DEFAULT_REGION = "US"
_ENRICH_BATCH_SIZE = ENRICH_BATCH_SIZE                 # 50: videos.list per-request id limit, a hard API constraint (not a tunable, so not in config)


# relevanceLanguage is a soft ranking bias, not a filter: with it set, English videos
# still appear, the target language just ranks higher. We apply the Mode's language only
# to queries that are actually written in CJK, so English queries are not skewed.
def _search_language_bias(relevance_language: str | None, query: str) -> str | None:
    """Return the relevanceLanguage to send for this search (None = omit the parameter)."""
    if not relevance_language:
        return None
    # CJK Unified Ideographs block: does the query contain at least one Han character?
    if any("一" <= ch <= "鿿" for ch in query):
        return relevance_language
    return None


# --- Dedup (pure function) ---
# After concurrent retrieval the same video_id can be hit by several strategies (e.g. a
# subscribed channel's video is also trending). Keep one entry per id; must run before
# enrich so the same id is never billed twice.

def dedup(videos: list[Video]) -> list[Video]:
    # Priority for source tags missing from SOURCE_PRIORITY: worse than every known source.
    UNKNOWN = max(SOURCE_PRIORITY.values()) + 1
    best: dict[str, Video] = {}
    for video in videos:
        prio = SOURCE_PRIORITY.get(video.source_strategy, UNKNOWN)
        kept = best.get(video.video_id)

        # First sighting is written directly; otherwise replace only if this entry has a
        # higher priority (smaller number).
        if kept is None or prio < SOURCE_PRIORITY.get(kept.source_strategy, UNKNOWN):
            # On collision the intent flag is OR-ed: the source tag is decided by priority,
            # but "retrieved in response to a user intent" is an identity that must survive
            # the tag being overridden.
            if kept is not None and kept.intent_sourced:
                video.intent_sourced = True
            best[video.video_id] = video
        elif video.intent_sourced and not kept.intent_sourced:
            kept.intent_sourced = True
    # Dicts preserve insertion order: output order = order of first appearance per id.
    return list(best.values())



# --- Quota accounting ---
# Single entry point for every charge in this file, so "charge immediately after each
# successful real call" has one home; execute_plan's budget guard reads remaining() too.

class QuotaCounter:
    """Thin wrapper over the storage layer's quota functions (timezone, accumulation and
    guards live in db; nothing is re-implemented here).

    Two ledgers are kept independently: the general pool ("general", 10,000 units/day)
    and the search bucket ("search", 100 calls/day). ``charge_tool`` routes each tool to
    its bucket via ``config.quota_bucket``."""

    def remaining(self, bucket: str = "general") -> int:
        """Remaining quota for today (Pacific time) in a bucket, floored at 0."""
        return db.quota_remaining(bucket)

    def used(self, day: str | None = None, bucket: str = "general") -> int:
        """Units used on a day ('YYYY-MM-DD'; default: today, Pacific time) in a bucket."""
        return db.quota_used(day, bucket)

    def cost_of(self, tool: str) -> int:
        """Cost of a tool/endpoint name from config.QUOTA_COST. Unknown names raise
        KeyError: the cost table is fixed, so an unknown name means an upstream bug and
        should fail loudly rather than silently charge 0 and skew reconciliation."""
        # Deliberately [] rather than .get(): the KeyError is the point.
        return QUOTA_COST[tool]

    def bucket_of(self, tool: str) -> str:
        """Tool/endpoint name -> ledger bucket (config.quota_bucket; unknown -> general)."""
        return quota_bucket(tool)

    def charge(self, cost: int, bucket: str = "general") -> None:
        """Record a charge of a known numeric cost into a bucket."""
        db.record_quota(cost, bucket)

    def charge_tool(self, tool: str) -> int:
        """Look up a tool's cost, record it in that tool's bucket, and return the cost.
        All endpoints (including channels.list) go through here with an explicit cost-table
        entry; there is no silent default, which would defeat cost_of's fail-loud rule."""
        cost = self.cost_of(tool)
        db.record_quota(cost, self.bucket_of(tool))
        return cost



# --- Pure response parsers ---

def _parse_duration_sec(iso: str | None) -> int:
    """Convert an ISO 8601 duration ("PT1H2M10S") to integer seconds.

    Downstream consumers (the S3 duration-window filter and S7 cards) need integer
    seconds. Missing or malformed values yield 0 (= unknown)."""
    if not iso:
        return 0
    try:
        return int(isodate.parse_duration(iso).total_seconds())
    except ValueError:
        return 0


def _aggregate_video_channels(resp: dict) -> list[dict]:
    """search.list type=video response -> candidate channels aggregated by channelId:
    [{"channel_id","title","hits","best_rank","sample_titles"}], sorted by hits desc,
    then best_rank asc.

    Used for seed-channel recommendations. The video results are engagement-ranked, so
    keyword-stuffed channels with zero subscribers have no ranking videos and drop out
    naturally — which is why recommendations are sourced from type=video rather than
    type=channel. Pure function, testable offline."""
    agg: dict[str, dict] = {}
    for rank, item in enumerate(resp.get("items", [])):
        sn = item.get("snippet", {})
        cid = sn.get("channelId")
        if not cid:
            continue
        slot = agg.get(cid)
        if slot is None:
            slot = agg[cid] = {"channel_id": cid, "title": sn.get("channelTitle"),
                               "hits": 1, "best_rank": rank, "sample_titles": []}
        else:
            slot["hits"] += 1
        # Keep up to EXPAND_SAMPLE_TITLES ranking video titles per channel (logging/debugging).
        vt = sn.get("title")
        if vt and len(slot["sample_titles"]) < _cfg.EXPAND_SAMPLE_TITLES:
            slot["sample_titles"].append(vt)
    return sorted(agg.values(), key=lambda d: (-d["hits"], d["best_rank"]))


def _parse_channel_search(resp: dict) -> list[dict]:
    """search.list type=channel response -> [{"channel_id","title","description"}].
    The id is a nested object {"kind":..,"channelId":..} (same shape as type=video)."""
    out: list[dict] = []
    for item in resp.get("items", []):
        cid = item.get("id", {}).get("channelId")
        if not cid:
            continue
        sn = item.get("snippet", {})
        out.append({"channel_id": cid, "title": sn.get("title"),
                    "description": sn.get("description")})
    return out

# Channel-pool expansion, second candidate source: creators' hand-picked "featured
# channels" sections. Rationale: creators tend to recommend channels of a similar kind.
# Costs 1 general-pool unit, a cheap complement to search_channels (which spends the
# scarce search bucket).
def _parse_multiple_channels(resp: dict) -> list[str]:
    """channelSections.list response -> featured channel ids. Only sections with
    snippet.type == 'multipleChannels' count; none / no contentDetails -> []."""
    ids: list[str] = []
    for item in resp.get("items", []):
        if item.get("snippet", {}).get("type") != "multipleChannels":
            continue
        ids.extend(item.get("contentDetails", {}).get("channels", []))
    return ids

# Category audit support. If a Mode's yt_category_ids allow-list contains a non-assignable
# category (assignable=false: uploaders can no longer pick it), retrieval wastes quota —
# trending returns 404 for it and search's videoCategoryId filter always returns empty.
def _parse_video_categories(resp: dict) -> dict[int, dict]:
    """videoCategories.list response -> {int(id): {"title", "assignable"}}.
    Legacy assignable=false categories are kept as-is (the reader sees the full list)."""
    return {int(it["id"]): {"title": it["snippet"]["title"],
                            "assignable": bool(it["snippet"]["assignable"])}
            for it in resp.get("items", [])}


# Topic extraction: compress topicDetails.topicCategories (Wikipedia URLs) into readable
# topic names that an LLM can use to judge what a channel is about.
def _topic_tails(urls: list[str] | None) -> list[str]:
    """topicDetails.topicCategories Wikipedia URLs -> trailing topic names."""
    return [u.rsplit("/", 1)[-1] for u in (urls or []) if u]


def _enriched_video(bare: Video, item: dict) -> Video:
    """Merge one videos.list item into a bare video -> complete Video.

    Fixed rules:
      - likeCount missing (hidden by the channel) -> like_count=0 and like_hidden=True;
      - commentCount missing (comments disabled) -> 0, no separate flag;
      - duration parsed via _parse_duration_sec.
    source_strategy is inherited from the bare video (dedup already decided on it;
    enrich must not rewrite it).
    """
    snippet = item.get("snippet", {})                 # title/description/tags/cover/category/publish time -> S3 filters + S7 cards
    stats = item.get("statistics", {})                # views/likes/comments -> S3 quality gates + ranker
    content = item.get("contentDetails", {})          # duration -> S3 mode duration window
    like_raw = stats.get("likeCount")
    like_hidden = like_raw is None

    # Cover: first available size in a "good enough first" order — medium is sufficient
    # for cards, then high/standard/default, maxres last (bandwidth trade-off, not
    # descending resolution).
    thumbs = snippet.get("thumbnails", {})
    cover_url = None
    for size in ("medium", "high", "standard", "default", "maxres"):
        if size in thumbs and thumbs[size].get("url"):
            cover_url = thumbs[size]["url"]
            break

    # publishedAt (ISO 8601 UTC, e.g. 2024-01-02T03:04:05Z) -> Unix seconds; malformed -> None.
    publish_time = None
    published = snippet.get("publishedAt")
    if published:
        try:
            from datetime import datetime
            publish_time = int(datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp())
        except ValueError:
            publish_time = None

    category_raw = snippet.get("categoryId")

    return Video(
        video_id=bare.video_id,
        channel_id=snippet.get("channelId") or bare.channel_id,            # prefer the API's channelId, fall back to the bare video's
        title=snippet.get("title", ""),
        duration_sec=_parse_duration_sec(content.get("duration")),
        source_strategy=bare.source_strategy,                              # source tag must not be rewritten by enrich
        intent_sourced=bare.intent_sourced,                                # intent flag is passed through the same way
        description=(snippet.get("description") or "")[:500],
        tags=snippet.get("tags") or [],                                    # key is absent on untagged videos
        cover_url=cover_url,
        category_id=int(category_raw) if category_raw is not None else None,   # API returns categoryId as a string
        view_count=int(stats.get("viewCount") or 0),
        like_count=int(like_raw) if like_raw is not None else 0,             # hidden state is tracked separately by like_hidden
        like_hidden=like_hidden,
        comment_count=int(stats.get("commentCount") or 0),
        publish_time=publish_time,
    )



# --- Persistence ---
# Foreign keys are enforced (PRAGMA foreign_keys=ON), so channels must exist before videos.
# enrich_stats already creates placeholders for channels it saw; this is a second safety net.
def persist_videos(videos: list[Video]) -> int:
    """Persist enriched videos in bulk and return the count written. Channels are
    written before videos; placeholders go through merge_channel so existing channel
    attributes are never overwritten."""
    # Channel ids already confirmed to exist, to skip repeat lookups for the same channel.
    ensured: set[str] = set()
    for v in videos:
        if v.channel_id not in ensured:
            if db.get_channel(v.channel_id) is None:
                db.merge_channel(Channel(channel_id=v.channel_id))          # placeholder; never clobbers authoritative data
            ensured.add(v.channel_id)
        db.upsert_video(v)
    return len(videos)




# --- YouTubeAdapter: the only part of this file that spends quota ---
# The YouTube implementation of SourceAdapter. Upper layers program against the abstract
# interface, so supporting another platform means adding an adapter, not touching the
# pipeline. Constructing an adapter is the line between offline code and code that spends
# quota: offline tests never need one.

class YouTubeAdapter(SourceAdapter):
    """YouTube Data API v3 adapter.

    The four retrieval methods call the real API and return "bare" videos
    (video_id / channel_id / source_strategy filled; stats and duration empty);
    enrich_stats fills them in batches. Each real call is charged right after it succeeds.
    """
    def __init__(self, api_key: str | None = None, user_id: int = DEFAULT_USER_ID,
                 region_code: str = DEFAULT_REGION,
                 relevance_language: str | None = None) -> None:
        # The API key comes from the environment (never stored in config or code).
        key = api_key or os.environ.get("FOCUSFLOW_YT_API_KEY")
        if not key:
            raise RuntimeError("缺少 YouTube API Key:请设置环境变量 FOCUSFLOW_YT_API_KEY")

        # Lazy import: google-api-python-client is heavy and not needed by the pure
        # dedup/accounting/parsing tests, which can run without it installed.
        from googleapiclient.discovery import build

        # _yt is private: code outside the adapter must not use the client directly,
        # otherwise calls would bypass quota accounting.
        self._yt = build("youtube", "v3", developerKey=key, cache_discovery=False)

        self.quota = QuotaCounter()               # all instances share one ledger (stored in SQLite)
        self.user_id = user_id

        # Set per Mode. execute_plan overwrites these before running a plan; one adapter
        # runs one plan at a time, so there are no concurrent writes.
        self.region_code = region_code
        self.relevance_language = relevance_language


    # The single bridge between the async world and the blocking SDK. Every API call goes
    # through it, so timeout (30s, never hang), thread pool (real concurrency) and
    # per-call Http isolation (thread safety) are handled in one place.

    async def _execute(self, request, tool: str):
        """Run a blocking ``request.execute()`` in a worker thread so concurrent strategies
        under ``asyncio.gather`` actually run in parallel instead of serially.

        Accounting happens inside the thread, right after ``execute()`` succeeds:
        ``to_thread`` cannot be cancelled, so if the awaiting coroutine is cancelled
        (client disconnect, mode switch) the request still completes and YouTube still
        bills it. Charging after the ``await`` would miss those calls. Failures raise and
        are not charged. Callers must not charge again after awaiting (double count).
        """
        import httplib2

        def _run():
            # A fresh Http per call: httplib2.Http is not thread-safe; timeout=30 avoids hangs.
            resp = request.execute(http=httplib2.Http(timeout=30))
            self.quota.charge_tool(tool)
            return resp

        return await asyncio.to_thread(_run)



    # Tool 1: latest uploads of a subscribed/preferred channel (playlistItems.list, 1 quota).
    async def fetch_channel_videos(self, channel_id: str, n: int) -> list[Video]:
        """Latest n uploads of one channel -> source_strategy='subscribed'.

        Uses the 1-quota playlistItems endpoint instead of the 100-quota search. The
        uploads playlist id is fetched once via channels.list (1) and cached in the DB,
        then reused for free. playlistItems pages cap at 50, so n > 50 is clamped.
        """
        uploads_id = await self._resolve_uploads_playlist_id(channel_id)
        if uploads_id is None:
            return []

        request = self._yt.playlistItems().list(
            part="contentDetails",                # only need videoId; stats come from enrich
            playlistId=uploads_id,
            maxResults=min(n, 50),
        )
        resp = await self._execute(request, "fetch_channel_videos")
        db.touch_channel_fetched(channel_id)             # feeds the cooldown window

        videos: list[Video] = []
        for item in resp.get("items", []):
            vid = item.get("contentDetails", {}).get("videoId")
            if not vid:
                continue
            videos.append(self._bare_video(vid, channel_id, "subscribed"))
        return videos




    async def _fetch_uploads_page(self, uploads_id: str, page_token: str | None
                                  ) -> tuple[list[str], str | None]:
        """Fetch one page of an uploads playlist -> (video ids, nextPageToken | None).
        One playlistItems.list call = 1 unit; page size CHANNEL_PAGE_SIZE (API max 50)."""
        kwargs = dict(part="contentDetails", playlistId=uploads_id,
                      maxResults=_cfg.CHANNEL_PAGE_SIZE)
        if page_token:
            kwargs["pageToken"] = page_token
        resp = await self._execute(self._yt.playlistItems().list(**kwargs), "fetch_channel_videos")
        ids = [it.get("contentDetails", {}).get("videoId") for it in resp.get("items", [])]
        return [v for v in ids if v], resp.get("nextPageToken") or None

    async def fetch_channel_videos_deep(self, channel_id: str, n: int, *,
                                        exclude: frozenset[str] = frozenset(),
                                        unfit=None) -> list[Video]:
        """Deep-paging channel fetch.

        Fetch the first page (newest uploads first). If it yields fewer than n *valid*
        items (not excluded, not already collected, not known to fall outside the mode's
        duration window), page deeper starting from a per (user, channel) cursor, up to
        CHANNEL_DEEP_PAGES_MAX pages per channel per refresh. Reaching the end marks the
        channel exhausted; it is not paged again for CHANNEL_DEEP_RESET_DAYS.

        Returns bare Videos tagged 'subscribed' (callers re-tag per path). On a paging
        error, the items collected so far are returned and the cursor is reset. Cursor
        read/write failures fail open to first-page-only behaviour.
        """
        uploads_id = await self._resolve_uploads_playlist_id(channel_id)
        if uploads_id is None:
            return []
        stats = getattr(self, "_deep_stats", None)
        pages_max = max(1, int(_cfg.CHANNEL_DEEP_PAGES_MAX))
        ids, next_token = await self._fetch_uploads_page(uploads_id, None)
        db.touch_channel_fetched(channel_id)
        seen: set[str] = set()
        valid = _pick_valid(ids, exclude, seen, unfit)
        if stats is not None:
            stats["channels"] += 1
        if len(valid) >= n or pages_max <= 1 or next_token is None:
            return [self._bare_video(v, channel_id, "subscribed") for v in valid[:n]]
        # ---- Deep pages: the cursor decides where to start ----
        try:
            cur = db.get_channel_cursor(self.user_id, channel_id)
        except Exception:                                   # noqa: BLE001 — fail-open
            logger.warning("深挖游标读取失败,退回只抓第一页: %s", channel_id)
            return [self._bare_video(v, channel_id, "subscribed") for v in valid[:n]]
        now = int(time.time())
        reset = False
        token = next_token                                  # default: start at page 2
        if cur is not None:
            if cur.get("exhausted_at"):
                if now - int(cur["exhausted_at"]) < _cfg.CHANNEL_DEEP_RESET_DAYS * 86400:
                    if stats is not None:
                        stats["resting"] += 1
                    return [self._bare_video(v, channel_id, "subscribed") for v in valid[:n]]
                reset = True                                # rest period over: start again from the top
            elif cur.get("next_page_token"):
                token = cur["next_page_token"]
        pages, added, failed = 1, 0, False
        while len(valid) < n and pages < pages_max and token:
            try:
                ids2, token2 = await self._fetch_uploads_page(uploads_id, token)
            except Exception as e:                          # noqa: BLE001 — one channel's paging failure must not block others
                logger.warning("深挖翻页失败(游标清零) %s: %s", channel_id, safe_repr(e))
                failed = True
                break
            pages += 1
            got = _pick_valid(ids2, exclude, seen, unfit)
            valid.extend(got)
            added += len(got)
            token = token2
        try:
            if failed:
                db.save_channel_cursor(self.user_id, channel_id, None, pages - 1, exhausted=False, reset=True)
            else:
                db.save_channel_cursor(self.user_id, channel_id, token, pages - 1,
                                       exhausted=(token is None), reset=reset)
        except Exception:                                   # noqa: BLE001
            logger.warning("深挖游标写入失败(忽略): %s", channel_id)
        if stats is not None:
            stats["paged"] += 1
            stats["pages"] += pages - 1
            stats["added"] += added
            if not failed and token is None:
                stats["exhausted"] += 1
        return [self._bare_video(v, channel_id, "subscribed") for v in valid[:n]]

    async def _resolve_uploads_playlist_id(self, channel_id: str) -> str | None:
        """Return the channel's uploads playlist id: from the DB if cached (free),
        otherwise via channels.list (1 quota) and then cached.

        The incidental channel write goes through merge_channel: it only adds the fields
        it fetched (title/subs/uploads id) and never clobbers is_small_pool or
        last_fetched — small-pool channels can reach this path too.
        """
        cached = db.get_channel(channel_id)
        if cached is not None and cached["uploads_playlist_id"]:
            return cached["uploads_playlist_id"]

        request = self._yt.channels().list(
            part="contentDetails,snippet,statistics", id=channel_id,
        )
        resp = await self._execute(request, "channels_list")
        items = resp.get("items", [])
        if not items:
            return None                                  # channel deleted or invalid id
        item = items[0]
        uploads_id = (
            item.get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )
        subs = item.get("statistics", {}).get("subscriberCount")      # may be hidden by the owner -> None
        db.merge_channel(Channel(
            channel_id=channel_id,
            channel_title=item.get("snippet", {}).get("title"),
            channel_subs=int(subs) if subs is not None else None,     # None -> merge keeps the stored value
            uploads_playlist_id=uploads_id,
        ))
        return uploads_id


    # Tool 2: keyword search (search.list, 1 call from the 100/day search bucket).
    async def search_by_keyword(self, query: str, n: int) -> list[Video]:
        """search.list for videos -> source_strategy='search'.

        This layer only records the search-bucket charge; the per-refresh search limit is
        enforced by the planner's validate_plan.
        """
        kwargs = dict(
            part="snippet",
            q=query,
            type="video",                   # videos only, no channels/playlists
            maxResults=min(n, 50),
            regionCode=self.region_code,    # per Mode, injected by execute_plan
        )
        # Language bias depends on the query's script; when there is no bias the parameter
        # must be omitted entirely (sending None is not the same as not sending it).
        bias = _search_language_bias(self.relevance_language, query)
        if bias:
            kwargs["relevanceLanguage"] = bias

        request = self._yt.search().list(**kwargs)
        resp = await self._execute(request, "search_by_keyword")

        videos: list[Video] = []
        for item in resp.get("items", []):
            # search.list ids are nested {"kind":..,"videoId":..} (videos.list uses a plain string).
            vid = item.get("id", {}).get("videoId")
            ch = item.get("snippet", {}).get("channelId")
            if not vid or not ch:
                continue
            videos.append(self._bare_video(vid, ch, "search"))
        return videos



    # Tool 3: regional most-popular chart for a category (videos.list chart=mostPopular).
    # Driven by the Mode's yt_category_ids; the main strategy for entertainment mode.
    async def fetch_trending_by_category(self, category_id: int, n: int) -> list[Video]:
        """videos.list?chart=mostPopular -> source_strategy='trending'.

        Only ids are used as bare videos; stats are still filled by enrich, keeping the
        "tools produce bare videos, enrich fills them" split. contentDetails is requested
        too (videos.list cost does not depend on part), so the real duration_sec is
        prefilled: trending is the only path that knows durations before truncation, which
        lets execute_plan drop out-of-window items for free before enrich. Enrich later
        overwrites the same values.
        """
        request = self._yt.videos().list(
            part="snippet,contentDetails",
            chart="mostPopular",
            videoCategoryId=str(category_id),                   # API expects a string
            regionCode=self.region_code,                        # per Mode, injected by execute_plan
            maxResults=min(n, 50),
        )
        resp = await self._execute(request, "fetch_trending_by_category")

        videos: list[Video] = []
        for item in resp.get("items", []):
            vid = item.get("id")                 # videos.list id is a plain string
            ch = item.get("snippet", {}).get("channelId")
            if not vid or not ch:
                continue
            v = self._bare_video(vid, ch, "trending")
            # Prefill the real duration (parse failure -> 0 = unknown, never wrongly dropped).
            v.duration_sec = _parse_duration_sec(
                item.get("contentDetails", {}).get("duration"))
            videos.append(v)
        return videos


    # Tool 4: sample k channels from the small-channel pool (< 200k subscribers) and fetch
    # their uploads concurrently -> source_strategy='hidden_gem'.
    async def mine_hidden_gems(self, k: int) -> list[Video]:
        """playlistItems.list on channels from db.sample_small_pool -> 'hidden_gem'.

        Cost: one playlistItems call (1) per channel; pool channels with no cached uploads
        id are resolved on the spot (one extra channels.list, first time only). Sampling,
        blacklist and cooldown SQL all live in the storage layer.
        """
        # Sampling in db: k random channels, blacklist excluded, 24h cooldown applied.
        pool = db.sample_small_pool(self.user_id, k)
        if not pool:
            return []

        async def _one(row) -> list[Video]:

            uploads_id = row["uploads_playlist_id"]
            if not uploads_id:
                uploads_id = await self._resolve_uploads_playlist_id(row["channel_id"])

            # Dead channels (deleted / no uploads list) are touched into the cooldown too;
            # otherwise they would never cool down and every sample would waste a
            # channels.list call. This caps the waste at 1 unit per channel per day.
            if not uploads_id:
                db.touch_channel_fetched(row["channel_id"])
                return []
            request = self._yt.playlistItems().list(
                part="contentDetails", playlistId=uploads_id, maxResults=50,
            )
            resp = await self._execute(request, "mine_hidden_gems")
            db.touch_channel_fetched(row["channel_id"])
            out: list[Video] = []
            for item in resp.get("items", []):
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    out.append(self._bare_video(vid, row["channel_id"], "hidden_gem"))
            return out

        # Fail-soft per channel: exceptions come back as results and are logged and skipped.
        results = await asyncio.gather(*[_one(r) for r in pool], return_exceptions=True)
        videos: list[Video] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("mine_hidden_gems 单频道失败(已跳过): %s", safe_repr(r))
            else:
                videos.extend(r)
        return videos


    # Enrichment: retrieval returns only identity fields; titles, durations and view/like
    # counts needed by S3 filters and S7 cards are backfilled here via videos.list —
    # 50 ids per call, 1 quota per batch.
    async def enrich_stats(self, videos: list[Video]) -> list[Video]:
        """videos.list?part=snippet,statistics,contentDetails, 50 per batch, 1 quota each.

        - Fills title/description/tags/cover/category/view/like/comment/duration/
          publish_time (see _enriched_video / _parse_duration_sec for the fixed rules).
        - Ids the API does not return (deleted / private / region-blocked) are dropped;
          those videos are not recommendable anyway.
        - Uses snippet.channelId/channelTitle to create placeholders for unseen channels
          (via merge_channel, never clobbering existing rows), so persist_videos cannot
          hit a foreign-key error.
        - Callers must pass the list *after* dedup (dedup first, then enrich, to save quota).
        """
        if not videos:
            return []
        by_id: dict[str, Video] = {v.video_id: v for v in videos}   # defensive dedup by id
        ids = list(by_id)
        enriched: list[Video] = []
        channel_titles: dict[str, str | None] = {}                  # channel id -> title (None still means "seen")

        for i in range(0, len(ids), _ENRICH_BATCH_SIZE):
            batch = ids[i: i + _ENRICH_BATCH_SIZE]
            request = self._yt.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
                # maxResults is not sent: it is ignored when querying by id.
            )
            # Batch-level fail-soft: a network blip loses only this batch. Otherwise the
            # exception would propagate through execute_plan and everything already spent
            # on retrieval would be wasted with nothing persisted.
            try:
                resp = await self._execute(request, "get_video_stats")
            except Exception as exc:
                logger.warning("enrich_stats 批次 %d~%d 失败(已跳过 %d 条): %s",
                               i, i + len(batch) - 1, len(batch), safe_repr(exc))
                continue                     # failed call is not charged; continue with the next batch

            for item in resp.get("items", []):
                bare = by_id.get(item.get("id"))
                if bare is None:
                    continue                           # id we never asked for (should not happen)
                video = _enriched_video(bare, item)
                enriched.append(video)

                # Remember channel titles; placeholders are written once per channel after the loop.
                if video.channel_id not in channel_titles:
                    channel_titles[video.channel_id] = item.get("snippet", {}).get("channelTitle")

        # Channel placeholders (FK: channel rows must exist before videos are persisted).
        # Write when the channel is missing, or when it exists without a title and we now have one.
        for ch_id, title in channel_titles.items():
            existing = db.get_channel(ch_id)
            if existing is None or (title and not existing["channel_title"]):
                db.merge_channel(Channel(channel_id=ch_id, channel_title=title))

        return enriched



    # --- YouTube-specific methods outside the SourceAdapter contract ---
    # Not planner tools and not part of the refresh pipeline: used by onboarding import,
    # offline channel-pool building, seed-channel recommendation and the channel preview UI.

    async def fetch_channels_meta(self, channel_ids: list[str]) -> list[Channel]:
        """channels.list for channel attributes in bulk (50 per batch, 1 quota each).

        Used by the Takeout import's optional enrich step and by channel-pool expansion.
        Builds and returns Channel objects without persisting; the caller decides the
        merge policy.
        """
        out: list[Channel] = []
        # No batch-level fail-soft here (unlike enrich_stats): a failed batch propagates.
        # These are one-off, operator-driven jobs that can simply be rerun.
        for i in range(0, len(channel_ids), _ENRICH_BATCH_SIZE):
            batch = channel_ids[i: i + _ENRICH_BATCH_SIZE]
            # Extra parts (description, branding keywords, topics) cost nothing more and give
            # the channel-classification judge the channel's self-description.
            request = self._yt.channels().list(
                part="contentDetails,snippet,statistics,topicDetails,brandingSettings",
                id=",".join(batch),
            )
            resp = await self._execute(request, "channels_list")

            for item in resp.get("items", []):
                subs = item.get("statistics", {}).get("subscriberCount")
                vc = item.get("statistics", {}).get("videoCount")
                sn = item.get("snippet", {})
                out.append(Channel(
                    channel_id=item["id"],
                    channel_title=sn.get("title"),
                    description=sn.get("description") or None,
                    keywords=(item.get("brandingSettings", {}).get("channel", {})
                              .get("keywords") or None),
                    channel_subs=int(subs) if subs is not None else None,
                    uploads_playlist_id=(
                        item.get("contentDetails", {})
                        .get("relatedPlaylists", {})
                        .get("uploads")
                    ),
                    video_count=int(vc) if vc is not None else None,      # used by the pool's hard-metric gate
                    topic_categories=_topic_tails(                        # topic signal for classification
                        item.get("topicDetails", {}).get("topicCategories")),
                ))
        return out


    # Offline channel-pool building: channel search.
    async def search_channels(self, query: str, limit: int = 10) -> list[dict]:
        """search.list type=channel (1 search-bucket call) -> candidate channels
        [{"channel_id","title","description"}]. Region/language bias as in search_by_keyword."""
        kwargs = dict(part="snippet", q=query, type="channel",
                      maxResults=min(limit, 50), regionCode=self.region_code)
        bias = _search_language_bias(self.relevance_language, query)
        if bias:
            kwargs["relevanceLanguage"] = bias
        request = self._yt.search().list(**kwargs)
        resp = await self._execute(request, "search_channels")
        return _parse_channel_search(resp)


    # Seed-channel recommendation source: channels aggregated from video search results.
    async def search_video_channels(self, query: str, n: int = 50) -> list[dict]:
        """search.list type=video (1 search-bucket call) -> candidates aggregated by
        channel [{"channel_id","title","hits","best_rank"}] (see _aggregate_video_channels).
        Region/language bias as in search_by_keyword; snippet only — channel stats are
        fetched separately via channels.list."""
        kwargs = dict(part="snippet", q=query, type="video",
                      maxResults=min(n, 50), regionCode=self.region_code)
        bias = _search_language_bias(self.relevance_language, query)
        if bias:
            kwargs["relevanceLanguage"] = bias
        request = self._yt.search().list(**kwargs)
        resp = await self._execute(request, "search_video_channels")
        return _aggregate_video_channels(resp)

    # Channel preview (frontend dialog).
    async def fetch_channel_preview(self, channel_id: str) -> dict | None:
        """channels.list snippet+statistics (1 unit) -> title/description/thumbnail/subscribers."""
        request = self._yt.channels().list(part="snippet,statistics", id=channel_id)
        resp = await self._execute(request, "channels_list")
        items = resp.get("items", [])
        if not items:
            return None
        it, sn = items[0], items[0].get("snippet", {})
        subs = it.get("statistics", {}).get("subscriberCount")
        vc = it.get("statistics", {}).get("videoCount")
        thumbs = sn.get("thumbnails", {})
        return {"channel_id": it["id"], "title": sn.get("title"),
                "description": sn.get("description") or "",
                "thumbnail": (thumbs.get("medium") or thumbs.get("default") or {}).get("url"),
                "subs": int(subs) if subs is not None else None,
                "video_count": int(vc) if vc is not None else None}

    # Channel page view: read-only fetchers. They do not produce Video objects or feed the
    # refresh pipeline; persistence is decided by the caller.
    @staticmethod
    def _parse_playlist_items(resp: dict) -> list[dict]:
        """playlistItems.list (snippet+contentDetails) -> small display-card dicts.
        channel_id comes from videoOwnerChannelId: user-made playlists can contain other
        channels' videos, and the FK must point at the real owner."""
        out: list[dict] = []
        for item in resp.get("items", []):
            cd = item.get("contentDetails", {})
            vid = cd.get("videoId")
            if not vid:
                continue
            sn = item.get("snippet", {})
            thumbs = sn.get("thumbnails", {})
            out.append({"video_id": vid,
                        "title": sn.get("title"),
                        "published_at": cd.get("videoPublishedAt") or sn.get("publishedAt"),
                        "thumbnail": (thumbs.get("medium") or thumbs.get("default") or {}).get("url"),
                        "channel_id": sn.get("videoOwnerChannelId")})
        return out

    async def fetch_playlist_items_page(self, playlist_id: str, page_token: str | None = None
                                        ) -> tuple[list[dict], str | None]:
        """One page of any playlist (1 unit, page size CHANNEL_PAGE_SIZE) ->
        (cards, nextPageToken | None). Uploads and user playlists share this entry point;
        charged as fetch_channel_videos (same playlistItems.list endpoint)."""
        kwargs = dict(part="snippet,contentDetails", playlistId=playlist_id,
                      maxResults=_cfg.CHANNEL_PAGE_SIZE)
        if page_token:
            kwargs["pageToken"] = page_token
        resp = await self._execute(self._yt.playlistItems().list(**kwargs), "fetch_channel_videos")
        return self._parse_playlist_items(resp), resp.get("nextPageToken") or None

    async def fetch_uploads_titles(self, channel_id: str, n: int) -> list[str]:
        """Titles of a channel's latest n uploads (playlistItems part=snippet, 1 unit,
        single page <= 50), used as portfolio evidence for the channel-expansion judge.
        Unresolvable uploads id -> []. Deliberately does *not* touch last_fetched: pool
        cooldown/rotation must not be altered by expansion lookups. A 404 (no public
        videos) is handled by the caller."""
        uploads_id = await self._resolve_uploads_playlist_id(channel_id)
        if uploads_id is None:
            return []
        request = self._yt.playlistItems().list(
            part="snippet", playlistId=uploads_id, maxResults=min(n, 50))
        resp = await self._execute(request, "fetch_channel_videos")
        out: list[str] = []
        for item in resp.get("items", []):
            t = item.get("snippet", {}).get("title")
            if t:
                out.append(t)
        return out

    async def fetch_channel_uploads_all(self, channel_id: str, *, video_count: int | None,
                                        full_max: int, pages_cap: int) -> tuple[list[dict], bool]:
        """Data for a channel's "Videos" tab. If video_count <= full_max, page through the
        whole uploads list (with a safety cap of ceil(full_max / page size) + 1 pages, in
        case videoCount disagrees with the list); otherwise, or if unknown, stop after
        pages_cap pages. Returns (videos, complete), complete = reached the last page.
        Unresolvable uploads id (deleted/invalid) -> ([], True)."""
        uploads_id = await self._resolve_uploads_playlist_id(channel_id)
        if uploads_id is None:
            return [], True
        if video_count is not None and video_count <= full_max:
            max_pages = math.ceil(full_max / _cfg.CHANNEL_PAGE_SIZE) + 1
        else:
            max_pages = max(1, pages_cap)
        out: list[dict] = []
        token: str | None = None
        pages = 0
        while pages < max_pages:
            items, token = await self.fetch_playlist_items_page(uploads_id, token)
            out.extend(items)
            pages += 1
            if token is None:
                break
        return out, token is None

    async def fetch_channel_playlists(self, channel_id: str) -> list[dict]:
        """playlists.list (1 unit, one page of 50) -> the channel's public playlists. None -> []."""
        request = self._yt.playlists().list(part="snippet,contentDetails",
                                            channelId=channel_id, maxResults=50)
        resp = await self._execute(request, "playlists_list")
        out: list[dict] = []
        for it in resp.get("items", []):
            sn = it.get("snippet", {})
            thumbs = sn.get("thumbnails", {})
            cnt = it.get("contentDetails", {}).get("itemCount")
            out.append({"playlist_id": it["id"], "title": sn.get("title"),
                        "thumbnail": (thumbs.get("medium") or thumbs.get("default") or {}).get("url"),
                        "item_count": int(cnt) if cnt is not None else None})
        return out

    async def fetch_videos_stats(self, video_ids: list[str]) -> dict[str, dict]:
        """videos.list contentDetails+statistics (1 unit per <= 50) ->
        {vid: {duration_sec, view_count}} for preview cards. Does not produce Video
        objects or feed the pipeline. Missing items (removed/private) are skipped."""
        if not video_ids:
            return {}
        request = self._yt.videos().list(
            part="contentDetails,statistics", id=",".join(video_ids[:50]))
        resp = await self._execute(request, "get_video_stats")   # must reuse an existing cost-table name: unknown names raise KeyError
        out: dict[str, dict] = {}
        for item in resp.get("items", []):
            views = item.get("statistics", {}).get("viewCount")
            out[item["id"]] = {
                "duration_sec": _parse_duration_sec(
                    item.get("contentDetails", {}).get("duration")),
                "view_count": int(views) if views is not None else None}
        return out


    # Offline channel-pool building: creators' featured-channel sections as a second candidate source.
    async def fetch_channel_sections(self, channel_id: str) -> list[str]:
        """channelSections.list (1 quota) -> channel ids from multipleChannels sections.
        A channel without such sections yields [] (coverage is inherently patchy)."""
        request = self._yt.channelSections().list(
            part="snippet,contentDetails", channelId=channel_id)
        resp = await self._execute(request, "channel_sections")
        return _parse_multiple_channels(resp)


    # Offline category audit. Returns the full list including assignability; the rule
    # ("no non-assignable categories in Mode allow-lists") lives in the audit tooling.
    # The source layer only fetches data.
    async def fetch_video_categories(self, region_code: str | None = None) -> dict[int, dict]:
        """videoCategories.list (1 quota) -> {category_id: {"title", "assignable"}}.
        Not a planner tool."""
        request = self._yt.videoCategories().list(
            part="snippet", regionCode=region_code or self.region_code)
        resp = await self._execute(request, "video_categories")
        return _parse_video_categories(resp)


    @staticmethod
    def _bare_video(video_id: str, channel_id: str, source_strategy: str) -> Video:
        """Common exit for all retrieval tools: guarantees video_id / channel_id /
        source_strategy (prerequisites for dedup and persistence). title and
        duration_sec are required Video fields, so they get placeholders (""/0)
        that enrich always overwrites."""
        return Video(
            video_id=video_id,
            channel_id=channel_id,
            title="",            # placeholder, filled by enrich
            duration_sec=0,      # placeholder, filled by enrich
            source_strategy=source_strategy,
        )


# ============================================================================
# execute_plan — the S2 executor
#
# Lives in the source layer on purpose: it has pure execution semantics (fetch ->
# dedup -> enrich -> persist, owning S2's own writes to videos/channels/quota). The
# orchestration bookkeeping (refresh id, plan logs, per-strategy counts) belongs to the
# orchestrator. Keeping the two apart preserves the layer boundary.
# ============================================================================

# Estimated cost of plans currently in flight: the reservation ledger for execute_plan's
# budget guard. Module-level, shared within one process; multi-process deployments each
# keep their own (intentionally not solved here). One reservation per quota bucket.
_INFLIGHT_EST = 0        # general-pool units
_INFLIGHT_SEARCH = 0     # search.list calls


def _channel_weight(pref: float, picks: int, affine: bool) -> float:
    """Sampling weight for a followed channel (pure function):
    exp(beta * (pref - 0.5)) * decay ** picks * (affinity if the channel matches this
    refresh's featured profile entry).

    A 0.1 difference in pref gives ~1.5x weight (preferred channels first and more often);
    each pick earlier the same day multiplies by the decay so others get a turn; channels
    recently attributed to the refresh's featured entry get the affinity boost."""
    pref = min(1.0, max(0.0, float(pref or 0.0)))     # pref_score lives in [0,1]; clamp to avoid exp overflow
    w = math.exp(_cfg.FOLLOW_PREF_BETA * (pref - 0.5))
    w *= _cfg.FOLLOW_PICK_DECAY ** max(0, int(picks or 0))
    if affine:
        w *= _cfg.FOLLOW_ENTRY_AFFINITY
    return max(w, 1e-6)


def _row_val(row, key: str, default=0):
    """Column access compatible with sqlite3.Row and dict; missing column -> default."""
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return default


def _weighted_sample_channels(rows: list, n: int, affinity: set | frozenset = frozenset()) -> list:
    """Weighted sampling without replacement of n followed channels.

    rows is the full rotation candidate set (with today's pick counts), weighted by
    _channel_weight. If n >= pool size, all rows are returned (small accounts). Every
    channel participates and same-day picks decay gently, so the whole set rotates
    within a day instead of the same top-N channels supplying every refresh."""
    if n >= len(rows):
        return rows
    pool = list(rows)
    picked: list = []
    while len(picked) < n and pool:
        weights = [_channel_weight(r["pref_score"], _row_val(r, "picks"),
                                   r["channel_id"] in affinity) for r in pool]
        idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
        picked.append(pool.pop(idx))
    return picked


def _pick_valid(ids: list[str], exclude: frozenset[str], seen: set[str], unfit) -> list[str]:
    """Filter page ids down to "valid" items for deep paging (pure, no API calls):
    not excluded, not already collected in this fetch, and known duration not outside
    the mode window (unfit(d) true -> drop; unknown/0 counts as valid, left to enrich).
    Order is preserved."""
    fresh: list[str] = []
    for v in ids:
        if v and v not in exclude and v not in seen:
            seen.add(v)                     # duplicates within a page count once
            fresh.append(v)
    if fresh and unfit is not None:
        known = db.video_durations(fresh)
        fresh = [v for v in fresh if not unfit(known.get(v, 0))]
    return fresh


async def _fetch_channel(adapter, channel_id: str, n: int,
                         exclude: frozenset[str], unfit) -> list[Video]:
    """Single-channel fetch shared by all channel paths. Uses deep paging when the adapter
    supports it (YouTubeAdapter); otherwise (test doubles, other platforms) falls back to
    SourceAdapter.fetch_channel_videos. Dispatching on the method's presence keeps the
    base ABC unchanged and existing stubs working."""
    deep = getattr(adapter, "fetch_channel_videos_deep", None)
    if deep is not None:
        return await deep(channel_id, n, exclude=exclude, unfit=unfit)
    return await adapter.fetch_channel_videos(channel_id, n)


async def _run_strategy(adapter: YouTubeAdapter, s: StrategyCall, *,
                        exclude: frozenset[str] = frozenset(), unfit=None) -> list[Video]:
    """Dispatch one StrategyCall to the matching adapter tool and truncate its result.

    params keys follow config.DEFAULT_PLANS (use_top_channels/n_channels, query,
    category_id, k, ...). An unknown tool raises ValueError: StrategyCall.tool's Literal
    already rejects it at the model layer, this is a second safeguard."""

    p = s.params
    # Overfetch: allow ceil(target * S2_OVERFETCH_FACTOR) so execute_plan's post-enrich
    # duration-window filter has headroom (within one 50-item page this costs no extra
    # retrieval quota). Intent strategies are not overfetched: each query is meant to
    # return exactly its quota of items, so enlarging it would break that contract.
    if s.intent_driven:
        eff = s.target_count
    else:
        eff = math.ceil(s.target_count * _cfg.S2_OVERFETCH_FACTOR)
    # Branch 1: channel-based retrieval (several variants selected by params)
    if s.tool == "fetch_channel_videos":
        if p.get("use_local_intent"):
            # Local intent recall: the orchestrator has already selected video_ids from the
            # full-text index (duration window and exclusions applied). Load rows from the
            # DB with zero API calls; tag as 'search' (same bucket as live intent search)
            # and mark them intent-sourced.
            rows = db.get_videos_by_ids(list(p.get("video_ids") or []))
            videos = []
            for r in rows:
                videos.append(Video(
                    video_id=r["video_id"], channel_id=r["channel_id"], title=r["title"] or "",
                    description=r["description"] or "", duration_sec=int(r["duration_sec"] or 0),
                    cover_url=r["cover_url"], category_id=r["category_id"],
                    view_count=r["view_count"] or 0, like_count=r["like_count"] or 0,
                    comment_count=r["comment_count"] or 0, publish_time=r["publish_time"],
                    source_strategy="search"))
            for v in videos:
                v.intent_sourced = True
            return videos[:eff]
        if p.get("use_library"):
            # Local high-quality candidate library, zero API: videos from high-quality
            # channels already in the DB that were not shown to completion, are not
            # excluded and were not judged unfit. Tagged by channel ownership (seed
            # channels -> seed_channel, others -> subscribed). Still goes through enrich
            # (1 unit / 50) to refresh stats.
            # Optional vector rerank: oversample by E1_LIBRARY_OVERSAMPLE, then within each
            # group (new pool items first) order vector hits by vector rank and append
            # misses in their original order. Only the order changes, never the set; with
            # the switch off / no vector index / no embedded entries, the original order holds.
            _vec_on = bool(_cfg.E1_VECTOR) and db.vector_available()
            lib_limit = eff * max(1, int(_cfg.E1_LIBRARY_OVERSAMPLE)) if _vec_on else eff
            rows = db.library_candidates(
                adapter.user_id, p.get("mode", ""), list(p.get("channel_ids") or []),
                exclude, p.get("fit_fingerprint", ""), int(_cfg.FIT_CACHE_TTL_DAYS), lib_limit)
            # E1_ENTRY_ISOLATION=1: rank in segments per profile entry; 0: RRF sum over all vectors.
            if _vec_on and rows:
                groups = db.entry_vec_groups_for_mode(adapter.user_id, p.get("mode", ""),
                                                      bool(_cfg.ES_VECTOR))   # vectors grouped by profile entry
                if groups:
                    ids = [r["video_id"] for r in rows]
                    if _cfg.E1_ENTRY_ISOLATION:
                        # Segments: primary entry -> featured secondary entries -> remaining
                        # entries by position -> misses in original order.
                        feat = list(p.get("featured_entry_ids") or [])
                        ranked = db.rerank_video_ids_by_entry_groups(
                            groups, feat, ids, float(_cfg.VEC_SUBTOPIC_WEIGHT))
                    else:
                        flat = [g.base for g in groups] + [s for g in groups for s in g.subs]
                        ranked = db.rerank_video_ids_by_vecs(flat, ids, eff)
                    pos = {vid: i for i, vid in enumerate(ranked)}
                    rows = sorted(rows, key=lambda r: (int(r["has_pool"] or 0),
                                                       0 if r["video_id"] in pos else 1,
                                                       pos.get(r["video_id"], 0)))
                    logger.info("vec[E1]: 候选库向量重排 u%s/%s 超采 %d 命中 %d 未命中 %d(条目 %d 方向 %d;%s;主推 %s)",
                                adapter.user_id, p.get("mode", ""), len(rows), len(ranked),
                                len(rows) - len(ranked), len(groups),
                                sum(len(g.subs) for g in groups),
                                "隔离分段" if _cfg.E1_ENTRY_ISOLATION else "RRF求和",
                                [e[:6] for e in (p.get("featured_entry_ids") or [])])
            seed_set = set(p.get("seed_channel_ids") or [])
            videos = []
            for r in rows:
                d = r["duration_sec"] or 0
                if unfit is not None and unfit(d):
                    continue
                videos.append(Video(
                    video_id=r["video_id"], channel_id=r["channel_id"], title=r["title"] or "",
                    description=r["description"] or "", duration_sec=d,
                    cover_url=r["cover_url"], category_id=r["category_id"],
                    view_count=r["view_count"] or 0, like_count=r["like_count"] or 0,
                    comment_count=r["comment_count"] or 0, publish_time=r["publish_time"],
                    source_strategy=("seed_channel" if r["channel_id"] in seed_set else "subscribed")))
            return videos[:eff]
        if p.get("use_seed_channels") or p.get("use_expanded_channels"):
            # Seed channels: channels hand-picked in profile entries. channel_ids are
            # injected by the orchestrator, already ordered by entry and capped. Fetched
            # concurrently per channel; one channel failing does not block the rest.
            # Tagged seed_channel (top of SOURCE_PRIORITY, so dedup keeps the seed
            # attribution on collisions). Blacklisted channels are filtered later in S3.
            # Expanded channels reuse the same logic with their own params and the
            # expanded_channel tag (ranked after subscribed, before hidden_gem).
            if p.get("use_seed_channels"):
                cids, label, n_default = p.get("seed_channel_ids") or [], "seed_channel", SEED_FETCH_PER_CHANNEL
            else:
                cids, label, n_default = (p.get("expanded_channel_ids") or [], "expanded_channel",
                                          _cfg.EXPAND_FETCH_PER_CHANNEL)
            if not cids:
                return []
            results = await asyncio.gather(
                *[_fetch_channel(adapter, c, p.get("n_per_channel", n_default),
                                 exclude, unfit)
                  for c in cids],
                return_exceptions=True)
            videos = []
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("%s 单频道失败(已跳过): %s", label, safe_repr(r))
                else:
                    videos.extend(r)
            for v in videos:
                v.source_strategy = label
            return videos
        elif p.get("use_type_pool"):
            # Type channel pool: sample the offline, curated pool by the user's interest
            # chips; uploads cost 1 unit per channel (uploads ids are cached). Re-tagged
            # channel_pool, ranked after hidden_gem in SOURCE_PRIORITY because the gem tag
            # identifies exploration slots and overriding it is costly.
            # Invariant: this path never writes user_channel_prefs.
            rows = db.pool_channels_for_chips(
                db.get_user_chips(adapter.user_id), p.get("n_channels", 8),
                # The pool is shared site-wide and built without anyone's prefs, so a
                # user's blacklisted channels can be in it. S3 would drop their videos,
                # but the slot and the 1-unit fetch would already be spent.
                exclude_ids=db.blacklisted_channel_ids(adapter.user_id))
            if not rows:
                logger.warning("use_type_pool:类型池为空,策略返回空")
                return []
            per_channel = max(1, math.ceil(eff / len(rows)))
            results = await asyncio.gather(
                *[_fetch_channel(adapter, r["channel_id"], per_channel, exclude, unfit)
                  for r in rows],
                return_exceptions=True)
            videos = []
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("use_type_pool 单频道失败(已跳过): %s", safe_repr(r))
                else:
                    videos.extend(r)
            for v in videos:
                v.source_strategy = "channel_pool"
        elif p.get("use_top_channels"):
            # Followed-channel rotation: weighted random sampling over the full candidate
            # set instead of always taking the top n, which would make every refresh
            # return the newest uploads of the same channels and keep supply static.
            n_ch = p.get("n_channels", 10)
            # Weights use same-day pick counts and the refresh's featured profile entries;
            # featured_positions/mode are rewritten into params by the orchestrator on
            # every refresh, so reused plans never go stale.
            _day = db.user_local_day(adapter.user_id)
            rows = db.rotation_channels(adapter.user_id, _day,                 # pool floor + "loved but not followed" line
                                        _cfg.FOLLOW_POOL_MIN_PREF, _cfg.FOLLOW_POOL_LOVED_PREF)
            if not rows:
                logger.warning("use_top_channels:该用户暂无偏好频道(冷启动未导入?),策略返回空")
                return []
            _aff = (db.channel_entry_affinity(adapter.user_id, p["mode"], p.get("featured_positions") or [])
                    if p.get("mode") else set())
            rows = _weighted_sample_channels(rows, n_ch, _aff)
            db.bump_channel_picks(adapter.user_id, [r["channel_id"] for r in rows], _day)
            logger.info("use_top_channels: FR-1 抽 %d 频道(贴本刷画像 %d,今日已中次数 %s)",
                        len(rows), sum(1 for r in rows if r["channel_id"] in _aff),
                        [int(_row_val(r, "picks")) for r in rows])

            # Split the target across channels; at least 1 per channel (a 0-item call
            # would still cost 1 quota).
            per_channel = max(1, math.ceil(eff / len(rows)))

            # Fail-soft per channel: exceptions are returned in place and skipped below.
            results = await asyncio.gather(
                *[_fetch_channel(adapter, r["channel_id"], per_channel, exclude, unfit) for r in rows],
                return_exceptions=True,
            )
            videos: list[Video] = []
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("fetch_channel_videos 单频道失败(已跳过): %s", safe_repr(r))
                else:
                    videos.extend(r)
        else:
            # Direct single-channel path: channel_id is required. p["channel_id"] (not .get)
            # so an incomplete plan fails loudly instead of silently drifting.
            videos = await _fetch_channel(adapter, p["channel_id"], eff, exclude, unfit)

    # Keyword search. A "queries" bundle (intent search) runs each query concurrently with
    # the target split evenly (ceil); the bundle's target is sized so each query returns
    # exactly its share. A single "query" is the plain path.
    elif s.tool == "search_by_keyword":
        queries = p.get("queries")
        if queries:
            per_q = max(1, math.ceil(eff / len(queries)))
            results = await asyncio.gather(
                *[adapter.search_by_keyword(q, per_q) for q in queries],
                return_exceptions=True)
            videos = []
            for q, r in zip(queries, results):
                if isinstance(r, Exception):
                    logger.warning("search 批单词失败(已跳过)「%s」: %s",
                                   q, safe_repr(r))
                else:
                    videos.extend(r)
        else:
            videos = await adapter.search_by_keyword(p["query"], eff)
    # Category trending chart: category_id is required.
    elif s.tool == "fetch_trending_by_category":
        videos = await adapter.fetch_trending_by_category(p["category_id"], eff)
    # Small-channel hidden gems: k (channels to sample) is optional.
    elif s.tool == "mine_hidden_gems":
        videos = await adapter.mine_hidden_gems(p.get("k", SMALL_POOL_SAMPLE_K))
    else:
        raise ValueError(f"未知检索工具: {s.tool}")
    # Common exit: truncate to the overfetch quota eff. Truncation keeps enrich from
    # billing for far more than the plan asked (hidden gems can return ~300 items for a
    # ~20-item target); the headroom above target_count is for execute_plan's duration
    # filter, after which results are trimmed back to the plan's ratios. Enrich cost for
    # the headroom is included in the costing estimate.
    # Intent strategies mark every result as intent-sourced (dedup preserves the flag).
    if s.intent_driven:
        for v in videos:
            v.intent_sourced = True
    return videos[:eff]




# ---- Ratio truncation helpers ----
def _strategy_tag(s: StrategyCall) -> str:
    """StrategyCall -> the source_strategy tag it produces (matches _run_strategy's re-tagging)."""
    if s.tool == "fetch_channel_videos":
        if s.params.get("use_local_intent"):
            return "search"                # local intent recall shares the live intent-search tag (intent results bypass ratios anyway)
        if s.params.get("use_library"):
            return "subscribed"            # aggregated under the subscribed target (per-video tags set in _run_strategy)
        if s.params.get("use_seed_channels"):
            return "seed_channel"
        if s.params.get("use_type_pool"):
            return "channel_pool"
        return "subscribed"
    return {"search_by_keyword": "search",
            "fetch_trending_by_category": "trending",
            "mine_hidden_gems": "hidden_gem"}.get(s.tool, s.tool)


def _truncate_by_target(videos: list[Video], plan: RetrievalPlan) -> list[Video]:
    """Trim back to the plan's ratios after overfetch + filtering: group by
    source_strategy and cap each tag at the sum of its strategies' target_count,
    preserving list order (uploads by recency / trending by popularity).

    Intent results pass through without consuming ratio slots (the tool already took
    exactly the intended amount). Unknown tags fail open and pass through (edge cases
    such as dedup re-tagging; better too many than too few — S3 and ranking absorb it)."""
    targets: dict[str, int] = {}
    for s in plan.strategies:
        if s.intent_driven:
            continue
        tag = _strategy_tag(s)
        targets[tag] = targets.get(tag, 0) + s.target_count
    # Total S2 delivery cap: with deep paging, channel paths deliver their full targets,
    # which would flood S3 -> S6 and the LLM judge. The cap only scales non-search sources
    # (search results already spent the scarce search bucket; cutting them wastes it
    # twice), proportionally, with a floor of 1 each.
    cap = int(getattr(_cfg, "S2_DELIVER_MAX", 0) or 0)
    capped_total = sum(t for tag, t in targets.items() if tag != "search")
    if cap > 0 and capped_total > cap:
        scale = cap / capped_total
        for tag in list(targets):
            if tag != "search":
                targets[tag] = max(1, int(targets[tag] * scale))
        logger.info("S2 交付上限: 配比目标 %d → %d(S2_DELIVER_MAX,search 不缩)", capped_total, cap)
    taken: dict[str, int] = {}
    out: list[Video] = []
    for v in videos:
        tag = v.source_strategy
        if v.intent_sourced or tag not in targets:
            out.append(v)
            continue
        if taken.get(tag, 0) < targets[tag]:
            taken[tag] = taken.get(tag, 0) + 1
            out.append(v)
    return out


async def execute_plan(
    plan: RetrievalPlan,
    adapter: YouTubeAdapter,
    persist: bool = True,
    excluded_video_ids: set[str] | None = None,
) -> list[Video]:
    """Execute a RetrievalPlan: budget guard -> mode injection -> concurrent strategies
    -> dedup -> pre-enrich filtering -> enrich -> persist -> duration filter, per-channel
    cap and ratio truncation. Input comes from the S1 planner (or a static default plan);
    output is the list of persisted Videos handed to S3.

    - asyncio.gather(return_exceptions=True): one strategy failing does not block others.
    - Order is fixed: dedup before enrich, so the same id is never billed twice.
    - persist=True writes channels before videos; tests/callers can manage persistence.
    - Plan logs, refresh id and per-strategy counts belong to the orchestrator, which
      aggregates counts from the returned Video.source_strategy.
    """
    # Duck typing: test doubles may have no quota counter, in which case the guard is skipped.
    quota = getattr(adapter, "quota", None)
    global _INFLIGHT_EST, _INFLIGHT_SEARCH
    reserved_g = reserved_s = 0
    if quota is not None:
        # Deterministic estimate from the official cost table, per bucket. The LLM's own
        # plan.est_quota is only logged; spending decisions never rely on it.
        est = estimate_plan_cost(plan)
        # Reservation-style guard: subtract in-flight plans' estimates before comparing,
        # and reserve with no await in between (atomic under single-process asyncio), so N
        # concurrent plans cannot all pass against the same remaining balance. Released in
        # finally. While a plan runs, its real charges and its reservation briefly overlap,
        # which only makes the guard more conservative — the safe direction.
        # Each bucket is compared, reserved and released separately; the error names the bucket.
        remaining_g = quota.remaining() - _INFLIGHT_EST
        remaining_s = quota.remaining("search") - _INFLIGHT_SEARCH
        if est.general > remaining_g:
            raise RuntimeError(
                f"计划预估通用成本 {est.general} 超过今日剩余 quota {remaining_g},拒绝执行"
                "(D9 最后防线;计划级预算校验在 agent/planner.validate_plan)")
        if est.search > remaining_s:
            raise RuntimeError(
                f"计划预估搜索次数 {est.search} 超过今日搜索桶剩余 {remaining_s},拒绝执行"
                "(D9 最后防线;搜索门槛在 orchestrator 注入侧)")
        _INFLIGHT_EST += est.general
        _INFLIGHT_SEARCH += est.search
        reserved_g, reserved_s = est.general, est.search

    try:
        # Inject region/language for plan.mode before gather, so all concurrent strategies
        # read the same settled values (no race).
        mode = MODES.get(plan.mode)
        if mode is not None:
            adapter.region_code = mode.region_code
            adapter.relevance_language = mode.relevance_language

        # The exclusion set and the mode's duration predicate are passed down to the fetch
        # layer, where deep paging uses them to judge "valid" items. The exclusion /
        # pre-check loop below stays as a safety net (search/trending/gems rely on it).
        _exclude = frozenset(excluded_video_ids or ())
        _unfit = ((lambda d: d > 0 and not mode.fits_duration(d)) if mode is not None else None)
        adapter._deep_stats = {"channels": 0, "paged": 0, "pages": 0, "added": 0,
                               "exhausted": 0, "resting": 0}
        # Strategy-level concurrency and fail-soft: each strategy hits the API independently.
        results = await asyncio.gather(
            *[_run_strategy(adapter, s, exclude=_exclude, unfit=_unfit) for s in plan.strategies],
            return_exceptions=True,
        )
        _ds = adapter._deep_stats
        if _ds["paged"] or _ds["resting"]:
            logger.info("S2 深挖: 频道 %d 个 / 触发翻页 %d 个 / 深页 %d 页(+%du) / 深页有效补 %d 条 / "
                        "翻到底 %d 个 / 歇息中 %d 个", _ds["channels"], _ds["paged"], _ds["pages"],
                        _ds["pages"], _ds["added"], _ds["exhausted"], _ds["resting"])
        videos: list[Video] = []
        # gather preserves order, so failures can be attributed to the right strategy.
        for s, r in zip(plan.strategies, results):
            if isinstance(r, Exception):
                logger.warning("策略 %s 整体失败(已跳过): %s", s.tool, safe_repr(r))
            else:
                videos.extend(r)

        # Fixed order: dedup before enrich.
        deduped = dedup(videos)

        # Pre-enrich filtering, zero cost:
        #  - Known out-of-window durations (prefilled by trending, or from earlier enrich in
        #    the videos table) are dropped before enrich, so repeated supply does not keep
        #    billing enrich batches. A stored 0 means "enrich failed / placeholder" and is
        #    treated as unknown (avoids permanently dropping e.g. live-stream placeholders).
        #  - Excluded ids (impression penalty / dislikes / already consumed, merged by the
        #    caller) are removed in the same loop, so they never take ratio slots that
        #    fresh candidates need.
        # Intent results are exempt from both (their handling lives in S3 / the orchestrator).
        dropped_pre = 0
        dropped_banned = 0
        banned = excluded_video_ids or set()
        if mode is not None or banned:
            unknown = ([v.video_id for v in deduped
                        if v.duration_sec == 0 and not v.intent_sourced]
                       if mode is not None else [])
            known = db.video_durations(unknown) if unknown else {}
            kept_fetch: list[Video] = []
            for v in deduped:
                if v.video_id in banned and not v.intent_sourced:
                    dropped_banned += 1
                    continue
                if mode is not None:
                    d = v.duration_sec or known.get(v.video_id, 0)
                    if d > 0 and not v.intent_sourced and not mode.fits_duration(d):
                        dropped_pre += 1
                        continue
                kept_fetch.append(v)
            deduped = kept_fetch

        enriched = await adapter.enrich_stats(deduped)
        # Out-of-window videos are persisted too: the videos table doubles as a free
        # duration cache for the next refresh's pre-check.
        if persist:
            persist_videos(enriched)

        # Post-enrich: filter by the mode's duration window, then trim to the plan's ratios.
        # Intent results pass through (S3 applies its own rules to them). Nothing is written
        # to filter_results here: the filter-accounting invariant starts at S3's input, so
        # S2-side removals are log-only. S3's duration rule remains as a backstop.
        if mode is None:
            return enriched                      # unknown mode: no filtering (matches S3's fallback)
        dropped_post = 0
        eligible: list[Video] = []
        for v in enriched:
            if v.intent_sourced or mode.fits_duration(v.duration_sec):
                eligible.append(v)
            else:
                dropped_post += 1
        # Per-channel delivery cap: keep the first S2_PER_CHANNEL_MAX per channel (newest
        # uploads first). Search and intent results are exempt (already spread across
        # queries). Without this, one sampled channel could fill 20+ slots of a refresh and
        # waste judge calls. Log-only, like the filters above.
        dropped_ch = 0
        _per_ch = int(getattr(_cfg, "S2_PER_CHANNEL_MAX", 0) or 0)
        if _per_ch > 0:
            _seen_ch: dict[str, int] = {}
            _capped: list[Video] = []
            for v in eligible:
                if v.intent_sourced or v.source_strategy == "search":
                    _capped.append(v)
                    continue
                _n = _seen_ch.get(v.channel_id, 0)
                if _n >= _per_ch:
                    dropped_ch += 1
                    continue
                _seen_ch[v.channel_id] = _n + 1
                _capped.append(v)
            eligible = _capped
        result = _truncate_by_target(eligible, plan)
        if dropped_pre or dropped_post or dropped_banned or dropped_ch or len(eligible) != len(result):
            logger.info(
                "S2 时长窗前移: 预检掉 %d / 禁推减集掉 %d / enrich 后掉 %d / 频道上限掉 %d / "
                "配比截断掉 %d → 交付 %d 条",
                dropped_pre, dropped_banned, dropped_post, dropped_ch,
                len(eligible) - len(result), len(result))
        return result
    finally:
        _INFLIGHT_EST -= reserved_g          # release reservations (on success, failure or cancellation)
        _INFLIGHT_SEARCH -= reserved_s
