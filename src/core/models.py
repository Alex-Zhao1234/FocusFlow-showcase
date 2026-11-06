"""Shared data models used across the pipeline, agent layer and orchestrator.

Covers the YouTube-facing entities (Channel, Video), per-stage verdicts and
intermediate results (QualityVerdict, Triage, ModeFitVerdict), the planner's
input/output contract (PlanContext, StrategyCall, RetrievalPlan), the user
profile, the outward-facing RecommendationCard, the Mode definition, and the
structured-output models for the offline channel-pool job.
"""
import logging
from typing import Literal
from pydantic import BaseModel, computed_field, field_validator, model_validator, Field

logger = logging.getLogger(__name__)

class Channel(BaseModel):
    """Objective channel attributes (shared across users, no user_id)."""
    channel_id:str
    channel_title:str|None = None
    channel_subs: int|None = None         # subscriber count
    uploads_playlist_id:str|None = None   # the channel's "uploads" playlist id
    is_small_pool: bool=False             # belongs to the small-channel ("hidden gems") pool
    last_fetched: int|None = None         # last fetch time (Unix seconds)
    # ---- In-memory only: not persisted by merge/upsert ----
    video_count: int | None = None            # statistics.videoCount, used by hard quality gates
    topic_categories: list[str] | None = None # topicDetails Wikipedia topic suffixes
    # Channel self-description fetched alongside channels.list, fed to the expansion judge.
    description: str | None = None            # snippet.description
    keywords: str | None = None               # brandingSettings.channel.keywords


class Video(BaseModel):
    """Unified video model. Fields describe the video itself as returned by the
    YouTube API; channel title/subscribers live on Channel, and storage metadata
    such as created_at belongs to the storage layer, not here."""
    # ---- Required ----
    video_id: str
    channel_id: str                       # foreign key to channels
    title: str
    duration_sec: int
    source_strategy: str                  # supply source: subscribed / channel_pool / search / trending / hidden_gem
    # ---- Optional ----
    platform: str = "youtube"
    description: str = ""                 # first 500 characters
    tags: list[str] = []
    cover_url: str | None = None
    category_id: int | None = None
    view_count: int = 0
    like_count: int = 0
    like_hidden: bool = False
    comment_count: int = 0
    publish_time: int | None = None
    # Pass for videos recalled by an intent search bundle. Transient within one
    # refresh, never persisted; drives the S3 exemption, the S6 separate pool and
    # S7 guaranteed seats. On dedup collisions the flags are OR-ed.
    intent_sourced: bool = False


    @computed_field
    @property
    def engagement_rate(self) -> float:
        """(like + 5*comment) / max(view, 1), stored raw. Platform-side public
        statistics; included in model_dump() as a read-only computed field."""
        return (self.like_count + 5 * self.comment_count) / max(self.view_count, 1)


class QualityVerdict(BaseModel):
    """Quality verdict for a video (S4/S5), cached in the quality_verdicts table."""
    video_id:str
    passed:bool = False
    quality_grade: Literal[0, 1, 2, 3] | None = None
    degraded: bool = False
    categories:list[str] = []
    confidence:float=0.0
    reason:str=""


class Triage(BaseModel):
    """S4 quality triage result, three buckets:

    - direct: statistical score >= QUALITY_PASS_THRESHOLD, passes without LLM review;
    - review: score too low, sent to S5 batched LLM review;
    - cached: 7-day cache hits (both passed and rejected). Triage only routes and
      never rules — recording rejections of cached failures is the job of S5 /
      the orchestrator.
    """
    direct:list[Video]
    review:list[Video]
    cached:list[tuple[Video,QualityVerdict]]


# ================================================================== #
# Agent-layer models
# ================================================================== #

class PlanContext(BaseModel):
    """S1 planner input. It is serialised into the LLM prompt, so every field must
    be JSON-serialisable: sqlite rows or custom objects are converted to plain
    types when the context is built, and the model can be model_dump_json()'d whole.
    """
    # Mode summary: {name, display_name, keywords, yt_category_ids, duration_window,
    # enable_discovery, strict_mode}. Blacklist keywords are applied in S3, not here.
    mode_config: dict
    # Top preferred channels by pref_score: [{channel_id, channel_title, pref_score}].
    channels: list[dict] = []
    # Recent explicit feedback per channel: {channel_id: [likes, dislikes]}.
    feedback: dict[str, list[int]] = {}
    quota_left: int                            # remaining general-pool quota
    search_left: int = 100                     # remaining search-bucket calls; defaults to a full bucket

    last_plan: dict | None = None              # previous plan {plan, per_strategy_counts, pass_rate, reused}; None on cold start

    time_stats: dict = {}                      # time-of-day distribution; currently always empty (placeholder)

    user_profile: dict = {}                    # persistent profile; {} when absent/disabled/unparseable (fail-soft), {"free_note": ...} when structuring failed

    session_intent: str = ""                   # active session intent, raw text; "" when none


class UserProfile(BaseModel):
    """Persistent structured profile. Every field must translate into an S1
    planning action; anything that cannot is left out. Deliberately has no
    demographic fields — age/gender/occupation are noisy proxies, and inferring
    undeclared attributes is off-limits.

    Token budget protection: the prompt asks for brevity, validators truncate
    with a warning instead of raising, and the profile is only consumed by S1.
    """
    interests: list[str] = []           # topics → search keywords / category leaning
    current_goals: list[str] = []       # time-bound goals (exam prep, learning English) → highest planning priority
    content_prefs: list[str] = []       # format preferences (long documentaries, tutorials, no talking) → strategy mix
    avoid: list[str] = []               # things the user said they don't want → negative planning constraint (not the S3 blacklist)
    languages: list[str] = []           # content language preferences
    free_note: str = ""                 # remaining statements that fit no field, quoted verbatim

    # Truncation limits live in config, but config imports this module to build
    # Mode/RetrievalPlan instances, so config is imported lazily inside the
    # validators. UserProfile is never constructed by config, so by the time a
    # validator runs config is fully loaded — no import cycle.
    @field_validator("interests", "current_goals", "content_prefs",
                     "avoid", "languages")
    @classmethod
    def _cap_list(cls, v: list[str], info):
        import config
        if len(v) > config.PROFILE_LIST_MAX:
            logger.warning("UserProfile.%s 超限(%d 项),截断到 %d",
                           info.field_name, len(v), config.PROFILE_LIST_MAX)
            return v[:config.PROFILE_LIST_MAX]
        return v

    @field_validator("free_note")
    @classmethod
    def _cap_note(cls, v: str):
        import config
        if len(v) > config.PROFILE_NOTE_MAX:
            logger.warning("UserProfile.free_note 超长(%d 字),截断到 %d",
                           len(v), config.PROFILE_NOTE_MAX)
            return v[:config.PROFILE_NOTE_MAX]
        return v


class ModeFitVerdict(BaseModel):
    """S6 mode-fit verdict produced by the LLM judge."""
    video_id: str
    fit: bool
    fit_score: float = 0.5              # 0~1, consumed by the ranker's W_FIT term; 0.5 is neutral
    content_type: str | None = None     # content type from the fixed vocabulary, judged in the same call; feeds the ranker's W_TYPE term
    degraded: bool = False              # True = generated by code fallback because the LLM was unavailable
    reason: str = ""                    # recommendation reason (fit=True) or rejection reason (fit=False), produced in the same call
    profile_entry: int | None = None    # which profile entry (1-based) of the current mode this card belongs to; None if none / intent batch / degraded
    cached: bool = False                # True = served from the 7-day verdict cache (no LLM call); bookkeeping notes get a "[cached]" prefix


# ================================================================== #
# Orchestrator output models
# ================================================================== #

class RecommendationCard(BaseModel):
    """Final S6 product, yielded in batches by the orchestrator; S7 and the UI
    consume it without modification. All fields are JSON-serialisable.

    Relation to RankedItem (the ranker's internal dataclass): RankedItem is an
    internal intermediate of the ranking step, while this card is the external
    contract. The orchestrator assembles cards from RankedItem plus the retained
    reason at the end; the two are intentionally kept separate.
    """
    video: Video
    entry_position: int | None = None   # profile-entry attribution, stored with the card for feedback attribution; None = unattributed
    reason: str                    # recommendation text shown on the card (the judge's fit=True reason)
    fit_score: float
    quality_norm: float            # quality normalised to 0~1: stat-score direct pass = DIRECT_PASS_QUALITY_NORM, LLM grade/3, cached grade, degraded path, or stat fallback
    pref_score: float              # channel preference score
    is_gem: bool
    final: float | None = None     # final weighted score; None for intermediate ("incoming") frames that have not been ranked yet
    # Streaming state: incoming = shown early from an intermediate wave; top = final
    # on-screen section (≤ TOP_N); more = final collapsed section.
    section: Literal["incoming", "top", "more"] = "incoming"
    # Content type passed through from the verdict, for explainability of the W_TYPE term.
    content_type: str | None = None
    overlay: bool = False          # card from a lightweight intent refresh: the frontend inserts it at the top of the current feed instead of swapping pages


# ================================================================== #
# Mode and plan models (config.py depends on these directly)
# ================================================================== #

class Mode(BaseModel):
    """Context-mode interface. The four mode instances are defined in config.py."""
    name: str                              # focus / entertainment / explore / wind_down
    display_name: str                      # display name
    yt_category_ids: list[int]
    keywords: list[str]
    blacklist_keywords: list[str]
    duration_min_sec: int
    duration_max_sec: int | None = None    # None = no upper bound
    region_code: str = "US"                # regionCode for search/trending (YouTube does not support CN)
    relevance_language: str | None = None  # search.list relevanceLanguage; None = any language
    enable_discovery: bool = False         # enable hidden-gem mining
    llm_system_prompt: str = ""            # unused: prompts live in the prompt registry; kept for signature stability
    strict_mode: bool = False              # wind_down=True: better to show fewer than show poor fits

    def fits_duration(self, duration_sec: int) -> bool:
        """Duration-window predicate shared by the S2 early filter and the S3
        prefilter. duration_sec=0 (enrichment-failure placeholder) is always below
        any mode minimum and returns False; max None = no upper bound."""
        if duration_sec < self.duration_min_sec:
            return False
        return self.duration_max_sec is None or duration_sec <= self.duration_max_sec


class StrategyCall(BaseModel):
    """A single retrieval strategy (S1 output)."""
    tool: Literal["fetch_channel_videos", "search_by_keyword",
                  "fetch_trending_by_category", "mine_hidden_gems"]
    params: dict
    target_count: int
    rationale: str                         # one-sentence reason, logged for planner-behaviour analysis
    # Intent-search-bundle flag. Only the orchestrator may set it to True; any value
    # produced by the planner is cleared in code, so the LLM cannot hallucinate a pass.
    intent_driven: bool = False


class RetrievalPlan(BaseModel):
    """Retrieval plan (S1 output)."""
    mode: Literal["focus", "entertainment", "explore", "wind_down"]
    strategies: list[StrategyCall]         # 2~6 strategies
    est_quota: int                         # the LLM's own estimate; logged only to observe its estimation ability
    # One-line acknowledgement to the user (≤60 chars), generated only when a session
    # intent is active. Observational: not validated; static fallback plans leave it empty.
    ack: str = ""


# ============================================================================
# Offline channel-pool job: LLM structured-output models (not on the refresh path)
# ============================================================================

class ChipSubtopicSet(BaseModel):
    """LLM output: sub-topic list for a chip. Only trimmed and de-duplicated here;
    the count limit lives in the prompt and is enforced by the caller via
    config.CHIP_SUBTOPICS_MAX (models does not import config, to avoid a cycle)."""

    subtopics: list[str]

    @model_validator(mode="after")
    def _clean(self):
        cleaned: list[str] = []
        seen: set[str] = set()
        for s in self.subtopics:
            s = s.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                cleaned.append(s)
        self.subtopics = cleaned
        return self


class ChannelSearchQuerySet(BaseModel):
    """LLM output: channel search queries, 3–5 distinct non-empty strings. The
    bounds mirror config.DISCOVERY_QUERIES_MIN/MAX as literals (models must not
    import config); a test keeps the two in sync."""

    queries: list[str]

    @model_validator(mode="after")             # runs after field validation, so queries is already list[str]
    def _clean_and_bound(self):
        cleaned: list[str] = []
        seen: set[str] = set()                 # case-insensitive de-dup keys
        for q in self.queries:
            q = q.strip()                      # LLM output often carries surrounding whitespace/newlines
            if q and q.lower() not in seen:
                seen.add(q.lower())
                cleaned.append(q)
        if not (3 <= len(cleaned) <= 5):
            raise ValueError(f"需要 3–5 个互异非空搜索词,清洗后得到 {len(cleaned)} 个")
        self.queries = cleaned
        return self


class ChannelCandidateVerdict(BaseModel):
    """LLM verdict for a single candidate channel."""
    channel_id: str
    relevance: float = Field(ge=0.0, le=1.0)    # topical fit; used for ordering only, never vetoes
    keep: bool                                  # whether to add the channel to the pool


class ChannelCandidateVerdicts(BaseModel):
    """Batch container for candidate verdicts. Id alignment is the caller's job:
    verdicts are consumed by id and missing ones are dropped."""
    verdicts: list[ChannelCandidateVerdict]


class ChipJudgeVerdict(BaseModel):
    """Chip judge output: selected channel ids. When in doubt the judge leaves a
    channel out, so the list may be empty; the caller truncates it to
    TYPE_POOL_TARGET_MAX."""
    selected_ids: list[str] = []
