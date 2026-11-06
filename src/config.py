"""Central configuration: every tunable parameter lives here.

Business code must not contain magic numbers; it reads them from this module.
The file sits at the bottom of the dependency graph (it imports only
core.models) and has three parts:

1. Global parameters — thresholds, weights, batch sizes, quota cost table, etc.
2. The four Mode instances: focus / entertainment / explore / wind_down.
3. DEFAULT_PLANS — a static retrieval plan per mode, used as the fallback when
   S1 planning fails.

Each constant carries one short comment naming the consuming module(s) and what
it controls.
"""
from core.models import Mode, RetrievalPlan, StrategyCall

# ================================================================== #
# 1. Global parameters
# ================================================================== #

# ---- Quality filtering (S4/S5) ----
QUALITY_CACHE_TTL_DAYS = 7  # triage · lifetime of cached S4/S5 quality verdicts

QUALITY_PASS_THRESHOLD = 0.65  # triage/orchestrator · S4 statistical score at or above which a video passes without LLM review
QUALITY_W_LIKE = 0.6  # triage · composite quality score: like-rate weight
QUALITY_W_COMMENT = 0.4  # triage · composite quality score: comment-rate weight

# orchestrator · per-mode override of the S4 direct-pass threshold; unlisted modes use the global value
MODE_DIRECT_THRESHOLDS: dict[str, float] = {"entertainment": 0.67, "explore": 0.67}

ALLOW_LIKE_HIDDEN_DIRECT = False  # triage · whether a video with hidden likes may pass directly on the comment signal alone
DIRECT_LIKE_TERM_FLOOR = 0.4  # triage · minimum like term required for a direct pass

QUALITY_LIKE_RATE_SCALE = 35  # triage · like-rate scaling: min(like_rate * 35, 1.0)
QUALITY_COMMENT_RATE_SCALE = 100  # triage · comment-rate scaling: min(comment_rate * 100, 1.0)

DEGRADE_PASS_THRESHOLD = 0.4  # judges · fail-open pass line on the statistical score when the LLM is unavailable

QUALITY_GRADE_PASS_MIN = 2  # judges · LLM quality grade (0-3 rubric); passed is derived in code as grade >= 2

FIT_SCORE_THRESHOLD: float | None = None  # orchestrator · optional minimum S6 fit_score; None = disabled

ESCALATE_CONF_MEAN_MIN = 0.0  # judges · escalate a batch to LLM_MODEL_FALLBACK when its mean confidence is below this


# ---- S6 final ranking weights ----
W_FIT = 0.35  # ranker · mode-fit score
W_TYPE = 0.15  # ranker · content-type bucket weight term

W_QUALITY = 0.3  # ranker · quality score
W_PREF = 0.2  # ranker · channel preference score (aggregated explicit feedback)
GEM_BONUS = 0.1  # ranker · bonus for hidden-gem (small channel) videos
# ranker · same-channel recency tie-break: two candidates from one channel whose final scores differ by
# <= this are treated as tied and the newer upload goes first (about a 0.05 fit difference; smaller than a
# quality or type step); 0 = off
RANK_RECENCY_TIE_DELTA = 0.02
RANK_CHANNEL_CAP = 5  # ranker · max on-screen seats per channel (top section only; reserved seats kept first; intent cards exempt); 0 = off
RANK_TYPE_CAP = 12  # ranker · max on-screen seats per content type (~40%; empty type not counted; intent cards exempt); 0 = off
DIRECT_PASS_QUALITY_NORM = 0.6  # orchestrator/ranker · constant quality_norm for direct-pass videos that have no LLM grade
EXPLORE_SLOT_FLOOR = 0.15  # ranker · minimum share of the top N reserved for hidden_gem + trending sources

# ================================================================== #
# Content-type axis
# ================================================================== #

# prompts/prefs · the 17-type content vocabulary; single source of truth for the type axis
CONTENT_TYPES = ["work_practice", "tech", "knowledge_pro", "knowledge_life",
                 "talk_podcast", "documentary", "news_politics", "comedy",
                 "music", "ambient", "cozy_life", "travel_outdoor",
                 "sports_gaming", "selfhelp_biz",
                 "film_anime", "fashion_beauty", "autos"]

# profiler/planner · fine-grained interest chips: slug → (display name, parent content type); 44 chips
INTEREST_CHIPS: dict[str, tuple[str, str]] = {
    # tech
    "ai": ("人工智能", "tech"),
    "programming": ("编程与开发", "tech"),
    "gadget_review": ("数码硬件评测", "tech"),
    "tech_business": ("科技商业资讯", "tech"),
    # work_practice
    "productivity_tools": ("效率工具与工作流", "work_practice"),
    "career_skills": ("职业技能教程", "work_practice"),
    "job_hunting": ("求职与职场经验", "work_practice"),
    # knowledge_pro
    "hard_science": ("硬核科普", "knowledge_pro"),
    "history_humanities": ("历史人文", "knowledge_pro"),
    "econ_finance": ("财经与经济学", "knowledge_pro"),
    "psychology": ("心理学", "knowledge_pro"),
    # knowledge_life
    "cooking": ("美食烹饪教程", "knowledge_life"),
    "fitness": ("健身与塑形", "knowledge_life"),
    "home_improve": ("家居收纳与装修", "knowledge_life"),
    # talk_podcast
    "deep_interview": ("人物深度访谈", "talk_podcast"),
    "casual_podcast": ("闲聊圆桌播客", "talk_podcast"),
    # documentary
    "nature_wildlife": ("自然与野生动物", "documentary"),
    "social_docu": ("社会人文纪实", "documentary"),
    "engineering_docu": ("科学与工程纪录", "documentary"),
    # news_politics
    "world_news": ("国际时事", "news_politics"),
    "investigative": ("深度调查报道", "news_politics"),
    # comedy
    "sketch_comedy": ("搞笑短剧与段子", "comedy"),
    "meme_remix": ("鬼畜与恶搞", "comedy"),
    "standup": ("脱口秀", "comedy"),
    # music
    "pop_music_mv": ("流行音乐与MV", "music"),
    "cover_instrument": ("翻唱与乐器演奏", "music"),
    # ambient
    "study_ambience": ("学习陪伴白噪音", "ambient"),
    "lofi": ("Lo-fi 与氛围音乐", "ambient"),
    # cozy_life
    "rural_slowlife": ("田园与乡村慢生活", "cozy_life"),
    "handcraft": ("手工与手作", "cozy_life"),
    "pets": ("宠物与萌宠日常", "cozy_life"),
    # travel_outdoor
    "travel_vlog": ("旅行Vlog", "travel_outdoor"),
    "hiking_camping": ("徒步露营与户外装备", "travel_outdoor"),
    "city_food_tour": ("城市探店与美食探访", "travel_outdoor"),
    # sports_gaming
    "gaming_esports": ("游戏实况与电竞", "sports_gaming"),
    "ball_sports": ("球类与体育赛事", "sports_gaming"),
    # selfhelp_biz
    "personal_finance": ("个人理财与投资", "selfhelp_biz"),
    "side_hustle": ("副业与创业案例", "selfhelp_biz"),
    "self_improvement": ("自我提升方法论", "selfhelp_biz"),
    # film_anime
    "anime_bangumi": ("动漫与番剧", "film_anime"),
    "film_commentary": ("影视剪辑与解说", "film_anime"),
    # fashion_beauty
    "fashion_outfit": ("时尚穿搭", "fashion_beauty"),
    "beauty_skincare": ("美妆护肤", "fashion_beauty"),
    # autos
    "cars_driving": ("汽车与驾驶", "autos"),
}


# ---- Type bucket weights ----
TYPE_BUCKET_WEIGHTS = [1.0, 0.7333, 0.4667, 0.2]  # type_weight_of · weights of the four buckets (primary / regular / occasional / marginal)
TYPE_WEIGHT_UNKNOWN = 0.6  # type_weight_of · weight for a missing, degraded or out-of-vocabulary type
TYPE_WEIGHT_OTHER = 0.2  # type_weight_of · weight when the LLM explicitly judged the video as "other"

# prefs/planner · default per-mode assignment of content types to the four buckets
MODE_TYPE_BUCKETS: dict[str, list[list[str]]] = {
    "focus":         [["work_practice", "tech", "knowledge_pro"],
                      ["documentary", "talk_podcast", "ambient", "music"],
                      ["knowledge_life", "cozy_life", "travel_outdoor", "comedy"],
                      ["sports_gaming", "selfhelp_biz", "news_politics",
                       "film_anime", "fashion_beauty", "autos"]],
    "explore":       [["knowledge_pro", "documentary", "tech"],
                      ["travel_outdoor", "talk_podcast", "knowledge_life", "music", "work_practice",
                       "autos"],
                      ["cozy_life", "comedy", "sports_gaming",
                       "film_anime", "fashion_beauty"],
                      ["news_politics", "ambient", "selfhelp_biz"]],
    "entertainment": [["comedy", "sports_gaming", "music",
                       "film_anime"],
                      ["travel_outdoor", "cozy_life", "documentary", "talk_podcast",
                       "fashion_beauty"],
                      ["knowledge_life", "tech", "knowledge_pro", "ambient",
                       "autos"],
                      ["news_politics", "work_practice", "selfhelp_biz"]],
    "wind_down":     [["cozy_life", "ambient", "music"],
                      ["travel_outdoor", "documentary", "talk_podcast", "knowledge_life"],
                      ["knowledge_pro", "comedy", "sports_gaming", "tech",
                       "film_anime", "fashion_beauty"],
                      ["work_practice", "news_politics", "selfhelp_biz",
                       "autos"]],
}


def normalize_content_type(raw: str) -> str:
    """Normalise how a content_type slug is written (shared by producers and
    consumers): Work_Practice / work practice / work-practice → work_practice.
    It only fixes spelling of the same slug; a wrong classification is not
    something normalisation can or should repair."""
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def type_weight_of(content_type: str | None, buckets: list[list[str]]) -> float:
    """Map a content type to its bucket weight. None → unknown weight, "other" →
    lowest weight, otherwise the weight of the bucket containing it; values
    outside the vocabulary are treated as unknown rather than penalised.
    Counting such values is done by the caller, which knows the prompt version."""
    if content_type is None:                       # degraded verdict or missing from LLM output
        return TYPE_WEIGHT_UNKNOWN
    ct = normalize_content_type(content_type)
    if ct == "other":
        return TYPE_WEIGHT_OTHER
    for w, bucket in zip(TYPE_BUCKET_WEIGHTS, buckets):
        if ct in bucket:
            return w
    return TYPE_WEIGHT_UNKNOWN                     # out of vocabulary = unknown, not penalised


# judges · YouTube category id → official English name (read side, full list including legacy ids)
YT_CATEGORY_NAMES: dict[int, str] = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
    15: "Pets & Animals", 17: "Sports", 18: "Short Movies",
    19: "Travel & Events", 20: "Gaming", 21: "Videoblogging",
    22: "People & Blogs", 23: "Comedy", 24: "Entertainment",
    25: "News & Politics", 26: "Howto & Style", 27: "Education",
    28: "Science & Technology", 29: "Nonprofits & Activism",

    # Legacy categories (assignable: false): uploaders can no longer choose them, but
    # older videos may still carry them, so the read side must know their names. They
    # must not be used on the write side (mode whitelists, trending queries).
    30: "Movies", 31: "Anime/Animation", 32: "Action/Adventure",
    33: "Classics", 34: "Comedy", 35: "Documentary", 36: "Drama",
    37: "Family", 38: "Foreign", 39: "Horror", 40: "Sci-Fi/Fantasy",
    41: "Thriller", 42: "Shorts", 43: "Shows", 44: "Trailers",
}

# ---- LLM batching and screen size ----
LLM_BATCH_SIZE = 15  # judges · S5/S6 LLM batch size
TOP_N = 30  # ranker/orchestrator · number of cards in the on-screen section

# ---- YouTube quota and search budgets ----
DAILY_QUOTA = 10_000  # storage · daily general-pool quota (search.list is counted separately)
DAILY_SEARCH_CALLS = 100  # storage · separate daily bucket for search.list calls
SEARCH_FLOOR_PROFILE = 20  # planner/orchestrator · below this many remaining search calls, profile rotation and free planner searches stop (refreshes still produce cards)
PROFILE_SEARCH_DAILY_MAX = 40  # orchestrator · per-user daily cap on profile-rotation plus free planner searches
INTENT_SEARCH_DAILY_MAX = 40  # orchestrator · per-user daily cap on intent searches (each refresh still ≤ INTENT_SEARCH_MAX)

DAILY_REFRESH_LIMIT = 1000  # prewarm/API · per-user daily refresh limit (effectively off; the global quota is the real limit)
MAX_SEARCH_PER_REFRESH = 2  # planner/sources · hard per-refresh cap on searches the planner may choose freely (intent bundles not counted)
PROFILE_SEARCH_TIER1_RESERVED = 1  # diversity · leading query slots reserved for profile entry 1; secondary entries rotate after it; 0 = off
EXCLUDE_EXEMPT_TOP_N = 2  # diversity/orchestrator · number of top profile entries exempt from excluded content types (seat and S7 side alike); 0 = off

# ---- Session intent search ----
INTENT_SEARCH_MAX = 7  # intent_search/orchestrator · max intent queries from the LLM (the verbatim intent first, never dropped, plus variants)
INTENT_PER_QUERY_KEEP = 10  # sources/orchestrator · results kept per intent query, in relevance order (50 = no truncation)
INTENT_MIN_DURATION_SEC = 60  # prefilter · anti-Shorts duration floor for intent results, which bypass the mode duration window
INTENT_RESERVED_SLOTS = 15  # ranker · seats reserved for intent cards at the head of the top N (not held empty if short)
INTENT_FIRST_WAVE_MAX = 15  # orchestrator · intent fast lane: intent items placed at the head of the first judge wave
INTENT_LITE_QUERIES = 6  # orchestrator · number of queries in a lightweight intent refresh
INTENT_LITE_TOP = 15  # orchestrator · number of final cards from a lightweight intent refresh (≤ INTENT_RESERVED_SLOTS)

# ---- Long-term viewing plans ----
VIEWING_PLAN_MODEL: str | None = "claude-sonnet-5"  # agent/viewing_plan · model for the planning agent (None = LLM_MODEL)
VIEWING_PLAN_SEARCH_MAX = 5      # viewing_plan_run · max search queries per daily plan batch
VIEWING_PLAN_MIN_QUERIES = 2     # viewing_plan_run · minimum queries the budget must allow, otherwise the plan refresh is skipped
VIEWING_PLAN_DAILY_DEFAULT = 5   # core/viewing_plan · default cards per day
VIEWING_PLAN_DAILY_MAX = 10      # core/viewing_plan · max cards per day
VIEWING_PLAN_DAYS_MAX = 30       # core/viewing_plan · max total plan length (days)
VIEWING_PLAN_STAGES_MAX = 8      # core/viewing_plan · max number of stages
VIEWING_PLAN_NAME_MAX = 20       # core/viewing_plan · max plan name length
VIEWING_PLAN_KEYWORDS_PER_STAGE = 5  # core/viewing_plan · max keywords per stage
VIEWING_PLAN_KEYWORD_MAX = 60    # core/viewing_plan · max characters per keyword
INTENT_FILL_BUCKET_TIERS = 2  # orchestrator · intent refresh fills remaining seats from the user's top N type buckets (no quota, no LLM)

# ---- Profile diversity seats ----
DIVERSITY_TIER1_SEATS = 8  # ranker · guaranteed seats for profile entry 1
DIVERSITY_SUB_SEATS = 4  # ranker · guaranteed seats per drawn secondary entry (doubled with recent-play boost)
DIVERSITY_SUB_BASE_WEIGHTS = [0.6, 0.45, 0.3, 0.2]  # diversity · draw weights for entries 2-5, decreasing by position
DIVERSITY_MAX_SUBS = 3  # diversity · max secondary entries drawn per refresh
DIVERSITY_ENTRY_CAP = 12  # ranker · max seats per profile entry in the top section (~40%)
DIVERSITY_PLAY_WINDOW_SEC = 600  # storage/diversity · recent-play window that boosts an entry's draw weight
DIVERSITY_PLAY_WEIGHT_MULT = 2.0  # diversity · draw-weight multiplier for a recently played entry (capped)
DIVERSITY_WEIGHT_CAP = 0.9  # diversity · draw-weight cap, so a boosted entry is never certain
DIVERSITY_MISS_FORCE = 3  # diversity · a secondary entry missed this many times in a row is forced in
SEAT_PROFILE_SHARE_PCT = 85  # diversity · share of seats given to profile entries in manual seat plans; the rest is a silent exploration reserve

# ---- Seed and expanded channels ----
EXPAND_UPLOAD_TITLES = 20  # profiler/sources · recent upload titles sent to the expansion judge per candidate channel
EXPAND_JUDGE_BATCH = 20  # profiler · candidates per expansion-judge chunk (chunks run concurrently)

ENTRY_CHIPS_MAX = 3  # profiler · max chips derived per profile entry, used for precise pool backfill
SEED_FETCH_PER_CHANNEL = 3  # orchestrator/sources · latest videos fetched per seed channel
SEED_FETCH_CHANNELS_MAX = 10  # orchestrator · max seed channels fetched per refresh (entry 1 first, then round-robin)
SEED_TIER1_FETCH_ALL = True  # orchestrator/storage · fetch all of entry 1's seed channels every refresh; False = pure round-robin
SEED_TIER1_SEATS = 3  # ranker · reserved seed seats for entry 1 (other entries use SEED_ENTRY_SEATS); 0 = same as SEED_ENTRY_SEATS
EXPAND_FETCH_PER_ENTRY = 3  # orchestrator/storage · expanded channels taken per primary entry per refresh (cursor rotation)
EXPAND_FETCH_CHANNELS_MAX = 6  # orchestrator · max expanded channels per refresh
EXPAND_FETCH_PER_CHANNEL = 3  # orchestrator/sources · latest videos fetched per expanded channel
SEED_ENTRY_SEATS = 2  # ranker · reserved seed-channel seats per profile entry, counted within its diversity seats; 0 = off

# costing/sources · unit cost per tool; validate_plan recomputes plan cost from this table instead of trusting the LLM
QUOTA_COST = {
    "fetch_channel_videos": 1,       # playlistItems.list
    "search_by_keyword": 1,          # search.list (separate search bucket, 1 per call)
    "fetch_trending_by_category": 1, # videos.list?chart=mostPopular
    "mine_hidden_gems": 1,           # playlistItems.list, per sampled channel
    "get_video_stats": 1,            # videos.list, 50 ids per call
    "channels_list": 1,              # channels.list, 50 ids per call
    "search_channels": 1,            # search.list type=channel (offline pool job) → search bucket
    "search_video_channels": 1,      # search.list type=video aggregated by channel (seed suggestions) → search bucket
    "channel_sections": 1,           # channelSections.list
    "playlists_list": 1,             # playlists.list (channel page playlists tab)
    "video_categories": 1,           # videoCategories.list
}




PLAN_QUOTA_BUDGET_RATIO = 0.5  # planner · a plan's general-pool cost may not exceed this share of remaining quota (search bucket gated separately)

# ---- Hidden gems (small-channel discovery; uses the general pool, not the search bucket) ----
SMALL_POOL_SAMPLE_K = 6  # planner/costing · small channels sampled per refresh
SMALL_CHANNEL_SUBS_MAX = 200_000  # ranker · "small channel" = fewer subscribers than this

GEM_PLAY_FACTOR_SWEET = 1.0  # ranker · view count inside the sweet spot: full factor
GEM_PLAY_FACTOR_LOW = 0.7  # ranker · below the sweet spot: unstable data, discounted
GEM_PLAY_FACTOR_HIGH = 0.5  # ranker · above the sweet spot: already promoted by the algorithm, heavily discounted
GEM_FAN_BONUS = 1.2  # ranker · multiplier for channels below SMALL_CHANNEL_SUBS_MAX
GEM_MEDIAN_MIN_N = 4  # ranker · median filtering of the gem pool only with at least this many items



GEM_VIEW_SWEET_MIN = 2_000  # ranker · view-count sweet spot lower bound
GEM_VIEW_SWEET_MAX = 30_000  # ranker · view-count sweet spot upper bound

# ---- S1 plan structure ----
PLAN_MIN_STRATEGIES = 2  # planner · minimum strategies per plan
PLAN_MAX_STRATEGIES = 6  # planner · maximum strategies per plan

# ---- LLM models ----
LLM_MODEL = "claude-haiku-4-5"  # agent/* · default model
LLM_MODEL_FALLBACK = "claude-sonnet-4-5"  # judges · stronger model used for escalation / harder modes

# ---- Agent layer ----
PLANNER_REPLAN_MAX = 1  # planner · re-plans allowed after a validation failure (with the reason attached)
LLM_MAX_SCHEMA_RETRIES = 1  # agent/* · PydanticAI structured-output retries
LLM_TIMEOUT_SEC = 60  # agent/* · per-call LLM timeout
COMPRESS_DESC_CHARS = 200  # judges · description truncation when compressing videos for the prompt
COMPRESS_TAGS_MAX = 8  # judges · tag truncation when compressing videos for the prompt
MODE_LLM_OVERRIDES: dict[str, str] = {}          # judges/planner · per-mode model override, e.g. {"wind_down": LLM_MODEL_FALLBACK}

# ---- Judge concurrency ----
LLM_MAX_CONCURRENCY = 4  # judges · global cap on concurrent judge LLM calls
FIRST_WAVE_MAX = 45  # orchestrator · first-wave size: top N by statistical score, the rest deferred to later waves (~1.5 × TOP_N)

# ---- S2 retrieval ----
S2_OVERFETCH_FACTOR = 2.0  # sources/costing · over-fetch ceil(target × factor) to leave room for the post-enrichment duration filter; intent strategies are not over-fetched; 1.0 = off
CHANNEL_DEEP_PAGES_MAX = 3   # sources/costing · max uploads pages per channel per refresh (incl. the first); 1 = no deep paging

# ---- S6 fit cache and local candidate library ----
FIT_CACHE_TTL_DAYS = 7      # orchestrator/storage · lifetime of cached S6 fit verdicts; 0 = cache off
LIBRARY_PER_REFRESH = 30    # orchestrator/sources · max zero-API library candidates injected per refresh; 0 = off
LIBRARY_TOP_K = 10          # orchestrator · library draws from the top K preferred channels (seed channels unlimited)

# ---- Planning state machine (steady vs. active) ----
ENABLE_ACTION_SIGNALS = True  # stability · use implicit action signals; False = explicit feedback only
SIGNAL_WINDOW_DAYS = 14  # prefs/stability · window for signal count and dislike rate
STABLE_MIN_SIGNALS = 10  # prefs/stability · minimum signals in the window to enter steady state
K_WINDOW = 5  # stability · refresh window for pass-rate convergence
PASS_RATE_STD_MAX = 0.15  # stability · pass-rate std above this = not converged → active
DISLIKE_CEIL = 0.30  # stability · dislike-rate ceiling
PLAY_DROP_K = 3  # stability · consecutive refreshes below the play-rate floor that trigger fallback
PLAY_RATE_FLOOR = 0.05  # stability · play-rate floor (plays / shown)
PLAN_REUSE_TTL_HOURS = 24  # stability · how long a steady-state plan may be reused

# ---- Card pool (instant first screen) ----
POOL_STALE_HOURS = 6  # prewarm · pooled cards older than this are marked stale (affects UI text only)

# ---- Impression penalty ----
IMPRESSION_TAIL_SEATS = 10  # orchestrator · penalised cards may only fill free seats among the last N of the top section, least-shown first
IMPRESSION_TAIL_FIXED_SEATS = 3  # orchestrator · tail seats always yielded to high-quality / seed-channel penalised cards; 0 = free seats only
TAIL_HIGH_QUALITY_MIN = 0.8  # orchestrator/storage · quality threshold for a fixed tail seat (seed channels qualify regardless)
IMPRESSION_RANK_DECAY = 0.8  # ranker · final *= decay ** recent impressions, so already-shown cards stop crowding out new ones; intent cards exempt
# neg_prefs · preset dislike reasons: key = value sent by the frontend, value = text passed to the judge
DISLIKE_CHIP_LABELS = {
    "channel_too_often": "这个频道推太多了",
    "channel_dislike": "不喜欢这个频道的内容",
    "topic_uninterested": "对这个题材不感兴趣",
    "low_quality": "内容质量差",
    "mode_mismatch": "不想在此模式看这类",
    "other": "其他",
}
# judges · fixed reason prefix for S6 negative-preference rejections (shown to users, counted exactly in reports)
NEG_PREF_HIT_PREFIX = {
    "zh": "不喜欢命中：",
    "en": "Matched your dislike: ",
}

# ---- Source mix ----
POOL_RATIO_NEW = 0.8  # prefs · type-pool share of supply for new users
POOL_RATIO_MATURE = 0.5  # prefs · type-pool share of supply for mature users

SOURCE_MIX_FLOOR = 0.35  # mix · soft floor: swap in cards when one side's share of channel cards on screen falls below this
SOURCE_MAX_SHARE = 0.5  # mix · cap on any single source_strategy's on-screen share (swap only; intent cards not counted); 0 = off

# ---- Followed channels ----
FOLLOW_ENTRY_SEATS_ON = True  # ranker · within an entry's seats, followed channels go after seed channels and before others
FOLLOW_SEAT_MIN_PREF = 0.6  # ranker · minimum preference score for a followed channel to get that priority; 0 = no threshold

# ---- Cold-start chip shelf and prewarm ----
CHIP_CANDIDATE_TTL_DAYS = 90  # prewarm/storage · expiry for cached chip candidates
PREWARM_MIN_CREDITS_LEFT = 2  # prewarm · only prewarm when at least this many refreshes remain today

# ---- Persistent profile ----
ENABLE_USER_PROFILE = True  # planner · master switch
PROFILE_LIST_MAX = 8  # models/prompts · max items per UserProfile list field
PROFILE_NOTE_MAX = 300  # models/planner · truncation of free_note / degraded raw summary

# ---- Session intent and chat ----
ENABLE_SESSION_INTENT = True  # planner/orchestrator · master switch
PROFILE_MODEL: str | None = "claude-sonnet-5"     # profiler · model (None = LLM_MODEL)

CHAT_MODEL: str | None = "claude-sonnet-5"  # chat · model (None = LLM_MODEL)

# ---- Trial account ----
TRIAL_SEARCH_FLOOR = 40         # orchestrator · below this many remaining site-wide search calls, trial-account refreshes skip all search paths

# ---- Chat-derived profile sections ----
CHAT_SHORT_MAX = 8  # profile_sections · max short-term entries (FIFO)
CHAT_LONG_MAX = 12  # profile_sections · max long-term entries per mode
CHAT_SHORT_TTL_DAYS = 30  # profile_sections · short-term entry lifetime (expired lazily on write)


# ================================================================== #
# 2. The four modes
# ================================================================== #

FOCUS = Mode(
    name="focus",
    display_name="专注",

    # 10=Music, 24=Entertainment (documentary-leaning; the LLM judge filters), 27=Education, 28=Science & Technology
    yt_category_ids=[10, 24, 27, 28],
    keywords=[
        "white noise", "ambient", "lo-fi study", "nature sounds", "documentary",
        "白噪音", "纯音乐", "纪录片",
        # work-oriented terms
        "AI tools", "coding tutorial", "productivity", "工作学习",
    ],
    blacklist_keywords=[
        "reaction", "reactions", "drama", "dramas", "shocking", "breaking news",
        "吐槽", "爆笑", "震惊",
    ],
    duration_min_sec=180,
    duration_max_sec=None,  # no upper bound
    region_code="US",
    relevance_language="zh-Hans",  # only applied to queries containing CJK
    enable_discovery=False,
    strict_mode=False,
)

ENTERTAINMENT = Mode(
    name="entertainment",
    display_name="娱乐",  # 23=Comedy, 24=Entertainment, 1=Film & Animation, 19=Travel, 15=Pets
    yt_category_ids=[23, 24, 1, 19, 15],
    keywords=[
        "funny", "comedy", "fun fact", "amazing", "travel", "vlog",
        "搞笑", "沙雕", "有趣", "趣味", "解说", "探店", "纪录片",
    ],
    blacklist_keywords=[  # core of this mode: no study/work/self-improvement content; inflections listed explicitly
        "tutorial", "tutorials", "course", "courses", "lecture", "lectures",
        "study", "studying", "how to learn",
        "productivity", "career", "careers", "job", "jobs", "interview prep",
        "教程", "课程", "网课", "考研", "学习方法",
        "职场", "求职", "面试", "效率", "工作汇报", "复习",
        # shared low-quality terms
        "shocking", "breaking news", "震惊", "塌房",
    ],
    duration_min_sec=10,
    duration_max_sec=1800,
    region_code="US",
    relevance_language="zh-Hans",  # only applied to queries containing CJK
    enable_discovery=True,  # hidden gems give niche fun content exposure
    strict_mode=False,
)

EXPLORE = Mode(
    name="explore",
    display_name="探索",  # 28=Science & Technology, 22=People & Blogs (popular science)
    yt_category_ids=[28, 22],
    keywords=[
        "explained", "why does", "how it works", "science",
        "冷知识", "科普", "原理",
    ],
    blacklist_keywords=[  # work anxiety
        "job", "jobs", "career", "careers", "layoff", "layoffs",
        "hustle", "get rich",
        "职场", "副业", "月入", "内卷", "35岁",
        # social conflict / gossip
        "divorce", "divorced", "cheating", "drama", "dramas", "撕", "塌房", "相亲",
    ],
    duration_min_sec=180,  # informative but not fragmentary
    duration_max_sec=10800,  # 3 hours
    region_code="US",
    relevance_language="zh-Hans",  # only applied to queries containing CJK
    enable_discovery=True,  # hidden-gem mining enabled
    strict_mode=False,
)

WIND_DOWN = Mode(
    name="wind_down",
    display_name="睡前",  # 22=People & Blogs, 15=Pets & Animals, 26=Howto & Style (cooking)
    yt_category_ids=[22, 15, 26],
    keywords=[
        "ASMR", "sleep", "calm", "slow living", "cooking",
        "治愈", "慢生活", "助眠",
    ],
    blacklist_keywords=[  # anything anxious, stimulating, conflict-driven or information-dense; `vs` is whole-word matched
        "shocking", "breaking news", "horror", "scary", "debate", "vs",
        "震惊", "恐怖", "惊悚", "吵架", "冲突", "爆肝", "干货", "考试", "面试",
    ],
    duration_min_sec=300,
    duration_max_sec=None,  # no upper bound
    region_code="US",
    relevance_language="zh-Hans",  # only applied to queries containing CJK
    enable_discovery=False,
    strict_mode=True,  # the only fail-closed mode: better to show fewer than show poor fits
)

MODES: dict[str, Mode] = {m.name: m for m in (FOCUS, ENTERTAINMENT, EXPLORE, WIND_DOWN)}  # all · mode name → Mode instance


# ================================================================== #
# 3. DEFAULT_PLANS: static fallback plans for S1
# ================================================================== #

# planner/costing · per-mode static plan used when LLM planning fails; params are interpreted by the sources layer
DEFAULT_PLANS: dict[str, RetrievalPlan] = {
    "focus": RetrievalPlan(
        mode="focus",
        strategies=[
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_top_channels": True, "n_channels": 10},
                         target_count=120, rationale="静态计划:专注频道列表为主"),
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_type_pool": True, "n_channels": 8},
                         target_count=40,
                         rationale="静态计划:类型池直抓(CS-8;新用户无偏好频道时的主力补给)"),
            StrategyCall(tool="search_by_keyword",
                         params={"query": "白噪音 纯音乐"},
                         target_count=40, rationale="静态计划:固定关键词补充"),
            StrategyCall(tool="search_by_keyword",
                         params={"query": "AI tools productivity"},
                         target_count=40, rationale="静态计划:工作向补充(#13(a),focus 新语义 work/AI 为正例;英文 query 不带语言偏置)"),
            StrategyCall(tool="fetch_trending_by_category",
                         params={"category_id": 10},
                         target_count=30, rationale="静态计划:音乐分类热门(普通单曲<10min 会被时长规则滤掉,配比压低)"),
        ],
        est_quota=66,  # informational: channel strategies priced at the 3-page worst case
    ),
    "entertainment": RetrievalPlan(
        mode="entertainment",
        strategies=[
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_top_channels": True, "n_channels": 10},
                         target_count=150, rationale="静态计划:搞笑/休闲/趣味频道为主"),
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_type_pool": True, "n_channels": 8},
                         target_count=40,
                         rationale="静态计划:类型池直抓(CS-8;新用户无偏好频道时的主力补给)"),
            StrategyCall(tool="fetch_trending_by_category",
                         params={"category_id": 23},
                         target_count=40, rationale="静态计划:Comedy 分类热门"),
            StrategyCall(tool="mine_hidden_gems",
                         params={"k": SMALL_POOL_SAMPLE_K},
                         target_count=40, rationale="静态计划:冷门趣味内容挖掘"),
        ],
        est_quota=72,  # informational: channel strategies priced at the 3-page worst case
    ),
    "explore": RetrievalPlan(
        mode="explore",
        strategies=[  # typical explore mix: ~60% channels + ~20% search + ~15% hidden gems
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_top_channels": True, "n_channels": 10},
                         target_count=150, rationale="静态计划:订阅科普频道 60%"),
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_type_pool": True, "n_channels": 8},
                         target_count=40,
                         rationale="静态计划:类型池直抓(CS-8;新用户无偏好频道时的主力补给)"),
            StrategyCall(tool="search_by_keyword",
                         params={"query": "科普 原理"},
                         target_count=50, rationale="静态计划:固定关键词(单页上限 50)"),
            StrategyCall(tool="mine_hidden_gems",
                         params={"k": SMALL_POOL_SAMPLE_K},
                         target_count=40, rationale="静态计划:小众宝藏挖掘"),
        ],
        est_quota=72,  # informational: channel strategies priced at the 3-page worst case
    ),
    "wind_down": RetrievalPlan(
        mode="wind_down",
        strategies=[
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_top_channels": True, "n_channels": 10},
                         target_count=100, rationale="静态计划:治愈系频道为主"),
            StrategyCall(tool="fetch_channel_videos",
                         params={"use_type_pool": True, "n_channels": 8},
                         target_count=40,
                         rationale="静态计划:类型池直抓(CS-8;新用户无偏好频道时的主力补给)"),
            StrategyCall(tool="fetch_trending_by_category",
                         params={"category_id": 26},
                         target_count=40, rationale="静态计划:Howto&Style 分类热门(烹饪/手作;分类 15 无热门榜)"),
            StrategyCall(tool="search_by_keyword",
                         params={"query": "治愈 慢生活"},
                         target_count=40, rationale="静态计划:固定关键词补充"),
        ],
        est_quota=64,  # informational: channel strategies priced at the 3-page worst case
    ),
}

# ---- User-facing fixed texts ----
# sse · fixed text of the error event when a refresh fails midway (by language)
PIPELINE_ERROR_TEXT = {
    "zh": "刷新中途异常,本次结果不完整,请稍后重试",
    "en": "The refresh hit an error midway — results are incomplete, please retry shortly",
}
# orchestrator · recommendation reason for cards exempted after an accepted appeal
APPEAL_EXEMPT_REASON = {
    "zh": "申诉通过,已放行",
    "en": "Appeal accepted — unblocked",
}


# ---- Vector search: embedding ----
EMBED_DESC_CHARS = 300                 # embed_text · description truncation for video embeddings
EMBED_CHANNEL_TITLES_MAX = 20          # embed_text · max recent upload titles in a channel embedding
RRF_K = 60                             # storage (hybrid_search) · RRF constant; English-only queries use K/2 on the FTS5 side

# ---- Vector search: local intent recall ----
INTENT_LOCAL_RECALL = 1        # orchestrator · query the local FTS index before the intent search bundle; 0 = off
INTENT_LOCAL_K = 45            # orchestrator/storage · max local recall results (INTENT_LITE_TOP × 3, leaving room for filtering)
INTENT_LITE_QUERIES_MIN = 1    # orchestrator · when local recall already yields ≥ INTENT_LITE_TOP, shrink to N queries (the verbatim intent is still searched)

# ---- Vector search: first-wave ordering ----
E7_VECTOR = 1                  # orchestrator · S6 first-wave order = RRF of statistical rank and profile-entry similarity rank; 0 = statistical only

# ---- Profile-entry sub-topics ----
PROFILE_SUBTOPICS = 1          # orchestrator · master switch: 0 = no derivation, rotation and expansion use only the entry's main query
PROFILE_SUBTOPIC_SEARCH = 1    # subtopics · 1 = rotate searches through sub-topics; 0 = derivation and vectors continue but searches use the main query (restart required)
ENTRY_SUBTOPICS_MIN = 8        # profiler · requested minimum sub-topics per entry (prompt only, not enforced)
ENTRY_SUBTOPICS_MAX = 12       # profiler · maximum sub-topics per entry (excess truncated)
PROFILE_SUBTOPIC_EN_EVERY = 0  # diversity · N > 0 = every Nth rotation of an entry searches the English sub-topic; 0 = entry language only
ES_VECTOR = 1                  # storage/sources/orchestrator · include sub-topic vectors in multi-vector retrieval and pool recall; 0 = entry vectors only

# ---- Vector search: per-entry candidate ranking ----
VEC_SUBTOPIC_WEIGHT = 0.85     # storage · per-entry score = max(entry sim, W × sub-topic sim); 1.0 = equal weight


# ---- Data sources, channel discovery and retrieval ----
TYPE_POOL_SUBS_MIN = 200_000  # discovery/channel-pool job · type-pool hard floor on subscribers
TYPE_POOL_TARGET_MAX = 60  # discovery/channel-pool job · max channels per chip in the type pool
TYPE_POOL_QUERIES_PER_CHIP = 3  # discovery · LLM search queries per chip per round
TYPE_POOL_SUBS_PREFERRED_MIN = 500_000  # discovery · soft preferred subscriber tier; used only in reports and query hints, not in the hard gate
CHIP_HARD_TO_EXPAND_ROUNDS = 2  # channel-pool job · a chip whose last N rounds added nothing gets bilingual queries
CHIP_SUBTOPICS_MIN = 8        # discovery · requested minimum sub-topics per chip (prompt only, not enforced)
CHIP_SUBTOPICS_MAX = 12       # discovery · maximum sub-topics per chip (excess truncated)
EXPAND_SAMPLE_TITLES = 3  # youtube · sample video titles kept per channel when aggregating video search results (logging/debug only)
QUOTA_BUCKET = {  # youtube/costing · tool name -> quota bucket; unlisted tools use "general"
    "search_by_keyword": "search",
    "search_channels": "search",
    "search_video_channels": "search",
}
def quota_bucket(tool: str) -> str:
    """Tool/endpoint name -> quota bucket ("general" or "search"); "general" is the default."""
    return QUOTA_BUCKET.get(tool, "general")
SOURCE_PRIORITY = {"seed_channel": -1, "subscribed": 0, "expanded_channel": 1,  # youtube · which source wins when dedup sees the same video_id (lower wins)
                   "hidden_gem": 2, "channel_pool": 3, "search": 4, "trending": 5}
DISCOVERY_MIN_SUBS = 10_000                  # channel-pool job · hard subscriber floor for discovered channels (matches SMALL_CHANNEL_SUBS_MIN)
DISCOVERY_QUERIES_MAX = 5                    # discovery · max search queries from the user-track query generator
DB_PATH_ENV = "FOCUSFLOW_DB_PATH"  # db · environment variable holding the database path
DEFAULT_USER_ID = 1  # sources/offline jobs · default user_id parameter for scripts and importers
CHANNEL_DEEP_RESET_DAYS = 30  # youtube/db · after paging to the end of a channel, days before paging restarts from the top
CHANNEL_PAGE_SIZE = 50        # youtube · playlistItems page size (API maximum; same cost, more candidates)
IMPRESSION_SUPPRESS_TTL_HOURS = 72  # db · impression-penalty window (distinct from POOL_TTL)
FOLLOW_PREF_BETA = 8.0  # youtube · followed-channel draw weight exp(β·(pref−0.5)): 0.6→2.2, 0.8→11, 0.9→25, 1.0→55; higher preference is drawn more often
FOLLOW_PICK_DECAY = 0.85  # youtube · weight × 0.85 per draw within the same day, so lower-scored channels get a turn; resets daily
FOLLOW_ENTRY_AFFINITY = 2.0  # youtube · weight × 2 for channels recently judged to belong to this refresh's primary profile entry
FOLLOW_POOL_MIN_PREF = 0.5  # db/youtube · floor for the followed-channel draw pool: channels below this (disliked) are excluded; 0 = off
FOLLOW_POOL_LOVED_PREF = 0.75  # db/youtube · unfollowed channels need at least this preference to enter the draw pool; 0 = off
EMBED_MODEL = "BAAI/bge-small-zh-v1.5" # embeddings · embedding model; changing it triggers a full re-embed
EMBED_DIM = 512                        # db · vector dimension, bound to EMBED_MODEL; fixed once the table exists
EMBED_MODEL_DIR = "models"             # embeddings · model cache directory (relative to project root; holds no user data)
VECTOR_SIM_MIN = 0.55                  # storage (hybrid_search) · cosine similarity threshold for a vector hit
E1_VECTOR = 1                  # youtube/db · rerank the candidate library by profile-entry vectors; 0 = default order
E1_LIBRARY_OVERSAMPLE = 4      # youtube · oversampling factor before vector reranking (take limit × N, keep top limit)
E1_ENTRY_ISOLATION = 1         # youtube/db · rerank per profile entry in tier order (primary, drawn secondary, rest, unmatched); 0 = summed multi-vector RRF


# ---- Retrieval delivery caps and vector switch ----
S2_DELIVER_MAX = 90           # youtube · cap on total non-search, non-intent items delivered by S2 (scaled per source ratio); 0 = off
S2_PER_CHANNEL_MAX = 8        # youtube · max items kept per channel before delivery (newest uploads first); 0 = off
VECTOR_SEARCH = 1             # retrieval · master switch: 0 = do not load sqlite-vec, skip vec tables, all callers fall back to non-vector paths
