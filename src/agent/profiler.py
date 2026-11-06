"""Profile structuring and profile-entry helpers (agent layer).

Turns a user's natural-language self-description into a structured
``UserProfile``, and provides the small LLM helpers used when users edit
their per-mode profile entries: merging long-term signals, translating
entries into search terms and content types, ranking followed channels,
deriving sub-topics, and judging candidate channels' fit.

This module is called only from the application/API layer (profile submit and
edit), never imported by any pipeline stage, so building profiles adds zero
LLM calls to a refresh.

Design points:

- No database access: like the judges, this module only calls the LLM;
  persistence belongs to the caller. The agent layer's only db edge stays the
  read-only one in ``planner.build_plan_context``.
- Errors propagate (the opposite of the planner). LLM exceptions are raised
  as-is; the caller always stores the user's raw text and sets the structured
  profile to NULL, and the consumer then falls back to a truncated free note.
  Only the caller can perform that fallback write, and swallowing the error
  would make it believe structuring had succeeded.
- Only what the user explicitly states is captured. The profile prompt
  forbids inferring undeclared attributes such as age, gender or occupation:
  such guesses are noisy proxies that would inject noise into planning.
- Model tier: ``PROFILE_MODEL`` (falls back to ``LLM_MODEL`` when None) for
  profile structuring, consolidation and entry translation; the cheaper
  ``LLM_MODEL`` for high-frequency helpers.
"""
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

from agent import _model_id, _model_settings
from agent.prompts import PROMPTS
from config import LLM_MAX_SCHEMA_RETRIES, LLM_MODEL, LLM_TIMEOUT_SEC
from core.models import UserProfile

_agent: Agent | None = None


def _get_agent() -> Agent:
    """Lazily build the profile Agent singleton.

    Not built at import time, so offline tests run without an API key. A
    missing key raises on first construction and propagates to the caller,
    consistent with ``build_profile``'s error policy.
    """
    global _agent
    if _agent is None:
        import config
        _agent = Agent(
            _model_id(config.PROFILE_MODEL or LLM_MODEL),
            output_type=UserProfile,
            system_prompt=PROMPTS["profile"].system,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(config.PROFILE_MODEL or LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _agent


async def build_profile(raw_text: str) -> UserProfile:
    """Structure the user's raw self-description into a ``UserProfile``.

    Over-length fields are truncated by the ``UserProfile`` validator (with a
    warning, not an error). LLM errors (missing key, network, timeout,
    exhausted schema retries) are not caught here.
    """
    result = await _get_agent().run(
        f"用户的自然语言声明如下,请结构化进 UserProfile:\n{raw_text}")
    return result.output


# ---- Long-term profile consolidation ---- #
class LongConsolidation(BaseModel):
    """Result of de-duplicating a new long-term signal.

    ``matched_indices``: 0-based indices of existing entries that mean the same
    kind of preference as the new signal. ``merged_text``: a single phrase
    covering those entries plus the new signal. Both empty when nothing matches.
    """
    matched_indices: list[int] = []
    merged_text: str | None = None


# The system prompt lives here rather than in agent/prompts.py, whose default
# prompts are kept byte-for-byte stable.
_CONSOLIDATE_SYSTEM = (
    "你是画像长期条目的归并判官。输入是一条新的用户偏好信号和现有长期条目列表"
    "(带序号,0 起)。判断新信号与哪些现有条目在语义上指同一类观看偏好——如"
    "「想看 AI 类视频」与「想看深度学习类视频」同类;把同类条目的序号填进"
    " matched_indices,并把它们与新信号统称成一条简洁偏好(不超过 50 字)写进"
    " merged_text。规则:宁缺毋滥,只有明确同类才归并,主题相近但指向不同的不算;"
    "喜好与回避绝不归并(「喜欢 AI 视频」与「不想看 AI 视频」是两回事);统称只做"
    "归纳不做泛化,不得把具体主题抬到更高层类目;没有同类时两个字段都留空。"
)

_consolidate_agent: Agent | None = None


def _get_consolidate_agent() -> Agent:
    """Lazily build the consolidation Agent (same tier as ``PROFILE_MODEL``)."""
    global _consolidate_agent
    if _consolidate_agent is None:
        import config
        _consolidate_agent = Agent(
            _model_id(config.PROFILE_MODEL or LLM_MODEL),
            output_type=LongConsolidation,
            system_prompt=_CONSOLIDATE_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(config.PROFILE_MODEL or LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _consolidate_agent


async def consolidate_long_signal(new_signal: str,
                                  existing: list[str]) -> LongConsolidation:
    """Ask which existing long-term entries are synonymous with ``new_signal``.

    Used when a new long-term preference is extracted (e.g. from chat) so
    synonymous entries can be merged into one. LLM errors propagate; the
    caller falls back to appending the signal as a new entry and is also
    responsible for validating out-of-range indices.
    """
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(existing))
    result = await _get_consolidate_agent().run(
        f"[新信号]\n{new_signal}\n\n[现有长期条目]\n{numbered}")
    return result.output


# ---- Profile-entry translation ---- #
class EntryDerivation(BaseModel):
    """Structured derivation of one user-written profile entry.

    All fields are empty when the entry cannot be translated.
    """
    base_types: list[str] = []           # up to 2 base content types (17-type vocabulary), by relevance
    search_query: str | None = None      # one representative YouTube search query, in the entry's language
    chips: list[str] = []                # fine-grained interest slugs (<= ENTRY_CHIPS_MAX); empty when unsure
    search_query_en: str | None = None   # the same preference as an English query
    text_en: str | None = None           # English translation of the entry (for the English UI)


# Prompt revision history is tracked outside the code.
_ENTRIES_SYSTEM = (
    "你是 FocusFlow 的画像条目翻译器。输入是某观看模式下用户亲笔声明的偏好"
    "条目列表(带序号,0 起)、17 类内容类型词表与 44 项细分兴趣词表(slug=显示名)。"
    "对每条输出:①base_types——从类型词表中选最贴合的 1~2 个类型(相关度降序,"
    "词表外的字符串一律不写);②search_query——一条能在 YouTube 搜到该偏好内容的"
    "具体搜索词,与条目原文同语言,宁具体勿宽泛(如「中国古风音乐 纯音乐」优于"
    "「音乐」);③search_query_en——同一偏好的**英文**搜索词(纯英文,不得含汉字;"
    "原文已是英文则与 search_query 相同),同样宁具体勿宽泛;④chips——从细分词表中"
    "选 0~3 个最贴合的 slug(只写 slug,相关度降序,词表外不写,拿不准宁空勿凑);"
    "⑤text_en——条目原文的英文译文(纯英文,忠实直译、不扩写不解释、保留原文的限定词;"
    "原文已是英文则逐字照抄)。"
    "输出条数与输入条数一致、按序对齐;某条实在无法翻译时该条五字段留空。"
)

_entries_agent: Agent | None = None


def _get_entries_agent() -> Agent:
    """Lazily build the entry-translation Agent (same tier as ``PROFILE_MODEL``)."""
    global _entries_agent
    if _entries_agent is None:
        import config
        _entries_agent = Agent(
            _model_id(config.PROFILE_MODEL or LLM_MODEL),
            output_type=list[EntryDerivation],
            system_prompt=_ENTRIES_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(config.PROFILE_MODEL or LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _entries_agent


def _has_cjk(text: str) -> bool:
    """True if the text contains at least one CJK unified ideograph."""
    return any("一" <= ch <= "鿿" for ch in text)


def _sanitize_derivations(items: list[EntryDerivation],
                          n_texts: int) -> list[EntryDerivation]:
    """Clean LLM output for entry translation (pure function).

    Drops base types outside the vocabulary and keeps at most 2 (order
    preserved); normalises blank queries to None; voids English fields that
    contain CJK characters; keeps only known, unique chips. The result is
    aligned to the input: extra items are truncated and missing ones are
    padded with empty derivations, so ``out[i]`` always matches entry ``i``.
    """
    import config
    out: list[EntryDerivation] = []
    for it in items[:n_texts]:
        bases = [b for b in it.base_types if b in config.CONTENT_TYPES][:2]

        # None, "" and whitespace-only all collapse to None.
        q = (it.search_query or "").strip() or None
        q_en = (it.search_query_en or "").strip() or None
        if q_en and _has_cjk(q_en):
            q_en = None
        t_en = " ".join((it.text_en or "").split()) or None
        if t_en and _has_cjk(t_en):
            t_en = None                          # the UI falls back to the original text
        chips: list[str] = []
        for c in it.chips or []:
            if c in config.INTEREST_CHIPS and c not in chips:
                chips.append(c)
        out.append(EntryDerivation(base_types=bases, search_query=q,
                                   chips=chips[:config.ENTRY_CHIPS_MAX],
                                   search_query_en=q_en, text_en=t_en))
    while len(out) < n_texts:
        out.append(EntryDerivation())
    return out


# ---- Followed-channel ranking ---- #
class FollowedRank(BaseModel):
    """LLM output: 0-based candidate indices, most relevant first (top N only).

    The LLM returns indices rather than channel names, so results map back to
    the input list unambiguously.
    """
    order: list[int] = []


_FOLLOWED_RANK_SYSTEM = (
    "你是 FocusFlow 的频道相关性排序器。输入是一条用户亲笔的偏好条目(含派生的类型/细分"
    "标签与搜索词)和用户已关注的 YouTube 频道列表(带序号,0 起;可能附频道的细分标签)。"
    "任务:按「该频道的内容与这条偏好的相关程度」降序输出序号,取前 N 个(候选不足 N 时"
    "全部输出,**不得因为不相关而省略**——这是排序不是筛选)。判断依据是频道名与标签所指的"
    "内容领域:同领域最优先(如偏好金融→财经/商业频道),其次相邻领域(金融→新闻时政、"
    "科技商业),再次其余泛知识类,明显无关的(如偏好金融时的美妆、游戏)排最后。"
    "只输出序号,不解释;序号不得越界、不得重复。"
)

_followed_rank_agent: Agent | None = None


def _get_followed_rank_agent() -> Agent:
    global _followed_rank_agent
    if _followed_rank_agent is None:
        _followed_rank_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=FollowedRank,
            system_prompt=_FOLLOWED_RANK_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _followed_rank_agent


def _sanitize_rank(order: list[int], n_items: int, top_n: int) -> list[int]:
    """Clean LLM rank output (pure function).

    Drops out-of-range, duplicate and non-integer indices, preserves order,
    and truncates to ``top_n``.
    """
    out: list[int] = []
    for i in order:
        if isinstance(i, int) and 0 <= i < n_items and i not in out:
            out.append(i)
        if len(out) >= top_n:
            break
    return out


async def rank_followed_channels(entry: dict, channels: list[dict], top_n: int) -> list[int]:
    """Rank the user's followed channels by relevance to one profile entry.

    Returns at most ``top_n`` indices into ``channels``, most relevant first.
    ``entry`` = {"text", "search_query", "base_types", "chips"};
    ``channels`` = [{"title", "chips"}]. LLM errors propagate; the caller
    falls back to chip/base-type matching.
    """
    lines = []
    for i, c in enumerate(channels):
        tags = f"(标签:{', '.join(c['chips'])})" if c.get("chips") else ""
        lines.append(f"{i}. {c.get('title') or '?'} {tags}".rstrip())
    prompt = (f"[偏好条目] {entry.get('text')}\n"
              f"[类型] {', '.join(entry.get('base_types') or [])} [细分] {', '.join(entry.get('chips') or [])}"
              f" [搜索词] {entry.get('search_query') or ''}\n"
              f"[N] {top_n}\n[关注频道]\n" + "\n".join(lines))
    result = await _get_followed_rank_agent().run(prompt)
    return _sanitize_rank(list(result.output.order), len(channels), top_n)


async def translate_mode_entries(mode: str,
                                 texts: list[str]) -> list[EntryDerivation]:
    """Translate one mode's profile entries (up to 6) in a single LLM call.

    Entries are numbered from 0 so the LLM returns an equal-length list
    aligned by position; output is cleaned by ``_sanitize_derivations``.
    LLM errors propagate; the caller stores NULL derivations.
    """
    import config
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts))

    # Vocabulary rendered as "slug=display name" pairs.
    chips_vocab = ", ".join(f"{slug}={disp}" for slug, (disp, _) in config.INTEREST_CHIPS.items())
    result = await _get_entries_agent().run(
        f"[观看模式] {mode}\n[类型词表] {', '.join(config.CONTENT_TYPES)}\n"
        f"[细分词表] {chips_vocab}\n"
        f"[用户条目]\n{numbered}")
    return _sanitize_derivations(result.output, len(texts))


# ---- Profile-entry sub-topic derivation ---- #
class SubtopicItem(BaseModel):
    """One sub-topic: name, query in the entry's language, English query."""
    subtopic: str
    query_zh: str
    query_en: str = ""


class EntrySubtopicSet(BaseModel):
    """LLM output: 8–12 sub-topics (the minimum is only a prompt request; extras are truncated)."""
    items: list[SubtopicItem]


_ENTRY_SUBTOPICS_SYSTEM = PROMPTS["entry_subtopics"].system
_entry_subtopics_agent: Agent | None = None


def _get_entry_subtopics_agent() -> Agent:
    """Lazily build the sub-topic Agent on ``LLM_MODEL`` (cost first: runs on every entry save)."""
    global _entry_subtopics_agent
    if _entry_subtopics_agent is None:
        _entry_subtopics_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=EntrySubtopicSet,
            system_prompt=_ENTRY_SUBTOPICS_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _entry_subtopics_agent


def _sanitize_subtopics(items: list[SubtopicItem]) -> list[dict]:
    """Clean sub-topic output (pure function).

    Drops items with a blank name or query; de-duplicates by whitespace-
    normalised ``query_zh`` preserving order; blanks ``query_en`` if it
    contains CJK; truncates to ``ENTRY_SUBTOPICS_MAX``. Returns dicts ready
    for the storage layer.
    """
    import config
    out: list[dict] = []
    seen: set[str] = set()
    for it in items:
        name = (it.subtopic or "").strip()
        zh = " ".join((it.query_zh or "").split())
        if not name or not zh or zh in seen:
            continue
        en = " ".join((it.query_en or "").split())
        if en and _has_cjk(en):
            en = ""
        seen.add(zh)
        out.append({"subtopic": name, "query_zh": zh, "query_en": en})
        if len(out) >= config.ENTRY_SUBTOPICS_MAX:
            break
    return out


def build_subtopics_prompt(mode: str, text: str, base_types: list[str] | None,
                           chips: list[str] | None, search_query: str | None,
                           search_query_en: str | None) -> str:
    """Build the sub-topic user message (pure function, testable offline)."""
    import config
    return (f"[观看模式] {mode}\n[偏好条目] {text}\n"
            f"[类型] {', '.join(base_types or []) or '无'} [细分] {', '.join(chips or []) or '无'}\n"
            f"[代表搜索词] {search_query or '无'} / {search_query_en or '无'}\n"
            f"请给出 {config.ENTRY_SUBTOPICS_MIN}~{config.ENTRY_SUBTOPICS_MAX} 个子方向。")


async def derive_entry_subtopics(mode: str, text: str, base_types: list[str] | None,
                                 chips: list[str] | None, search_query: str | None,
                                 search_query_en: str | None) -> list[dict]:
    """Derive a list of sub-topics from one profile entry.

    Uses no YouTube quota. Errors propagate; the caller treats failure as
    "no sub-topics" and proceeds without them.
    """
    result = await _get_entry_subtopics_agent().run(
        build_subtopics_prompt(mode, text, base_types, chips, search_query, search_query_en))
    return _sanitize_subtopics(list(result.output.items))


# ---- Candidate-channel fit judge (expansion / recommendation) ---- #
class FitItem(BaseModel):
    """Verdict for one candidate: ``i`` = index within the chunk (from 0), ``fit`` in {0, 1, 2}."""
    i: int
    fit: int


class ChannelFit(BaseModel):
    """LLM output: one ``{i, fit}`` per candidate index.

    Verdicts are keyed by index rather than aligned by position: with large
    candidate lists a bare positional list drifts out of alignment and
    mis-scores channels.
    """
    fits: list[FitItem]


# Scores each candidate channel 0/1/2 by how much of its recent output matches
# the preference, deciding whether it is worth adding to the user's supply.
_EXPAND_FIT_SYSTEM = (
    "你是 FocusFlow 的频道贴合判官。输入是一条用户亲笔的偏好条目(含派生的类型/细分标签"
    "与中英搜索词)和候选 YouTube 频道列表(带序号,0 起;每个频道给出主题分类、简介、"
    "关键词和近期上传的视频标题)。任务:对每个频道,根据**近期视频标题与简介**估计"
    "「与这条偏好直接相关的内容占比」,输出一个档位:2=大多数(约三分之二以上)内容就是"
    "这个偏好,频道主业即此;1=经常出现但不是主业(约四分之一到三分之二);0=只是偶然沾边"
    "或无关。例:偏好「AI、大模型」——近期 20 条里 15 条讲 LLM/AI 工具的频道=2;科技商业"
    "频道 20 条里 5 条谈 AI=1;芯片测评频道做了一期 AI 新闻=0。判断只看内容本身:**不看"
    "订阅数、不看语言(中文英文同等对待)、不看频道名字面、不猜频道来源**;标题为空时"
    "只按简介与主题判。输出 fits 列表:**每个候选一条 {i: 候选序号, fit: 0/1/2}**,"
    "序号必须逐一对应输入(0 到 候选数-1,一个不缺、不重复、不越界)。"
)

_expand_fit_agent: Agent | None = None


def _get_expand_fit_agent() -> Agent:
    """Lazily build the channel-fit Agent on ``LLM_MODEL`` (cost first)."""
    global _expand_fit_agent
    if _expand_fit_agent is None:
        _expand_fit_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=ChannelFit,
            system_prompt=_EXPAND_FIT_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _expand_fit_agent


def _build_fit_prompt(entry: dict, cands: list[dict]) -> str:
    """Build the channel-fit user message (pure function, testable offline).

    An entry header followed by one block per candidate. Candidate blocks
    deliberately omit where the candidate came from (search vs. pool), so the
    judge cannot be biased by source. Descriptions are cut to 200 chars and
    titles to ``EXPAND_UPLOAD_TITLES``.
    ``entry`` = {"text", "base_types", "chips", "search_query", "search_query_en"};
    ``cands`` = [{"title", "topics", "description", "keywords", "titles"}].
    """
    import config
    blocks = []
    for i, c in enumerate(cands):
        titles = [t for t in (c.get("titles") or []) if t][:config.EXPAND_UPLOAD_TITLES]
        desc = " ".join((c.get("description") or "").split())[:200]
        lines = [f"{i}. {c.get('title') or '?'}",
                 f"主题:{', '.join(c.get('topics') or []) or '无'}",
                 f"简介:{desc or '无'}",
                 f"关键词:{(c.get('keywords') or '无')[:120]}",
                 "近期视频:" + (" | ".join(titles) if titles else "无")]
        blocks.append("\n".join(lines))
    head = (f"[偏好条目] {entry.get('text')}\n"
            f"[类型] {', '.join(entry.get('base_types') or [])} "
            f"[细分] {', '.join(entry.get('chips') or [])}\n"
            f"[搜索词] {entry.get('search_query') or ''} | {entry.get('search_query_en') or ''}\n"
            f"[候选数] {len(cands)}\n[候选频道]\n")
    return head + "\n\n".join(blocks)


def _sanitize_fits(items: list, n: int) -> list[int]:
    """Clean channel-fit output into a list aligned with the ``n`` candidates.

    Unlike the other sanitizers this one raises: an out-of-range or missing
    index means the LLM did not answer the given candidate list, so the whole
    chunk is untrustworthy (``ValueError``; the caller treats it as a judge
    failure). Duplicate indices: the last one wins. ``fit`` is clamped to 0..2.
    """
    by_i: dict[int, int] = {}
    for it in items:
        i, f = int(it.i), int(it.fit)
        if not 0 <= i < n:
            raise ValueError(f"fits 序号越界 {i}/{n}")
        by_i[i] = max(0, min(2, f))
    missing = [i for i in range(n) if i not in by_i]
    if missing:
        raise ValueError(f"fits 缺序号 {missing[:5]}…/{n}")
    return [by_i[i] for i in range(n)]


async def _judge_chunk(entry: dict, chunk: list[dict]) -> list[int]:
    """One judge round-trip for a chunk of candidates."""
    result = await _get_expand_fit_agent().run(_build_fit_prompt(entry, chunk))
    return _sanitize_fits(list(result.output.fits), len(chunk))


async def judge_channel_fit(entry: dict, cands: list[dict]) -> list[int]:
    """Score candidate channels 0/1/2 by the share of recent output matching the entry.

    Candidates are split into chunks of ``config.EXPAND_JUDGE_BATCH`` (indices
    restart at 0 per chunk) and judged concurrently — sending too many at once
    causes misalignment. Results are merged back in ``cands`` order. Any LLM
    error or malformed chunk propagates; the caller falls back (e.g. treating
    all candidates as fit=1 and marking them unjudged).
    """
    if not cands:
        return []

    import config
    size = max(1, int(getattr(config, "EXPAND_JUDGE_BATCH", 20)))
    chunks = [cands[i:i + size] for i in range(0, len(cands), size)]
    parts = await asyncio.gather(*(_judge_chunk(entry, ch) for ch in chunks))
    return [f for part in parts for f in part]
