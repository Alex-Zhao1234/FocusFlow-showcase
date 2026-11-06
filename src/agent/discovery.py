"""LLM-driven channel discovery for the offline channel pools.

This module belongs to the agent layer but is never imported by any pipeline
stage: it is used only by the offline channel-pool job, so a refresh incurs
zero additional LLM calls because of it.

It exposes lazily constructed singleton agents (same pattern as the profiler),
each paired with a pure-code fallback:

User track (expand from a user's preferred-channel pool profile):
  1. ``generate_search_queries`` -> ``ChannelSearchQuerySet`` (3–5 queries);
     fallback ``fallback_queries`` uses the seeds' topic categories as queries.
  2. ``judge_candidates`` -> ``ChannelCandidateVerdicts`` (one verdict per
     candidate); fallback ``fallback_rank`` sorts by hard metrics.

Chip track (build the per-interest-chip type pool):
  3. ``generate_chip_queries`` -> ``list[str]``; fallback
     ``fallback_chip_queries`` returns the chip display name.
  4. ``judge_chip_candidates`` -> selected channel ids; fallback is again
     ``fallback_rank``.
  5. ``generate_chip_subtopics`` -> a sub-topic map used to rotate query
     directions across rounds.

The module touches no database. LLM exceptions propagate unchanged and the
caller triggers the fallback in its exception handler: swallowing them here
would make the caller believe it got a structured result and silently lose
the fallback path.
"""
from pydantic_ai import Agent

from agent import _model_id, _model_settings
from agent.prompts import PROMPTS
from config import (CHIP_HARD_TO_EXPAND_ROUNDS,
                    CHIP_SUBTOPICS_MIN, CHIP_SUBTOPICS_MAX,
                    DISCOVERY_MIN_SUBS, DISCOVERY_QUERIES_MAX, INTEREST_CHIPS,
                    LLM_MAX_SCHEMA_RETRIES, LLM_MODEL, LLM_TIMEOUT_SEC,
                    SMALL_CHANNEL_SUBS_MAX,
                    TYPE_POOL_QUERIES_PER_CHIP, TYPE_POOL_SUBS_MIN,
                    TYPE_POOL_SUBS_PREFERRED_MIN, TYPE_POOL_TARGET_MAX)
from core.models import (Channel, ChannelCandidateVerdicts, ChannelSearchQuerySet,
                         ChipJudgeVerdict, ChipSubtopicSet)

_query_agent: Agent | None = None
_judge_agent: Agent | None = None
_chip_query_agent: Agent | None = None
_chip_judge_agent: Agent | None = None
_chip_subtopics_agent: Agent | None = None


_CHIP_QUERY_SYSTEM = PROMPTS["chip_query"].system
_CHIP_JUDGE_SYSTEM = PROMPTS["chip_judge"].system
_CHIP_SUBTOPICS_SYSTEM = PROMPTS["chip_subtopics"].system


# ---- User track: expand from a user's preferred-channel pool profile ----

def _get_query_agent() -> Agent:
    """Lazy singleton: importing needs no API key; a missing key fails loudly on first use."""
    global _query_agent
    if _query_agent is None:
        _query_agent = Agent(
            _model_id(LLM_MODEL),                                                  # adds the provider prefix PydanticAI expects
            output_type=ChannelSearchQuerySet,
            system_prompt=PROMPTS["discovery_query"].system,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _query_agent


def _get_judge_agent() -> Agent:
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=ChannelCandidateVerdicts,
            system_prompt=PROMPTS["discovery_judge"].system,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _judge_agent


async def generate_search_queries(pool_profile: str) -> ChannelSearchQuerySet:
    """Pool profile -> 3–5 channel search queries. Exceptions propagate.

    The user prompt ends with a subscriber-tier hint. The DISCOVERY_MIN_SUBS
    gate is applied only after searching, so queries that surface small
    channels waste scarce search-bucket calls. The hint therefore states a
    preferred tier (SMALL_CHANNEL_SUBS_MAX, the large-channel lower bound)
    plus the hard floor (DISCOVERY_MIN_SUBS); mid-tier channels between the
    two are still accepted. Only the user prompt carries this hint; the system
    prompt is left untouched.
    """
    prompt = (f"用户偏好频道池画像如下,请生成频道搜索词:\n{pool_profile}\n"
              f"要求:【订阅数越高越优先】——{SMALL_CHANNEL_SUBS_MAX:,} 以上最佳,"
              f"请优先给能搜到该题材知名/头部创作者的搜索词;"
              f"{DISCOVERY_MIN_SUBS:,} 是入池硬闸,低于此数的频道搜回来也进不了池;"
              f"介于两者之间的中腰部频道仍然要收,不必刻意回避;"
              f"避免「新人」「刚起步」「独立小众」等指向零起步频道的表述。\n")
    result = await _get_query_agent().run(prompt)
    return result.output


async def judge_candidates(pool_profile: str,
                           candidates_block: str) -> ChannelCandidateVerdicts:
    """Judge candidate relevance against the pool profile (keep flag + relevance score).

    ``candidates_block`` is the text rendering of candidates that already passed
    the hard gate. Exceptions propagate; the caller falls back to hard-metric ranking.
    """
    result = await _get_judge_agent().run(
        f"## 池画像\n{pool_profile}\n\n## 候选频道\n{candidates_block}")
    return result.output


def fallback_queries(seeds: list[Channel]) -> list[str]:
    """Fallback for ``generate_search_queries``: seed topic categories as queries.

    Deduplicated in order and capped at DISCOVERY_QUERIES_MAX. May return fewer
    than three queries; the degraded path enforces no minimum (search whatever
    exists; with zero queries the round relies on other sources only).
    """
    seen: list[str] = []
    for c in seeds:
        for t in (c.topic_categories or []):
            t = t.replace("_", " ")
            if t not in seen:
                seen.append(t)
    return seen[:DISCOVERY_QUERIES_MAX]


def fallback_rank(cands: list[Channel]) -> list[Channel]:
    """Fallback for the judges: sort by subscribers, then video count, descending."""
    return sorted(cands, key=lambda c: ((c.channel_subs or 0),
                                        (c.video_count or 0)), reverse=True)


# ---- Chip track: per-interest-chip type pool (same lazy-singleton pattern) ----

def _get_chip_query_agent() -> Agent:
    """Lazy singleton: importing needs no API key; a missing key fails loudly on first use."""
    global _chip_query_agent
    if _chip_query_agent is None:
        _chip_query_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=ChannelSearchQuerySet,
            system_prompt=_CHIP_QUERY_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _chip_query_agent


def _get_chip_judge_agent() -> Agent:
    global _chip_judge_agent
    if _chip_judge_agent is None:
        _chip_judge_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=ChipJudgeVerdict,
            system_prompt=_CHIP_JUDGE_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _chip_judge_agent


def _get_chip_subtopics_agent() -> Agent:
    """Lazy singleton, same rules as ``_get_chip_query_agent``."""
    global _chip_subtopics_agent
    if _chip_subtopics_agent is None:
        _chip_subtopics_agent = Agent(
            _model_id(LLM_MODEL),
            output_type=ChipSubtopicSet,
            system_prompt=_CHIP_SUBTOPICS_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _chip_subtopics_agent


async def generate_chip_subtopics(chip_slug: str) -> list[str]:
    """Chip topic -> 8–12 sub-topics (a direction map, generated once per chip and stored).

    Costs no YouTube quota. Exceptions propagate; the caller records the round
    as degraded and proceeds without directions. The minimum is only requested
    in the prompt (better fewer than padded); the maximum is enforced by truncation.
    """
    prompt = (f"chip 主题:{chip_scope(chip_slug)}\n"
              f"请给出 {CHIP_SUBTOPICS_MIN}~{CHIP_SUBTOPICS_MAX} 个子领域。")
    result = await _get_chip_subtopics_agent().run(prompt)
    return result.output.subtopics[:CHIP_SUBTOPICS_MAX]


def _chip_candidates_block(cands: list[Channel]) -> str:
    """Render candidates for the chip judge: one line of id / title / subscribers / topics.

    There is no description column: ``Channel`` carries no description field
    and this layer does no database lookups to fetch one.
    """
    lines = []
    for c in cands:
        topics = "/".join(c.topic_categories or []) or "无"
        lines.append(f"- {c.channel_id} | {c.channel_title} | "
                     f"订阅 {c.channel_subs} | 主题 {topics}")
    return "\n".join(lines)


def chip_scope(chip_slug: str) -> str:
    """Pool scope for a chip = its INTEREST_CHIPS display name.

    Topic breadth is governed by the stored sub-topic map: query generation
    receives the round's directions and the judge receives the same set, so
    both sides stay equally wide. This function remains the single entry point
    for all three consumers (queries, judge, sub-topic generation).
    """
    return INTEREST_CHIPS[chip_slug][0]


async def generate_chip_queries(chip_slug: str, existing_titles: list[str], reroll: bool = False,
                                bilingual: bool = False,
                                subtopics: list[str] | None = None,
                                used_queries: list[str] | None = None) -> list[str]:
    """Chip topic -> quality-oriented channel search queries.

    This is the convergence point for all query strategies; the decisions on
    when to apply each one live in the caller, this function only assembles
    the user prompt from optional sections:

    - ``subtopics``: this round's directions; the query quota is split across them.
    - ``used_queries``: recently used queries to avoid (including mere rewordings).
    - ``existing_titles``: channels already in the pool; look for different styles.
    - ``reroll``: previous queries hit nothing new; change angle entirely.
    - ``bilingual``: the chip has stalled for several rounds; require Chinese and
      English queries.
    - A head-channel hint (preferred / hard subscriber floors) is always appended.

    Exceptions propagate; the caller falls back to ``fallback_chip_queries``.
    Output is truncated to TYPE_POOL_QUERIES_PER_CHIP (each query costs one
    search-bucket call).
    """
    prompt = f"chip 主题:{chip_scope(chip_slug)}\n"

    # Direction section: split the per-chip query quota across the round's sub-topics.
    if subtopics:
        base, rem = divmod(TYPE_POOL_QUERIES_PER_CHIP, len(subtopics))
        alloc = "、".join(f"「{st}」出 {base + (1 if i < rem else 0)} 条"
                          for i, st in enumerate(subtopics))
        prompt += (f"本轮请围绕以下子领域出词:{alloc}"
                   f"(合计 {TYPE_POOL_QUERIES_PER_CHIP} 条)。\n")

    # Avoidance section: recently used queries; rewording them does not count as new.
    if used_queries:
        prompt += (f"以下搜索词近期已用过,不要重复,也不要只换措辞"
                   f"(如 computer vision→CV→计算机视觉):"
                   f"{'、'.join(used_queries)}\n")

    # Gap-filling section: steer away from styles the pool already covers.
    if existing_titles:
        prompt += (f"以下频道已入池,找与它们不同风格/子领域的:"
                    f"{'、'.join(existing_titles)}\n")

    # Reroll section: can coexist with the direction section (that one says where
    # to go, this one insists on changing angle).
    if reroll:
        prompt += ("注意:该类型历史搜索词连续零命中(命中的频道全部已在池内或"
                   "不过硬指标),请彻底更换搜索角度(换子领域/换措辞/换视角),"
                   "并至少包含 1 条英文搜索词。\n")

    # Bilingual section for hard-to-expand chips. Its wording deliberately avoids
    # the reroll trigger phrase so tests can tell the two sections apart.
    if bilingual:
        prompt += (f"注意:该类型近 {CHIP_HARD_TO_EXPAND_ROUNDS} 轮扩建净增都不达标,"
                   f"中文侧可挖存量已稀薄——本轮请出【中英双语】搜索词,"
                   f"{TYPE_POOL_QUERIES_PER_CHIP} 条里中文与英文各至少 1 条;"
                   f"以频道质量与订阅数为先,不要为了凑中文词而降低订阅档次。\n")

    # Head-channel section (always present): preferred tier and hard floor.
    prompt += (f"要求:本词表面向头部频道建设——请给能搜到知名/头部创作者的搜索词"
               f"(订阅 {TYPE_POOL_SUBS_PREFERRED_MIN:,} 以上最佳,{TYPE_POOL_SUBS_MIN:,} 为硬底线),"
               f"避免「小众」「冷门」「独立」等长尾表述;该主题的头部创作者若以英文为主,"
               f"请直接给英文搜索词。\n")
    result = await _get_chip_query_agent().run(prompt)
    return result.output.queries[:TYPE_POOL_QUERIES_PER_CHIP]


def fallback_chip_queries(chip_slug: str) -> list[str]:
    """Fallback for ``generate_chip_queries``: the chip display name only (nothing invented)."""
    display, _base = INTEREST_CHIPS[chip_slug]
    return [display]


async def judge_chip_candidates(chip_slug: str, cands: list[Channel],
                              subtopics: list[str] | None = None) -> list[str]:
    """Chip topic + candidates -> selected channel ids (relevant and good quality; fewer is fine).

    The result is guaranteed to be a subset of the input ids: hallucinated ids
    are filtered here, since this is the only data source before the pool job
    persists and the caller should be able to trust it without re-validating.
    The list is then truncated to TYPE_POOL_TARGET_MAX. Exceptions propagate;
    the caller falls back to ``fallback_rank``.
    """
    block = _chip_candidates_block(cands)

    # Optional section: when directions are given, matching any of them counts as relevant.
    sub_sec = (f"\n\n## 本轮子领域\n候选按以下方向召回:{'、'.join(subtopics)}"
               f"——与 chip 主题或这些子领域相关即算相关。" if subtopics else "")
    result = await _get_chip_judge_agent().run(
        f"## chip 主题\n{chip_scope(chip_slug)}{sub_sec}\n\n## 候选频道\n{block}")
    valid = {c.channel_id for c in cands}
    selected = [i for i in result.output.selected_ids if i in valid]   # drop ids outside the candidate set
    return selected[:TYPE_POOL_TARGET_MAX]
