"""Expand a user's viewing intent into a pack of YouTube search queries.

A thin agent-layer member consumed only by the orchestrator (stage S1.5, the
intent search pack). The first query is always the user's intent text
verbatim, so explicit user input is honoured as a literal keyword search; the
rest are variants from one lightweight LLM call (Chinese/English word forms,
sub-angles, synonyms).

Unlike planner/chat, which re-raise LLM errors, expansion failures are
absorbed here and degrade to the verbatim query alone: intent retrieval must
never drop to zero because the expansion step failed, and the first search is
a deterministic action guaranteed by code. No database, no YouTube API. The
prompt lives in this file rather than in ``prompts.py``. The caller trims to
its search budget; this module only caps at ``INTENT_SEARCH_MAX``.
"""
import logging

from pydantic import BaseModel
from pydantic_ai import Agent

from agent import _model_id, _model_settings

logger = logging.getLogger(__name__)


class IntentQuerySet(BaseModel):
    """Structured output of the expansion call; cleanup happens in ``intent_queries``."""
    queries: list[str]

_SYSTEM = (
    "你是 FocusFlow 的意图检索词扩展器。输入是用户此刻的观看意图声明(原文),"
    "任务:产出一组 YouTube 视频搜索词变体,与原文一起构成多角度检索包。\n"
    "要求:①每条变体必须仍精确指向用户点名的具体主题,不得泛化成上位类目"
    "(『神经网络底层基础讲解』的变体可以是『反向传播原理』『neural network "
    "from scratch』,不可以是『AI 视频』);②中英双语词形都出;③覆盖子角度、"
    "同义表述、经典关键词;④不要复述原文本身;⑤每条是可直接提交给 YouTube "
    "搜索框的短词组,不带解释。"
)

_agent: Agent | None = None

def _get_agent() -> Agent:
    """Lazily build the singleton Agent, so importing needs no API key."""
    global _agent
    import config
    if _agent is None:
        _agent = Agent(
            _model_id(config.LLM_MODEL),
            output_type=IntentQuerySet,
            system_prompt=_SYSTEM,
            retries=config.LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(config.LLM_MODEL, timeout=config.LLM_TIMEOUT_SEC),
        )
    return _agent

async def _expand(intent_text: str, lang: str) -> list[str]:
    """LLM half of the expansion. Errors propagate and are absorbed by
    ``intent_queries``; tests patch this function."""
    import config
    n = config.INTENT_SEARCH_MAX - 1              # Number of variants; the first slot is reserved for the verbatim intent
    result = await _get_agent().run(
        f"[意图原文] {intent_text}\n[变体数量] 至多 {n} 条")
    return result.output.queries

async def intent_queries(intent_text: str, lang: str = "zh") -> list[str]:
    """Intent -> search query pack.

    The first query is the verbatim intent (never trimmed), followed by
    deduplicated variants in LLM order, capped at ``INTENT_SEARCH_MAX`` in
    total. If expansion fails, only the verbatim query is returned, so intent
    retrieval never drops to zero."""
    import config
    text = " ".join(intent_text.split())        # Collapse all whitespace runs to single spaces
    queries = [text]
    try:
        variants = await _expand(text, lang)
    except Exception as e:                      # Degrade to the verbatim query only
        logger.warning("意图扩展失败(降级只跑首搜): %s", e)
        return queries
    # Dedupe against the verbatim text and earlier variants; normalise whitespace,
    # coerce to str, skip empties. Order is preserved so the LLM's best variants come first.
    seen = {text}
    for q in variants:
        q = " ".join(str(q).split())
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    return queries[: config.INTENT_SEARCH_MAX]
