"""Real-time adjudicator for "dislike" feedback.

Given a disliked video plus the reasons the user ticked, and taking into
account the user's own positive profile for that mode and their existing
negative preferences, the LLM generalises the feedback into one topic-level
negative preference, or merges it into an existing one. Follows the same
pattern as ``profiler.consolidate_long_signal``: a BaseModel output, an
in-file system prompt, and a lazy singleton Agent.

The prompt lives in this file rather than in ``agent/prompts.py``. No
database access: persistence belongs to the API layer.
"""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from agent import _model_id, _model_settings
from config import LLM_MAX_SCHEMA_RETRIES, LLM_MODEL, LLM_TIMEOUT_SEC


class NegAdjudication(BaseModel):
    """Adjudication result.

    ``matched_index`` is the 0-based index of an existing negative preference
    with the same meaning (-1 = none, create a new entry). ``text`` is the new
    entry or the merged umbrella phrasing; empty means no entry this time."""
    matched_index: int = -1         # Index of the existing entry to merge into; -1 = new entry
    text: str = ""                  # Entry text; empty = produce nothing


_ADJUDICATE_SYSTEM = (
    "你是观看负偏好的裁定官。输入:一条用户点了「不喜欢」并给出理由的视频"
    "(标题/频道/所处观看模式)、用户为该模式亲笔声明的正向偏好、既有负偏好列表"
    "(带序号,0 起)。任务:把这次不喜欢归纳成一条负偏好。三条纪律:"
    "①一句话、类型/主题级颗粒度——不写频道名(频道好恶另有账本),也不复述"
    "单个视频;②防过宽——归纳宽度以「换一个同类视频该结论仍成立」为限,"
    "不得因一个标题党励志视频就写「不看励志类」;③结合画像推理真实边界——"
    "正向偏好与不喜欢理由对照,分辨用户不喜欢的到底是哪个侧面(如爱看深度"
    "评测者嫌某视频「太浅」,该归纳为「不喜欢浅尝辄止的资讯速览」而非整个"
    "类目)。与既有负偏好语义同类时填 matched_index 并把统称写进 text(宁缺"
    "毋滥,明确同类才归并);无同类则 matched_index=-1、text 写新条;理由信息"
    "太少归纳不出时 text 留空。text 不超过 50 字。"
    "勾选理由决定颗粒度(DC 批):「对这个题材不感兴趣」→直接以题材归纳,再按画像分辨是"
    "整个题材还是某个侧面;「内容质量差」→含标题党/内容农场/空洞口水,归纳成「不喜欢××"
    "类型的低质呈现」而非题材本身;「不想在此模式看这类」→措辞带上模式限定;"
    "「其他」→只凭补充说明归纳,信息不足则 text 留空。"
)

_adjudicate_agent: Agent | None = None


def _get_adjudicate_agent() -> Agent:
    """Lazily build the singleton Agent; uses the cost-oriented PROFILE_MODEL tier."""
    global _adjudicate_agent
    if _adjudicate_agent is None:
        import config
        _adjudicate_agent = Agent(
            _model_id(config.PROFILE_MODEL or LLM_MODEL),
            output_type=NegAdjudication,
            system_prompt=_ADJUDICATE_SYSTEM,
            retries=LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(config.PROFILE_MODEL or LLM_MODEL, timeout=LLM_TIMEOUT_SEC),
        )
    return _adjudicate_agent


def sanitize_adjudication(adj: NegAdjudication, n_existing: int) -> NegAdjudication:
    """Normalise raw LLM output (pure function, testable offline).

    ``n_existing`` is the number of existing entries enumerated in the prompt,
    so a valid ``matched_index`` lies in ``[0, n_existing)``; anything else
    becomes -1 ("new entry"). ``text`` is collapsed to one line and capped at
    80 chars; an empty text is passed through (= no entry)."""
    idx = adj.matched_index if 0 <= adj.matched_index < n_existing else -1
    text = " ".join(adj.text.split())[:80]
    return NegAdjudication(matched_index=idx, text=text)


async def adjudicate_dislike(*, video_title: str, channel_title: str, mode: str,
                             chips: list[str], free_text: str | None,
                             profile_lines: list[str],
                             existing: list[str], lang: str = "zh") -> NegAdjudication:
    """Run one adjudication call.

    ``chips`` are the reason keys ticked in the UI (mapped to their labels via
    ``config.DISLIKE_CHIP_LABELS``); ``free_text`` is the optional note;
    ``profile_lines`` is the user's positive profile for this mode; the order
    of ``existing`` defines the indices ``matched_index`` refers to.

    LLM errors propagate; the API layer handles them fail-soft. For
    ``lang == "en"`` an English-output requirement is appended to the user
    prompt (the system prompt is shared)."""
    import config
    labels = "、".join(config.DISLIKE_CHIP_LABELS.get(c, c) for c in chips)      # Unknown keys fall back to the raw key
    prof = "\n".join(f"- {t}" for t in profile_lines) or "(无)"
    negs = "\n".join(f"{i}. {t}" for i, t in enumerate(existing)) or "(无)"
    result = await _get_adjudicate_agent().run(
        f"[视频] {video_title}(频道:{channel_title})\n[观看模式] {mode}\n"
        f"[勾选理由] {labels}\n[补充说明] {free_text or '(无)'}\n"
        f"[该模式正向画像]\n{prof}\n[既有负偏好]\n{negs}"
        + ("\n\n[Output requirement] Write `text` in English." if lang == "en" else ""))
    return result.output
