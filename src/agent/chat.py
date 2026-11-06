"""In-app AI chat companion.

A thin member of the agent layer, consumed directly by the API layer (as with
``agent.profiler``). A single LLM call does two jobs at once: a friendly,
in-persona reply, and a silent analysis that extracts a refresh-intent
candidate plus short- and long-term profile signals.

The chatbot is a pure collector: it never operates the product and never
writes to the database. Persistence, rate limiting and profile merging all
live in the API layer. The persona prompt is kept in this file rather than in
``agent/prompts.py``, whose default prompts are locked byte-for-byte.

LLM errors are always re-raised; the API layer falls back to
``fallback_turn`` so chat never returns a 5xx.
"""
import logging

from pydantic import BaseModel
from pydantic_ai import Agent

from agent import _model_id, _model_settings

logger = logging.getLogger(__name__)


class LongSignal(BaseModel):
    """A long-term profile signal.

    ``mode`` is the mode key the user explicitly tied it to (focus/explore/
    wind_down/entertain). When unset or invalid, the API layer falls back to
    the current chat mode; validation happens there, not here."""
    text: str                           # Signal text, e.g. "long-time fan of astronomy explainers"; stripped and capped at 200 chars before storage
    mode: str | None = None             # Mode this signal belongs to


class ChatTurn(BaseModel):
    """One chat turn: the reply plus the silent analysis (raw input for the
    layered profile writes done by the API layer)."""
    reply: str                            # Reply shown to the user (required)
    intent_candidate: str | None = None   # Only for an explicit wish to watch something; small talk / mood / unsure -> None
    short_signals: list[str] = []         # Time-bound states (exam prep, recently into X) -> short-term profile section
    long_signals: list[LongSignal] = []   # Stable traits (hobbies, standing preferences/aversions) -> long-term section, grouped by mode


_SYSTEM = {
    "zh": (
        "你是「小福」,FocusFlow 的站内聊天伙伴。FocusFlow 帮用户按自己的意图看"
        " YouTube,对抗算法沉迷。\n"
        "性格:活泼、温暖、简短自然,偶尔用 emoji,不油腻、不说教;回复不超过三句话。\n"
        "职责一·陪聊:围绕用户说的话回应,别自顾自扯远。\n"
        "职责二·意图:只有明确的观看意愿(想看/想学/想找某类视频)才算意图——把它改写成"
        "一句简洁可执行的意图放进 intent_candidate;情绪表达、闲聊、提问都不算;拿不准就留空"
        "(确认卡骚扰比漏检更伤)。改写必须保留用户点名的具体主题限定词,只删礼貌与语气词,"
        "不得把具体主题泛化成上位类目(『神经网络底层基础讲解』不能改成『AI 类视频』)。\n"
        "职责三·画像信号:short_signals 收带时效的状态(在备考/最近迷上 X),long_signals 收"
        "稳定特征(长期爱好/明确的持续偏好或回避),每条为 {text, mode}——用户明确指向某个"
        "模式的偏好(如「睡前模式我想听白噪音」「探索模式多来历史纪录片」),mode 填对应"
        "模式键(focus/explore/wind_down/entertain);没有指名模式就留空,不要猜。"
        "宁缺毋滥,单轮可以都为空。\n"
        "边界:你不能替用户操作产品(不能切模式、不能触发刷新、不能改设置);被要求时"
        "友善说明用户可以自己动手,别答应做不到的事。"
    ),
    "en": (
        "You are Flo, the in-app chat buddy of FocusFlow — a tool that helps users watch"
        " YouTube on their own terms instead of the algorithm's.\n"
        "Personality: playful, warm, brief and natural; light emoji use; never preachy."
        " Keep replies within three sentences.\n"
        "Duty 1 — chat: respond to what the user actually said.\n"
        "Duty 2 — intent: only an explicit desire to WATCH something (want to watch/learn/"
        "find some kind of videos) counts; rewrite it as one concise actionable line in"
        " intent_candidate. Moods, small talk and questions don't count; when unsure, leave"
        " it null (a wrong confirm card hurts more than a miss). The rewrite must keep the"
        " user's specific topic qualifiers verbatim — drop only politeness fillers, and never"
        " generalize a specific topic into its parent category ('neural network fundamentals'"
        " must not become 'AI videos').\n"
        "Duty 3 — profile signals: short_signals for time-bound states (studying for an exam,"
        " recently into X); long_signals for stable traits (lasting hobbies, explicit standing"
        " preferences or aversions), each as {text, mode} — when the user explicitly ties a"
        " preference to a mode (e.g. 'for wind-down I want white noise'), set mode to that mode"
        " key (focus/explore/wind_down/entertain); leave it null when no mode is named — never"
        " guess. Be conservative; both may be empty.\n"
        "Boundary: you cannot operate the product (no mode switching, no refresh, no settings)."
        " If asked, kindly point out the user can do it themselves; never promise actions."
    ),
}

_agents: dict[str, Agent] = {}

def _get_agent(lang: str) -> Agent:
    """Lazily build one Agent per language, so importing needs no API key."""
    import config
    if lang not in _agents:
        _agents[lang] = Agent(
            _model_id(config.CHAT_MODEL or config.LLM_MODEL),
            output_type=ChatTurn,
            system_prompt=_SYSTEM[lang],
            retries=config.LLM_MAX_SCHEMA_RETRIES,                  # PydanticAI schema-validation retries
            model_settings=_model_settings(config.CHAT_MODEL or config.LLM_MODEL, timeout=config.LLM_TIMEOUT_SEC),
        )
    return _agents[lang]


# Build the user prompt: history, current mode, optional plan context, then the new message.
def _compose_prompt(history: list[dict], text: str, mode_name: str | None, lang: str,
                    plan_context: str | None = None) -> str:
    lines = ["[对话历史(旧→新)]" if lang == "zh" else "[History (old→new)]"]
    for m in history:
        lines.append(f"{m['role']}: {m['text']}")                          # role is "user" or "assistant"
    mode_line = (f"[当前模式] {mode_name or '未选'}" if lang == "zh"
                 else f"[Current mode] {mode_name or 'none'}")
    lines.append(mode_line)

    # Plan context ("plan name / stage / goal") is only added when chatting from
    # the plan page; other callers pass None and the section is skipped.
    if plan_context:
        lines.append((f"[当前计划] {plan_context}" if lang == "zh"
                      else f"[Current plan] {plan_context}"))
    new_line = f"[用户新消息] {text}" if lang == "zh" else f"[New message] {text}"   # New message always goes last
    lines.append(new_line)
    return "\n".join(lines)


async def run_chat(history: list[dict], text: str, mode_name: str | None,
                   lang: str, plan_context: str | None = None) -> ChatTurn:
    """Run one chat turn (reply + silent analysis).

    ``history`` is ordered old to new and already windowed by the caller.
    Errors propagate; the API layer handles fallback."""
    result = await _get_agent(lang).run(
        _compose_prompt(history, text, mode_name, lang, plan_context))
    return result.output


# Catch-all fallback for any LLM failure: upstream HTTP errors (429 rate limit /
# quota, 401/403 bad key, 5xx overload), network errors and timeouts (wrapped as
# ModelAPIError), schema-validation failures after retries
# (UnexpectedModelBehavior), and content-filter refusals.
def fallback_turn(lang: str) -> ChatTurn:
    """Fallback reply used by the API layer when the LLM call fails:
    no signals, no intent, and chat never returns a 5xx."""
    reply = ("呜,我刚刚走神了…再说一遍好吗?" if lang == "zh"
             else "Oops, I spaced out — could you say that again?")
    return ChatTurn(reply=reply)
