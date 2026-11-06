"""Planning agent for long-term viewing plans.

Part of the agent layer: no SQL, no imports from ``sources/``. Each turn is a
single LLM call producing ``PlanDraftTurn{reply, draft, ready}``. Draft
validation lives in ``core.viewing_plan.sanitize_draft`` (invoked by the API
layer); this module does no post-processing.

The planning conversation is stored separately and passed in by the API layer
as ``history`` (items ``{role, text, draft_json}``). The most recent non-empty
``draft_json`` is included as the previous draft, so the LLM revises it rather
than starting over.
"""
from __future__ import annotations

from pydantic_ai import Agent

from agent import _model_id, _model_settings
from agent.prompts import PROMPTS
from core.viewing_plan import PlanDraftTurn

_agents: dict[str, Agent] = {}


def _get_agent(lang: str) -> Agent:
    """Lazily build one Agent per language (no API key needed at import).
    The system prompt is shared across languages."""
    import config
    if lang not in _agents:
        model = config.VIEWING_PLAN_MODEL or config.LLM_MODEL
        _agents[lang] = Agent(
            _model_id(model),
            output_type=PlanDraftTurn,
            system_prompt=PROMPTS["viewing_plan"].system,
            retries=config.LLM_MAX_SCHEMA_RETRIES,
            model_settings=_model_settings(model, timeout=config.LLM_TIMEOUT_SEC),
        )
    return _agents[lang]


def compose_draft_prompt(history: list[dict], background: str, text: str, lang: str) -> str:
    zh = lang == "zh"
    lines = [("[背景]" if zh else "[Background]"), background, ""]
    lines.append("[对话历史(旧→新)]" if zh else "[History (old→new)]")
    last_draft: str | None = None
    for m in history:
        lines.append(f"{m['role']}: {m['text']}")
        if m.get("draft_json"):
            last_draft = m["draft_json"]
    if last_draft:
        lines += ["", ("[上一版草案]" if zh else "[Previous draft]"), last_draft]
    lines += ["", (f"[用户新消息] {text}" if zh else f"[New message] {text}")]
    return "\n".join(lines)


async def run_draft_turn(history: list[dict], background: str, text: str,
                         lang: str) -> PlanDraftTurn:
    """Run one planning turn. Errors propagate; the API layer falls back to
    ``fallback_turn``."""
    result = await _get_agent(lang).run(compose_draft_prompt(history, background, text, lang))
    return result.output


def fallback_turn(lang: str) -> PlanDraftTurn:
    reply = ("呜,我刚刚卡了一下…再说一遍好吗?" if lang == "zh"
             else "Oops, I glitched — could you say that again?")
    return PlanDraftTurn(reply=reply, draft=None, ready=False)
