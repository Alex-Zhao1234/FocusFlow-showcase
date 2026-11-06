# Agent layer: the single place in the project where LLM calls happen.
# Module rules: planner and judges never import each other (both are driven by the
# orchestrator); prompts.py is a pure-data module imported by both; the agent layer
# never imports sources/ and issues no SQL (database access is confined to
# planner.build_plan_context; judges are database-free).


def _model_id(name: str) -> str:
    """Return the PydanticAI model identifier (``provider:model``) for ``name``.

    The provider prefix is decided only here. It lives in the package
    ``__init__`` because planner and judges both need it but must not import
    each other; this is their only common ancestor (prompts.py is restricted to
    importing pydantic only).

    Provider routing is by model name: ``deepseek-*`` models go through the
    DeepSeek provider (OpenAI-compatible endpoint, ``DEEPSEEK_API_KEY``);
    everything else uses Anthropic. Setting any ``*_MODEL`` config constant to
    a DeepSeek model switches that tier without code changes."""
    provider = "deepseek" if name.startswith("deepseek") else "anthropic"
    return f"{provider}:{name}"


def _model_settings(name: str, *, timeout: float) -> dict:
    """Shared ``model_settings`` for every Agent construction site.

    DeepSeek models enable "thinking" by default, which must be explicitly
    disabled in the request body: otherwise large structured-output calls
    (judges, planner) slow down by an order of magnitude, hit the request
    timeout, and the reasoning text can corrupt the structured output.
    Anthropic models get no ``extra_body`` and behave unchanged."""
    settings: dict = {"timeout": timeout}
    if name.startswith("deepseek"):
        settings["extra_body"] = {"thinking": {"type": "disabled"}}
    return settings
