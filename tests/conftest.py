import pytest


@pytest.fixture(autouse=True)
def _stub_intent_expand(monkeypatch):
    """Pin the intent-expansion LLM call to a failure so the test suite never
    reaches a real model, even if an API key happens to be exported. Tests that
    need custom expansion behaviour assign intent_search._expand themselves."""
    from agent import intent_search

    async def _no_llm(text, lang):
        raise RuntimeError("LLM calls are disabled in tests")

    monkeypatch.setattr(intent_search, "_expand", _no_llm)
