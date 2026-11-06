"""Tests for the viewing-plan drafting agent: prompt assembly, pass-through and fallback.

The agent is fully mocked; no external calls are made.
"""
import asyncio
import os
import tempfile
os.environ.setdefault("FOCUSFLOW_DB_PATH", os.path.join(tempfile.mkdtemp(), "boot.db"))
import pytest
from agent import viewing_plan as vp
from agent.prompts import PROMPTS
from core.viewing_plan import PlanDraft, PlanDraftTurn, PlanStage


def test_prompt_注册与要点():
    p = PROMPTS["viewing_plan"]
    assert p.name == "viewing_plan" and p.version == "v3"
    for frag in ("由浅入深", "一次只问一个", "ready", "keywords"):
        assert frag in p.system, frag
    # plans are about viewing, not study; two kinds; theme is always emitted
    for frag in ("长期观看计划", "progressive", "variety", "每天不重样", "theme"):
        assert frag in p.system, frag
    assert "小福·学习规划" not in p.system


def test_compose_prompt_含背景历史草案新消息():
    hist = [{"role": "user", "text": "我想学机器学习", "draft_json": None},
            {"role": "assistant", "text": "基础如何?", "draft_json": None},
            {"role": "user", "text": "零基础", "draft_json": None},
            {"role": "assistant", "text": "先出个草案", "draft_json": '{"name":"ML 入门"}'}]
    p = vp.compose_draft_prompt(hist, "[模式] 专注 | 时长窗 ≥180s", "每天 5 个吧", "zh")
    for frag in ("专注", "我想学机器学习", "零基础", '{"name":"ML 入门"}', "每天 5 个吧", "[上一版草案]"):
        assert frag in p, frag
    assert p.index("[上一版草案]") < p.index("每天 5 个吧")


def test_run_draft_turn_透传(monkeypatch):
    captured = {}

    class _R:
        output = PlanDraftTurn(reply="好", ready=True, draft=PlanDraft(
            name="ML 入门", goal="g", level="零基础", daily_count=5,
            stages=[PlanStage(title="w1", days=7, keywords=["linear regression"], goal="回归")]))

    class _A:
        async def run(self, prompt, **kw):
            captured["p"] = prompt
            return _R()

    monkeypatch.setattr(vp, "_get_agent", lambda lang: _A())
    turn = asyncio.run(vp.run_draft_turn([], "bg", "hi", "en"))
    assert turn.ready and turn.draft.name == "ML 入门"
    assert "hi" in captured["p"]


def test_run_draft_turn_异常上抛(monkeypatch):
    class _Boom:
        async def run(self, prompt, **kw):
            raise RuntimeError("llm down")
    monkeypatch.setattr(vp, "_get_agent", lambda lang: _Boom())
    with pytest.raises(RuntimeError):
        asyncio.run(vp.run_draft_turn([], "bg", "hi", "zh"))


def test_fallback_turn_双语零草案():
    zh, en = vp.fallback_turn("zh"), vp.fallback_turn("en")
    assert zh.draft is None and not zh.ready and zh.reply
    assert en.reply != zh.reply
