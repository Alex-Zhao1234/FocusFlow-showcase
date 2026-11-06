"""Tests for the viewing-plan ruling section injected into the S6 fit judge.

Captures the prompt prefix passed to _judge_batch; no external calls are made.
"""
import asyncio
import os
import tempfile
os.environ.setdefault("FOCUSFLOW_DB_PATH", os.path.join(tempfile.mkdtemp(), "boot.db"))
import pytest
import config
from agent import judges
from core.models import Video
from core.viewing_plan import PlanRuling


def _v(i):
    return Video(video_id=f"v{i}", channel_id="c", title=f"t{i}", duration_sec=600,
                 source_strategy="search", intent_sourced=True)


def _capture(monkeypatch):
    seen = {}

    async def _fake_judge_batch(get_agent, batch, channel_titles, degrade, validate, *,
                                prompt_prefix="", **kw):
        seen["prefix"] = prompt_prefix
        return []

    monkeypatch.setattr(judges, "_judge_batch", _fake_judge_batch)
    return seen


async def _drain(gen):
    return [b async for b in gen]


def test_plan_ruling_前缀进prompt且不带画像负偏好段(monkeypatch):
    seen = _capture(monkeypatch)
    ruling = PlanRuling(plan_name="ML 入门", stage_title="第 1 周", stage_goal="理解线性回归",
                        level="零基础", taste_lines=["英文讲解优先"])
    asyncio.run(_drain(judges.judge_mode_fit(
        [_v(1)], config.MODES["focus"], plan_ruling=ruling,
        profile_lines=["不该出现的画像行"], neg_pref_lines=["不该出现的负偏好"])))
    p = seen["prefix"]
    assert "【计划裁定】" in p and "理解线性回归" in p and "英文讲解优先" in p
    assert "不该出现" not in p


def test_plan_ruling_与意图互斥():
    ruling = PlanRuling(plan_name="a", stage_title="b", stage_goal="c", level="")
    with pytest.raises(ValueError):
        asyncio.run(_drain(judges.judge_mode_fit(
            [_v(1)], config.MODES["focus"], intent_text="x", plan_ruling=ruling)))


def test_无plan_ruling_行为不变(monkeypatch):
    seen = _capture(monkeypatch)
    asyncio.run(_drain(judges.judge_mode_fit([_v(1)], config.MODES["focus"],
                                             profile_lines=["画像行"])))
    assert "画像行" in seen["prefix"] and "【计划裁定】" not in seen["prefix"]
