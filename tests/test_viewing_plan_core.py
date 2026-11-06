"""Tests for viewing-plan core pure functions: stage location, draft sanitizing,
taste summaries, ruling prefixes, edit merging and detail views. No I/O.
"""
import os
import tempfile
os.environ.setdefault("FOCUSFLOW_DB_PATH", os.path.join(tempfile.mkdtemp(), "boot.db"))
import pytest
from pydantic import ValidationError

import config
from core.viewing_plan import (Located, PlanDraft, PlanRuling, PlanStage,
                               locate_stage, plan_summary, sanitize_draft, taste_lines)


def _stages(days=(3, 4, 5)):
    return [PlanStage(title=f"阶段{i}", days=d, keywords=[f"k{i}"], goal=f"目标{i}")
            for i, d in enumerate(days)]


def test_locate_stage_按日历落段():
    st = _stages()
    assert locate_stage(st, "2026-09-06", 0, "2026-09-06").stage_index == 0
    assert locate_stage(st, "2026-09-06", 0, "2026-09-08").stage_index == 0
    loc = locate_stage(st, "2026-09-06", 0, "2026-09-09")
    assert loc.stage_index == 1 and loc.day_index == 3 and loc.total_days == 12
    assert locate_stage(st, "2026-09-06", 0, "2026-09-17").stage_index == 2
    assert not locate_stage(st, "2026-09-06", 0, "2026-09-17").finished


def test_locate_stage_越界末段finished与负偏移():
    st = _stages()
    loc = locate_stage(st, "2026-09-06", 0, "2026-09-18")    # day 12 is past the end
    assert loc.stage_index == 2 and loc.finished
    loc = locate_stage(st, "2026-09-06", -2, "2026-09-09")   # day 3 - 2 = 1 -> stage 0
    assert loc.stage_index == 0 and loc.day_index == 1
    loc = locate_stage(st, "2026-09-06", 0, "2026-09-01")    # negative day clamped to 0
    assert loc.day_index == 0 and loc.stage_index == 0


def test_sanitize_draft_裁剪钳制去重():
    d = PlanDraft(name="x" * 40, goal="g", level="", daily_count=15, stages=[
        PlanStage(title="a", days=0, keywords=["k1", " ", "k1", "k2", "k3", "k4", "k5", "k6"], goal="g1"),
        PlanStage(title="b", days=2, keywords=["k9"], goal="g2"),
    ])
    out = sanitize_draft(d)
    assert out is not None
    assert len(out.name) == config.VIEWING_PLAN_NAME_MAX
    assert out.daily_count == config.VIEWING_PLAN_DAILY_MAX
    assert out.stages[0].days == 1
    assert out.stages[0].keywords == ["k1", "k2", "k3", "k4", "k5"]


def test_sanitize_draft_关键词截60字():
    """The LLM occasionally emits a whole goal sentence as a keyword; overly long
    queries return nothing useful and still cost search quota, so they are truncated."""
    long_kw = "线" * 80
    d = PlanDraft(name="n", goal="g", level="l", daily_count=3,
                  stages=[PlanStage(title="t", days=3, keywords=[" " + long_kw + " ", "短词"],
                                    goal="g")])
    out = sanitize_draft(d)
    assert out.stages[0].keywords[0] == "线" * config.VIEWING_PLAN_KEYWORD_MAX
    assert len(out.stages[0].keywords[0]) == 60 and out.stages[0].keywords[1] == "短词"


def test_sanitize_draft_零阶段或零关键词返回None():
    assert sanitize_draft(PlanDraft(name="n", goal="g", daily_count=5, stages=[])) is None
    only_blank = PlanDraft(name="n", goal="g", daily_count=5, stages=[
        PlanStage(title="a", days=1, keywords=["  "], goal="g")])
    assert sanitize_draft(only_blank) is None


def test_sanitize_draft_daily_count_下钳与总天数上限():
    d = PlanDraft(name="n", goal="g", daily_count=0, stages=[
        PlanStage(title="a", days=100, keywords=["k"], goal="g")])
    out = sanitize_draft(d)
    assert out.daily_count == 1
    assert sum(s.days for s in out.stages) <= config.VIEWING_PLAN_DAYS_MAX


def test_taste_lines_空与两条上限():
    from core.profile_sections import LONG_HEADER
    assert taste_lines("", "focus") == []
    raw = ("- 英文讲解优先\n- 不喜欢 vlog\n\n" + LONG_HEADER + "\n"
           "- [2026-09-01|x3|focus] 喜欢可视化讲解\n- [2026-09-01|x1|wind_down] 喜欢白噪音")
    lines = taste_lines(raw, "focus")
    assert len(lines) == 2
    assert lines[0] == "英文讲解优先"
    assert "可视化" in lines[1]
    assert not any("白噪音" in l for l in lines)


def test_plan_ruling_prefix_含阶段目标与口味句可选():
    r = PlanRuling(plan_name="ML 入门", stage_title="第 1 周", stage_goal="理解线性回归",
                   level="零基础", taste_lines=[])
    p = r.prompt_prefix()
    assert "【计划裁定】" in p and "理解线性回归" in p and "零基础" in p
    assert "口味" not in p
    r2 = r.model_copy(update={"taste_lines": ["英文讲解优先"]})
    assert "英文讲解优先" in r2.prompt_prefix() and "不得因口味否决" in r2.prompt_prefix()


def test_plan_summary_形状():
    plan = {"id": 7, "mode": "focus", "name": "ML 入门", "start_date": "2026-09-06",
            "day_offset": 0, "stages": [s.model_dump() for s in _stages()]}
    s = plan_summary(plan, "2026-09-09")
    # legacy rows without kind/theme default to progressive and theme=name on read
    assert s == {"id": 7, "mode": "focus", "name": "ML 入门", "kind": "progressive",
                 "theme": "ML 入门", "day_index": 3,
                 "total_days": 12, "finished": False, "stage_title": "阶段1"}


# ================================================================== #
# Plan kinds (progressive / variety) and theme
# ================================================================== #
def test_sanitize_draft_kind与theme校验():
    base = dict(name="脱口秀轮播", goal="g", level="爱看单口",
                stages=[PlanStage(title="a", days=3, keywords=["standup"], goal="g1")])
    ok = sanitize_draft(PlanDraft(**base, kind="variety", theme="脱口秀"))
    assert ok.kind == "variety" and ok.theme == "脱口秀"
    # kind is a Literal: pydantic rejects invalid values at the boundary
    # (both for structured LLM output and API input)
    with pytest.raises(ValidationError):
        PlanDraft(**base, kind="随便")
    # paths that bypass validation (model_construct) still fall back to progressive
    assert sanitize_draft(PlanDraft.model_construct(**base, kind="随便", theme="")).kind \
        == "progressive"
    assert sanitize_draft(PlanDraft.model_construct(**base, kind="", theme="")).kind \
        == "progressive"
    # theme truncated to 40 chars; blank theme falls back to the sanitized name
    assert sanitize_draft(PlanDraft(**base, theme="主" * 60)).theme == "主" * 40
    assert sanitize_draft(PlanDraft(**base, theme="   ")).theme == "脱口秀轮播"
    assert sanitize_draft(PlanDraft(**dict(base, name="n" * 40), theme="")).theme \
        == "n" * config.VIEWING_PLAN_NAME_MAX


def test_plan_ruling_prefix_两态文案():
    prog = PlanRuling(plan_name="ML 入门", stage_title="第 1 周", stage_goal="理解线性回归",
                      level="零基础", kind="progressive", theme="机器学习")
    p = prog.prompt_prefix()
    assert "长期观看计划" in p and "主题:机器学习" in p
    assert "难度与用户背景匹配" in p and "用户背景:零基础" in p
    assert "子主题" not in p
    var = prog.model_copy(update={"kind": "variety", "theme": "脱口秀"})
    v = var.prompt_prefix()
    assert "属于主题「脱口秀」" in v and "本阶段子主题" in v
    assert "难度" not in v
    # empty theme -> falls back to the plan name (legacy plans)
    assert "主题:ML 入门" in prog.model_copy(update={"theme": ""}).prompt_prefix()


# ================================================================== #
# Plan directory: merge_edit / plan_detail
# ================================================================== #
from core.viewing_plan import merge_edit, plan_detail  # noqa: E402


def _stored_plan(**over):
    base = {"id": 7, "mode": "focus", "name": "ML 入门", "goal": "g", "kind": "variety",
            "theme": "机器学习", "level": "零基础", "daily_count": 5, "start_date": "2026-09-06",
            "day_offset": 0, "status": "active", "last_regen_day": None,
            "stages": [s.model_dump() for s in _stages()]}
    base.update(over)
    return base


def test_merge_edit_只改四字段_kind_theme_level_取存量并过sanitize():
    patch = {"name": "  改名 ", "goal": "新目标", "daily_count": 99,
             "kind": "progressive", "theme": "别的", "level": "别的",
             "stages": [{"title": "a", "days": 0, "keywords": ["k", "k", ""], "goal": "x"}]}
    d = merge_edit(_stored_plan(), patch)
    assert d.name == "改名" and d.goal == "新目标"
    assert d.daily_count == config.VIEWING_PLAN_DAILY_MAX     # clamped
    assert d.kind == "variety" and d.theme == "机器学习" and d.level == "零基础"   # the three read-only fields ignore the patch
    assert d.stages[0].days == 1 and d.stages[0].keywords == ["k"]


def test_merge_edit_存量无theme回落name_零关键词返回None():
    d = merge_edit(_stored_plan(theme="", kind=None), {"name": "N", "goal": "", "daily_count": 3,
                                                       "stages": [{"title": "a", "days": 2, "keywords": ["q"], "goal": ""}]})
    assert d.theme == "N" and d.kind == "progressive"
    assert merge_edit(_stored_plan(), {"name": "N", "goal": "", "daily_count": 3,
                                       "stages": [{"title": "a", "days": 2, "keywords": [""], "goal": ""}]}) is None


def test_plan_detail_形状_与_regen_used_today():
    p = _stored_plan(last_regen_day="2026-09-09")
    d = plan_detail(p, "2026-09-09")
    assert d["id"] == 7 and d["kind"] == "variety" and d["theme"] == "机器学习"
    assert d["day_index"] == 3 and d["total_days"] == 12 and d["stage_index"] == 1
    assert d["start_date"] == "2026-09-06" and d["day_offset"] == 0
    assert d["daily_count"] == 5 and d["level"] == "零基础" and d["goal"] == "g"
    assert [s["title"] for s in d["stages"]] == ["阶段0", "阶段1", "阶段2"]
    assert d["regen_used_today"] is True and d["finished"] is False
    assert plan_detail(_stored_plan(), "2026-09-09")["regen_used_today"] is False
