"""Long-term viewing plans: shared models and pure logic.

Pure models and functions (no I/O, no db, no agent). Provides the shared types
for viewing plans plus the pure pieces of logic: locating the current stage,
sanitising an LLM-produced draft, extracting taste hints from the profile, and
the summary/detail projections. Used by the planning agent, the S6 judge, the
plan-run pipeline step and the API layer.

A plan has one of two kinds: ``progressive`` (a course that goes from basic to
advanced) or ``variety`` (non-repeating sub-topics rotating under one theme).
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

import config
from core.profile_sections import long_texts, split_sections


class PlanStage(BaseModel):
    """One plan stage: title, length in days, search keywords and goal."""
    title: str
    days: int
    keywords: list[str]
    goal: str


class PlanDraft(BaseModel):
    """A plan draft as produced by the planning agent or the edit path."""
    name: str
    goal: str
    # kind: progressive = basic-to-advanced course; variety = non-repeating
    # sub-topics under one theme. theme is a short phrase used by profile signals,
    # the judge prompt and UI subtitles; empty falls back to name. level is the
    # user's background (baseline for learning plans, taste otherwise). Older rows
    # lack kind/theme and read as progressive / theme=name.
    # kind is a Literal rather than str because this class doubles as the
    # structured-output schema: the enum must appear in the tool schema to
    # constrain the LLM, otherwise near-miss values would silently fall back to
    # progressive in sanitize_draft and the user's intent would be lost.
    kind: Literal["progressive", "variety"] = "progressive"
    theme: str = ""
    level: str = ""
    daily_count: int = config.VIEWING_PLAN_DAILY_DEFAULT
    stages: list[PlanStage]


class PlanDraftTurn(BaseModel):
    """One planning-agent turn: reply text, current draft (optional), and whether
    the draft is ready for review."""
    reply: str
    draft: PlanDraft | None = None
    ready: bool = False


class Located(BaseModel):
    """Result of locate_stage: current stage and day position within the plan."""
    stage_index: int
    stage: PlanStage
    day_index: int
    total_days: int
    finished: bool


class PlanRuling(BaseModel):
    """Inputs for the plan-ruling section prepended to the S6 judge prompt.
    The criteria text differs by plan kind."""
    plan_name: str
    stage_title: str
    stage_goal: str
    level: str
    kind: str = "progressive"
    theme: str = ""
    taste_lines: list[str] = []

    def prompt_prefix(self) -> str:
        theme = self.theme or self.plan_name
        # Progressive plans: fit the stage goal and match the user's level.
        # Variety plans: belong to the theme and this stage's sub-topic (no ordering, no difficulty).
        if self.kind == "variety":
            criterion = (f"内容属于主题「{theme}」且属于本阶段子主题即 fit=true,"
                         "按贴合度给 fit_score")
        else:
            criterion = ("内容与阶段目标贴合、难度与用户背景匹配即 fit=true,"
                         "按贴合度与难度匹配度给 fit_score")
        head = (
            f"【计划裁定】用户正在执行长期观看计划「{self.plan_name}」(主题:{theme}),"
            f"当前阶段「{self.stage_title}」,阶段目标:{self.stage_goal}。"
            f"用户背景:{self.level or '未说明'}。本批视频均为该阶段"
            f"关键词召回。判定标准改为:{criterion};【不再按本模式常规判据拒绝】。仍然照常:"
            "content_type 判类型;哗众取宠/空洞内容照拒。fit=true 的 reason 用「本阶段」口吻"
            "点明它对应阶段目标的哪一点。")
        if self.taste_lines:
            joined = ";".join(self.taste_lines[:2])
            head += (f"同阶段候选取舍时可参考用户口味:{joined}。口味只用于取舍,"
                     "不得因口味否决贴合阶段的视频。")
        return head + "\n\n"


def locate_stage(stages: list[PlanStage], start_date: str, day_offset: int,
                 today: str) -> Located:
    """day_index = (today − start) + offset, clamped to ≥ 0; the stage whose day
    range contains it is current. Past the end → last stage with finished=True."""
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(today)
    day_index = max(0, (d1 - d0).days + day_offset)
    total = sum(max(1, s.days) for s in stages)
    cum = 0
    for i, st in enumerate(stages):
        if day_index < cum + max(1, st.days):
            return Located(stage_index=i, stage=st, day_index=day_index,
                           total_days=total, finished=False)
        cum += max(1, st.days)
    return Located(stage_index=len(stages) - 1, stage=stages[-1], day_index=day_index,
                   total_days=total, finished=True)


def sanitize_draft(draft: PlanDraft) -> PlanDraft | None:
    """Normalise untrusted LLM output: trim keywords, clamp numbers, truncate names.
    Returns None (draft invalid) if there are no stages or any stage has no
    keywords left after cleaning."""
    stages: list[PlanStage] = []
    for st in draft.stages[: config.VIEWING_PLAN_STAGES_MAX]:
        seen: list[str] = []
        for k in st.keywords:
            # Cap keyword length: the LLM occasionally emits a whole goal sentence as
            # a keyword, which finds nothing yet still spends a search call.
            k = k.strip()[: config.VIEWING_PLAN_KEYWORD_MAX]
            if k and k not in seen:
                seen.append(k)
        if not seen:
            return None
        stages.append(PlanStage(
            title=st.title.strip()[:60] or "阶段",
            days=max(1, int(st.days)),
            keywords=seen[: config.VIEWING_PLAN_KEYWORDS_PER_STAGE],
            goal=st.goal.strip()[:200]))
    if not stages:
        return None
    # Cap total days: trim from the last stage backwards, keeping each stage ≥ 1 day.
    over = sum(s.days for s in stages) - config.VIEWING_PLAN_DAYS_MAX
    for st in reversed(stages):
        if over <= 0:
            break
        cut = min(over, st.days - 1)
        st.days -= cut
        over -= cut
    # Invalid kind falls back to progressive (the Literal already guards normal
    # input; this covers paths that bypass validation, e.g. model_construct).
    # theme is capped at 40 chars and falls back to the cleaned name, so
    # downstream consumers can always treat it as non-empty.
    name = draft.name.strip()[: config.VIEWING_PLAN_NAME_MAX] or "我的计划"
    return PlanDraft(
        name=name,
        goal=draft.goal.strip()[:200],
        kind=draft.kind if draft.kind in ("progressive", "variety") else "progressive",
        theme=draft.theme.strip()[:40] or name,
        level=draft.level.strip()[:100],
        daily_count=min(config.VIEWING_PLAN_DAILY_MAX, max(1, int(draft.daily_count))),
        stages=stages)


def taste_lines(raw_text: str, mode: str) -> list[str]:
    """Taste hints for the judge: the first line of the user's own profile section
    plus the first long-term entry for this mode, at most two; [] if none."""
    if not raw_text:
        return []
    out: list[str] = []
    own, _short, _long = split_sections(raw_text)
    for line in own.splitlines():
        line = line.strip().lstrip("-").strip()
        if line:
            out.append(line[:80])
            break
    longs = long_texts(raw_text, mode)
    if longs:
        out.append(longs[0][:80])
    return out[:2]


def plan_summary(plan: dict, today: str) -> dict:
    """Lightweight projection for the active-plan badge/list. ``plan`` is the dict
    returned by the storage layer, with stages already parsed into list[dict]."""
    stages = [PlanStage(**s) for s in plan["stages"]]
    loc = locate_stage(stages, plan["start_date"], plan.get("day_offset", 0), today)
    return {"id": plan["id"], "mode": plan["mode"], "name": plan["name"],
            # Older rows without kind/theme → progressive / theme=name
            "kind": plan.get("kind") or "progressive",
            "theme": plan.get("theme") or plan["name"],
            "day_index": loc.day_index, "total_days": loc.total_days,
            "finished": loc.finished, "stage_title": loc.stage.title}


# ================================================================== #
# Editing an active plan
# ================================================================== #
_EDITABLE = ("name", "goal", "daily_count", "stages")


def merge_edit(plan: dict, patch: dict) -> PlanDraft | None:
    """Apply a user edit to an existing plan. Only name/goal/daily_count/stages are
    editable; kind/theme/level always come from the stored row (the theme was
    already written into the profile on activation, and changing kind would
    change the meaning of every stage — either amounts to a new plan). The result
    goes through the same sanitize_draft as newly created plans, so both paths
    obey identical constraints. None = invalid draft."""
    draft = PlanDraft(
        name=str(patch.get("name", "")),
        goal=str(patch.get("goal", "")),
        kind=plan.get("kind") or "progressive",
        theme=plan.get("theme") or "",
        level=plan.get("level") or "",
        daily_count=int(patch.get("daily_count", config.VIEWING_PLAN_DAILY_DEFAULT)),
        stages=[PlanStage(**s) if not isinstance(s, PlanStage) else s
                for s in patch.get("stages", [])])
    return sanitize_draft(draft)


def plan_detail(plan: dict, today: str) -> dict:
    """Full projection for the plan detail view (superset of plan_summary): full
    stages including keywords, start_date/day_offset (the frontend derives each
    stage's date range from them), and regen_used_today (the once-per-day
    "regenerate today's cards" flag)."""
    out = plan_summary(plan, today)
    stages = [PlanStage(**s) for s in plan["stages"]]
    loc = locate_stage(stages, plan["start_date"], plan.get("day_offset", 0), today)
    out.update({
        "goal": plan.get("goal", ""),
        "level": plan.get("level", ""),
        "daily_count": int(plan["daily_count"]),
        "stages": [s.model_dump() for s in stages],
        "start_date": plan["start_date"],
        "day_offset": int(plan.get("day_offset", 0)),
        "stage_index": loc.stage_index,
        "regen_used_today": plan.get("last_regen_day") == today,
    })
    return out
