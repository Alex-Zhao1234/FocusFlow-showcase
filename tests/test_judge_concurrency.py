"""Tests for judge-layer LLM concurrency.

Covers the concurrency gate (a lazily built semaphore sized by
``config.LLM_MAX_CONCURRENCY``), inter-batch parallelism in the fit/quality
judges, order-preserving streaming and cancellation, and the orchestrator's
first-wave truncation and whole-batch draining helpers.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FOCUSFLOW_DB_PATH", os.path.join(tempfile.mkdtemp(), "boot.db"))

import config                                                        # noqa: E402
from agent import judges                                             # noqa: E402
from core.models import QualityVerdict, Video                        # noqa: E402


# ------------------------------------------------------------------ #
# Helpers: fake videos, a well-behaved fake LLM and an offline judge layer.
# ------------------------------------------------------------------ #
def _mk(vid: str, **kw) -> Video:
    base = dict(channel_id="ch", title="t", duration_sec=600,
                source_strategy="search", view_count=1000, like_count=0,
                comment_count=0)
    base.update(kw)
    return Video(video_id=vid, **base)


def _mk_batch(n: int) -> list[Video]:
    return [_mk(f"v{i:02d}") for i in range(n)]


def _prompt_ids(prompt: str) -> list[str]:
    import json as _json
    return [item["id"] for item in _json.loads(prompt[prompt.index("["):])]


def _echo(prompt: str, *, grade=3, confidence=0.9, degraded=False) -> list:
    """Well-behaved fake LLM: one verdict per id in the prompt, full grade."""
    return [QualityVerdict(video_id=i, passed=True, quality_grade=grade,
                           confidence=confidence, degraded=degraded, reason="ok")
            for i in _prompt_ids(prompt)]


@pytest.fixture()
def _offline(monkeypatch):
    """Offline judge layer: records model overrides and prompts; handler is swappable."""
    holder = {"handler": _echo, "calls": [], "models": []}

    def _fake_get_agent(prompt_name, mode, output_type, model_override=None):
        holder["models"].append(model_override)
        return None

    async def _fake_call(agent, prompt):
        holder["calls"].append(prompt)
        h = holder["handler"]
        if isinstance(h, Exception):
            raise h
        return h(prompt)

    monkeypatch.setattr(judges, "_get_agent", _fake_get_agent)
    monkeypatch.setattr(judges, "_call_agent", _fake_call)
    return holder


def _collect(agen) -> list:
    async def _run():
        return [b async for b in agen]
    return asyncio.run(_run())


@pytest.fixture(autouse=True)
def _restore_llm_concurrency():
    """Several tests mutate config.LLM_MAX_CONCURRENCY directly. Restore it in a
    finally block so a failing assertion cannot leak the value into later tests
    in the same process."""
    original = config.LLM_MAX_CONCURRENCY
    try:
        yield
    finally:
        config.LLM_MAX_CONCURRENCY = original


def test_闸值可调且按需重建():
    """The gate is built lazily from config.LLM_MAX_CONCURRENCY and rebuilt on
    next access after the config changes, so tests need no manual reset."""
    config.LLM_MAX_CONCURRENCY = 4
    s1 = judges._llm_semaphore()
    assert s1 is judges._llm_semaphore(), "同闸值下应复用同一实例"
    config.LLM_MAX_CONCURRENCY = 2
    s2 = judges._llm_semaphore()
    assert s2 is not s1, "闸值变化应重建"


def test_受闸出口真的限流():
    """Under concurrent _call_agent_guarded calls, in-flight calls never exceed the gate."""
    config.LLM_MAX_CONCURRENCY = 2
    judges._llm_semaphore()                  # force rebuild
    live = 0
    peak = 0

    async def _fake_call(agent, prompt):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return []

    async def _drive():
        orig = judges._call_agent
        judges._call_agent = _fake_call
        try:
            await asyncio.gather(*[
                judges._call_agent_guarded(None, f"p{i}") for i in range(8)])
        finally:
            judges._call_agent = orig

    asyncio.run(_drive())
    assert peak <= 2, f"峰值并发 {peak} 超过闸值 2"


def test_闸值1等于完全串行():
    """Rollback switch: with LLM_MAX_CONCURRENCY=1 the peak is always 1 (fully serial)."""
    config.LLM_MAX_CONCURRENCY = 1
    judges._llm_semaphore()
    live = 0
    peak = 0

    async def _fake_call(agent, prompt):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return []

    async def _drive():
        orig = judges._call_agent
        judges._call_agent = _fake_call
        try:
            await asyncio.gather(*[
                judges._call_agent_guarded(None, f"p{i}") for i in range(5)])
        finally:
            judges._call_agent = orig

    asyncio.run(_drive())
    assert peak == 1


def test_fit批间真并发(_offline):
    """judge_mode_fit dispatches batches concurrently: peak > 1 (bounded by the gate)."""
    config.LLM_MAX_CONCURRENCY = 4
    judges._llm_semaphore()
    live = 0
    peak = 0
    orig = judges._call_agent

    async def _slow(agent, prompt):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.03)
        live -= 1
        return await orig(agent, prompt)

    judges._call_agent = _slow
    try:
        videos = _mk_batch(45)                       # 3 batches (LLM_BATCH_SIZE=15)
        batches = _collect(judges.judge_mode_fit(videos, config.EXPLORE))
    finally:
        judges._call_agent = orig
    assert len(batches) == 3, "批数不变"
    assert peak >= 2, f"峰值并发 {peak},批间未并发"


def test_并发结果与串行逐位一致(_offline):
    """Ordering is preserved: the same input yields identical batch order and
    in-batch item order whether run serially or concurrently. The ranker's
    stable sort depends on insertion order; drift would randomize tie-breaks
    at the TOP_N boundary."""
    videos = _mk_batch(45)

    config.LLM_MAX_CONCURRENCY = 1
    judges._llm_semaphore()
    serial = _collect(judges.judge_mode_fit(videos, config.EXPLORE))

    config.LLM_MAX_CONCURRENCY = 4
    judges._llm_semaphore()
    concurrent = _collect(judges.judge_mode_fit(videos, config.EXPLORE))

    assert [[vd.video_id for vd in b] for b in serial] \
        == [[vd.video_id for vd in b] for b in concurrent]


def test_quality批间真并发(_offline):
    """judge_quality is concurrent too (shares the same gate as fit)."""
    config.LLM_MAX_CONCURRENCY = 4
    judges._llm_semaphore()
    live = 0
    peak = 0
    orig = judges._call_agent

    async def _slow(agent, prompt):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.03)
        live -= 1
        return await orig(agent, prompt)

    judges._call_agent = _slow
    try:
        videos = _mk_batch(30)                       # 2 batches
        stat = {v.video_id: 0.5 for v in videos}
        batches = _collect(judges.judge_quality(videos, config.EXPLORE,
                                                stat_scores=stat))
    finally:
        judges._call_agent = orig
    assert len(batches) == 2
    assert peak >= 2


def test_逐批流式吐(_offline):
    """Finished batches are yielded immediately instead of waiting for the
    slowest one. Running everything to completion before yielding would make
    the first batch arrive as late as the last.

    Three batches are built and only batches 2/3 are slowed down; the gap
    between first and last arrival must be clearly positive. If results were
    buffered and emitted at once, the gap would approach zero."""
    config.LLM_MAX_CONCURRENCY = 4
    judges._llm_semaphore()
    videos = _mk_batch(45)                       # 3 batches (LLM_BATCH_SIZE=15)
    slow_ids = {v.video_id for v in videos[15:]}  # batches 2/3
    orig = judges._call_agent

    async def _selective_slow(agent, prompt):
        if slow_ids & set(_prompt_ids(prompt)):
            await asyncio.sleep(0.08)
        return await orig(agent, prompt)

    judges._call_agent = _selective_slow
    arrivals: list[float] = []

    async def _drive():
        start = time.monotonic()
        async for _b in judges.judge_mode_fit(videos, config.EXPLORE):
            arrivals.append(time.monotonic() - start)

    try:
        asyncio.run(_drive())
    finally:
        judges._call_agent = orig

    assert len(arrivals) == 3
    assert arrivals[-1] - arrivals[0] > 0.05, \
        f"首末批到达间隔仅 {arrivals[-1] - arrivals[0]:.3f}s,疑似退化为攒批一次吐"


def test_取消不丢已算批且不再发起新调用(_offline):
    """The consumer closes the generator right after the first batch: the
    yielded batch stays intact and no new LLM calls are started afterwards.
    asyncio.CancelledError derives from BaseException, so it is not swallowed
    by the ``except Exception`` in _judge_batch; cleanup relies on the
    cancel + gather in the generator's finally block.

    Batches 2/3 sleep before their call is issued, so at aclose time they have
    not yet called the LLM. Cancellation must ensure they never do, rather
    than merely interrupting them mid-call."""
    config.LLM_MAX_CONCURRENCY = 4
    judges._llm_semaphore()
    videos = _mk_batch(45)                       # 3 batches
    slow_ids = {v.video_id for v in videos[15:]}  # batches 2/3
    orig = judges._call_agent

    async def _selective_slow(agent, prompt):
        if slow_ids & set(_prompt_ids(prompt)):
            await asyncio.sleep(0.5)              # generous margin so cancellation wins
        return await orig(agent, prompt)

    judges._call_agent = _selective_slow

    async def _drive():
        agen = judges.judge_mode_fit(videos, config.EXPLORE)
        first = await agen.__anext__()
        calls_at_first = len(_offline["calls"])
        await agen.aclose()
        return first, calls_at_first

    try:
        first, calls_at_first = asyncio.run(_drive())
    finally:
        judges._call_agent = orig

    assert [vd.video_id for vd in first] == [v.video_id for v in videos[:15]], \
        "已 yield 的首批数据应完好无损"
    assert calls_at_first == 1, "首批到达时,慢批不应已发起调用"
    assert len(_offline["calls"]) == calls_at_first, \
        "aclose 取消后不应再有新的 LLM 调用发起"


# ------------------------------------------------------------------ #
# First-wave truncation
# ------------------------------------------------------------------ #
def test_首波截断_按统计分取前N():
    """Keep the top N by statistical quality score; the rest are deferred, not dropped."""
    from pipeline.orchestrator import _truncate_first_wave
    vs = _mk_batch(10)
    score = {v.video_id: float(i) for i, v in enumerate(vs)}   # later items score higher
    kept, deferred = _truncate_first_wave(vs, lambda v: score[v.video_id], 3)
    assert len(kept) == 3 and len(deferred) == 7
    assert [v.video_id for v in kept] == [vs[9].video_id, vs[8].video_id,
                                          vs[7].video_id], "取分最高的三条"
    assert set(v.video_id for v in kept) | set(v.video_id for v in deferred) \
        == set(v.video_id for v in vs), "顺延者不得丢失"


def test_首波截断_None不截断():
    """Rollback switch: FIRST_WAVE_MAX=None disables truncation and keeps input order."""
    from pipeline.orchestrator import _truncate_first_wave
    vs = _mk_batch(10)
    kept, deferred = _truncate_first_wave(vs, lambda v: 1.0, None)
    assert [v.video_id for v in kept] == [v.video_id for v in vs], "原序原样返回"
    assert deferred == []


def test_首波截断_不足N时全留():
    from pipeline.orchestrator import _truncate_first_wave
    vs = _mk_batch(3)
    kept, deferred = _truncate_first_wave(vs, lambda v: 1.0, 30)
    assert len(kept) == 3 and deferred == []


# ------------------------------------------------------------------ #
# Follow-up waves
# ------------------------------------------------------------------ #






def test_后续波一次送尽整批():
    """Follow-up waves send all complete batches at once (not one batch at a
    time) so the judges' inter-batch concurrency actually kicks in. Verified
    on the pure splitting helper without driving the whole pipeline."""
    from pipeline.orchestrator import _drain_whole_batches
    # 75 items: send all 75 (5 full batches), nothing left
    sent, rest = _drain_whole_batches(list(range(75)), 15)
    assert len(sent) == 75 and rest == []
    # 80 items: send 75, keep the remaining 5 buffered
    sent, rest = _drain_whole_batches(list(range(80)), 15)
    assert len(sent) == 75 and len(rest) == 5
    # less than one batch: send nothing (never send partial batches)
    sent, rest = _drain_whole_batches(list(range(9)), 15)
    assert sent == [] and len(rest) == 9
    # exactly one batch: send it
    sent, rest = _drain_whole_batches(list(range(15)), 15)
    assert len(sent) == 15 and rest == []
