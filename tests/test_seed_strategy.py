"""Tests for the seed-channel retrieval strategy.

Covers:
  - config.SOURCE_PRIORITY ranks seed_channel highest (-1)
  - core.costing.estimate_plan_cost prices the use_seed_channels branch
  - sources.youtube._run_strategy's use_seed_channels branch (relabels to
    seed_channel, fetches channels concurrently, one failing channel does not
    block the rest)
  - sources.youtube.dedup keeps seed_channel on video_id collisions
  - pipeline.mix._side / apply_pool_ratio leave seed strategies untouched
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

os.environ.setdefault("FOCUSFLOW_SESSION_SECRET", "test-secret")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FOCUSFLOW_DB_PATH", os.path.join(tempfile.mkdtemp(), "boot.db"))

import config  # noqa: E402
from core.costing import estimate_plan_cost  # noqa: E402
from core.models import RetrievalPlan, StrategyCall, Video  # noqa: E402


def test_source_priority_seed_on_top():
    assert config.SOURCE_PRIORITY["seed_channel"] == -1     # highest priority
    assert min(config.SOURCE_PRIORITY.values()) == -1


def test_costing_counts_seed_channels():
    plan = RetrievalPlan(mode="focus", strategies=[StrategyCall(
        tool="fetch_channel_videos",
        params={"use_seed_channels": True, "seed_channel_ids": ["UCa", "UCb"],
                "n_per_channel": 3},
        target_count=6, rationale="t")], est_quota=6)
    # 2 channels × 1u × CHANNEL_DEEP_PAGES_MAX (worst-case deep paging) + enrich ⌈6/50⌉ × 1u
    assert estimate_plan_cost(plan).general == 2 * config.CHANNEL_DEEP_PAGES_MAX + 1


class _FakeAdapter:
    user_id = 1

    async def fetch_channel_videos(self, channel_id, n):
        if channel_id == "UCbad" + "x" * 16:
            raise RuntimeError("boom")
        return [Video(video_id=f"vid_{channel_id[-6:]}{i:05d}"[:11].ljust(11, "x"),
                      channel_id=channel_id, title="t", description="",
                      duration_sec=600, source_strategy="subscribed")
                for i in range(min(n, 2))]


def test_run_strategy_seed_branch(monkeypatch):
    from sources.youtube import _run_strategy
    good_a = "UCa" + "a" * 20
    good_b = "UCb" + "b" * 20
    bad = "UCbad" + "x" * 16
    fake = _FakeAdapter()
    s = StrategyCall(tool="fetch_channel_videos",
                     params={"use_seed_channels": True,
                             "seed_channel_ids": [good_a, good_b, bad],
                             "n_per_channel": 2},
                     target_count=10, rationale="测试")
    videos = asyncio.run(_run_strategy(fake, s))
    # a failing channel is skipped; good_a/good_b still return results
    assert videos, "坏频道不应导致整批返回空"
    assert all(v.source_strategy == "seed_channel" for v in videos), "seed 分支须重标 seed_channel"
    assert {v.channel_id for v in videos} == {good_a, good_b}, "异常频道被跳过,不进结果"


def test_dedup_seed_wins_subscribed():
    from sources.youtube import dedup
    a = Video(video_id="vid_dup00001", channel_id="UC" + "z" * 22, title="", description="",
              duration_sec=1, source_strategy="subscribed")
    b = Video(video_id="vid_dup00001", channel_id="UC" + "z" * 22, title="", description="",
              duration_sec=1, source_strategy="seed_channel")
    kept = dedup([a, b])
    assert kept[0].source_strategy == "seed_channel", "seed_channel 置顶:撞车须保钦点归因"


# ================================================================== #
# Seed strategies add no extra seat type in source mixing
# ================================================================== #
def test_seed_neutral_in_source_mix():
    from pipeline.mix import _side
    from pipeline.ranker import RankedItem
    v = Video(video_id="vid_seed00001", channel_id="UC" + "s" * 22, title="t",
              description="", duration_sec=600, source_strategy="seed_channel")
    item = RankedItem(video=v, fit_score=0.9, quality_score_norm=0.9,
                      channel_pref_score=0.5, is_gem=False, final=0.9,
                      content_type=None, type_w=1.0)
    assert _side(item) is None, "seed_channel 卡不入两侧、不参与屏上软保底换位"


def test_apply_pool_ratio_ignores_seed_strategy():
    from pipeline.mix import apply_pool_ratio
    seed_s = StrategyCall(tool="fetch_channel_videos",
                          params={"use_seed_channels": True,
                                  "seed_channel_ids": ["UCa", "UCb"],
                                  "n_per_channel": 3},
                          target_count=6, rationale="测试")
    top_s = StrategyCall(tool="fetch_channel_videos",
                         params={"use_top_channels": True, "n_channels": 10},
                         target_count=120, rationale="测试")
    plan = RetrievalPlan(mode="focus", strategies=[top_s, seed_s], est_quota=126)
    p1 = apply_pool_ratio(plan, 0.5, has_pref_channels=True)
    seeds = [s for s in p1.strategies if s.params.get("use_seed_channels")]
    assert len(seeds) == 1, "seed 策略应原样保留,不被滑动条重分/移除"
    assert seeds[0].target_count == 6 and seeds[0].params == seed_s.params, \
        "target_count 与 params 均不变(滑动条正交)"
