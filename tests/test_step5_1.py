"""Tests for the content-type axis of the mode system.

Sections: M1 constants/models/pure functions, M1.5 category whitelist hygiene,
M1.6 FOCUS supply-side keywords, M2 fit prompt structure, M3 consumption chain
(judges, orchestrator, ranker).

Numeric assertion policy:
  - The 17-slug type vocabulary, the four bucket weights and the UNKNOWN/OTHER
    weights are frozen values and are asserted literally. Changing them is a
    deliberate design change, and a failing test here makes that explicit.
  - W_FIT/W_TYPE are tunable, so only two invariants are asserted: the four
    ranking weights sum to 1.0, and W_FIT + W_TYPE == 0.5 (the fit axis is
    split between them). Retuning them requires no change here.
"""
import json
from pathlib import Path

import pytest

import config
from core.models import ModeFitVerdict, RecommendationCard, Video

# ================================================================== #
# M1 · Constants, models and pure functions
# ================================================================== #

_MODES4 = ("focus", "entertainment", "explore", "wind_down")

# Frozen type vocabulary, asserted as a set (ordering semantics live in the
# per-mode buckets). film_anime/fashion_beauty/autos exist so that whole content
# families do not fall into the low-weight "other" bucket.
_FROZEN_TYPES = {
    "work_practice", "tech", "knowledge_pro", "knowledge_life", "talk_podcast",
    "documentary", "news_politics", "comedy", "music", "ambient",
    "cozy_life", "travel_outdoor", "sports_gaming", "selfhelp_biz",
    "film_anime", "fashion_beauty", "autos",
}


def test_m1_bucket_weights_frozen_and_equidistant():
    """The four tier weights are equidistant, from 1.0 down to 0.2 (frozen values)."""
    w = config.TYPE_BUCKET_WEIGHTS
    assert w == [1.0, 0.7333, 0.4667, 0.2]
    steps = [w[i] - w[i + 1] for i in range(3)]
    assert all(abs(s - steps[0]) < 1e-3 for s in steps), "四档须等距"
    assert config.TYPE_WEIGHT_UNKNOWN == 0.6                 # unknown is not the same as low-value
    assert config.TYPE_WEIGHT_OTHER == 0.2                   # only "other" is penalized to the bottom tier


def test_m1_content_types_wellformed():
    """The vocabulary has 17 unique, non-empty slugs, excludes "other", and uses
    lowercase snake_case with no whitespace."""
    ts = config.CONTENT_TYPES
    assert len(ts) == 17 and len(set(ts)) == 17
    assert set(ts) == _FROZEN_TYPES
    assert "other" not in ts                                 # "other" is a fallback tier only, not a vocabulary entry
    for t in ts:
        assert t and t == t.strip().lower() and " " not in t and "-" not in t, \
            f"词表书写红线:{t!r}"


def test_m1_mode_buckets_complete():
    """Each mode's buckets contain exactly the 17 slugs, in four non-empty tiers
    with no duplicates."""
    assert set(config.MODE_TYPE_BUCKETS) == set(_MODES4)
    for mode_name, buckets in config.MODE_TYPE_BUCKETS.items():
        assert len(buckets) == 4, f"{mode_name}: 须四档"
        assert all(len(b) > 0 for b in buckets), f"{mode_name}: 四桶均须非空"
        flat = [t for b in buckets for t in b]
        assert len(flat) == 17 and set(flat) == _FROZEN_TYPES, \
            f"{mode_name}: 桶内容与词表不恰合"


def test_m1_type_weight_three_branches():
    """Three branches: None -> 0.6, "other" -> 0.2, out-of-vocabulary -> 0.6;
    otherwise the bucket's tier weight (checked for every mode, bucket and slug)."""
    for buckets in config.MODE_TYPE_BUCKETS.values():
        assert config.type_weight_of(None, buckets) == config.TYPE_WEIGHT_UNKNOWN
        assert config.type_weight_of("other", buckets) == config.TYPE_WEIGHT_OTHER
        assert config.type_weight_of("definitely_not_a_type", buckets) \
            == config.TYPE_WEIGHT_UNKNOWN
        for tier_w, bucket in zip(config.TYPE_BUCKET_WEIGHTS, buckets):
            for t in bucket:
                assert config.type_weight_of(t, buckets) == tier_w, (t, tier_w)


def test_m1_type_weight_normalization():
    """Normalization only fixes spelling variants (case, surrounding whitespace,
    hyphens/spaces -> underscores); 'Other' variants also hit the other branch."""
    buckets = config.MODE_TYPE_BUCKETS["focus"]
    canon = config.type_weight_of("work_practice", buckets)
    for variant in ("Work_Practice", "work practice", "work-practice",
                    " WORK_PRACTICE ", "Work Practice"):
        assert config.type_weight_of(variant, buckets) == canon, variant
    assert config.type_weight_of(" Other ", buckets) == config.TYPE_WEIGHT_OTHER


def test_m1_weight_invariants():
    """Scale sanity: the four ranking weights sum to 1.0 and W_FIT + W_TYPE is
    0.5 (both invariants must survive retuning). The direct-pass quality norm
    ordering invariant is independent of the fit-axis split."""
    assert config.W_FIT + config.W_TYPE + config.W_QUALITY + config.W_PREF \
        == pytest.approx(1.0)
    assert config.W_FIT + config.W_TYPE == pytest.approx(0.5)
    assert config.W_TYPE > 0
    assert 0 < 1 / 3 < config.DIRECT_PASS_QUALITY_NORM < 2 / 3 < 1


def test_m1_category_names_full_roster():
    """YT_CATEGORY_NAMES holds the full official list, including the legacy
    30-44 range for read-side mapping; retrieval whitelists are restricted to
    assignable categories separately."""
    names = config.YT_CATEGORY_NAMES
    assert len(names) == 32
    assert all(isinstance(k, int) for k in names)
    assert all(isinstance(v, str) and v for v in names.values())
    assert names[28] == "Science & Technology"
    assert names[10] == "Music"
    assert names[35] == "Documentary"                        # legacy, kept for read-side mapping
    assert set(range(30, 45)) <= set(names), "legacy 段 30~44 须全量收录"


def test_m1_models_content_type_default_none():
    """content_type defaults to None on both models, so existing callers are
    unaffected (None means degraded output or a path without typing)."""
    vd = ModeFitVerdict(video_id="v1", fit=True)
    assert vd.content_type is None
    v = Video(video_id="v1", channel_id="UC" + "x" * 22, title="t",
              duration_sec=600, source_strategy="subscribed")
    card = RecommendationCard(video=v, reason="", fit_score=0.5,
                              quality_norm=0.6, pref_score=0.5, is_gem=False)
    assert card.content_type is None


# ================================================================== #
# M1.5 · Category whitelist hygiene
# ================================================================== #

_AUDIT_CACHE = (Path(__file__).resolve().parents[1]
                / "scripts" / "cache" / "backtest_cache" / "assignable_categories.json")

# Fallback set so the assertions are meaningful without the audit cache.
# When the cache exists it takes precedence; drift between the two is caught
# by test_m15_names_consistent_with_cache.
_ASSIGNABLE_FALLBACK = {1, 2, 10, 15, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29}


def _assignable_ids() -> set[int]:
    if _AUDIT_CACHE.exists():
        data = json.loads(_AUDIT_CACHE.read_text(encoding="utf-8"))
        return {int(k) for k, v in data["categories"].items() if v["assignable"]}
    return set(_ASSIGNABLE_FALLBACK)


def test_m15_whitelists_all_assignable():
    """Every mode's category whitelist is a subset of the assignable set; dead
    entries waste quota on trending/search calls that can never return results."""
    assignable = _assignable_ids()
    for mode in config.MODES.values():
        dead = set(mode.yt_category_ids) - assignable
        assert not dead, f"{mode.name}: 白名单含非 assignable 条目 {sorted(dead)}"


def test_m15_focus_plus_28_explore_minus_35():
    """Spot checks: FOCUS includes 28 (Science & Technology) to widen supply;
    EXPLORE excludes 35 (a non-assignable dead entry)."""
    assert 28 in config.FOCUS.yt_category_ids
    assert 35 not in config.EXPLORE.yt_category_ids


# ================================================================== #
# M2 · Fit prompts and compress
# ================================================================== #

_FIT_NAMES = tuple(f"fit_{m}" for m in _MODES4)


def test_m2_prompts_v2_with_type_section():
    """Fit prompt versions match the expected table (any prompt text change
    bumps the version). The type-classification section is assembled from the
    single _TYPE_SECTION constant to avoid drift across the four prompts, and
    contains all 17 slugs, the decision ladder, the no-invented-labels rule and
    the category-prior sentence."""
    from agent import prompts
    expected_versions = {"fit_focus": "v5"}          # the other three are v4
    for name in _FIT_NAMES:
        p = prompts.PROMPTS[name]
        assert p.version == expected_versions.get(name, "v4"), name
        assert prompts._TYPE_SECTION in p.system, f"{name}: 判类型段须为常量拼装"
    ts = prompts._TYPE_SECTION
    for slug in _FROZEN_TYPES:
        assert slug in ts, slug
    assert "命中即止" in ts                       # decision ladder
    assert "禁止发明" in ts                       # no invented labels
    assert "category" in ts and "创作者自报" in ts  # category prior sentence
    assert "content_type" in prompts._FIT_COMMON  # output schema line (defined once)


def test_m2_t_d12_no_user_profession_wording():
    """work_practice is an intrinsic property of the content: fit prompts must
    not mention the user's profession, since S6 type classification has no
    user context."""
    from agent import prompts
    for name in _FIT_NAMES:
        sys_text = prompts.PROMPTS[name].system
        assert "用户的专业" not in sys_text and "用户的职业" not in sys_text, name


def test_m2_wind_down_rubric_untouched():
    """The wind_down rubric body is unchanged (only the type section was added);
    key sentences must still be present verbatim."""
    from agent import prompts
    sys_text = prompts.PROMPTS["fit_wind_down"].system
    for sentence in ("内容只许安抚,不许唤醒",
                     "【宁缺毋滥】:本模式拒绝倾向最高",
                     "深夜食堂:一碗汤面的慢制作",
                     "悬疑解说:消失的房客"):
        assert sentence in sys_text, sentence


def test_m2_rubric_adjudications_present():
    """Spot-check rubric rules: focus has work positives, a hard exclusion of
    sensitive topics and sleep-aid negatives; explore is permissive (income
    promises/anxiety are not rejection reasons) but rejects pure ads;
    entertainment has two exclusion examples."""
    from agent import prompts
    focus = prompts.PROMPTS["fit_focus"].system
    assert "工作" in focus and "正例" in focus            # work/AI positive examples
    assert "一律拒" in focus and "历史/政治" in focus      # hard exclusion of sensitive topics
    assert "讲述" in focus                               # presentation-style gate kept for non-sensitive topics
    assert "睡眠引导" in focus or "助眠" in focus          # sleep-aid negative examples
    assert "时长" in focus and "不按时长" in focus         # duration is not a fit criterion
    explore = prompts.PROMPTS["fit_explore"].system
    assert "不构成拒因" in explore                        # permissive
    assert "纯广告" in explore                            # the only remaining rejection family
    assert "焦虑" in explore                              # anxiety explicitly not a rejection reason
    ent = prompts.PROMPTS["fit_entertainment"].system
    assert "宣传" in ent                                  # exclusion 1: corporate/product promotion
    assert "严肃" in ent                                  # exclusion 2: heavy serious narratives


def test_m5_final_adjudications():
    """Spot-check calibration decisions: focus judges white-noise videos by
    content rather than title marketing; explore allows serious history/politics
    documentaries but rejects clickbait; explore puts work_practice in tier 2;
    news_politics is narrowed to current affairs, with history/politics
    documentaries classified as documentary. Also: no fit threshold and
    W_TYPE = 0.15."""
    from agent import prompts
    focus = prompts.PROMPTS["fit_focus"].system
    assert "判内容不判标题营销" in focus                  # white noise: judge content, not title
    assert "引导入睡口播" in focus                        # reject only spoken sleep guidance
    explore = prompts.PROMPTS["fit_explore"].system
    assert "严肃的历史/政治纪录片" in explore              # serious documentaries allowed
    assert "标题党" in explore and "煽动对立" in explore   # precise boundary of what is still rejected
    ts = prompts._TYPE_SECTION
    assert "当下时事" in ts and "纪录片属性" in ts         # ladder: current affairs vs documentary
    w = config.type_weight_of("work_practice", config.MODE_TYPE_BUCKETS["explore"])
    assert w == config.TYPE_BUCKET_WEIGHTS[1]            # tier 2
    assert config.FIT_SCORE_THRESHOLD is None            # no fit threshold
    assert config.W_TYPE == 0.15                         # type weight


def test_m2_compress_category_key():
    """compress adds a category prior key only when a mapping exists (omitted
    otherwise to save tokens); legacy ids such as 35 still map on read."""
    from agent.judges import compress
    base = dict(video_id="v1", channel_id="UC" + "x" * 22, title="t",
                duration_sec=600, source_strategy="subscribed")
    assert compress(Video(**base, category_id=28))["category"] == "Science & Technology"
    assert compress(Video(**base, category_id=35))["category"] == "Documentary"
    assert "category" not in compress(Video(**base))          # category_id=None


def test_m2_degrade_fit_content_type_none():
    """Degraded fit verdicts always have content_type None (a degraded result
    must not pretend to have a type), for both strict and non-strict modes."""
    from agent.judges import _degrade_fit
    v = Video(video_id="v1", channel_id="UC" + "x" * 22, title="t",
              duration_sec=600, source_strategy="subscribed")
    for mode in (config.WIND_DOWN, config.FOCUS):
        for vd in _degrade_fit([v], mode):
            assert vd.content_type is None


# ================================================================== #
# M3 · Consumption chain: ranker type weight, orchestrator collection/logging
# ================================================================== #


def _mk_video(vid: str, src: str = "search") -> Video:
    return Video(video_id=vid, channel_id="UC" + "m3" * 11, title=f"t-{vid}",
                 duration_sec=900, source_strategy=src)




def test_m3_type_missing_batch_costs_one_call(monkeypatch):
    """A missing content_type must not trigger batch retries or model
    escalation: with no content_type in any verdict, exactly one LLM call is made."""
    import asyncio
    from agent import judges
    videos = [_mk_video(f"b{i}") for i in range(3)]
    calls = []

    async def fake_call(agent, prompt):
        calls.append(1)
        return [ModeFitVerdict(video_id=v.video_id, fit=True, fit_score=0.9)
                for v in videos]                            # content_type missing (None) everywhere

    monkeypatch.setattr(judges, "_get_agent", lambda *a, **k: object())
    monkeypatch.setattr(judges, "_call_agent", fake_call)

    async def run():
        out = []
        async for batch in judges.judge_mode_fit(videos, config.FOCUS):
            out.extend(batch)
        return out

    verdicts = asyncio.run(run())
    assert len(calls) == 1 and len(verdicts) == 3
    assert all(vd.content_type is None and not vd.degraded for vd in verdicts)


def test_m3_oov_flush_warning_with_prompt_version(caplog):
    """Out-of-vocabulary types are aggregated into one warning per refresh that
    includes the fit prompt version."""
    import logging
    from collections import Counter
    from pipeline import orchestrator
    from agent.prompts import PROMPTS
    with caplog.at_level(logging.WARNING, logger="pipeline.orchestrator"):
        orchestrator._flush_type_oov(config.FOCUS, Counter(vlog_daily=3, foo=1))
    assert len(caplog.records) == 1                        # exactly one aggregated record
    msg = caplog.records[0].getMessage()
    assert "vlog_daily" in msg and PROMPTS["fit_focus"].version in msg
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pipeline.orchestrator"):
        orchestrator._flush_type_oov(config.FOCUS, Counter())
    assert not caplog.records                              # no OOV, no log


def test_m3_final_card_content_type_passthrough():
    """_final_card passes content_type through to the public card contract."""
    from pipeline.orchestrator import _final_card
    from pipeline.ranker import RankedItem
    item = RankedItem(video=_mk_video("vc"), fit_score=0.8, quality_score_norm=0.6,
                      channel_pref_score=0.5, is_gem=False, final=0.7,
                      content_type="tech", type_w=1.0)
    card = _final_card(item, "top", {"vc": "理由"})
    assert card.content_type == "tech" and card.reason == "理由"


# ================================================================== #
# M5 · Labeling helpers
# ================================================================== #


def _load_gold_labels_module():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "scripts" / "evals" / "gold_labels.py"
    spec = importlib.util.spec_from_file_location("gold_labels_m5", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod




# ================================================================== #
# M1.6 · FOCUS supply-side keywords
# ================================================================== #

_WORK_TERMS = ("AI tools", "coding tutorial", "productivity", "工作学习")


def test_m16_focus_keywords_work_terms():
    """FOCUS keywords include work-oriented terms so search is not limited to
    background audio; none of them hits the mode's blacklist (substring check,
    stricter than the S3 word-boundary match)."""
    for kw in _WORK_TERMS:
        assert kw in config.FOCUS.keywords, kw
        low = kw.lower()
        assert not any(b in low for b in config.FOCUS.blacklist_keywords), kw


def test_m16_plan_context_carries_new_terms():
    """The planner's mode summary carries the new terms into the S1 prompt
    (read-only check; no DB, no LLM)."""
    from agent import planner
    kws = planner._mode_summary(config.FOCUS)["keywords"]
    for kw in _WORK_TERMS:
        assert kw in kws, kw


def test_m16_default_plan_focus_work_search():
    """The static fallback plan has a work-oriented search and is still valid:
    search count <= MAX_SEARCH_PER_REFRESH and validate_plan passes with a full
    quota."""
    from agent.planner import validate_plan
    plan = config.DEFAULT_PLANS["focus"]
    searches = [s for s in plan.strategies if s.tool == "search_by_keyword"]
    assert len(searches) <= config.MAX_SEARCH_PER_REFRESH
    assert any(any(t in s.params["query"] for t in ("AI", "coding", "工作"))
               for s in searches), "静态计划须含 work 向 search 条目(#13(a))"
    ok, why = validate_plan(plan, config.DAILY_QUOTA)
    assert ok, why


def test_m15_names_consistent_with_cache():
    """YT_CATEGORY_NAMES matches the official category list. Checked only when
    the audit cache is present; skipped otherwise."""
    if not _AUDIT_CACHE.exists():
        pytest.skip("assignable_categories.json 未生成(先跑 scripts/evals/category_audit.py)")
    data = json.loads(_AUDIT_CACHE.read_text(encoding="utf-8"))
    for k, v in data["categories"].items():
        cid = int(k)
        assert cid in config.YT_CATEGORY_NAMES, f"官方清单含 {cid},NAMES 缺失"
        assert config.YT_CATEGORY_NAMES[cid] == v["title"], \
            f"{cid}: NAMES={config.YT_CATEGORY_NAMES[cid]!r} 官方={v['title']!r}"
