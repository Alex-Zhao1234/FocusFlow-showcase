"""Tests for sanitizing profiler-derived entry fields and the channel-fit judge.

Pure functions; no API key required.
"""
from __future__ import annotations

import config
from agent.profiler import EntryDerivation, _sanitize_derivations


def test_sanitize_chips_filters_dedupes_caps():
    it = EntryDerivation(base_types=["tech"], search_query="q",
                         chips=["ai", "bogus", "ai", "lofi", "programming", "cooking"])
    out = _sanitize_derivations([it], 1)
    assert out[0].chips == ["ai", "lofi", "programming"]      # drop unknown chips, dedupe, cap at ENTRY_CHIPS_MAX
    assert len(out[0].chips) <= config.ENTRY_CHIPS_MAX


def test_sanitize_rank_drops_out_of_range_and_dupes_caps():
    from agent.profiler import _sanitize_rank
    assert _sanitize_rank([3, 1, 9, 1, -1, 0, 2], n_items=4, top_n=3) == [3, 1, 0]
    assert _sanitize_rank([], 4, 3) == []


def test_sanitize_chips_default_empty_and_padding():
    out = _sanitize_derivations([EntryDerivation(base_types=["tech"])], 2)
    assert out[0].chips == [] and out[1].chips == []


# ─── English search-query derivation and channel-portfolio fit judge ───

def test_sanitize_search_query_en_drops_cjk_and_blank():
    items = [EntryDerivation(base_types=["tech"], search_query="人工智能 大模型",
                             search_query_en="AI large language models"),
             EntryDerivation(search_query="白噪音", search_query_en="白噪音 white noise"),  # copied the source text
             EntryDerivation(search_query="x", search_query_en="   ")]
    out = _sanitize_derivations(items, 3)
    assert out[0].search_query_en == "AI large language models"
    assert out[1].search_query_en is None        # contains CJK -> None (otherwise the zh bias returns)
    assert out[2].search_query_en is None


def test_sanitize_fits_by_index_and_clamp():
    """The judge returns {i, fit} pairs by index: out-of-order and duplicate items
    (last wins) are normalized; a missing or out-of-range index is an error."""
    import pytest
    from agent.profiler import FitItem, _sanitize_fits
    items = [FitItem(i=2, fit=0), FitItem(i=0, fit=5), FitItem(i=1, fit=-1),
             FitItem(i=1, fit=1), FitItem(i=3, fit=2)]
    assert _sanitize_fits(items, 4) == [2, 1, 0, 2]
    with pytest.raises(ValueError):
        _sanitize_fits([FitItem(i=0, fit=2), FitItem(i=1, fit=2)], 3)     # index 2 missing
    with pytest.raises(ValueError):
        _sanitize_fits([FitItem(i=0, fit=2), FitItem(i=7, fit=2)], 2)     # out of range


def test_judge_channel_fit_chunks_and_merges(monkeypatch):
    """Candidates are judged concurrently in chunks of EXPAND_JUDGE_BATCH, indexed
    from 0 within each chunk; merged results stay aligned with the candidates."""
    import asyncio
    from agent import profiler
    prompts = []

    class _Out:
        def __init__(self, n): self.output = profiler.ChannelFit(
            fits=[profiler.FitItem(i=i, fit=(2 if i == 0 else 1)) for i in range(n)])

    class _Agent:
        async def run(self, prompt):
            prompts.append(prompt)
            n = int(prompt.split("[候选数] ")[1].split("\n")[0])
            return _Out(n)
    monkeypatch.setattr(profiler, "_get_expand_fit_agent", lambda: _Agent())
    monkeypatch.setattr(config, "EXPAND_JUDGE_BATCH", 20)
    cands = [{"title": f"c{i}", "titles": [], "topics": []} for i in range(45)]
    fits = asyncio.run(profiler.judge_channel_fit({"text": "x"}, cands))
    assert len(prompts) == 3 and len(fits) == 45
    assert fits[0] == 2 and fits[20] == 2 and fits[40] == 2 and fits[1] == 1   # first of each chunk = in-chunk index 0
    assert "0. c20" in prompts[1] and "19. c39" in prompts[1] and "4. c44" in prompts[2]


def test_build_fit_prompt_no_source_and_titles_capped():
    from agent.profiler import _build_fit_prompt
    entry = {"text": "AI、大模型", "base_types": ["tech"], "chips": ["ai"],
             "search_query": "人工智能 大模型", "search_query_en": "AI LLM"}
    cands = [{"title": "老石谈芯", "src": "search", "topics": ["Technology"],
              "description": "芯片" * 300, "keywords": "chip",
              "titles": [f"t{i}" for i in range(config.EXPAND_UPLOAD_TITLES + 5)]},
             {"title": "StatQuest", "src": "pool", "topics": [], "description": None,
              "keywords": None, "titles": []}]
    p = _build_fit_prompt(entry, cands)
    assert "search" not in p and "pool" not in p and "来源" not in p   # the judge must not see the candidate source
    assert "0. 老石谈芯" in p and "1. StatQuest" in p
    assert f"t{config.EXPAND_UPLOAD_TITLES - 1}" in p and f"t{config.EXPAND_UPLOAD_TITLES}" not in p
    assert len([l for l in p.splitlines() if l.startswith("简介:")][0]) <= 210   # description truncated to 200 chars
    assert "AI LLM" in p and "人工智能 大模型" in p
