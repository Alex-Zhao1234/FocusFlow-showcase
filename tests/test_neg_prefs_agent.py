"""Tests for sanitizing the negative-preference adjudicator's output.

Pure, offline functions only. The LLM call itself is out of scope for unit
tests; real connectivity is covered by live smoke tests.
"""
from agent.neg_prefs import NegAdjudication, sanitize_adjudication


def test_sanitize_index_bounds():
    assert sanitize_adjudication(NegAdjudication(matched_index=5, text="x"), 3).matched_index == -1
    assert sanitize_adjudication(NegAdjudication(matched_index=-2, text="x"), 3).matched_index == -1
    assert sanitize_adjudication(NegAdjudication(matched_index=2, text="x"), 3).matched_index == 2


def test_sanitize_text_oneline_and_cap():
    adj = sanitize_adjudication(
        NegAdjudication(matched_index=-1, text="  不喜欢\n标题党  混剪  " + "长" * 100), 0)
    assert "\n" not in adj.text
    assert len(adj.text) <= 80
    assert adj.text.startswith("不喜欢 标题党 混剪")


def test_sanitize_empty_text_passthrough():
    assert sanitize_adjudication(NegAdjudication(), 0).text == ""


def test_system_prompt_has_chip_granularity_section():
    from agent.neg_prefs import _ADJUDICATE_SYSTEM
    for key in ("对这个题材不感兴趣", "内容质量差", "不想在此模式看这类", "其他"):
        assert key in _ADJUDICATE_SYSTEM
