"""Tests for the ranker's per-channel and per-content-type caps.

Caps only swap items between the top and overflow lists: the combined set is
unchanged and nothing is recorded as filtered. Seeded items are kept first,
intent cards are neither counted nor swapped, and cap=0 disables the cap.
"""
from __future__ import annotations

import config
from core.models import Video
from pipeline.ranker import RankedItem, _apply_channel_cap, _apply_type_cap


def CH(c):
    return "UC" + c * 22


def _it(vid, final, ch="a", ctype="tech", intent=False):
    v = Video(video_id=vid, channel_id=CH(ch), title="t", duration_sec=600,
              source_strategy="subscribed", intent_sourced=intent)
    return RankedItem(video=v, fit_score=0.5, quality_score_norm=0.5, channel_pref_score=0.5,
                      is_gem=False, final=final, content_type=ctype)


def _ids(items):
    return [it.video.video_id for it in items]


def test_channel_cap_demotes_and_refills(monkeypatch):
    monkeypatch.setattr(config, "RANK_CHANNEL_CAP", 2)
    top = [_it(f"a{i}", 0.9 - i * 0.01, "a") for i in range(4)] + [_it("b0", 0.5, "b")]
    more = [_it("c0", 0.4, "c"), _it("a9", 0.39, "a"), _it("d0", 0.3, "d")]
    n_top, n_more = _apply_channel_cap(top, more, set())
    assert sum(1 for it in n_top if it.video.channel_id == CH("a")) == 2
    assert set(_ids(n_top)) == {"a0", "a1", "b0", "c0", "d0"}      # refill skips a9 since channel a is full
    assert len(n_top) == 5 and set(_ids(n_top + n_more)) == set(_ids(top + more))
    finals = [it.final for it in n_top]
    assert finals == sorted(finals, reverse=True)                  # both lists re-sorted by score


def test_channel_cap_keeps_seeded_first(monkeypatch):
    monkeypatch.setattr(config, "RANK_CHANNEL_CAP", 1)
    top = [_it("a_hi", 0.9, "a"), _it("a_seed", 0.1, "a")]
    n_top, n_more = _apply_channel_cap(top, [], {"a_seed"})
    assert _ids(n_top) == ["a_seed"] and _ids(n_more) == ["a_hi"]


def test_channel_cap_off_and_intent_exempt(monkeypatch):
    monkeypatch.setattr(config, "RANK_CHANNEL_CAP", 0)
    top = [_it(f"a{i}", 0.9, "a") for i in range(6)]
    assert _apply_channel_cap(top, [], set())[0] == top
    monkeypatch.setattr(config, "RANK_CHANNEL_CAP", 1)
    top = [_it("a0", 0.9, "a"), _it("ai1", 0.8, "a", intent=True), _it("ai2", 0.7, "a", intent=True)]
    n_top, _ = _apply_channel_cap(top, [], set())
    assert set(_ids(n_top)) == {"a0", "ai1", "ai2"}                # intent cards are neither counted nor swapped


def test_type_cap_none_not_counted_and_refill(monkeypatch):
    monkeypatch.setattr(config, "RANK_TYPE_CAP", 2)
    top = [_it(f"t{i}", 0.9 - i * 0.01, chr(97 + i), "tech") for i in range(3)] + [_it("n0", 0.5, "z", None)]
    more = [_it("m0", 0.45, "y", "music"), _it("t9", 0.44, "x", "tech")]
    n_top, n_more = _apply_type_cap(top, more)
    assert sum(1 for it in n_top if it.content_type == "tech") == 2
    assert set(_ids(n_top)) == {"t0", "t1", "n0", "m0"}
    assert set(_ids(n_top + n_more)) == set(_ids(top + more))


def test_type_cap_off(monkeypatch):
    monkeypatch.setattr(config, "RANK_TYPE_CAP", 0)
    top = [_it(f"t{i}", 0.9, chr(97 + i), "tech") for i in range(5)]
    assert _apply_type_cap(top, [])[0] == top


def test_type_cap_protects_seeded(monkeypatch):
    """Seeded cards are not removed by the type cap (same rule as the channel cap)."""
    monkeypatch.setattr(config, "RANK_TYPE_CAP", 2)
    top = [_it("t_hi1", 0.9, "a", "tech"), _it("t_hi2", 0.8, "b", "tech"), _it("t_seed", 0.1, "c", "tech")]
    n_top, _ = _apply_type_cap(top, [], {"t_seed"})
    assert "t_seed" in _ids(n_top) and "t_hi2" not in _ids(n_top)


# ---- Later refill/swap steps must not break the per-channel cap ----
def test_type_cap_refill_respects_channel_cap(monkeypatch):
    monkeypatch.setattr(config, "RANK_CHANNEL_CAP", 2)
    monkeypatch.setattr(config, "RANK_TYPE_CAP", 2)
    top = [_it("a0", 0.9, "a", "tech"), _it("a1", 0.89, "a", "tech"), _it("b0", 0.88, "b", "tech"), _it("m0", 0.5, "m", "music")]
    more = [_it("a2", 0.87, "a", "music"), _it("c0", 0.4, "c", "music")]
    n_top, _ = _apply_type_cap(top, more)
    assert sum(1 for it in n_top if it.video.channel_id == CH("a")) == 2      # refill skips a2 since channel a is full
    assert "c0" in _ids(n_top)
