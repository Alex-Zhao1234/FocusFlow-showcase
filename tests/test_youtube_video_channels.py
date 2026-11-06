"""Tests for aggregating search.list (type=video) responses into channels, plus channel-meta and uploads-title fetching."""
from __future__ import annotations

from sources.youtube import _aggregate_video_channels


def _item(cid, title="t"):
    return {"id": {"videoId": "v"}, "snippet": {"channelId": cid, "channelTitle": title}}


def test_aggregate_counts_hits_and_best_rank_and_sorts():
    resp = {"items": [_item("UCb", "B"), _item("UCa", "A"), _item("UCb"),
                      {"id": {}, "snippet": {}},            # no channelId: skipped
                      _item("UCc", "C"), _item("UCa")]}
    out = _aggregate_video_channels(resp)
    assert [o["channel_id"] for o in out] == ["UCb", "UCa", "UCc"]   # hits 2,2,1; ties broken by best_rank
    assert out[0] == {"channel_id": "UCb", "title": "B", "hits": 2, "best_rank": 0,
                      "sample_titles": []}       # no video titles -> empty sample list
    assert out[1]["best_rank"] == 1 and out[2]["hits"] == 1


def test_aggregate_empty():
    assert _aggregate_video_channels({}) == []


# ─── Sample titles in aggregation; channel description/keywords; uploads titles ───

def _vitem(cid, vtitle, ctitle="t"):
    return {"id": {"videoId": "v"},
            "snippet": {"channelId": cid, "channelTitle": ctitle, "title": vtitle}}


def test_aggregate_keeps_sample_titles_capped():
    import config
    resp = {"items": [_vitem("UCa", f"a{i}") for i in range(config.EXPAND_SAMPLE_TITLES + 2)]
            + [_vitem("UCb", "b0")]}
    out = _aggregate_video_channels(resp)
    a = next(o for o in out if o["channel_id"] == "UCa")
    assert a["sample_titles"] == [f"a{i}" for i in range(config.EXPAND_SAMPLE_TITLES)]
    assert next(o for o in out if o["channel_id"] == "UCb")["sample_titles"] == ["b0"]


def _bare_adapter():
    from sources.youtube import YouTubeAdapter
    a = object.__new__(YouTubeAdapter)
    a.user_id = 1
    a.region_code = "US"
    a.relevance_language = None
    a.calls = []

    class _Req:
        def __init__(self, kw): self.kw = kw

    class _Res:
        def list(self, **kw): return _Req(kw)

    class _Yt:
        def channels(self): return _Res()
        def playlistItems(self): return _Res()
    a._yt = _Yt()
    return a


def test_channels_meta_carries_description_keywords_topics():
    import asyncio
    a = _bare_adapter()

    async def _exec(request, tool):
        a.calls.append((tool, request.kw))
        return {"items": [{
            "id": "UCx",
            "snippet": {"title": "X", "description": "专注 AI 与大模型"},
            "statistics": {"subscriberCount": "12000", "videoCount": "30"},
            "contentDetails": {"relatedPlaylists": {"uploads": "UUx"}},
            "topicDetails": {"topicCategories": ["https://en.wikipedia.org/wiki/Technology"]},
            "brandingSettings": {"channel": {"keywords": "ai llm \"machine learning\""}},
        }]}
    a._execute = _exec
    metas = asyncio.run(a.fetch_channels_meta(["UCx"]))
    m = metas[0]
    assert m.description == "专注 AI 与大模型" and m.keywords == 'ai llm "machine learning"'
    assert m.topic_categories == ["Technology"] and m.uploads_playlist_id == "UUx"
    assert "brandingSettings" in a.calls[0][1]["part"] and a.calls[0][0] == "channels_list"


def test_fetch_uploads_titles_parses_and_no_touch(monkeypatch):
    import asyncio
    from sources import youtube as yt_mod
    a = _bare_adapter()
    touched = []
    monkeypatch.setattr(yt_mod.db, "touch_channel_fetched", lambda cid: touched.append(cid))

    async def _resolve(cid):
        return "UUx"
    a._resolve_uploads_playlist_id = _resolve

    async def _exec(request, tool):
        a.calls.append((tool, request.kw))
        return {"items": [{"snippet": {"title": "t1"}}, {"snippet": {"title": "t2"}},
                          {"snippet": {}}]}
    a._execute = _exec
    titles = asyncio.run(a.fetch_uploads_titles("UCx", 20))
    assert titles == ["t1", "t2"]
    tool, kw = a.calls[0]
    assert tool == "fetch_channel_videos" and kw["part"] == "snippet" and kw["maxResults"] == 20
    assert kw["playlistId"] == "UUx" and touched == []      # must not touch last_fetched (pool cooldown is unaffected)


def test_fetch_uploads_titles_unresolved_returns_empty():
    import asyncio
    a = _bare_adapter()

    async def _resolve(cid):
        return None
    a._resolve_uploads_playlist_id = _resolve
    assert asyncio.run(a.fetch_uploads_titles("UCdead", 20)) == []
