"""
How the degraded paths are tested.

Every agent has a coded path for when the model fails, and every one of
those paths has a test of this shape: make the model raise, call the real
endpoint through the test client, and assert three things:

  1. the HTTP status is 200, never 500;
  2. the response carries a `degraded` flag so the client can say so;
  3. nothing the user typed was lost.

Two representative tests follow, lightly trimmed. The first covers the
profile-structuring agent. The second covers the vector extension: if
`sqlite-vec` is not loadable on a machine, the schema is created without the
vector table and the feature reports itself unavailable, instead of the app
failing to start.
"""

from core.models import UserProfile
from storage import db


def test_profile_endpoint_survives_llm_outage(monkeypatch):
    _reset_db()
    uid, headers = _seed_user("bob")
    client = _client()

    async def boom(raw_text: str) -> UserProfile:
        raise RuntimeError("LLM unavailable (simulated)")

    import api.routes as routes
    monkeypatch.setattr(routes, "build_profile", boom)

    resp = client.post("/api/profile",
                       json={"raw_text": "I like history content"},
                       headers=headers)

    assert resp.status_code == 200            # an LLM outage is never a 500
    body = resp.json()
    assert body["degraded"] is True
    assert body["profile"] is None

    row = db.get_user_profile(uid)
    assert row is not None
    assert row["raw_text"] == "I like history content"   # raw input is stored first and survives
    assert row["profile_json"] is None


def test_missing_vector_extension_is_fail_soft(monkeypatch):
    monkeypatch.setattr(db, "_sqlite_vec", None)        # pretend the extension is absent
    table_names = _fresh_db()

    assert db.vector_available() is False
    assert "video_vec" not in table_names                # the vec0 virtual table is skipped
    assert "video_vec_map" in table_names                # the plain mapping table still exists
