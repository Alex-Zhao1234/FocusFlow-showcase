"""Retrieval layer: hybrid vector + full-text search over the local video library.

This module owns everything the recommender needs to find content *locally*,
without spending YouTube API quota:

* connection setup (``get_conn``), which also loads the sqlite-vec extension;
* the vector / full-text table DDL (``_init_vector_tables``);
* the S6 mode-fit verdict cache, so the LLM judge does not re-score videos it
  has already judged under the same user profile;
* the high-quality candidate library (recent uploads from trusted channels the
  user has not yet been shown);
* the hybrid retrieval primitives: sqlite-vec KNN, FTS5 keyword search, RRF
  fusion, per-profile-entry reranking, channel similarity and intent recall.

The retrieval paths are referred to by short labels that match their config
switches: E1 = candidate-library rerank by profile entries, E2/E5 = local recall
for user intents and pool-channel matching, E3/E4 = channel ranking for seed
recommendations and the type pool, E7 = first-wave ordering in the orchestrator.

Key design points
-----------------
* **Map tables vs. vec0 virtual tables.** A vec0 table stores nothing but the
  embedding column. Identity and bookkeeping (which video / channel / entry a
  vector belongs to, the model name, the hash of the embedded text) live in
  ordinary "map" tables that share the same ``rowid``. Incremental embedding,
  reverse lookup of KNN hits and orphan cleanup all go through the map tables.
  vec0 has no UPSERT, so writes are "upsert map row, then delete + insert the
  vector under the same rowid".
* **Scopes become rowid sets.** vec0 only supports restricting a KNN query by
  ``rowid``, so every scope ("only these videos", "only channels", "only this
  kind") is first translated into a rowid set via the map tables. For
  non-empty scopes the code computes ``vec_distance_cosine`` directly instead
  of using ``MATCH ... AND k = ?``: with a ``rowid IN (...)`` filter, vec0
  returns nothing when ``k = 1``, and scopes are small (tens to hundreds of
  rows), so direct computation is both correct and cheap.
* **FTS5 own-content table with jieba-segmented columns.** ``videos_fts`` is
  not an external-content table over ``videos``: the videos primary key is
  TEXT (no stable rowid), and the indexed columns hold *derived* text. Chinese
  has no spaces, so titles / descriptions / tags are pre-segmented with jieba
  and stored space-separated, which lets the stock ``unicode61`` tokenizer
  index them as words.
* **Graceful degradation.** If sqlite-vec is not installed or the
  ``VECTOR_SEARCH`` switch is off, ``vector_available()`` is False: vec0
  tables are not created, vector writes are silent no-ops, vector reads return
  empty results, and hybrid search falls back to FTS5 only. Map tables and
  FTS5 are always created since they rely on SQLite built-ins only. Callers
  check ``vector_available()`` instead of guessing whether tables exist.
* **Non-symmetric embedding model.** bge-small-zh embeds documents (videos,
  channels) and queries (profile entries, chips, negative preferences,
  subtopics) differently; the embedding side is handled in
  ``sources/embeddings.py``. A newly saved entry gets its vector on the next
  nightly embedding run, so every reader tolerates missing vectors.
"""
from __future__ import annotations
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from config import DB_PATH_ENV

try:
    import sqlite_vec as _sqlite_vec
except ImportError:
    _sqlite_vec = None

_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = Path(os.environ.get(DB_PATH_ENV, _PROJECT_ROOT / "focusflow.db"))
if not DB_PATH.is_absolute():
    DB_PATH = _PROJECT_ROOT / DB_PATH


def vector_available() -> bool:
    """True when the sqlite-vec extension is importable and VECTOR_SEARCH is on.

    This is the single check callers use; business code never probes for tables.
    """
    return _sqlite_vec is not None and bool(getattr(config, "VECTOR_SEARCH", 0))


def get_conn()->sqlite3.Connection:
    """Open a database connection; the single entry point for all DB access."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if vector_available():                           # extensions are per-connection: load on every connect
        conn.enable_load_extension(True)
        _sqlite_vec.load(conn)
        conn.enable_load_extension(False)            # disable again right away to shrink the attack surface

    conn.row_factory = sqlite3.Row
    return conn


def _init_vector_tables(conn: sqlite3.Connection) -> None:
    """Create the vector and full-text tables (called from schema init).

    Map tables and FTS5 are always created (SQLite built-ins). vec0 virtual
    tables are created only when ``vector_available()``: without the extension
    ``CREATE VIRTUAL TABLE ... USING vec0`` would fail, whereas empty map
    tables are harmless. FTS uses its own content table rather than
    ``content='videos'`` because videos has a TEXT primary key (unstable rowid)
    and the indexed columns store jieba-segmented derived text.
    """

    # Registry for video vectors. video_vec holds only the embedding, so the
    # owning video, model and source-text hash live here, joined by rowid.
    # Used for incremental embedding, KNN reverse lookup and orphan cleanup.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_vec_map(
            rowid        INTEGER PRIMARY KEY,           -- 与 video_vec 同 rowid
            video_id     TEXT UNIQUE NOT NULL,          --视频 id,唯一。同一视频重嵌是覆盖不是新增
            model        TEXT NOT NULL,                 -- 嵌入模型名;与 config.EMBED_MODEL 不同即重嵌
            text_hash    TEXT NOT NULL,                 -- 嵌入文本(标题 + 简介截断 + 标签拼成的模板)的 sha1;作业每次重算哈希,相等就跳过,所以每晚只嵌新增和改动的视频
            embedded_at  INTEGER NOT NULL               -- 嵌入时间戳,报表和排障用。
        )""")
    # Registry for all non-video vectors in text_vec. Five object kinds share
    # one vector table (channel, entry, chip, negpref, subtopic); (kind, key)
    # identifies the object. A kind-scoped KNN first collects that kind's
    # rowids here, searches text_vec within them, then maps rowids back to keys.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS text_vec_map(
            rowid        INTEGER PRIMARY KEY,           -- 与 text_vec 同 rowid
            kind         TEXT NOT NULL,                 -- 对象类别。channel 频道(E3/E4/E5 的被匹配方)、entry 画像条目、chip 44 个兴趣词、negpref 负偏好、ES 批新增 subtopic 子方向。
            key          TEXT NOT NULL,                 -- 该类别内的业务键,按类别各自约定:频道是 channel_id,条目是 {userid}:{entry_id},子方向是 {userid}:{entry_id}:{position},
                                                        -- 负偏好是 {uid}:{id},chip 是 slug。       UNIQUE(kind, key) 保证一个对象一条向量,重嵌是覆盖
            model        TEXT NOT NULL,                 -- 嵌入模型名
            text_hash    TEXT NOT NULL,                 -- 嵌入文本
            embedded_at  INTEGER NOT NULL,              -- 嵌入时间
            UNIQUE(kind, key)
        )""")

    # Full-text index over videos, used for local intent recall: when enough
    # local hits exist, online search for the intent can be cut back to save
    # search quota. The nightly embedding job writes it alongside the vectors.
    # video_id is UNINDEXED (stored only, used to join back to videos); the
    # *_seg columns hold jieba-segmented, space-separated text so unicode61
    # can split Chinese into words.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5(
            video_id UNINDEXED, title_seg, desc_seg, tags_seg,
            tokenize='unicode61 remove_diacritics 2')""")

    # The vec0 tables that actually store vectors: video_vec for videos,
    # text_vec for channels / entries / subtopics / chips / negative prefs.
    if vector_available():
        dim = int(config.EMBED_DIM)        # must match the embedding model's output dimension
        # Cosine distance: similarity = 1 - distance, and VECTOR_SIM_MIN is applied to that similarity.
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS video_vec USING vec0(emb float[{dim}] distance_metric=cosine)")
        conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS text_vec USING vec0(emb float[{dim}] distance_metric=cosine)")


def get_videos_by_ids(video_ids: list[str]) -> list[sqlite3.Row]:
    """Fetch video rows by id, preserving input order and skipping missing ids.

    An empty list returns immediately (``IN ()`` is a syntax error). Values are
    bound as parameters; only the placeholder count is interpolated.
    """
    if not video_ids:
        return []
    conn = get_conn()
    try:
        placeholders = ",".join("?" * len(video_ids))
        rows = conn.execute(
            f"SELECT * FROM videos WHERE video_id IN ({placeholders})",
            tuple(video_ids),
        ).fetchall()
        # IN (...) returns rows in SQLite's order, so re-order by the input list and drop missing ids.
        by_id = {r["video_id"]: r for r in rows}
        return [by_id[i] for i in video_ids if i in by_id]
    finally:
        conn.close()


def _chunks(seq: list, n: int = 500):
    """Yield slices of at most ``n`` items, keeping IN lists under SQLite's variable limit."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def get_fit_verdicts(user_id: int, mode: str, video_ids: list[str], fingerprint: str,
                     ttl_days: int) -> dict[str, sqlite3.Row]:
    """Return cached S6 fit verdicts as ``{video_id: Row}``.

    Rows judged under a different profile fingerprint or older than
    ``ttl_days`` are ignored, so a profile change invalidates the cache.
    Queried in chunks of 500 to stay under the placeholder limit.
    """
    if not video_ids or ttl_days <= 0:
        return {}
    cutoff = int(time.time()) - ttl_days * 86400
    out: dict[str, sqlite3.Row] = {}
    conn = get_conn()
    try:
        for i in range(0, len(video_ids), 500):
            chunk = video_ids[i:i + 500]
            rows = conn.execute(
                f"SELECT * FROM mode_fit_verdicts WHERE user_id = ? AND mode = ? AND fingerprint = ? "
                f"AND judged_at >= ? AND video_id IN ({','.join('?' * len(chunk))})",
                (user_id, mode, fingerprint, cutoff, *chunk)).fetchall()
            for r in rows:
                out[r["video_id"]] = r
        return out
    finally:
        conn.close()


def save_fit_verdicts(user_id: int, mode: str, fingerprint: str, verdicts) -> None:
    """Upsert a batch of S6 verdicts in one write.

    Callers must filter out degraded (fallback) verdicts first so that a
    judge outage is never cached as a real decision.
    """
    if not verdicts:
        return
    now = int(time.time())
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO mode_fit_verdicts(user_id, mode, video_id, fit, fit_score, "
            "content_type, reason, profile_entry, fingerprint, judged_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            [(user_id, mode, vd.video_id, 1 if vd.fit else 0, float(vd.fit_score),
              vd.content_type, vd.reason or "", vd.profile_entry, fingerprint, now)
             for vd in verdicts])
        conn.commit()
    finally:
        conn.close()


def prune_fit_verdicts(days: int) -> int:
    """Delete verdicts older than ``days`` (at least 1); returns the number removed."""
    conn = get_conn()
    try:
        cur = conn.execute("DELETE FROM mode_fit_verdicts WHERE judged_at < ?",
                           (int(time.time()) - max(1, days) * 86400,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def seed_channels_for_mode(user_id: int, mode: str) -> list[str]:
    """All seed channels attached to any entry of this mode, de-duplicated, in insertion order.

    Feeds the candidate library's high-quality channel set; not limited to the
    featured profile entries.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT channel_id FROM user_seed_channels WHERE user_id = ? AND mode = ? "
            "ORDER BY added_at", (user_id, mode)).fetchall()
        return [r["channel_id"] for r in rows]
    finally:
        conn.close()


def library_candidates(user_id: int, mode: str, channel_ids: list[str], exclude: frozenset,
                       fingerprint: str, ttl_days: int, limit: int) -> list[sqlite3.Row]:
    """Zero-API candidates from the local library for high-quality channels.

    A video qualifies when it is from one of ``channel_ids``, is not in
    ``exclude``, and has no unexpired ``fit = 0`` verdict under the current
    fingerprint. Videos already in the card pool are normally excluded, but
    they flow back once *every* pool row has been shown and the latest showing
    is older than the impression-suppression window, so good videos return
    after a cooldown at no quota cost. Prefetched-but-unshown rows and rows
    shown inside the window stay excluded; watched / liked / penalised videos
    are handled by the caller's ``exclude`` set.

    Ordering: never-pooled videos first, then recycled ones; within each,
    not-yet-judged before judged-fit-but-unseated; then newest first. The
    exclude set is applied in Python because it can hold thousands of ids.
    """
    if not channel_ids or limit <= 0:
        return []
    cutoff = int(time.time()) - max(0, ttl_days) * 86400
    cutoff72 = int(time.time()) - config.IMPRESSION_SUPPRESS_TTL_HOURS * 3600
    conn = get_conn()
    try:
        cur = conn.execute(
            f"""
            SELECT v.*,
                   EXISTS(SELECT 1 FROM mode_fit_verdicts f WHERE f.user_id = ? AND f.mode = ?
                          AND f.video_id = v.video_id AND f.fit = 1 AND f.fingerprint = ?
                          AND f.judged_at >= ?) AS has_fit,
                   EXISTS(SELECT 1 FROM card_pool p WHERE p.user_id = ? AND p.mode = ?
                          AND p.video_id = v.video_id) AS has_pool
            FROM videos v
            WHERE v.channel_id IN ({','.join('?' * len(channel_ids))})
              AND NOT EXISTS (SELECT 1 FROM card_pool p WHERE p.user_id = ? AND p.mode = ?
                              AND p.video_id = v.video_id
                              AND (p.last_served_at IS NULL OR p.last_served_at >= ?))
              AND NOT EXISTS (SELECT 1 FROM mode_fit_verdicts f WHERE f.user_id = ? AND f.mode = ?
                              AND f.video_id = v.video_id AND f.fit = 0 AND f.fingerprint = ?
                              AND f.judged_at >= ?)
            ORDER BY has_pool ASC, has_fit ASC, COALESCE(v.publish_time, 0) DESC
            """,
            (user_id, mode, fingerprint, cutoff, user_id, mode, *channel_ids,
             user_id, mode, cutoff72, user_id, mode, fingerprint, cutoff))
        out: list[sqlite3.Row] = []
        while len(out) < limit:
            rows = cur.fetchmany(200)
            if not rows:
                break
            for r in rows:
                if r["video_id"] not in exclude:
                    out.append(r)
                    if len(out) >= limit:
                        break
        return out
    finally:
        conn.close()


def _rowids_for_videos(conn: sqlite3.Connection, video_ids: list[str]) -> list[int]:
    """Translate video ids to video_vec rowids (vec0 only understands rowids)."""
    out: list[int] = []
    for chunk in _chunks(list(video_ids)):
        ph = ",".join("?" * len(chunk))
        out.extend(r[0] for r in conn.execute(
            f"SELECT rowid FROM video_vec_map WHERE video_id IN ({ph})", tuple(chunk)))
    return out


def _rowids_for_texts(conn: sqlite3.Connection, kind: str, keys: list[str]) -> list[int]:
    """Translate ``(kind, key)`` pairs to text_vec rowids."""
    out: list[int] = []
    for chunk in _chunks(list(keys)):
        ph = ",".join("?" * len(chunk))
        out.extend(r[0] for r in conn.execute(
            f"SELECT rowid FROM text_vec_map WHERE kind = ? AND key IN ({ph})", (kind, *chunk)))
    return out


def video_vec_hashes(model: str) -> dict[str, str]:
    """``{video_id: text_hash}`` for videos embedded with ``model``.

    The embedding job re-hashes each video's text and only re-embeds on a
    mismatch or absence. A model change yields an empty dict and therefore a
    full re-embed.
    """
    conn = get_conn()
    try:
        return {r[0]: r[1] for r in conn.execute(
            "SELECT video_id, text_hash FROM video_vec_map WHERE model = ?", (model,))}
    finally:
        conn.close()


def upsert_video_vec(video_id: str, model: str, text_hash: str, emb: bytes,
                     conn: sqlite3.Connection | None = None) -> None:
    """Write one video vector: upsert the map row, then replace the vec0 row under its rowid.

    When ``conn`` is passed the caller owns the transaction (no commit / close),
    which lets the embedding job batch many writes. No-op when vectors are
    unavailable.
    """
    if not vector_available():
        return
    own = conn is None
    conn = conn or get_conn()
    try:
        conn.execute(
            """INSERT INTO video_vec_map(video_id, model, text_hash, embedded_at) VALUES(?,?,?,?)
               ON CONFLICT(video_id) DO UPDATE SET model=excluded.model, text_hash=excluded.text_hash,
                                                   embedded_at=excluded.embedded_at""",
            (video_id, model, text_hash, int(time.time())))
        rid = conn.execute("SELECT rowid FROM video_vec_map WHERE video_id = ?", (video_id,)).fetchone()[0]

        # vec0 has no UPSERT: delete the old vector, then insert under the same rowid.
        conn.execute("DELETE FROM video_vec WHERE rowid = ?", (rid,))
        conn.execute("INSERT INTO video_vec(rowid, emb) VALUES(?, ?)", (rid, emb))
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def delete_video_vecs(video_ids: list[str], conn: sqlite3.Connection | None = None) -> int:
    """Delete video vectors (map + vec0); returns the number of map rows removed.

    Map rows are deleted even when the extension is unavailable, since the
    vec0 table cannot be accessed without it.
    """
    if not video_ids:
        return 0
    own = conn is None
    conn = conn or get_conn()
    try:
        n = 0
        for chunk in _chunks(list(video_ids)):
            rids = _rowids_for_videos(conn, chunk)
            if rids:
                ph = ",".join("?" * len(rids))
                if vector_available():
                    conn.execute(f"DELETE FROM video_vec WHERE rowid IN ({ph})", tuple(rids))
                n += conn.execute(f"DELETE FROM video_vec_map WHERE rowid IN ({ph})", tuple(rids)).rowcount
        if own:
            conn.commit()
        return n
    finally:
        if own:
            conn.close()


def prune_video_vec_orphans() -> int:
    """Remove vector and FTS rows whose video no longer exists in ``videos``.

    Retention deletes source rows; this cleans up the derived data afterwards.
    Vector and FTS orphans are collected separately because the two indexes
    need not cover the same videos. Returns the number of vector rows removed.
    """
    conn = get_conn()
    try:
        gone = [r[0] for r in conn.execute(
            "SELECT m.video_id FROM video_vec_map m LEFT JOIN videos v ON v.video_id = m.video_id "
            "WHERE v.video_id IS NULL")]
        gone_fts = [r[0] for r in conn.execute(
            "SELECT f.video_id FROM videos_fts f LEFT JOIN videos v ON v.video_id = f.video_id "
            "WHERE v.video_id IS NULL")]
        n = delete_video_vecs(gone, conn)
        delete_video_fts(gone_fts, conn)
        conn.commit()
        return n
    finally:
        conn.close()


def upsert_video_fts(video_id: str, title_seg: str, desc_seg: str, tags_seg: str,
                     conn: sqlite3.Connection | None = None) -> None:
    """Write or replace a video's FTS row (delete + insert; FTS5 has no UPSERT).

    Inputs must already be segmented, space-separated text.
    """
    own = conn is None
    conn = conn or get_conn()
    try:
        conn.execute("DELETE FROM videos_fts WHERE video_id = ?", (video_id,))
        conn.execute("INSERT INTO videos_fts(video_id, title_seg, desc_seg, tags_seg) VALUES(?,?,?,?)",
                     (video_id, title_seg or "", desc_seg or "", tags_seg or ""))
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def delete_video_fts(video_ids: list[str], conn: sqlite3.Connection | None = None) -> int:
    """Delete FTS rows for the given videos; returns the number removed."""
    if not video_ids:
        return 0
    own = conn is None
    conn = conn or get_conn()
    try:
        n = 0
        for chunk in _chunks(list(video_ids)):
            ph = ",".join("?" * len(chunk))
            n += conn.execute(f"DELETE FROM videos_fts WHERE video_id IN ({ph})", tuple(chunk)).rowcount
        if own:
            conn.commit()
        return n
    finally:
        if own:
            conn.close()


def text_vec_hashes(kind: str, model: str) -> dict[str, str]:
    """``{key: text_hash}`` for objects of ``kind`` embedded with ``model`` (incremental-embedding check)."""
    conn = get_conn()
    try:
        return {r[0]: r[1] for r in conn.execute(
            "SELECT key, text_hash FROM text_vec_map WHERE kind = ? AND model = ?", (kind, model))}
    finally:
        conn.close()


def upsert_text_vec(kind: str, key: str, model: str, text_hash: str, emb: bytes,
                    conn: sqlite3.Connection | None = None) -> None:
    """Write one non-video vector; same pattern as ``upsert_video_vec`` on text_vec_map / text_vec."""
    if not vector_available():
        return
    own = conn is None
    conn = conn or get_conn()
    try:
        conn.execute(
            """INSERT INTO text_vec_map(kind, key, model, text_hash, embedded_at) VALUES(?,?,?,?,?)
               ON CONFLICT(kind, key) DO UPDATE SET model=excluded.model, text_hash=excluded.text_hash,
                                                    embedded_at=excluded.embedded_at""",
            (kind, key, model, text_hash, int(time.time())))
        rid = conn.execute("SELECT rowid FROM text_vec_map WHERE kind = ? AND key = ?", (kind, key)).fetchone()[0]
        conn.execute("DELETE FROM text_vec WHERE rowid = ?", (rid,))
        conn.execute("INSERT INTO text_vec(rowid, emb) VALUES(?, ?)", (rid, emb))
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def delete_text_vecs(kind: str, keys: list[str], conn: sqlite3.Connection | None = None) -> int:
    """Delete vectors of one ``kind`` by key (map + vec0); returns the number of map rows removed."""
    if not keys:
        return 0
    own = conn is None
    conn = conn or get_conn()
    try:
        n = 0
        for chunk in _chunks(list(keys)):
            rids = _rowids_for_texts(conn, kind, chunk)
            if rids:
                ph = ",".join("?" * len(rids))
                if vector_available():
                    conn.execute(f"DELETE FROM text_vec WHERE rowid IN ({ph})", tuple(rids))
                n += conn.execute(f"DELETE FROM text_vec_map WHERE rowid IN ({ph})", tuple(rids)).rowcount
        if own:
            conn.commit()
        return n
    finally:
        if own:
            conn.close()


def text_vec_ready(kind: str, key: str) -> bool:
    """Whether this object has a vector under the current model.

    The single check callers use; a vector from a different model counts as missing.
    """
    if not vector_available():
        return False
    conn = get_conn()
    try:
        return conn.execute("SELECT 1 FROM text_vec_map WHERE kind = ? AND key = ? AND model = ?",
                            (kind, key, config.EMBED_MODEL)).fetchone() is not None
    finally:
        conn.close()


def get_text_vec(kind: str, key: str) -> bytes | None:
    """Return the stored vector for ``(kind, key)``, or None if absent."""
    if not vector_available():
        return None
    conn = get_conn()
    try:
        r = conn.execute(
            "SELECT v.emb FROM text_vec_map m JOIN text_vec v ON v.rowid = m.rowid "
            "WHERE m.kind = ? AND m.key = ?", (kind, key)).fetchone()
        return bytes(r[0]) if r else None                  # normalise memoryview/buffer to bytes
    finally:
        conn.close()


def _knn(conn: sqlite3.Connection, table: str, query_vec: bytes, k: int,
         rowids: list[int] | None) -> list[tuple[int, float]]:
    """Nearest neighbours on a vec0 table: ``[(rowid, sim)]`` with sim = 1 - cosine distance.

    Results below ``VECTOR_SIM_MIN`` are dropped; ordered by distance ascending.

    * ``rowids=None``: unrestricted vec0 KNN (``emb MATCH ? AND k = ?``).
    * ``rowids=[]``: the scope is an empty set, so the result is empty (not "unrestricted").
    * non-empty ``rowids``: no KNN; ``vec_distance_cosine`` is computed for every
      row in scope, then sorted and cut to ``k``. vec0's
      ``MATCH ... AND k = ? AND rowid IN (...)`` returns nothing when ``k = 1``,
      which would silently drop single-candidate scopes. Scopes are small, so
      direct computation is robust and not slower.
    """
    floor = 1.0 - float(config.VECTOR_SIM_MIN)          # sim >= MIN  <=>  distance <= 1 - MIN
    if rowids is None:
        sql = f"SELECT rowid, distance FROM {table} WHERE emb MATCH ? AND k = ?"
        return [(r[0], 1.0 - r[1]) for r in conn.execute(sql, (query_vec, int(k))) if r[1] <= floor]
    if not rowids:
        return []
    hits: list[tuple[int, float]] = []
    for chunk in _chunks(list(rowids)):
        sql = (f"SELECT rowid, vec_distance_cosine(emb, ?) AS d FROM {table} "
               f"WHERE rowid IN ({','.join('?' * len(chunk))})")
        hits.extend((r[0], r[1]) for r in conn.execute(sql, (query_vec, *chunk)) if r[1] <= floor)
    hits.sort(key=lambda x: (x[1], x[0]))
    return [(rid, 1.0 - d) for rid, d in hits[:int(k)]]


def vec_search_videos(query_vec: bytes, k: int,
                      video_ids: list[str] | None = None) -> list[tuple[str, float]]:
    """Video vector search: ``[(video_id, sim)]`` by descending similarity, sim >= VECTOR_SIM_MIN.

    When ``video_ids`` is given, search is restricted to that set (via rowids).
    """
    if not vector_available() or k <= 0:
        return []
    conn = get_conn()
    try:
        rowids = _rowids_for_videos(conn, video_ids) if video_ids is not None else None
        hits = _knn(conn, "video_vec", query_vec, k, rowids)
        if not hits:
            return []
        ph = ",".join("?" * len(hits))
        # Map hit rowids back to video ids in a single query.
        idmap = {r[0]: r[1] for r in conn.execute(
            f"SELECT rowid, video_id FROM video_vec_map WHERE rowid IN ({ph})", tuple(h[0] for h in hits))}
        return [(idmap[rid], sim) for rid, sim in hits if rid in idmap]
    finally:
        conn.close()


def vec_search_texts(kind: str, query_vec: bytes, k: int,
                     keys: list[str] | None = None) -> list[tuple[str, float]]:
    """Vector search over text_vec restricted to ``kind``: ``[(key, sim)]``.

    The kind restriction is a rowid set taken from the map table (text_vec is
    small: thousands of channels, hundreds of entries); ``keys`` narrows it further.
    """
    if not vector_available() or k <= 0:
        return []
    conn = get_conn()
    try:
        if keys is not None:
            rowids = _rowids_for_texts(conn, kind, keys)
        else:
            rowids = [r[0] for r in conn.execute("SELECT rowid FROM text_vec_map WHERE kind = ?", (kind,))]
        hits = _knn(conn, "text_vec", query_vec, k, rowids)
        if not hits:
            return []
        ph = ",".join("?" * len(hits))
        keymap = {r[0]: r[1] for r in conn.execute(
            f"SELECT rowid, key FROM text_vec_map WHERE rowid IN ({ph})", tuple(h[0] for h in hits))}
        return [(keymap[rid], sim) for rid, sim in hits if rid in keymap]
    finally:
        conn.close()


def _fts_query(query_seg: str) -> str:
    """Build a safe FTS5 query from segmented text, e.g. ``"a" OR "b" OR "c"``.

    Each token is double-quoted (inner quotes doubled) and the tokens are
    OR-ed, so punctuation and words like AND / NOT are taken literally rather
    than parsed as FTS5 syntax. Pure-punctuation tokens are dropped; returns
    "" when nothing usable remains.
    """
    toks = [t.replace('"', '""') for t in (query_seg or "").split() if t.strip()]
    toks = [t for t in toks if any(ch.isalnum() for ch in t)]
    return " OR ".join(f'"{t}"' for t in toks)


def fts_search_videos(query_seg: str, k: int,
                      video_ids: list[str] | None = None) -> list[tuple[str, float]]:
    """Keyword search over title / description / tags: ``[(video_id, score)]``.

    ``score = -bm25`` so that larger means more relevant, matching the
    direction of vector similarity. ``query_seg`` must already be segmented.
    """
    q = _fts_query(query_seg)
    if not q or k <= 0:
        return []
    if video_ids is not None and not video_ids:
        return []
    conn = get_conn()
    try:
        # MATCH on the table name searches all indexed columns; bm25() is lower = more relevant.
        sql = "SELECT video_id, bm25(videos_fts) AS r FROM videos_fts WHERE videos_fts MATCH ?"
        args: list = [q]

        if video_ids is not None:
            ids = list(video_ids)[:900]                       # stay under SQLite's variable limit; callers narrow larger scopes
            sql += f" AND video_id IN ({','.join('?' * len(ids))})"
            args.extend(ids)
        sql += " ORDER BY r LIMIT ?"
        args.append(int(k))
        return [(r[0], -float(r[1])) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def hybrid_search(query_vec: bytes | None, query_seg: str | None, k: int = 30,
                  video_ids: list[str] | None = None, cjk: bool = True) -> list[tuple[str, float]]:
    """Hybrid search: vector and FTS5 paths each take their top ``3k``, fused with RRF.

    Returns ``[(video_id, fused_score)]`` descending, cut to ``k``.

    * ``query_vec=None`` uses FTS5 only; empty ``query_seg`` uses vectors only;
      when vectors are unavailable the vector path returns nothing, so the
      search degrades to FTS5 automatically.
    * ``cjk=False`` (pure-English query) gives the FTS5 path ``K/2``, boosting
      literal matches because bge-small-zh is weak on English.
    * The two paths have incompatible score scales (unbounded bm25 vs. cosine
      in [0, 1]); RRF uses ranks only, so no weight tuning is needed.
    """
    if k <= 0:
        return []
    n = 3 * k
    K = float(config.RRF_K)                 # larger K flattens the gap between top and lower ranks
    fused: dict[str, float] = {}
    if query_vec is not None:
        for rank, (vid, _sim) in enumerate(vec_search_videos(query_vec, n, video_ids), 1):
            fused[vid] = fused.get(vid, 0.0) + 1.0 / (K + rank)
    if query_seg:
        kf = K if cjk else K / 2
        for rank, (vid, _s) in enumerate(fts_search_videos(query_seg, n, video_ids), 1):
            # Videos hit by both paths accumulate two contributions and rise to the top.
            fused[vid] = fused.get(vid, 0.0) + 1.0 / (kf + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]     # ties broken by id for stable output


def vec_stats() -> dict:
    """Row counts per table and model distribution, for reports and diagnostics."""
    conn = get_conn()
    try:
        out: dict = {"available": vector_available()}
        out["video_vec"] = conn.execute("SELECT COUNT(*) FROM video_vec_map").fetchone()[0]
        out["text_vec"] = {r[0]: r[1] for r in conn.execute(
            "SELECT kind, COUNT(*) FROM text_vec_map GROUP BY kind")}
        out["fts"] = conn.execute("SELECT COUNT(*) FROM videos_fts").fetchone()[0]
        out["model"] = {r[0]: r[1] for r in conn.execute(
            "SELECT model, COUNT(*) FROM video_vec_map GROUP BY model")}
        return out
    finally:
        conn.close()


def videos_for_embedding(limit: int | None = None) -> list[sqlite3.Row]:
    """All videos' id / title / description / tags, newest first (so a limited run embeds the latest)."""
    conn = get_conn()
    try:
        sql = "SELECT video_id, title, description, tags FROM videos ORDER BY COALESCE(created_at, 0) DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def entries_for_embedding() -> list[sqlite3.Row]:
    """All users' profile entries (user_id, entry_id, text, search_query)."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT user_id, entry_id, text, search_query FROM user_mode_profile_entries"
            " WHERE entry_id IS NOT NULL").fetchall()
    finally:
        conn.close()


def subtopics_for_embedding() -> list[sqlite3.Row]:
    """All users' entry subtopics (user_id, entry_id, position, subtopic, query_zh)."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT user_id, entry_id, position, subtopic, query_zh FROM entry_subtopics"
            " ORDER BY user_id, entry_id, position").fetchall()
    finally:
        conn.close()


def neg_prefs_for_embedding() -> list[sqlite3.Row]:
    """All users' negative preferences (user_id, id, text)."""
    conn = get_conn()
    try:
        return conn.execute("SELECT user_id, id, text FROM mode_negative_prefs").fetchall()
    finally:
        conn.close()


@dataclass(frozen=True)      # immutable: groups are read-only during reranking
class EntryVecGroup:
    """Vectors of one profile entry.

    ``base`` is the entry's search-query vector (kind ``'entry'``); ``subs``
    are its subtopic vectors (kind ``'subtopic'``) in position order, empty
    when subtopics are not requested.
    """
    entry_id: str
    position: int           # rank within the mode; drives primary / secondary / rest segmentation
    base: bytes
    subs: list[bytes] = field(default_factory=list)


def entry_vec_groups_for_mode(user_id: int, mode: str,
                              include_subtopics: bool = False) -> list[EntryVecGroup]:
    """Embedded profile-entry vectors for one user and mode, grouped per entry, by position.

    Entries without a base vector under the current model are omitted
    entirely (a newly saved entry only gets its vector on the next embedding
    run); subtopics attach only to groups that have a base vector.
    Consumers: E1 segmented rerank and E7 first-wave ordering.
    """
    if not vector_available():
        return []
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT e.entry_id, e.position, v.emb FROM user_mode_profile_entries e                     --选择entry_id,position,emb
               JOIN text_vec_map m ON m.kind = 'entry' AND m.key = (e.user_id || ':' || e.entry_id)      --映射表的键格式是用户:条目
                                   AND m.model = ?
               JOIN text_vec v ON v.rowid = m.rowid                                                      -- JOIN 是内连接,基类没嵌的条目整行不出现,对应 docstring 说的「保存当刻只落文本,向量次日生效」 
               WHERE e.user_id = ? AND e.mode = ? AND e.entry_id IS NOT NULL                             
               ORDER BY e.position""", (config.EMBED_MODEL, user_id, mode)).fetchall()                   # position order; segmented reranking relies on it
        subs: dict[str, list[bytes]] = {}
        if include_subtopics and rows:
            for r in conn.execute(
                    """SELECT s.entry_id, v.emb FROM entry_subtopics s
                       JOIN user_mode_profile_entries e ON e.user_id = s.user_id AND e.entry_id = s.entry_id
                       JOIN text_vec_map m ON m.kind = 'subtopic'
                            AND m.key = (s.user_id || ':' || s.entry_id || ':' || s.position)
                            AND m.model = ?
                       JOIN text_vec v ON v.rowid = m.rowid
                       WHERE s.user_id = ? AND e.mode = ?
                       ORDER BY e.position, s.position""", (config.EMBED_MODEL, user_id, mode)):
                subs.setdefault(r[0], []).append(bytes(r[1]))
        return [EntryVecGroup(r[0], int(r[1]), bytes(r[2]), subs.get(r[0], [])) for r in rows]
    finally:
        conn.close()


def subtopic_vecs_for_entry(user_id: int, entry_id: str) -> list[bytes]:
    """Embedded subtopic vectors of one entry, in position order; missing ones are skipped.

    Used by E5: matching channels with every subtopic as well as the entry
    itself widens coverage.
    """
    if not vector_available():
        return []
    conn = get_conn()
    try:
        rows = conn.execute(
            """SELECT v.emb FROM entry_subtopics s
               JOIN text_vec_map m ON m.kind = 'subtopic'
                    AND m.key = (s.user_id || ':' || s.entry_id || ':' || s.position)
                    AND m.model = ?
               JOIN text_vec v ON v.rowid = m.rowid
               WHERE s.user_id = ? AND s.entry_id = ? ORDER BY s.position""",
            (config.EMBED_MODEL, user_id, entry_id)).fetchall()
        return [bytes(r[0]) for r in rows]
    finally:
        conn.close()


def rerank_video_ids_by_vecs(query_vecs: list[bytes], video_ids: list[str], k: int) -> list[str]:
    """Rank one candidate set against several query vectors and fuse with RRF.

    Each query vector runs a KNN (``k``) over ``video_ids``; the ranks are
    summed with RRF, so videos relevant to several entries accumulate score.
    Returns hit ids by fused score; non-hits are omitted (the caller appends
    them). This is the fallback path when per-entry isolation
    (``E1_ENTRY_ISOLATION``) is off.
    """
    if not query_vecs or not video_ids or k <= 0 or not vector_available():
        return []
    K = float(config.RRF_K)
    fused: dict[str, float] = {}
    for qv in query_vecs:
        # Unlike hybrid_search, RRF here merges several queries on the vector path only.
        for rank, (vid, _sim) in enumerate(vec_search_videos(qv, k, video_ids), 1):
            fused[vid] = fused.get(vid, 0.0) + 1.0 / (K + rank)
    return [vid for vid, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))]


def entry_video_scores(group: EntryVecGroup, video_ids: list[str],
                       sub_weight: float) -> dict[str, float]:
    """Score each candidate against one entry: ``max(sim_base, sub_weight * max(sim_sub))``.

    This is the scoring unit of the segmented rerank. Videos below
    ``VECTOR_SIM_MIN`` (after the subtopic discount) are absent. The base
    vector and each subtopic run one KNN with ``k`` = number of candidates.
    """
    if not video_ids or not vector_available():
        return {}
    out: dict[str, float] = {}
    n = len(video_ids)
    for vid, sim in vec_search_videos(group.base, n, video_ids):
        out[vid] = sim
    floor = float(config.VECTOR_SIM_MIN)

    for sv in group.subs:
        for vid, sim in vec_search_videos(sv, n, video_ids):
            w = sim * float(sub_weight)                       # subtopic matches are discounted
            if w >= floor and w > out.get(vid, -1.0):         # must still clear the threshold; keep the max
                out[vid] = w
    return out


def rerank_video_ids_by_entry_groups(groups: list[EntryVecGroup], featured_entry_ids: list[str],
                                     video_ids: list[str], sub_weight: float) -> list[str]:
    """E1 rerank: greedy segmentation of candidates by profile entry.

    Segment order is ``featured_entry_ids`` (in the given order, skipping ids
    without a group), then the remaining groups by position. Each segment
    claims the videos that score above the threshold and were not claimed by
    an earlier segment, ordered by score descending (ties by id). Returns the
    claimed ids in order; unclaimed videos are omitted for the caller to append.

    Unlike ``rerank_video_ids_by_vecs``, scores are not summed across entries:
    each video belongs only to the earliest segment that wants it, so the
    primary entry is never crowded out by videos that weakly match many others.
    """
    if not groups or not video_ids or not vector_available():
        return []
    by_id = {g.entry_id: g for g in groups}
    order: list[EntryVecGroup] = []
    for eid in featured_entry_ids:
        g = by_id.get(eid)
        if g is not None and g not in order:
            order.append(g)

    order.extend(g for g in sorted(groups, key=lambda g: g.position) if g not in order)
    claimed: set[str] = set()
    out: list[str] = []

    for g in order:
        # Later segments only see what earlier segments left unclaimed.
        scores = entry_video_scores(g, [v for v in video_ids if v not in claimed], sub_weight)
        for vid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0])):
            claimed.add(vid)
            out.append(vid)
    return out


def channel_sims(entry_vec: bytes, channel_ids: list[str]) -> dict[str, float]:
    """Cosine similarity of one entry vector to each channel vector: ``{channel_id: sim}``.

    Channels without a vector are absent (callers fall back or append them).
    No ``VECTOR_SIM_MIN`` threshold: E3/E4 need a full ordering, not hit detection.
    """
    if not channel_ids or not vector_available():
        return {}
    conn = get_conn()
    try:
        rowids = _rowids_for_texts(conn, "channel", list(channel_ids))
        if not rowids:
            return {}
        # Compute distances directly within the scope rather than via KNN,
        # for the same k=1 reason documented in _knn.
        hits: list[tuple[int, float]] = []
        keymap: dict[int, str] = {}

        for chunk in _chunks(list(rowids)):
            ph = ",".join("?" * len(chunk))
            hits.extend((r[0], r[1]) for r in conn.execute(
                f"SELECT rowid, vec_distance_cosine(emb, ?) FROM text_vec WHERE rowid IN ({ph})",
                (entry_vec, *chunk)) if r[1] is not None)

            keymap.update({r[0]: r[1] for r in conn.execute(
                f"SELECT rowid, key FROM text_vec_map WHERE rowid IN ({ph})", tuple(chunk))})
        return {keymap[rid]: 1.0 - dist for rid, dist in hits if rid in keymap}
    finally:
        conn.close()


def intent_local_candidates(query_seg: str, k: int, exclude: set[str] | frozenset,
                            dmin: int | None, dmax: int | None) -> list[sqlite3.Row]:
    """Local recall for a user intent, at zero API cost.

    Searches ``videos_fts`` with the segmented intent text, applies the
    mode's duration window (``dmin`` / ``dmax`` in seconds, None = unbounded)
    and the exclude set (watched / liked / disliked / penalised), and returns
    up to ``k`` video rows in FTS relevance order. Unknown duration (0) passes,
    consistent with S3; it is re-checked after enrichment. The exclude set is
    applied in Python because it can hold thousands of ids.
    """
    if k <= 0:
        return []
    hits = fts_search_videos(query_seg, k * 3)          # oversample: filters below drop some
    if not hits:
        return []
    rows = get_videos_by_ids([vid for vid, _ in hits])   # preserves relevance order
    out: list[sqlite3.Row] = []
    for r in rows:
        if r["video_id"] in exclude:
            continue
        d = int(r["duration_sec"] or 0)
        if d > 0 and ((dmin is not None and d < dmin) or (dmax is not None and d > dmax)):
            continue
        out.append(r)
        if len(out) >= k:
            break
    return out


def video_sims_max(groups: list[EntryVecGroup], video_ids: list[str],
                   sub_weight: float) -> dict[str, float]:
    """E7: per video, the best ``entry_video_scores`` across all entries ``{video_id: sim}``.

    Within an entry the subtopic discount applies; across entries the max is
    taken, not the sum. Videos with no vector or below ``VECTOR_SIM_MIN`` are
    absent (the caller ranks them on statistical score only).
    """
    if not groups or not video_ids or not vector_available():
        return {}
    out: dict[str, float] = {}
    for g in groups:
        for vid, s in entry_video_scores(g, video_ids, sub_weight).items():
            if s > out.get(vid, -1.0):
                out[vid] = s
    return out


def pool_channels_by_vecs(vecs: list[bytes], k_each: int, k: int) -> list[dict]:
    """Multi-vector pool-channel recall (entry base vector plus each subtopic).

    Each vector takes its ``k_each`` nearest pool channels; results are
    unioned by channel_id keeping the max sim, sorted by sim descending and
    cut to ``k``. With a single vector it matches ``pool_channels_by_vec``.
    """
    if not vecs or k <= 0 or not vector_available():
        return []
    best: dict[str, dict] = {}

    for qv in vecs:
        for r in pool_channels_by_vec(qv, k_each):
            cur = best.get(r["channel_id"])
            if cur is None or r["sim"] > cur["sim"]:
                best[r["channel_id"]] = r
    return sorted(best.values(), key=lambda r: (-r["sim"], r["channel_id"]))[:k]


def pool_channels_by_vec(entry_vec: bytes, k: int) -> list[dict]:
    """Nearest ``k`` *pool* channels for one entry vector: channel rows plus a ``sim`` column.

    A channel is in the pool when ``pool_source`` or ``chip_tags`` is set.
    This is E5's semantic path for pool backfill, unioned by the caller with
    exact chip matches.
    """
    if k <= 0 or not vector_available():
        return []
    # Oversample 3x: subscribed and seed channels are embedded too and get filtered out below.
    hits = vec_search_texts("channel", entry_vec, k * 3)

    if not hits:
        return []
    conn = get_conn()
    try:
        ids = [cid for cid, _ in hits]
        ph = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""SELECT * FROM channels WHERE channel_id IN ({ph})
                AND (pool_source IS NOT NULL OR chip_tags IS NOT NULL)""", tuple(ids)).fetchall()
        by_id = {r["channel_id"]: r for r in rows}    # IN does not preserve order; walk hits instead
        out: list[dict] = []

        for cid, sim in hits:
            r = by_id.get(cid)
            if r is None:
                continue
            d = dict(r)
            d["sim"] = sim
            out.append(d)
            if len(out) >= k:                         # hits are sorted, so the first k survivors are the best k
                break
        return out
    finally:
        conn.close()
