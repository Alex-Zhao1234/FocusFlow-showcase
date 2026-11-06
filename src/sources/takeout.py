"""Google Takeout subscriptions importer (one-shot cold-start seeding).

Independent of the refresh pipeline: no pipeline stage imports it. It is
invoked by the onboarding flow, runs once and exits. It parses the
``subscriptions.csv`` from a Takeout export (or the whole ZIP) into channel
seeds, writing the shared ``channels`` table (objective attributes) and the
per-user ``user_channel_prefs`` table (pref_score=0.6, source='takeout_seed').

Four phases:
  Parse    -> read channel_id (column 1) and channel_title (column 3) by
              position, not by header name (headers vary with account locale);
              the first row is always skipped; a full ZIP is accepted and the
              CSV located inside it.
  Validate -> channel_id must start with 'UC' and be 24 characters; dedupe
              preserving order; malformed rows are skipped and counted.
  Enrich   -> optional: reuse ``YouTubeAdapter.fetch_channels_meta`` to fill
              subscriber counts and uploads_playlist_id (1 quota unit per batch
              of 50). Off by default; the first refresh can fill these lazily,
              which avoids a quota spike at import time.
  Persist  -> channels via ``db.merge_channel`` (merge-preserving, never
              overwrites existing objective attributes); preference seeds via
              ``db.upsert_channel_preference``.

Seeds are written only when the user has no preference row for the channel.
``upsert_channel_preference`` overwrites pref_score and source unconditionally,
so re-importing Takeout would otherwise reset learned scores
(source='self_learned') back to 0.6. A seed is meant to be the prior before
the first real feedback arrives, so an existing row means it is no longer
needed; such rows are counted in ``pref_skipped``.

Precondition: the ``users`` row for ``user_id`` must already exist (foreign key
of ``user_channel_prefs``); the onboarding flow guarantees this.
"""
import asyncio
import csv
import io
import logging
import zipfile
from pathlib import Path

from core.models import Channel
from config import DEFAULT_USER_ID
from storage import db

logger = logging.getLogger(__name__)

# Takeout channel_id shape: 'UC' prefix + 22 characters = length 24.
_CID_PREFIX = "UC"
_CID_LEN = 24
_SEED_PREF_SCORE = 0.6           # cold-start prior, slightly above the neutral 0.5
# Marks seed placeholders so they are distinguishable from scores learned from feedback ('self_learned').
_SEED_SOURCE = "takeout_seed"


# ============================================================================
# Parse + Validate (pure code, unit-testable offline)
# ============================================================================
def _read_csv_text(source: str | Path | bytes) -> str:
    """Normalize the three input forms into CSV text.

    - ``.csv`` path -> read directly;
    - ``.zip`` path or ZIP bytes (whole-archive upload) -> locate the
      subscriptions CSV inside the archive;
    - CSV bytes (single-file upload) -> decode.

    Decoding uses utf-8-sig: Takeout CSVs carry a BOM that plain utf-8 would
    glue onto the first column.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".zip":
            data = path.read_bytes()
        else:
            return path.read_text(encoding="utf-8-sig")
    else:
        data = source

    # Bytes: try ZIP first (whole-archive upload), otherwise decode as CSV text.
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            # Prefer the standard name; else accept the archive's only CSV; else fail.
            candidates = [n for n in csvs if n.lower().endswith("subscriptions.csv")] \
                         or (csvs if len(csvs) == 1 else [])
            if not candidates:
                raise ValueError(
                    "ZIP 内未找到订阅 csv(确认导出时勾选了 YouTube 订阅内容);"
                    f"包内 csv: {csvs or '无'}")
            # Decompressed in memory; nothing is written to disk.
            return zf.read(candidates[0]).decode("utf-8-sig")
    return data.decode("utf-8-sig")


def parse_subscriptions(source: str | Path | bytes) -> tuple[list[tuple[str, str | None]], int]:
    """Parse + Validate: return ``([(channel_id, channel_title)], malformed_row_count)``.

    - By position: column 1 (index 0) = channel_id, column 3 (index 2) =
      channel_title; header names are not trusted; the first row is skipped.
    - Rows with fewer than 3 columns or an invalid id are counted as malformed.
    - Deduplicated preserving order; duplicates are not counted as malformed.
    """
    text = _read_csv_text(source)
    rows = list(csv.reader(io.StringIO(text)))

    seen: set[str] = set()
    out: list[tuple[str, str | None]] = []
    skipped = 0
    for row in rows[1:]:                        # first row is always the header
        if len(row) < 3:
            skipped += 1
            continue
        cid = row[0].strip()
        title = row[2].strip() or None
        if not (cid.startswith(_CID_PREFIX) and len(cid) == _CID_LEN):
            skipped += 1
            continue
        if cid in seen:
            continue
        seen.add(cid)
        out.append((cid, title))
    return out, skipped


# ============================================================================
# Enrich (optional, costs quota) + Persist
# ============================================================================
def import_subscriptions(
    source: str | Path | bytes,
    user_id: int = DEFAULT_USER_ID,
    enrich: bool = False,                                # costs quota; off by default
    adapter=None,                                        # sources.youtube.YouTubeAdapter (untyped to avoid a hard import)
) -> dict:
    """Import Takeout subscriptions once.

    Synchronous; async callers should wrap it in ``asyncio.to_thread``.

    Returns stats: ``{"imported": channels written, "seeded": new seeds,
    "pref_skipped": seeds skipped because a preference row existed,
    "malformed": malformed rows, "enriched": channels enriched}``.
    Only writes to the database; triggers no pipeline logic.
    """
    pairs, malformed = parse_subscriptions(source)

    # Enrich (optional): batched channels.list via the adapter, 1 quota unit per 50 ids.
    meta: dict[str, Channel] = {}
    if enrich:
        if adapter is None:
            raise ValueError("enrich=True 需传入 YouTubeAdapter 实例(API 调用只在 youtube.py)")
        channels = asyncio.run(adapter.fetch_channels_meta([cid for cid, _ in pairs]))
        meta = {c.channel_id: c for c in channels}

    imported = seeded = pref_skipped = 0
    for cid, title in pairs:
        m = meta.get(cid)
        # Objective attributes -> shared channels table; merge semantics keep existing rows intact.
        db.merge_channel(Channel(
            channel_id=cid,
            channel_title=(m.channel_title if m and m.channel_title else title),
            channel_subs=m.channel_subs if m else None,
            uploads_playlist_id=m.uploads_playlist_id if m else None,
        ))
        imported += 1

        # Preference seed -> per-user table, only when no row exists (never overwrite learned scores).
        if db.get_channel_pref(user_id, cid) is None:
            db.upsert_channel_preference(
                user_id, cid, pref_score=_SEED_PREF_SCORE, source=_SEED_SOURCE,
            )
            seeded += 1
        else:
            pref_skipped += 1
        # A subscription means "followed": set the flag on new and skipped rows alike.
        # The flag update leaves score/source untouched, so re-importing can tag
        # existing learned rows as subscribed.
        db.set_channel_subscribed(user_id, cid, True)

    # seeded + pref_skipped == imported; enriched <= imported (deleted channels are not returned).
    stats = {
        "imported": imported,
        "seeded": seeded,
        "pref_skipped": pref_skipped,
        "malformed": malformed,
        "enriched": len(meta),
    }
    logger.info("Takeout 导入完成: %s", stats)
    return stats
