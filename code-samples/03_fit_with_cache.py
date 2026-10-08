"""
The mode-fit judge (S6) behind a 7-day, per-user verdict cache.

Read path:
  * session-intent refreshes and a TTL of 0 bypass the cache entirely;
  * cache hits are grouped into batch zero and yielded immediately with no
    model call, so cached cards reach the screen first;
  * misses go to the judge in batches and are yielded as they come back.

Write path:
  * only non-degraded verdicts are written back (a verdict produced on the
    fail-open path is not worth remembering for a week);
  * the cache key includes a fingerprint of prompt version, model, language
    and the user's own profile text, so changing any of those invalidates it.

Failure policy: a cache read error counts as "all misses", a cache write
error is a warning. The judge still runs. Nothing on this path can take the
refresh down.
"""

import asyncio
import logging

import config
from core import fit_cache
from core.logsafe import redacted_traceback
from core.models import Video
from storage import db

logger = logging.getLogger(__name__)


async def _fit_with_cache(subwave: list[Video], intent_text: str, *, judge,
                          user_id: int, mode_name: str, fingerprint: str,
                          stats: dict):
    ttl = int(config.FIT_CACHE_TTL_DAYS)
    if intent_text or ttl <= 0 or not subwave:
        async for batch in judge(subwave):
            stats["llm"] += len(batch)
            yield batch
        return

    ids = [v.video_id for v in subwave]
    try:
        rows = await asyncio.to_thread(db.get_fit_verdicts, user_id, mode_name,
                                       ids, fingerprint, ttl)
    except Exception:                                   # noqa: BLE001
        logger.warning("S6 cache read failed, treating as all misses\n%s",
                       redacted_traceback())
        rows = {}

    hits = [fit_cache.verdict_from_row(rows[i]) for i in ids if i in rows]
    if hits:
        stats["cached"] += len(hits)
        yield hits                                       # batch zero: free

    misses = [v for v in subwave if v.video_id not in rows]
    if not misses:
        return

    async for batch in judge(misses):
        stats["llm"] += len(batch)
        keep = [vd for vd in batch if not vd.degraded]
        if keep:
            try:
                await asyncio.to_thread(db.save_fit_verdicts, user_id, mode_name,
                                        fingerprint, keep)
            except Exception:                           # noqa: BLE001
                logger.warning("S6 cache write failed, ignored\n%s",
                               redacted_traceback())
        yield batch
