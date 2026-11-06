"""Background "shelf" prewarming for instant refreshes.

For each ``(user_id, mode)`` at most one background task runs a full
``orchestrator.run`` (frames are discarded; the final results are written to
the card pool inside ``run``), so that when the user taps "N new items" the
next batch can be swapped in instantly.

Prewarming is charged against the daily refresh quota: ``try_count_refresh``
is the single charging point, and one pipeline run is one charge regardless of
who triggered it.

The only trigger is "a batch was delivered" (SSE done / shelf swap). A
completed prewarm never triggers another one: during quiet periods (a reused
steady plan yields only old videos, so the shelf stays empty) self-triggering
would burn the whole quota in a chain.

The event loop is injected by the app lifespan. Without it (tests, offline
scripts) ``maybe_start`` always returns False and has no side effects.
"""
from __future__ import annotations

import asyncio
import logging
import os

import config
from pipeline import orchestrator
from storage import db

logger = logging.getLogger(__name__)

# Event loop handle, injected at startup.
_LOOP: asyncio.AbstractEventLoop | None = None
# At most one running task per (user_id, mode_key).
_TASKS: dict[tuple[int, str], asyncio.Task] = {}


def set_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Inject the event loop handle (called from the app lifespan).

    Sync FastAPI routes run in worker threads that have no running loop, so
    ``asyncio.get_running_loop()`` would raise there; background tasks can
    only be scheduled through this stored handle via ``call_soon_threadsafe``.
    """
    global _LOOP
    _LOOP = loop


def is_running(user_id: int, mode_key: str) -> bool:
    t = _TASKS.get((user_id, mode_key))
    # done() covers normal completion, exception and cancellation.
    return t is not None and not t.done()


def cancel_all() -> None:
    """Cancel all running prewarm tasks (app shutdown).

    Charged quota is not refunded -- same as a user closing the page mid-stream.
    """
    for t in _TASKS.values():
        if not t.done():
            t.cancel()


async def _run(user_id: int, mode_key: str, lang: str) -> None:
    from api.sse import new_refresh_id   # deferred import to avoid a cycle (api -> pipeline is the main direction)
    mode = config.MODES[mode_key]
    refresh_id = new_refresh_id()
    logger.info("prewarm[u%s/%s/%s]: 预备批起跑 refresh_id=%s",
                user_id, mode_key, lang, refresh_id)
    try:
        # Frames are discarded: only the side effect (results written to the
        # card pool) matters. consume_intent=False: background batches only
        # prepare regular supply and must never consume a session intent the
        # user is waiting for in the foreground -- intent refreshes are always
        # served by the SSE stream.
        async for _ in orchestrator.run(user_id, mode, refresh_id, lang=lang,
                                        consume_intent=False):
            pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("prewarm[u%s/%s]: 预备失败(额度已扣,fail-soft):%r",
                       user_id, mode_key, exc)


def maybe_start(user_id: int, mode_key: str, lang: str) -> bool:
    """Decide whether to start a prewarm; thread-safe. Returns True if started.

    Gates:
      1. the event loop is available;
      2. no prewarm is running for this key;
      3. trial accounts are skipped;
      4. the shelf has no fresh unserved batch (stay at most one batch ahead);
      5. at least ``PREWARM_MIN_CREDITS_LEFT`` refreshes remain today (the
         last ones are reserved for the user).

    Known trade-off: the checks and the task creation are not atomic, so a
    concurrent window could charge twice. Charging itself is atomic, so the
    worst case is one extra batch without exceeding the limit -- not worth a
    lock.
    """

    # 1. No injected loop (tests / offline scripts): do nothing.
    if _LOOP is None or _LOOP.is_closed():
        return False

    # 2. One task per user and mode, to avoid burning quota twice.
    if is_running(user_id, mode_key):
        return False

    # 3. Trial accounts: background runs have no device key, so per-device/IP
    # limits cannot apply; an invisible extra pipeline per refresh is
    # unacceptable. An existing shelf is still served (no LLM involved).
    row = db.get_user_by_id(user_id)
    if row is not None and row["is_trial"]:
        return False

    # 4. Shelf already holds an unserved, non-stale batch.
    if db.unserved_card_pool_meta(user_id, mode_key, lang,
                                  config.POOL_STALE_HOURS)[0] > 0:
        return False
    # 5. Daily quota headroom.
    limit = int(os.environ.get("FOCUSFLOW_DAILY_REFRESH_LIMIT",
                               config.DAILY_REFRESH_LIMIT))
    used = db.user_daily_usage(user_id)["refreshes"]

    if limit - used < config.PREWARM_MIN_CREDITS_LEFT:
        return False
    # Atomic charge; prewarming counts as a refresh.
    if not db.try_count_refresh(user_id, limit):
        return False

    # call_soon_threadsafe takes a plain callable, not a coroutine, hence the closure.
    def _spawn() -> None:
        _TASKS[(user_id, mode_key)] = _LOOP.create_task(
            _run(user_id, mode_key, lang))

    # get_running_loop() raises (rather than returning None) off the loop thread.
    try:
        same_loop = asyncio.get_running_loop() is _LOOP
    except RuntimeError:
        same_loop = False

    if same_loop:
        _spawn()
    else:
        _LOOP.call_soon_threadsafe(_spawn)    # the only thread-safe way to schedule onto the loop
    return True
