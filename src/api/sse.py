"""SSE adapter for the refresh endpoint (API presentation layer).

Imports are deliberately narrow: ``pipeline.orchestrator``, ``core.models`` and
``api.schemas``, plus two thin exceptions: ``pipeline.prewarm`` (start
preparing the next batch after ``done``) and ``storage.db`` (mark final cards
as served). It never touches ``agent/``, individual pipeline stages or the
YouTube source; that is the orchestrator's job.

Responsibility boundary: the orchestrator is the only producer and this module
the only consumer. It translates one-to-one and never re-orchestrates: no
regrouping, dedup or reordering of cards here.

Event set: ``card`` / ``done`` / ``error``, i.e. ``card*N -> (done | error)``.
``orchestrator.run()`` is an async generator yielding batches of
``RecommendationCard``: zero or more progressive preview frames
(``section="incoming"``, ``final=None``) followed by the final snapshot
(``section`` in {"top", "more"}, ``final`` set); consumers treat the last
frame as authoritative. ``bridge()`` flattens batches into single cards
without cross-batch dedup. Frame types are distinguished by ``CardOut.final``
and the frontend rebuilds the final state; see ``api.schemas.CardOut``.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import traceback
import uuid
from typing import AsyncIterator

import config                                 # PIPELINE_ERROR_TEXT is looked up per language
from core.logsafe import redact, safe_repr    # Redact exceptions/tracebacks (API keys) before logging
from core.models import Mode, RecommendationCard
from pipeline import orchestrator, prewarm
from storage import db

from api.schemas import CardOut

logger = logging.getLogger(__name__)

# Per-user registry of the task driving the current refresh stream. Starting a new
# stream cancels the old one, closing the window where an old pipeline keeps running
# after the frontend has closed it. Active cancellation is used instead of a per-user
# lock: when cancellation hits a generator suspended at `yield`, its `finally` runs
# at a GC-dependent time, so a lock might never be released and new streams would
# hang. Cancellation has no such risk and matches the frontend semantics ("switching
# mode abandons the old stream"; newest wins). In-process only (single-process
# deployment; multiple workers would each have their own registry).
_ACTIVE_STREAMS: dict[int, asyncio.Task] = {}


def sse(event: str, data: dict) -> str:
    """Encode one SSE frame: ``event`` and ``data`` lines followed by a blank line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def new_refresh_id() -> str:
    """Generate a refresh_id: the first 16 hex chars (64 bits) of a uuid4.

    32 bits would make collisions likely after tens of thousands of refreshes,
    which would mix up per-refresh data across users. Downstream code treats
    the id as an opaque string (cache key / log correlation) and never parses
    it or relies on its length."""
    return uuid.uuid4().hex[:16]


async def heartbeat(frames: AsyncIterator[str], interval: float = 15.0) -> AsyncIterator[str]:
    """Emit an SSE comment line after ``interval`` seconds of silence so reverse
    proxies do not drop the connection as idle.

    EventSource ignores comment lines, so the frontend needs no changes. On
    close, ``aclose`` is propagated to the inner iterator so
    ``refresh_stream``'s ``finally`` (stream deregistration) still runs.

    Context propagation: ``create_task`` without ``context=`` copies the
    current context into each new Task, so a ``ContextVar.set`` made inside one
    ``__anext__`` (e.g. ``llmmeter.start()`` at the top of
    ``orchestrator.run()``) would be invisible to producers created in a later
    ``__anext__``, and their LLM token usage would silently be recorded as 0.
    The fix is to copy the context once and run every ``__anext__`` with
    ``context=ctx``: sets made during ``Context.run`` are written to that same
    object and remain visible. Do not introduce any implicit Task creation
    without ``context=`` between ``start()`` and producer creation.

    Disconnect handling: if the consumer is cancelled while ``nxt`` (the Task
    wrapping the inner ``__anext__``) is still pending, ``CancelledError`` lands
    on ``asyncio.wait`` and we enter ``finally``. ``nxt.cancel()`` only
    *requests* cancellation; calling ``it.aclose()`` immediately would raise
    ``RuntimeError: aclose(): asynchronous generator is already running`` and
    mask the CancelledError. So we first ``await nxt`` (swallowing only its own
    CancelledError/StopAsyncIteration) until the inner call has fully unwound,
    then ``aclose()``. Our own CancelledError is not swallowed and re-raises
    after ``finally``."""
    it = frames.__aiter__()
    # All Tasks driving __anext__() share one Context object, so the ContextVar set
    # by llmmeter.start() stays visible to producers created in later __anext__ calls.
    ctx = contextvars.copy_context()
    nxt: asyncio.Future | None = None
    try:
        while True:
            if nxt is None:
                nxt = asyncio.get_running_loop().create_task(it.__anext__(), context=ctx)
            done, _ = await asyncio.wait({nxt}, timeout=interval)
            if not done:
                yield ": ping\n\n"
                continue
            try:
                frame = nxt.result()
            except StopAsyncIteration:
                return
            nxt = None
            yield frame
    finally:
        if nxt is not None and not nxt.done():
            nxt.cancel()
            try:
                await nxt
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        if hasattr(it, "aclose"):
            await it.aclose()


async def bridge(frames: AsyncIterator[list[RecommendationCard]],
                 ) -> AsyncIterator[RecommendationCard]:
    """Flatten the orchestrator's batch iterator into a single-card iterator.

    This is where "translate one-to-one, never re-orchestrate" is enforced: no
    dedup, no reordering, cards are forwarded in arrival order."""
    async for batch in frames:
        for card in batch:
            yield card


async def refresh_stream(user_id: int, mode: Mode, lang: str = "zh",
                         trial: bool = False) -> AsyncIterator[str]:
    """Core generator behind ``GET /api/refresh``. Event sequence:
    ``card*N -> (done | error)``.

    ``lang`` is passed straight through to ``orchestrator.run``; validation
    happens in the route handler, not here. A ``plan_ack`` event is not
    emitted because ``orchestrator.run()`` does not expose the RetrievalPlan.

    Failure handling (fail-soft):

    - Client disconnect: Starlette injects ``asyncio.CancelledError`` at the
      current await point. It is deliberately *not* swallowed, so it propagates
      through ``bridge()`` into the orchestrator and stops a pipeline that would
      otherwise keep burning API quota after a mode switch or page close.
    - Any other exception from ``orchestrator.run()``: headers were already sent
      with the first frame, so the status code can no longer become a 500.
      Instead an ``error`` event is emitted and the stream ends cleanly, without
      ``done`` (the refresh did not complete). The orchestrator never reaches its
      final card_pool write on this path, so a partial result cannot replace the
      previous good one. The ``error`` payload still carries ``refresh_id`` so the
      failed stream can be correlated in the frontend and in logs.
    """
    refresh_id = new_refresh_id()
    channel_titles: dict[str, str | None] = {}   # Per-stream channel-title cache, reused across frames
    final_ids: list[str] = []   # video_ids of final cards, marked served after `done`

    # Cancel this user's previous stream (see _ACTIVE_STREAMS). `task` is the response
    # task driving this generator; the identity check keeps two sequential streams on
    # the same task (direct-drive tests) from cancelling themselves.
    task = asyncio.current_task()
    prev = _ACTIVE_STREAMS.get(user_id)
    if prev is not None and prev is not task and not prev.done():
        logger.info("refresh[u%s]: 新流 %s 启动,收掉旧流任务", user_id, refresh_id)
        prev.cancel()
    if task is not None:
        _ACTIVE_STREAMS[user_id] = task
    try:
        try:
            # Pass `trial` only for trial accounts, so the normal call shape is unchanged.
            async for card in bridge(orchestrator.run(user_id, mode, refresh_id, lang=lang,
                                                      **({"trial": True} if trial else {}))):
                # On a cache miss from_item looks up the channel in the db; run it in a
                # thread so write-lock contention cannot freeze the event loop.
                out = await asyncio.to_thread(CardOut.from_item, card, channel_titles)
                if out.final:
                    final_ids.append(out.video_id)
                yield sse("card", out.model_dump())
            yield sse("done", {"refresh_id": refresh_id})
            # Final cards have reached the screen, so mark them served; otherwise they
            # would count as fresh stock and the prewarm gate would never fire.
            if final_ids:
                await asyncio.to_thread(db.mark_card_pool_served,
                                        user_id, mode.name, lang, final_ids)
            # One batch delivered -> start preparing the next one (the gate itself
            # checks quota, an in-flight job and existing stock).
            await asyncio.to_thread(prewarm.maybe_start, user_id, mode.name, lang)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Redact before logging: the traceback's last line repeats str(exc), which
            # may contain a full URL with the API key. exc_info=True would bypass
            # redaction, so the traceback is formatted manually and redacted as a whole
            # (stack frames are kept, so no debugging detail is lost).
            logger.warning(
                "orchestrator[%s/u%s] refresh_id=%s 中途异常,降级为 error 事件收尾:%s\n%s",
                mode.name, user_id, refresh_id, safe_repr(exc),
                redact("".join(traceback.format_exception(exc))))
            # The error payload is a fixed, whitelisted text: str(exc) is never sent to
            # the browser (HTTP errors embed the full request URL including the API
            # key; sqlite errors embed SQL). The raw exception only goes to the log
            # above; the client gets the fixed text plus refresh_id for correlation.
            yield sse("error", {
                "stage": "pipeline",
                "text": config.PIPELINE_ERROR_TEXT.get(lang, config.PIPELINE_ERROR_TEXT["zh"]),
                "degraded": True, "refresh_id": refresh_id,
            })
            return
        # The final snapshot is written to card_pool by the orchestrator on its last frame.
    finally:
        # Only remove our own entry; a newer stream may already have replaced it.
        if task is not None and _ACTIVE_STREAMS.get(user_id) is task:
            del _ACTIVE_STREAMS[user_id]
