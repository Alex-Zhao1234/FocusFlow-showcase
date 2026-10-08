"""
Abridged shape of one refresh.

The real function is several hundred lines long: it also handles session
intents, user-authored profile entries, seed and followed channels, seat
planning, the candidate library and a dozen edge cases. This excerpt keeps
only the spine so the streaming contract and the stage order are visible.

Contract for callers:
  * intermediate yields are partial batches, section="incoming";
  * the LAST yield is always the complete final snapshot;
  * the plan, timings, cost and conservation check are written to the ledger
    BEFORE the final snapshot leaves, so a client that disconnects early
    cannot lose the accounting.
"""

import asyncio
from collections.abc import AsyncIterator

import config
from agent import judges, planner
from core.models import Mode, RecommendationCard
from pipeline import stability
from pipeline.prefilter import prefilter
from pipeline.ranker import rank
from pipeline.triage import triage
from sources import youtube
from storage import db


async def run(user_id: int, mode: Mode, refresh_id: str,
              lang: str = "zh") -> AsyncIterator[list[RecommendationCard]]:
    # ---- S0: where is this (user, mode) in its state machine? -------------
    state = await asyncio.to_thread(stability.mode_state, user_id, mode)

    # ---- S1: retrieval plan (LLM with a hand-written default per mode) ----
    if state.can_reuse_plan:
        plan = await asyncio.to_thread(db.get_last_plan, user_id, mode.name)
    else:
        plan, plan_source = await planner.make_plan(user_id, mode)

    # ---- S2: fetch. Consumed / disliked / over-exposed ids are removed -----
    #      BEFORE the fetch so quota is not spent on cards that would be dropped.
    exclude = await asyncio.to_thread(db.pre_fetch_exclusions, user_id, mode.name)
    videos = await youtube.execute_plan(plan, user_id, exclude=exclude)
    n_in = len(videos)                      # left-hand side of the conservation check

    # ---- S3: free rules. Every drop writes a ledger row. --------------------
    survivors = prefilter(videos, mode, refresh_id)

    # ---- S4: cost gate. direct / cached / review ----------------------------
    t = triage(survivors, direct_threshold=config.MODE_DIRECT_THRESHOLDS.get(mode.name))

    # ---- S6 wave machine: a closure shared by both producers ----------------
    candidates, reasons = [], {}

    async def s6_wave(wave):
        async for verdicts in judges.judge_mode_fit(wave, mode, user_id, lang):
            cards = []
            for v in verdicts:
                ok, rule = _fit_effective(v)
                if ok:
                    candidates.append(v)
                    reasons[v.video_id] = v.reason
                    cards.append(_incoming_card(v))
                else:
                    db.add_filter_result(refresh_id, v.video_id, stage="S6", rule=rule)
            if cards:
                yield cards

    # ---- First wave runs CONCURRENTLY with S5. ------------------------------
    #      direct + cached survivors do not need the quality judge, so they go
    #      to S6 immediately. The review bucket goes to S5 and feeds S6 in
    #      later waves as batches fill up.
    first_wave, carried = _truncate_first_wave(t.direct + [v for v, _ in t.cached],
                                               limit=config.FIRST_WAVE_MAX)

    async def produce_first_wave(out):
        async for cards in s6_wave(first_wave):
            await out.put(cards)
        await out.put(None)                                  # sentinel

    async def produce_from_s5(out):
        buf = list(carried)
        async for verdicts in judges.judge_quality(t.review):
            for v in verdicts:
                if v.passed:
                    buf.append(v.video)
                else:
                    db.add_filter_result(refresh_id, v.video_id, stage="S5", rule="quality")
            while len(buf) >= config.LLM_BATCH_SIZE:
                batch, buf = buf[:config.LLM_BATCH_SIZE], buf[config.LLM_BATCH_SIZE:]
                async for cards in s6_wave(batch):
                    await out.put(cards)
        if buf:
            async for cards in s6_wave(buf):
                await out.put(cards)
        await out.put(None)

    queue: asyncio.Queue = asyncio.Queue()
    tasks = [asyncio.create_task(produce_first_wave(queue)),
             asyncio.create_task(produce_from_s5(queue))]
    done = 0
    while done < len(tasks):
        item = await queue.get()
        if item is None:
            done += 1
            continue
        yield item                                           # section="incoming"

    # ---- S7: the user's own excluded content types --------------------------
    kept, excluded = _partition_excluded(candidates, user_id, mode.name)
    for v in excluded:
        db.add_filter_result(refresh_id, v.video_id, stage="S7", rule="user_excluded_type")

    # ---- rank once, then seat rules, then split top / fold -----------------
    ranked = rank(kept, mode, user_id, refresh_id)
    snapshot = _final_snapshot(ranked, reasons)

    # ---- accounting BEFORE the final frame ----------------------------------
    n_top = sum(1 for c in snapshot if c.section == "top")
    n_more = sum(1 for c in snapshot if c.section == "more")
    check = conservation_check(refresh_id, n_in, n_top, n_more)
    await asyncio.to_thread(db.log_plan, user_id, mode.name, refresh_id, plan, check)

    yield snapshot                                           # the authoritative final frame
