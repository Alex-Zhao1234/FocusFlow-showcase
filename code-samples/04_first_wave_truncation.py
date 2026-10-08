"""
First-wave truncation.

Part of the measurement-driven pass that took a full refresh from 164.9 s to
58.3 s. Before it, the first wave sent to the mode-fit judge was every
direct-pass and cached candidate, so the first cards waited for all of them.
After it, the first wave is capped: the top `limit` by statistical quality
go first, and the rest are CARRIED into later waves. Nothing is dropped and
nothing is written to the ledger here; this function only reorders.

Two later additions, both no-ops when their argument is absent:

  * intent_head: when the user typed an intent, the candidates that came from
    that intent search are placed at the HEAD of the first wave without
    competing for `limit`. Search results typically have weaker engagement
    numbers than subscribed channels, so a pure statistical cut would push
    the user's own request into later waves.

  * sims: a map {video_id: cosine similarity to the user's profile entries}
    from the vector index. When present, the order of the normal pool becomes
    reciprocal rank fusion of (statistical rank, similarity rank). Videos
    without a vector yet (fetched today, embedded tonight) keep their
    statistical rank only, so the first screen is not all library stock.

`limit=None` is the rollback switch: the function then behaves exactly as it
did before the optimisation.
"""

from collections.abc import Callable

import config
from core.models import Video


def _truncate_first_wave(first_wave: list[Video],
                         stat_of: Callable[[Video], float],
                         limit: int | None, *,
                         intent_head: int = 0,
                         sims: dict[str, float] | None = None,
                         ) -> tuple[list[Video], list[Video]]:
    intent_pool: list[Video] = []
    normal_pool = first_wave
    if intent_head > 0:
        intent_pool = sorted((v for v in first_wave if v.intent_sourced),
                             key=stat_of, reverse=True)
        normal_pool = [v for v in first_wave if not v.intent_sourced]
        head, intent_overflow = intent_pool[:intent_head], intent_pool[intent_head:]
    else:
        head, intent_overflow = [], []

    if limit is None or len(normal_pool) <= limit:
        return head + list(normal_pool), list(intent_overflow)

    ordered = sorted(normal_pool, key=stat_of, reverse=True)   # stable: ties keep insertion order
    if sims:
        k = float(config.RRF_K)
        fused = {v.video_id: 1.0 / (k + i) for i, v in enumerate(ordered, 1)}
        by_sim = sorted((v for v in ordered if v.video_id in sims),
                        key=lambda v: sims[v.video_id], reverse=True)
        for i, v in enumerate(by_sim, 1):
            fused[v.video_id] += 1.0 / (k + i)
        ordered = sorted(ordered, key=lambda v: fused[v.video_id], reverse=True)

    return head + ordered[:limit], ordered[limit:] + intent_overflow
