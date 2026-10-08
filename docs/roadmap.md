# Roadmap

A short, honest account of where the project is and what comes next. Dates are 2026.

## Shipped

| When | What |
|---|---|
| June | Data model, YouTube adapter, the pure-code pipeline stages, the first judge, the orchestrator. |
| July | Filter system with ledger and appeals, mode system, channel-pool building, the planner. |
| August | React frontend with SSE streaming, i18n, favourites, settings, cold-start wizard, instant first paint, chat assistant. Public deployment with TLS, rate limiting and consent. Thesis submitted. |
| September | Negative preferences, user-authored profile entries with seat allocation, seed and followed channels, viewing plans, judge caches and the candidate library, quota dual-ledger, compliance batch (deletion, export, retention), security hardening (CSP, headers, secret handling), hybrid vector + full-text retrieval, mobile layout pass. |

## Current cadence

Small batches, roughly one per day. Each batch:

1. starts as a written spec with numbered decisions,
2. lands with tests (the suite is at 1,347 backend + 497 frontend),
3. is deployed the same day,
4. is checked against production logs for a week before the spec is closed.

## In progress or under observation

- Before/after evaluation of the hybrid retrieval upgrade, stage by stage, now that the embedding job has been repaired and backfilled.
- Observation windows on several September batches: diversity seats, channel cooldown after dislikes, impression-based soft demotion.
- Screenshots and a short demo video for this showcase.

## Deliberately not planned

- **An automated weekly feedback loop.** Designed, then cut to ship. A manual CLI review exists instead.
- **Multi-tenant scale-out.** One process and one SQLite file are a choice, not a limitation hit so far.
- **A visible user profile.** See decision 10 in [decisions.md](decisions.md).
- **Open-sourcing the core.** The prompts, ranking rules and pool-building method stay private. This showcase is the public surface.
