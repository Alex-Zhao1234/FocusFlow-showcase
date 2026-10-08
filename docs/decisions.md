# Design decisions

Twelve decisions that shaped the product, in the order a reader is most likely to ask about them. Each one follows the same shape: the problem, the choice, what it cost, and what happened.

---

### 1. Intent before content, instead of a smarter feed

**Problem.** Every recommender I looked at, including the ones I could build, optimises for engagement because that is what the data rewards. A "better" feed is still a feed.

**Choice.** Make the user declare intent first (a mode, optionally a sentence), and treat everything outside that intent as out of scope rather than low-ranked. One refresh yields a finite batch. No infinite scroll.

**Cost.** Friction. A mode picker before the first card is a step most products would A/B away. Some users will type an intent that the catalogue cannot satisfy.

**Result.** The product has a reason to exist. Every downstream decision (the ledger, the judges, the seat rules) only makes sense once "out of scope" is a first-class outcome.

### 2. A ledger for every rejection, and a conservation check

**Problem.** A gatekeeper that hides things is only trustworthy if it can show what it hid and why.

**Choice.** Every drop at every stage writes a row (video, stage, rule, reason). The UI shows the ledger. Users can appeal; accepted appeals become standing exemptions. After every refresh: `in == shown + folded + ledger rows`, asserted in code and in acceptance tests.

**Cost.** One more table on the hot path and a discipline that every new stage has to follow.

**Result.** The conservation check is the first thing read whenever card counts look wrong. When output dropped after a performance change, the check showed the two were unrelated and pointed the investigation elsewhere. Transparency turned out to be a debugging tool as much as a product feature.

### 3. One process, one SQLite file

**Problem.** The complexity budget is finite. Where should it go?

**Choice.** FastAPI, a single process, SQLite in WAL mode, the frontend served from the same process. No queue, no cache server, no second database. All SQL in one module; schema migration is add-only and idempotent.

**Cost.** Vertical scaling only. Nightly jobs share the machine. Some things (vector search, full-text search) had to be done inside SQLite rather than with a dedicated service.

**Result.** Months of iteration spent on filtering decisions rather than infrastructure. When hybrid retrieval was needed, `sqlite-vec` and FTS5 did the job inside the same file.

### 4. Every LLM call in one layer, every call with a coded fallback

**Problem.** A product that depends on a model API will see that API fail, slow down or change behaviour.

**Choice.** All model calls live in one package. Each agent ships with a pure-code path for failure: default plans, fail-open verdicts flagged as degraded, literal-text fallbacks, raw input stored before structuring. The API layer returns a degraded response with a flag, never a 500. Tests assert the degraded path for every agent.

**Cost.** Roughly double the code per agent, and the degraded paths need their own tests and their own logging so you notice when they are running.

**Result.** Model outages and rate limits have never taken the site down. Degradation rate is a metric, not an incident.

### 5. A pure-code cost gate in front of the judges

**Problem.** Running a model on every candidate is both slow and expensive, and most candidates do not need it.

**Choice.** Triage (S4) splits candidates into direct pass, cached verdict and needs-review using free signals and two 7-day caches. Only the review bucket reaches the quality judge. The mode-fit judge checks its cache before every wave.

**Cost.** Two caches to invalidate correctly. The mode-fit cache is keyed by a fingerprint of prompt version, model, language and the user's profile text, so any of those changing invalidates it.

**Result.** In a tracked production refresh, 62 of 102 candidates skipped the quality judge and the fit cache answered half of the fit rulings for free.

### 6. Stream waves, paint the last result first

**Problem.** A full refresh takes tens of seconds. A spinner for that long loses the user.

**Choice.** Server-Sent Events, one wave per judged batch. The first wave is the direct-pass and cached candidates, which do not wait for the quality judge. Before any of that, the previous session's final frame is painted from a local pool. A visible progress ribbon shows which stage is working.

**Cost.** A frame model in the client (preview frames settle on an authoritative final frame) and careful ordering rules so that a card shown early can still be demoted or dropped by the final rank.

**Result.** Measured first paint of 21 ms on a return visit. The ribbon became the product's signature.

### 7. Measure, then parallelise

**Problem.** The full refresh took 164.9 s on real data. Guessing where the time went would have been easy and wrong.

**Choice.** Instrument every stage with wall-clock laps. The data showed the two LLM stages were serialised against each other and against their own batches. Three changes: run the first wave concurrently with the quality judge, judge batches concurrently with a concurrency cap, and truncate the first wave by statistical score so the first cards do not wait for the whole set.

**Cost.** Async generators that yield from inside concurrent producers, which took one careful rewrite to get right.

**Result.** 164.9 s → 58.3 s, a 65 % reduction, verified on the same data.

### 8. Treat API quota as a product constraint, not an ops detail

**Problem.** YouTube's API has daily quotas, and the search endpoint has its own much smaller one. Running out mid-day means an empty product.

**Choice.** Two separate ledgers, every call charged by the adapter, the planner clears both budgets before returning a plan, offline jobs run against a fixed reserve. A nightly pre-warmed shelf, a zero-cost local candidate library, and verdict caches all exist primarily to spend less quota per refresh.

**Cost.** Accounting code everywhere an API call happens, and a planner that has to reason about budget.

**Result.** The real bottleneck turned out to be the search bucket, not the headline quota. Finding that out required the ledgers.

### 9. Hybrid retrieval inside SQLite, with a before/after evaluation

**Problem.** Keyword search alone misses relevant videos, especially across Chinese and English. A vector database would break decision 3.

**Choice.** `sqlite-vec` for dense vectors from a small multilingual model, FTS5 with `jieba` pre-segmentation for lexical search, reciprocal rank fusion, then a per-interest segmentation so one dominant interest cannot crowd out the others. Every piece behind a flag.

**Cost.** A nightly embedding job and a one-time backfill. One early finding: the default FTS tokenizer treats a run of Chinese characters as a single token, which is why `jieba` is there.

**Result.** Each stage was evaluated against the pre-upgrade baseline before being left on. The evaluation also uncovered that the nightly job had silently stopped for a week, which is its own lesson about monitoring cron.

### 10. No visible profiling panel

**Problem.** Personalisation needs a model of the user. Showing that model ("we think you like…") makes people feel watched.

**Choice.** A product red line: no panel that displays an inferred profile. Personalisation is driven by entries the user writes in their own words, per mode, and can reorder, edit or delete. Inferred state exists internally and shapes seat allocation, but is never surfaced as "this is you".

**Cost.** Less explainability of *why* a card appeared, offset by the rejection ledger which explains why things did *not* appear.

**Result.** The red line was later narrowed, not removed: user-authored entries are visible and editable because the user wrote them; inferred state still is not.

### 11. Cost-tiered models, with one failed experiment

**Problem.** The best model for every call is the most expensive model for every call.

**Choice.** A small model by default for the judges and planner, escalation only where measurement showed it mattered, a larger model for conversational surfaces where multi-turn context is the point. A same-day trial of a cheaper third-party model was rolled back the same day: slower by 1.2–1.6×, lower judge quality, and a new class of ID-mismatch errors.

**Cost.** Two model configurations to keep in sync.

**Result.** The LLM bill scales with the review bucket, not with traffic.

### 12. Cut scope to ship, then iterate in small batches

**Problem.** The original plan had an eighth stage, a fully automated weekly feedback loop. It was the most academically interesting piece and the least likely to change what users saw.

**Choice.** Ship without it. Keep a manual weekly review with a CLI instead. After the core was stable, switch the cadence to small daily batches, each with a written spec, tests and a production check, rather than large campaigns.

**Cost.** A less complete story on paper.

**Result.** The product reached production and has been iterated on continuously since. The small-batch cadence produced 664 commits in under four months, with production incidents measured in minutes.
