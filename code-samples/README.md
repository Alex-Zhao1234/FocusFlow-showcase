# Selected code

Six excerpts from the private codebase, cleaned for reading. Comments were rewritten in English and internal batch references removed. Signatures and control flow are faithful to the real code; some arguments and logging were dropped for length. **These files are not runnable on their own.**

| File | What it shows |
|---|---|
| [01_orchestrator_skeleton.py](01_orchestrator_skeleton.py) | The shape of one refresh: S0 → S7 as an async generator that streams waves and always ends with a final snapshot. Abridged from a much longer function. |
| [02_conservation_check.py](02_conservation_check.py) | The invariant that closes every refresh: `in == shown + folded + ledger rows`. |
| [03_fit_with_cache.py](03_fit_with_cache.py) | The mode-fit judge wrapped in a 7-day cache: hits are emitted first with no model call, misses go to the judge, non-degraded verdicts are written back, every failure fails open. |
| [04_first_wave_truncation.py](04_first_wave_truncation.py) | The first-wave cut from the 164.9 s → 58.3 s optimisation, including the later addition of rank fusion with vector similarity. |
| [05_triage.py](05_triage.py) | The S4 cost gate, complete: three buckets, zero writes, no user identity, no magic numbers. |
| [06_fail_soft_tests.py](06_fail_soft_tests.py) | How the degraded paths are tested: the model is made to fail, the endpoint must still return 200 with a flag, and the user's raw input must survive. |

All thresholds, weights and budgets referenced here live in a single config module in the private repo and are intentionally not shown.
