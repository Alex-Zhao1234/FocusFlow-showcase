"""Quota cost estimation for a RetrievalPlan — the single source of truth.

Lives in the core layer (depends only on config and core.models) so both the
agent layer (planner's validate_plan) and the sources layer (execute_plan's
pre-flight assertion) price a plan with exactly the same formula. The agent
layer is not allowed to import sources/, and duplicating the function would
let the two copies drift.

The LLM planner reports its own ``est_quota``, but that number is not trusted:
cost is always recomputed here from ``config.QUOTA_COST``. The YouTube quota is
modelled as two buckets — a general unit pool and a separate per-call bucket
for ``search.list`` — so the estimate is returned as a ``PlanCost`` pair.
"""
import math
from dataclasses import dataclass

import config as _cfg                              # read S2_OVERFETCH_FACTOR / CHANNEL_DEEP_PAGES_MAX at call time
from config import QUOTA_COST, SMALL_POOL_SAMPLE_K
from core.models import RetrievalPlan

# Enrichment is billed per videos.list batch of up to 50 ids. The sources layer
# imports this constant for its real batching (enrich_stats / channels.list), so
# the billing formula and the actual batching can never diverge.
ENRICH_BATCH_SIZE = 50


@dataclass(frozen=True)
class PlanCost:
    """Two-bucket estimate for one plan: general = general-pool units,
    search = number of search.list calls. Frozen because it is a deterministic
    value derived from constants and must not be mutated downstream."""
    general: int
    search: int

    @property
    def total(self) -> int:
        return self.general + self.search

# Used by the planner's validation step to assert the plan is affordable
# (compared against PLAN_QUOTA_BUDGET_RATIO of the remaining quota); an
# over-budget plan is sent back for re-planning or replaced by a static plan.
def estimate_plan_cost(plan: RetrievalPlan) -> PlanCost:
    """Recompute a plan's two-bucket cost from config.QUOTA_COST, ignoring est_quota.

    General pool: retrieval strategies plus enrichment batches; search bucket:
    search.list calls. Channel fetches are priced per channel and hidden-gem
    mining per sampled channel. Incidental channels.list lookups that cannot be
    predicted up front are not counted — the assertion only needs the right order
    of magnitude, not unit-exact accounting."""
    general = 0
    search = 0
    for s in plan.strategies:
        p = s.params
        if s.tool == "fetch_channel_videos":
            if p.get("use_library") or p.get("use_local_intent"):
                n_ch = 0                                          # local candidate library / local intent recall: zero API (enrichment still counted below)
            elif p.get("use_seed_channels"):
                n_ch = len(p.get("seed_channel_ids") or [])      # seed channels: 1 unit per channel
            elif p.get("use_expanded_channels"):
                n_ch = len(p.get("expanded_channel_ids") or [])  # expanded-channel track: 1 unit per channel
            elif p.get("use_top_channels") or p.get("use_type_pool"):
                # Defaults when the plan omits n_channels: type pool 8, top channels 10.
                n_ch = p.get("n_channels", 8 if p.get("use_type_pool") else 10)
            else:
                n_ch = 1                     # neither flag set: a single explicitly named channel
            # One playlistItems.list call per channel, priced at the worst case of
            # CHANNEL_DEEP_PAGES_MAX pages (1 = deep paging disabled), so the budget
            # gate and the in-flight reservation are both conservative.
            general += n_ch * QUOTA_COST["fetch_channel_videos"] * max(1, int(_cfg.CHANNEL_DEEP_PAGES_MAX))
        elif s.tool == "mine_hidden_gems":
            # Priced per sampled channel k (defaults to SMALL_POOL_SAMPLE_K).
            general += p.get("k", SMALL_POOL_SAMPLE_K) * QUOTA_COST["mine_hidden_gems"]
        elif s.tool == "search_by_keyword":
            # A query batch costs one search.list per query; a single query costs one.
            search += QUOTA_COST[s.tool] * max(1, len(p.get("queries") or []) or 1)
        else:
            # Everything else (e.g. trending by category) is one API call per strategy.
            general += QUOTA_COST[s.tool]
            
    # Enrichment = ceil(total over-fetched items / 50) batches. Mirrors the
    # over-fetch truncation in the sources layer: non-intent strategies fetch
    # ceil(target × S2_OVERFETCH_FACTOR), intent-driven ones are not over-fetched.
    # Batches saved by the duration pre-check are not deducted, so the estimate
    # errs on the high side and the gate only gets more conservative.
    total_target = sum(
        s.target_count if s.intent_driven
        else math.ceil(s.target_count * _cfg.S2_OVERFETCH_FACTOR)
        for s in plan.strategies)
    general += math.ceil(total_target / ENRICH_BATCH_SIZE) * QUOTA_COST["get_video_stats"]
    return PlanCost(general=general, search=search)
