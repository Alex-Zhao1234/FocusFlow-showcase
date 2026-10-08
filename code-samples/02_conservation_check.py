"""
The invariant that closes every refresh.

    len(videos entering S3) == len(top) + len(more) + rows in filter_results for this refresh

Every stage that drops a video must write a ledger row. If any stage drops
silently, or any stage writes twice, the equation breaks and the orchestrator
logs an error with both sides. The same function is called by the acceptance
tests, so there is exactly one definition of "conserved".

It is cheap (one COUNT query), runs on every production refresh, and is the
first thing read whenever card counts look wrong. When output dropped after a
performance change, this check showed the two were unrelated.
"""

from storage import db


def conservation_check(refresh_id: str, n_s3_input: int,
                       n_top: int, n_more: int) -> dict:
    n_filtered = len(db.get_filter_results(refresh_id))
    rhs = n_top + n_more + n_filtered
    return {
        "ok": n_s3_input == rhs,
        "lhs": n_s3_input,
        "rhs": rhs,
        "n_top": n_top,
        "n_more": n_more,
        "n_filtered": n_filtered,
    }


# In the orchestrator, after rank and before the final frame is emitted:
#
#     check = conservation_check(refresh_id, n_in, n_top, n_more)
#     if not check["ok"]:
#         logger.error("conservation broken: %d != top+more+filtered %d (%s)",
#                      check["lhs"], check["rhs"], check)
#
# Two things are deliberately OUTSIDE the equation and documented as such:
#   * appeal exemptions are routed around S3..S7 before the count starts, so
#     the left-hand side never contains them;
#   * "tail seats" (soft-demoted cards shown after the fold) are not counted as
#     shown, because they were already accounted for when first demoted.
