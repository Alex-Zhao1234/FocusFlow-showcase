"""Storage layer facade.

The retrieval functions (hybrid vector + full-text search, the mode-fit cache and
the local candidate library) are real and live in ``storage.retrieval``. Everything
else -- users, auth, quota ledgers, preferences, filter records and other CRUD -- is
part of the private codebase and is represented here by signature-only stubs that
raise ``NotImplementedError``."""
from __future__ import annotations
from typing import Any

from storage.retrieval import (  # noqa: F401
    EntryVecGroup,
    _chunks,
    _fts_query,
    _init_vector_tables,
    _knn,
    _rowids_for_texts,
    _rowids_for_videos,
    channel_sims,
    delete_text_vecs,
    delete_video_fts,
    delete_video_vecs,
    entries_for_embedding,
    entry_vec_groups_for_mode,
    entry_video_scores,
    fts_search_videos,
    get_conn,
    get_fit_verdicts,
    get_text_vec,
    get_videos_by_ids,
    hybrid_search,
    intent_local_candidates,
    library_candidates,
    neg_prefs_for_embedding,
    pool_channels_by_vec,
    pool_channels_by_vecs,
    prune_fit_verdicts,
    prune_video_vec_orphans,
    rerank_video_ids_by_entry_groups,
    rerank_video_ids_by_vecs,
    save_fit_verdicts,
    seed_channels_for_mode,
    subtopic_vecs_for_entry,
    subtopics_for_embedding,
    text_vec_hashes,
    text_vec_ready,
    upsert_text_vec,
    upsert_video_fts,
    upsert_video_vec,
    vec_search_texts,
    vec_search_videos,
    vec_stats,
    vector_available,
    video_sims_max,
    video_vec_hashes,
    videos_for_embedding,
)


def get_user_by_id(user_id) -> Any:
    raise NotImplementedError

def get_user_subtopic_search(user_id) -> Any:
    raise NotImplementedError

def get_channel(channel_id) -> Any:
    raise NotImplementedError

def touch_channel_fetched(channel_id) -> Any:
    raise NotImplementedError

def merge_channel(channel) -> Any:
    raise NotImplementedError

def upsert_video(video) -> Any:
    raise NotImplementedError

def video_durations(video_ids) -> Any:
    raise NotImplementedError

def save_quality_verdict(verdict) -> Any:
    raise NotImplementedError

def get_quality_verdicts(video_ids, ttl_days=None) -> Any:
    raise NotImplementedError

def upsert_channel_preference(user_id, channel_id, pref_score=None, source=None, trusted=None, blacklisted=None) -> Any:
    raise NotImplementedError

def get_channel_pref(user_id, channel_id) -> Any:
    raise NotImplementedError

def effective_channel_pref(user_id, channel_id) -> Any:
    raise NotImplementedError

def list_channel_prefs(user_id) -> Any:
    raise NotImplementedError

def top_channels(user_id, limit=None, now=None) -> Any:
    raise NotImplementedError

def sample_small_pool(user_id, k=None) -> Any:
    raise NotImplementedError

def blacklisted_channel_ids(user_id) -> Any:
    raise NotImplementedError

def trusted_channel_ids(user_id) -> Any:
    raise NotImplementedError

def user_local_day(user_id) -> Any:
    raise NotImplementedError

def rotation_channels(user_id, day, min_pref=None, loved_pref=None, now=None) -> Any:
    raise NotImplementedError

def bump_channel_picks(user_id, channel_ids, day) -> Any:
    raise NotImplementedError

def channel_entry_affinity(user_id, mode, positions, days=None) -> Any:
    raise NotImplementedError

def subscribed_channel_ids(user_id) -> Any:
    raise NotImplementedError

def set_channel_subscribed(user_id, channel_id, subscribed) -> Any:
    raise NotImplementedError

def aggregate_recent_feedback(user_id, n=None) -> Any:
    raise NotImplementedError

def get_actions(user_id, mode=None, since=None) -> Any:
    raise NotImplementedError

def hourly_action_stats(user_id, mode, days=None) -> Any:
    raise NotImplementedError

def accepted_appeal_video_ids(user_id) -> Any:
    raise NotImplementedError

def get_negative_prefs(user_id, mode=None) -> Any:
    raise NotImplementedError

def explicit_dislike_video_ids(user_id, mode) -> Any:
    raise NotImplementedError

def cooled_channel_ids(user_id, now) -> Any:
    raise NotImplementedError

def save_retrieval_plan(user_id, mode, plan_json, per_strategy_counts=None, pass_rate=None, reused=None) -> Any:
    raise NotImplementedError

def get_latest_retrieval_plan(user_id, mode=None) -> Any:
    raise NotImplementedError

def get_user_profile(user_id) -> Any:
    raise NotImplementedError

def get_active_intent(user_id, mode) -> Any:
    raise NotImplementedError

def count_signals(user_id, mode, days=None) -> Any:
    raise NotImplementedError

def recent_dislike_rate(user_id, mode, days=None) -> Any:
    raise NotImplementedError

def recent_pass_rate_std(user_id, mode, k) -> Any:
    raise NotImplementedError

def play_rate_series(user_id, mode, k) -> Any:
    raise NotImplementedError

def add_filter_result(user_id, refresh_id, video_id, stage, rule, reason) -> Any:
    raise NotImplementedError

def get_filter_results(refresh_id, user_id=None) -> Any:
    raise NotImplementedError

def _pacific_today() -> Any:
    raise NotImplementedError

def record_quota(cost, bucket=None) -> Any:
    raise NotImplementedError

def quota_used(day=None, bucket=None) -> Any:
    raise NotImplementedError

def quota_remaining(bucket=None) -> Any:
    raise NotImplementedError

def try_count_refresh(user_id, limit) -> Any:
    raise NotImplementedError

def bump_user_usage(user_id, yt_units, llm_in, llm_out) -> Any:
    raise NotImplementedError

def user_search_used(user_id, kind) -> Any:
    raise NotImplementedError

def bump_user_search(user_id, kind, n) -> Any:
    raise NotImplementedError

def user_daily_usage(user_id, day=None) -> Any:
    raise NotImplementedError

def get_mode_prefs(user_id, mode) -> Any:
    raise NotImplementedError

def expanded_channels_by_entry(user_id, entry_ids, per_entry, limit, exclude) -> Any:
    raise NotImplementedError

def advance_expand_cursor(user_id, entry_id, n) -> Any:
    raise NotImplementedError

def get_mode_profile_entries(user_id, mode) -> Any:
    raise NotImplementedError

def entry_subtopics_for_mode(user_id, mode) -> Any:
    raise NotImplementedError

def advance_entry_subtopic_cursor(user_id, entry_ids, field=None) -> Any:
    raise NotImplementedError

def bump_entry_feature_ledger(user_id, mode, featured) -> Any:
    raise NotImplementedError

def recent_played_entry_positions(user_id, mode, window_sec) -> Any:
    raise NotImplementedError

def advance_search_cursor(user_id, mode, n) -> Any:
    raise NotImplementedError

def pool_channels_for_chips(chips, limit, exclude_ids=None) -> Any:
    raise NotImplementedError

def pool_channels_for_mode(user_id, allowed_bases, exclude_ids, prefer_chips, limit) -> Any:
    raise NotImplementedError

def get_user_chips(user_id) -> Any:
    raise NotImplementedError

def upsert_card_pool(user_id, mode, lang, rows) -> Any:
    raise NotImplementedError

def mark_card_pool_served(user_id, mode, lang, video_ids) -> Any:
    raise NotImplementedError

def unserved_card_pool_meta(user_id, mode, lang, within_hours) -> Any:
    raise NotImplementedError

def user_card_pool_video_ids(user_id) -> Any:
    raise NotImplementedError

def suppressed_video_ids(user_id) -> Any:
    raise NotImplementedError

def impression_tail_rows(user_id, mode, lang, limit, exclude_ids=None, *, min_quality=None, seed_channels=None) -> Any:
    raise NotImplementedError

def window_impressions(user_id, video_ids) -> Any:
    raise NotImplementedError

def consumed_suppress_sets(user_id) -> Any:
    raise NotImplementedError

def get_chip_candidates(chips, lang, max_age_days) -> Any:
    raise NotImplementedError

def seed_channels_by_entry(user_id, entry_ids, limit, first_all=None) -> Any:
    raise NotImplementedError

def plan_served_video_ids(plan_id) -> Any:
    raise NotImplementedError

def get_channel_cursor(user_id, channel_id) -> Any:
    raise NotImplementedError

def save_channel_cursor(user_id, channel_id, next_page_token, pages_advanced, exhausted, reset=None) -> Any:
    raise NotImplementedError
