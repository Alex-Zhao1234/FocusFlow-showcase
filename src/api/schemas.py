from __future__ import annotations
from typing import Literal

from pydantic import BaseModel

from core.models import RecommendationCard
from storage import db


class CardOut(BaseModel):
    """SSE card payload for ``GET /api/refresh``.

    A presentation-layer view, not an alias of RecommendationCard; some fields
    come from elsewhere:

    - ``channel_title``: not part of the Video model (it lives in the channels
      table), so it is looked up by ``channel_id`` via ``storage.db.get_channel``.
    - ``categories``: RecommendationCard does not carry category data
      (``QualityVerdict.categories`` exists only for rejected videos), so this
      is always an empty list rather than invented data.
    - ``quality_score``: taken from ``card.quality_norm`` (the normalised
      quality score).
    - ``slot``: ``RecommendationCard.section`` has three states
      (incoming/top/more) while the payload defines two. ``incoming`` cards are
      progressive preview frames (not yet finally ranked) and are shown in the
      top area for now; the ``more`` fold is decided only by the final snapshot.
      On the first frame with ``final=True`` the frontend clears the previews
      and rebuilds the list from the final frames (the last frame wins).
    """
    video_id: str
    title: str
    channel_id: str
    channel_title: str | None = None
    duration_sec: int
    cover_url: str | None = None
    view_count: int = 0
    like_count: int = 0
    source_strategy: str
    publish_time: int | None = None   # Upload time (Unix seconds), shown as a date on the card; None for older cached cards
    quality_score: float | None = None
    categories: list[str] = []
    reason: str = ""
    slot: Literal["top", "more"]
    final: bool = False          # False = incoming preview frame; True = final snapshot frame
                                 # (derived from RecommendationCard.final is not None)
    overlay: bool = False        # Final card of a lightweight intent refresh, overlaid onto the feed by the frontend

    @classmethod
    def from_item(cls, card: RecommendationCard,
                  channel_titles: dict[str, str | None] | None = None) -> "CardOut":
        """The single translation point from RecommendationCard (the S6 output)
        to CardOut (the SSE presentation projection); every card flattened by
        ``api.sse.bridge`` passes through here once.

        ``channel_titles`` is a per-stream cache owned by the caller, so multiple
        cards from the same channel in one refresh hit the db only once. The API
        layer builds its own cache rather than reaching into pipeline state.
        Pass ``None`` to look up every card individually (convenient in offline
        tests)."""
        video = card.video
        if channel_titles is not None and video.channel_id in channel_titles:
            channel_title = channel_titles[video.channel_id]
        else:
            row = db.get_channel(video.channel_id)
            channel_title = row["channel_title"] if row is not None else None
            if channel_titles is not None:
                channel_titles[video.channel_id] = channel_title
        slot: Literal["top", "more"] = "more" if card.section == "more" else "top"
        return cls(
            video_id=video.video_id, title=video.title, channel_id=video.channel_id,
            channel_title=channel_title, duration_sec=video.duration_sec,
            cover_url=video.cover_url, view_count=video.view_count,
            like_count=video.like_count, source_strategy=video.source_strategy,
            publish_time=video.publish_time,
            quality_score=card.quality_norm, categories=[], reason=card.reason,
            slot=slot, final=card.final is not None, overlay=card.overlay,
        )


# ============================================================================
# Video favorites
# ============================================================================
FOLDER_NAME_MAX = 50   # Folder-name length limit (keeps the list UI intact); longer names get a 422, never silently truncated
