"""Abstract data-source interface.

``SourceAdapter`` is the tool contract between the agent layer and concrete
platforms: supporting a new platform means adding an adapter, with no change
to the pipeline skeleton. Depends only on ``core.models``.

Design constraints:
  1. Five abstract methods: four retrieval tools plus one completion step
     (``enrich_stats``).
  2. The four retrieval method names match the four ``StrategyCall.tool``
     literals exactly, so the orchestrator can dispatch by tool name.
  3. Every method returns ``list[Video]``. Retrieval results may carry
     placeholder statistics (filled in later by ``enrich_stats``), but
     ``video_id``, ``channel_id`` and ``source_strategy`` must be set: they are
     required for deduplication and persistence.
  4. The ABC contains no implementation, not even quota accounting, which is
     platform-specific and belongs to the concrete adapter.
"""
from abc import ABC, abstractmethod

from core.models import Video


class SourceAdapter(ABC):
    """Multi-platform data-source interface.

    The four retrieval methods correspond to the four ``StrategyCall.tool``
    literals; ``enrich_stats`` stands apart because it completes statistics
    rather than retrieving. Subclasses must override all five methods or
    instantiation raises ``TypeError``.

    The interface is async from the base up: the data layer runs several
    strategies concurrently and is latency- and quota-sensitive, so requests
    should overlap instead of waiting serially.
    """

    @abstractmethod
    async def fetch_channel_videos(self, channel_id: str, n: int) -> list[Video]:
        """Latest videos from a subscribed/preferred channel -> source_strategy='subscribed'."""
        ...

    @abstractmethod
    async def search_by_keyword(self, query: str, n: int) -> list[Video]:
        """Keyword search -> source_strategy='search'."""
        ...

    @abstractmethod
    async def fetch_trending_by_category(self, category_id: int, n: int) -> list[Video]:
        """Regional trending videos for a category -> source_strategy='trending'."""
        ...

    @abstractmethod
    async def mine_hidden_gems(self, k: int) -> list[Video]:
        """Underrated videos mined from the small-channel pool -> source_strategy='hidden_gem'."""
        ...

    @abstractmethod
    async def enrich_stats(self, videos: list[Video]) -> list[Video]:
        """Batch-fill statistics/contentDetails onto bare videos (completion, not retrieval)."""
        ...
