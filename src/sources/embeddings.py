"""Embedding models for vector retrieval.

This is the single place where text is turned into vectors. The storage layer
(``storage/retrieval.py``) only ever sees the packed ``bytes`` produced here.

Design points:

* **Lazy model loading.** ``fastembed`` (ONNX runtime) is imported only when
  ``FastEmbedder`` first embeds something. The web server process imports just
  the protocol, ``FakeEmbedder`` and ``to_blob``, and never loads the model,
  which takes roughly 900 MB resident -- too much to keep alongside the API on
  a small host. Embedding therefore runs as a separate offline job.
* **Two entry points.** bge-small-zh is an asymmetric model: documents use
  ``embed_docs`` and queries use ``embed_queries`` (which adds the model's
  query instruction prefix). The two must never be mixed.
* **FakeEmbedder** produces deterministic vectors derived from a SHA-1 of the
  text, with zero dependencies, for tests and dry runs.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol

import config


def to_blob(vec) -> bytes:
    """Pack a float sequence as L2-normalised float32 bytes, the format vec0 expects.

    Normalising makes cosine / inner-product comparisons independent of vector
    length. A zero vector is packed unchanged. ``struct`` uses native byte
    order, which is little-endian on x86 and ARM.
    """
    xs = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in xs))
    if norm > 0:
        xs = [x / norm for x in xs]
    return struct.pack(f"{len(xs)}f", *xs)


class Embedder(Protocol):
    """Interface every embedder implements; callers depend only on this.

    Both methods return one packed vector (see ``to_blob``) per input text, in
    input order, ready to be stored as-is.
    """
    model: str                  # model name; also stored as the vector's version tag
    dim: int                    # vector dimension, needed to create vec0 tables

    def embed_docs(self, texts: list[str]) -> list[bytes]: ...
    def embed_queries(self, texts: list[str]) -> list[bytes]: ...


class FakeEmbedder:
    """Deterministic fake embedder: same text, same vector; different texts are nearly orthogonal.

    Loads no model and needs no network, so tests and model-less environments
    can exercise the full embed-store-KNN path. ``dim`` defaults to
    ``config.EMBED_DIM``. Satisfies ``Embedder`` structurally.
    """

    def __init__(self, dim: int | None = None, model: str = "fake"):
        self.dim = int(dim or config.EMBED_DIM)
        self.model = model

    def _one(self, text: str) -> bytes:
        h = hashlib.sha1(text.encode("utf-8")).digest()
        # Stretch the 20-byte digest into a dim-length pseudo-random vector.
        raw = [((h[i % len(h)] + i * 31) % 251) / 251.0 - 0.5 for i in range(self.dim)]
        return to_blob(raw)

    def embed_docs(self, texts: list[str]) -> list[bytes]:
        """Embed documents (the content being searched, e.g. videos, channels)."""
        return [self._one(t) for t in texts]

    def embed_queries(self, texts: list[str]) -> list[bytes]:
        """Embed queries (what is searched with, e.g. profile entries, chips, negative prefs)."""
        return [self._one(t) for t in texts]


class FastEmbedder:
    """Production embedder running the real bge model via fastembed (ONNX).

    Construction loads nothing; the first embed call imports fastembed and
    downloads or reads the model from ``EMBED_MODEL_DIR``. ``threads`` defaults
    to 1 so that inference on a single-core host does not starve the web server;
    offline jobs on bigger machines can raise it.
    """

    def __init__(self, model: str | None = None, dim: int | None = None, threads: int = 1):
        self.model = model or config.EMBED_MODEL
        self.dim = int(dim or config.EMBED_DIM)
        self._threads = threads
        self._impl = None                          # loaded model; None until first use

    def _load(self):
        """Load the model on first use and cache it on the instance."""
        if self._impl is None:
            from pathlib import Path
            from fastembed import TextEmbedding      # deferred import: see module docstring
            cache = Path(config.EMBED_MODEL_DIR)
            if not cache.is_absolute():
                cache = Path(__file__).resolve().parents[1] / cache
            cache.mkdir(parents=True, exist_ok=True)

            # The first run downloads the ONNX model and tokenizer into cache_dir; later runs read locally.
            self._impl = TextEmbedding(model_name=self.model, cache_dir=str(cache),
                                       threads=self._threads)
        return self._impl

    def embed_docs(self, texts: list[str]) -> list[bytes]:
        """Embed documents; text is passed to the model unchanged."""
        if not texts:
            return []
        return [to_blob(v) for v in self._load().embed(texts, batch_size=16)]

    def embed_queries(self, texts: list[str]) -> list[bytes]:
        """Embed queries; fastembed prepends bge's retrieval instruction prefix.

        The same text therefore yields different vectors here and in
        ``embed_docs``, so the embedding job tracks query-side hashes separately.
        """
        if not texts:
            return []
        return [to_blob(v) for v in self._load().query_embed(texts)]
