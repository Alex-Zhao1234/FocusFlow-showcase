"""Contract tests for sources.embeddings: fake-embedder determinism, normalization, lazy loading of the real model."""
from __future__ import annotations

import math
import struct
import sys

import config
from sources import embeddings as em


def test_to_blob_normalizes():
    b = em.to_blob([3, 4])
    x, y = struct.unpack("2f", b)
    assert math.isclose(math.hypot(x, y), 1.0, rel_tol=1e-6)
    assert em.to_blob([0, 0]) == struct.pack("2f", 0, 0)


def test_fake_embedder_is_deterministic_and_sized(monkeypatch):
    monkeypatch.setattr(config, "EMBED_DIM", 8)
    e = em.FakeEmbedder()
    a, b = e.embed_docs(["x", "x"])
    assert a == b and len(a) == 8 * 4
    assert e.embed_docs(["y"])[0] != a
    assert e.embed_queries(["x"])[0] == a


def test_fast_embedder_does_not_import_fastembed_on_construct():
    sys.modules.pop("fastembed", None)
    e = em.FastEmbedder()
    assert e.model == config.EMBED_MODEL and e.dim == config.EMBED_DIM
    assert "fastembed" not in sys.modules
    assert e.embed_docs([]) == [] and "fastembed" not in sys.modules
