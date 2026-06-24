"""Embedding utilities for routers that need similarity over free text.

This module is not part of the spec's section 6 interface list, but is a
necessary supporting piece for ``StaticSemanticRouter`` (and, later, the
intent-complexity feature used by the context-aware router).

``HashingEmbedder`` is a deterministic, dependency-free stand-in for a real
embedding model (e.g. sentence-transformers, an OpenAI/Voyage embedding
endpoint). It exists so the benchmark suite can run fully offline and
reproducibly; swapping in a real embedding provider only requires
implementing the ``Embedder`` protocol and passing it to the router's
constructor.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

__all__ = ["Embedder", "HashingEmbedder", "cosine_similarity"]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class Embedder(Protocol):
    """Structural type for anything that can turn text into a fixed-size vector."""

    def embed(self, text: str) -> Sequence[float]:
        """Return a fixed-length numeric vector representing ``text``."""
        ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing vectorizer.

    Tokenizes on word boundaries, hashes each token via SHA-256 into one of
    ``n_dims`` buckets, and accumulates term counts. Hashing (rather than
    Python's built-in ``hash()``) keeps the embedding stable across
    processes regardless of ``PYTHONHASHSEED``, which matters for
    reproducible benchmark runs.
    """

    def __init__(self, n_dims: int = 256) -> None:
        if n_dims <= 0:
            raise ValueError("n_dims must be positive")
        self.n_dims = n_dims

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.n_dims
        for token in _tokenize(text):
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.n_dims
            vector[bucket] += 1.0
        return tuple(vector)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1], or 0.0 if either vector has zero norm."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
