import pytest

from routing_benchmark.embedding import HashingEmbedder, cosine_similarity


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(n_dims=64)
    a = embedder.embed("summarize this document for me")
    b = embedder.embed("summarize this document for me")
    assert a == b


def test_hashing_embedder_rejects_non_positive_dims():
    with pytest.raises(ValueError):
        HashingEmbedder(n_dims=0)


def test_hashing_embedder_empty_text_is_zero_vector():
    embedder = HashingEmbedder(n_dims=32)
    vector = embedder.embed("   ")
    assert vector == (0.0,) * 32


def test_hashing_embedder_distinguishes_different_texts():
    embedder = HashingEmbedder(n_dims=256)
    a = embedder.embed("calculate the monthly compound interest")
    b = embedder.embed("write a poem about the ocean")
    assert a != b


def test_cosine_similarity_identical_vectors_is_one():
    v = (1.0, 2.0, 3.0)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    assert cosine_similarity((0.0, 0.0), (1.0, 1.0)) == 0.0


def test_cosine_similarity_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 2.0), (1.0, 2.0, 3.0))
