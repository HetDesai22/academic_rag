import numpy as np


def test_shape_and_normalisation(embedder):
    v = embedder.encode(["Explain the backpropagation algorithm.", "Define PCA."])
    assert v.shape == (2, embedder.dim)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)


def test_cache_prevents_recomputation(embedder):
    texts = ["A unique test sentence about gradient boosting 8931.", "Another one about k-medoids 8931."]
    embedder.encode(texts)
    before = embedder.encoded_count
    again = embedder.encode(texts)
    assert embedder.encoded_count == before          # served from cache
    assert again.shape[0] == 2


def test_query_and_document_modes_are_distinct(embedder):
    t = "What questions were asked about SVM?"
    q, d = embedder.encode([t], query=True)[0], embedder.encode([t])[0]
    assert not np.allclose(q, d)


def test_semantic_similarity_orders_paraphrases_first(embedder):
    a = embedder.encode(["Explain the k-means clustering algorithm."])[0]
    b = embedder.encode(["Describe how K-Means groups data into clusters."])[0]
    c = embedder.encode(["State Bayes theorem with an example."])[0]
    assert a @ b > a @ c


def test_empty_input(embedder):
    assert embedder.encode([]).shape == (0, embedder.dim)


def test_device_is_auto_detected(embedder):
    assert embedder.device in ("cuda", "cpu")
