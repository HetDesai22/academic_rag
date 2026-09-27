from src.topic_modeling.classifier import SubjectClassifier, confidence_band, keyword_matches, split_segments
from src.topic_modeling.taxonomy import flatten, validate_source

SUBJECT = "Machine Learning"


def test_taxonomy_is_valid(taxonomy_src, log):
    validate_source(taxonomy_src, {SUBJECT}, log)
    rows = flatten(taxonomy_src)
    assert all(r["description"] for r in rows)


def test_semantic_assignment_without_topic_keyword(taxonomy_src, embedder, cfg):
    clf = SubjectClassifier(SUBJECT, taxonomy_src, embedder, cfg)
    res = clf.classify([
        "How are the connection weights of a multi-layer network updated using the error propagated from the output layer?",
        "Describe how data points are grouped around centroids that are recomputed iteratively.",
    ])
    assert res[0]["primary_topic"] == "Neural Network"
    assert res[1]["primary_topic"] == "Unsupervised Learning"
    for r in res:
        assert 0.0 <= r["topic_confidence"] <= 1.0
        assert r["subtopic"]


def test_most_specific_keyword_wins(taxonomy_src):
    rows = [r for r in flatten(taxonomy_src) if r["subject"] == "Artificial Intelligence and Machine Learning"]
    hits = keyword_matches("What is stochastic gradient descent?", rows)
    matched = {rows[i]["subtopic"] for i in hits}
    assert "Stochastic and Mini-batch Gradient Descent" in matched
    assert "Gradient Descent" not in matched


def test_multi_topic_question_gets_secondary_topic(taxonomy_src, embedder, cfg):
    clf = SubjectClassifier("Applied Machine Learning", taxonomy_src, embedder, cfg)
    res = clf.classify(["Write short note on: a) Generative Adversarial Networks b) Deep Learning"])[0]
    topics = {res["primary_topic"], *res["secondary_topics"]}
    assert {"Deep Learning", "GANs, Deep RL and Adversarial Attacks"} <= topics


def test_generic_follow_up_does_not_create_secondary(taxonomy_src, embedder, cfg):
    clf = SubjectClassifier(SUBJECT, taxonomy_src, embedder, cfg)
    res = clf.classify(["Explain the backpropagation algorithm. What are the limitations of this algorithm?"])[0]
    assert res["secondary_topics"] == []


def test_off_syllabus_question_is_not_confident(taxonomy_src, embedder, cfg):
    clf = SubjectClassifier(SUBJECT, taxonomy_src, embedder, cfg)
    res = clf.classify(["Explain the purpose of the TCP three-way handshake in computer networks."])[0]
    assert confidence_band(res["topic_confidence"], cfg["confidence_thresholds"]) != "high"


def test_human_exemplar_changes_prediction(taxonomy_src, embedder, cfg):
    q = "Write down different techniques for data preprocessing."
    base = SubjectClassifier("Introduction of Machine learning", taxonomy_src, embedder, cfg).classify([q])[0]
    ex = [("Dimensionality Reduction", "Dimensionality Reduction Concepts", q)]
    taught = SubjectClassifier("Introduction of Machine learning", taxonomy_src, embedder, cfg, exemplars=ex).classify([q])[0]
    assert taught["subtopic"] == "Dimensionality Reduction Concepts"
    assert taught["topic_confidence"] >= base["topic_confidence"] - 1e-6


def test_segment_split_and_bands(cfg):
    assert split_segments("Define CNN. b) Enlist 4 applications of CNN. c) Explain working of CNN.")
    assert split_segments("Explain SVM.") == []
    th = cfg["confidence_thresholds"]
    assert confidence_band(0.99, th) == "high" and confidence_band(0.1, th) == "low"
    assert confidence_band((th["high"] + th["low"]) / 2, th) == "medium"
