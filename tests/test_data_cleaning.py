import pandas as pd

from src.preprocessing.cleaning import clean_text, dedup_key, is_variant, mark_duplicates, preprocess


def test_empty_and_non_text_inputs():
    assert clean_text(None)[0] == ""
    assert clean_text(float("nan"))[0] == ""
    assert clean_text("   ")[0] == ""


def test_whitespace_unicode_and_hyphen_breaks():
    text, rules = clean_text("Explain  K- Means’ clustering − steps .")
    assert text == "Explain K-Means' clustering - steps."
    assert {"char_map", "hyphen_break", "whitespace"} <= set(rules)


def test_header_footer_and_page_numbers_removed():
    assert clean_text("GUJARAT TECHNOLOGICAL UNIVERSITY Explain SVM. Page 2 of 3")[0] == "Explain SVM."
    assert clean_text("Explain the concept of sampling distribution. Page of")[0] == \
        "Explain the concept of sampling distribution."


def test_duplicated_text_removed():
    assert clean_text("Explain PCA. Explain PCA.")[0] == "Explain PCA."


def test_academic_terms_and_stopwords_preserved():
    # No stemming / stopword removal: meaning-bearing words stay intact.
    text, _ = clean_text("What is the role of the kernel in a non-linear SVM?")
    assert text == "What is the role of the kernel in a non-linear SVM?"


def test_math_italic_letters_normalised():
    text, rules = clean_text("Find the inverse of the matrix \U0001D434= [ 2 0 −2 ]")
    assert "A=" in text and "-2" in text
    assert "unicode_nfkc" in rules


def test_preprocess_keeps_raw_and_cleaned(cfg, log):
    df = pd.DataFrame({"question_id": ["a", "b", "c"], "question_text": ["Explain  SVM", None, "Define PCA."]})
    out = preprocess(df, "question_text", cfg, log)
    assert list(out["raw_question_text"][[0, 2]]) == ["Explain  SVM", "Define PCA."]
    assert out.loc[0, "cleaned_question_text"] == "Explain SVM"
    assert bool(out.loc[1, "flag_empty"]) and not bool(out.loc[0, "flag_empty"])


def test_duplicates_are_grouped_not_deleted(cfg):
    df = pd.DataFrame({"question_id": list("abcd"),
                       "cleaned_question_text": ["Explain SVM.", "explain svm", "Explain backpropagation.",
                                                 "Explain back propagation."]})
    out, stats = mark_duplicates(df, "cleaned_question_text", 85)
    assert len(out) == 4                                    # nothing removed
    assert out.loc[0, "exact_duplicate_group"] == out.loc[1, "exact_duplicate_group"] != ""
    assert out.loc[2, "near_duplicate_group"] == out.loc[3, "near_duplicate_group"] != ""


def test_variant_guard_rejects_different_concepts():
    assert not is_variant(dedup_key("Define CNN."), dedup_key("Define RNN."))
    assert not is_variant(dedup_key("Types of supervised learning"), dedup_key("Types of unsupervised learning"))
    assert not is_variant(dedup_key("What are the advantages of KNN?"), dedup_key("What are the disadvantages of KNN?"))
    assert is_variant(dedup_key("What is naive Bayes classifier?"), dedup_key("What is neive Bayes classifier?"))
