import pytest

from inference.accuracy_scorer import AccuracyScorer


def test_perfect_match():
    scorer = AccuracyScorer()
    score = scorer.score_candidate("hello world", "hello world", 1.2)
    assert score["accuracy_score"] == pytest.approx(1.0)
    assert score["word_error_rate"] == pytest.approx(0.0)


def test_normalization_numbers_and_symbols():
    scorer = AccuracyScorer()

    # percent vs %
    score1 = scorer.score_candidate("The value is 25 percent", "The value is 25%", 1.0)
    assert score1["accuracy_score"] > 0.95

    # spelled numbers vs digits
    score2 = scorer.score_candidate("I have twenty-five dollars", "I have 25 dollars", 1.0)
    assert score2["accuracy_score"] > 0.95
