import numpy as np
import pytest

from src.metrics import (
    average_precision_at_k,
    evaluate_rankings,
    precision_at_k,
    ranking_metrics,
    recall_at_k,
)


def test_known_ranking_metrics_have_correct_denominators():
    scores = ranking_metrics(["a", "x", "b", "y", "c"], {"a", "b", "c", "d"})
    assert scores["precision@5"] == pytest.approx(3 / 5)
    assert scores["precision@10"] == pytest.approx(3 / 10)
    assert scores["recall@10"] == pytest.approx(3 / 4)
    assert scores["ap@10"] == pytest.approx((1 + 2 / 3 + 3 / 5) / 4)


def test_ap_at_10_uses_minimum_of_total_relevance_and_cutoff():
    relevant = set(range(20))
    scores = ranking_metrics(list(range(10)), relevant)
    assert scores["ap@10"] == 1
    assert scores["recall@10"] == 0.5
    assert scores["precision@10"] == 1


def test_relevant_items_after_cutoff_do_not_inflate_metrics():
    assert average_precision_at_k(["x", "y", "a"], {"a"}, 2) == 0
    assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0
    assert precision_at_k(["x", "y", "a"], {"a"}, 2) == 0


def test_short_rankings_keep_fixed_precision_denominator():
    scores = ranking_metrics(["a"], {"a"})
    assert scores == {"precision@5": 0.2, "precision@10": 0.1, "recall@10": 1, "ap@10": 1}


def test_no_relevance_is_undefined_but_missing_results_are_zero():
    assert all(value is None for value in ranking_metrics(["a"], set()).values())
    assert all(value == 0 for value in ranking_metrics([], {"a"}).values())


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "10"])
@pytest.mark.parametrize("function", [precision_at_k, recall_at_k, average_precision_at_k])
def test_nonpositive_or_noninteger_cutoffs_rejected(k, function):
    with pytest.raises(ValueError, match="positive integer"):
        function(["a"], {"a"}, k)


def test_numpy_integer_cutoff_is_valid():
    assert precision_at_k(["a"], {"a"}, np.int64(5)) == 0.2


@pytest.mark.parametrize("ranking", [["a", "a"], ["a", "b", "a"], "abc", [["unhashable"]]])
def test_invalid_rankings_are_rejected_even_if_duplicates_beyond_cutoff(ranking):
    with pytest.raises(ValueError):
        average_precision_at_k(ranking, {"a"}, 1)


def test_map_macroaverages_only_evaluable_queries_and_reports_coverage():
    report = evaluate_rankings({"q1": ["a"], "q2": ["x"], "q3": ["a"]},
                               {"q1": {"a"}, "q2": {"b"}, "q3": set()})
    assert report.summary["map@10"] == 0.5
    assert report.summary["precision@5"] == 0.1
    assert report.summary["recall@10"] == 0.5
    assert report.summary["queries_total"] == 3
    assert report.summary["queries_evaluated"] == 2
    assert report.summary["queries_without_relevant"] == 1
    assert report.summary["coverage"] == pytest.approx(2 / 3)
    assert report.per_query[2]["ap@10"] is None
    assert report.per_query[2]["evaluable"] is False


def test_no_evaluable_queries_return_null_means_not_fabricated_zeros():
    report = evaluate_rankings({"q": []}, {})
    assert report.summary["coverage"] == 0
    assert report.summary["queries_without_relevant"] == 1
    assert report.summary["map@10"] is None
    assert report.summary["precision@5"] is None


def test_query_universe_retains_missing_rankings_and_relevance():
    report = evaluate_rankings({"ranked": ["x"]}, {"missing_ranking": {"a"}})
    assert report.summary["queries_total"] == 2
    assert report.summary["queries_evaluated"] == 1
    by_query = {row["query_id"]: row for row in report.per_query}
    assert by_query["ranked"]["evaluable"] is False
    assert by_query["missing_ranking"]["ap@10"] == 0
    assert by_query["missing_ranking"]["returned_count"] == 0


def test_explicit_query_universe_retains_queries_absent_from_both_mappings():
    report = evaluate_rankings({}, {}, query_ids=["q1", "q2"])
    assert report.summary["queries_total"] == 2
    assert len(report.per_query) == 2
    with pytest.raises(ValueError, match="outside"):
        evaluate_rankings({"q3": []}, {}, query_ids=["q1"])


def test_empty_evaluation_is_explicit():
    report = evaluate_rankings({}, {})
    assert report.summary["queries_total"] == 0
    assert report.summary["coverage"] == 0
    assert report.summary["map@10"] is None
