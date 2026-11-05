"""Retrieval measures on explicit, self-excluded gallery relevance sets.

Precision@k always divides by k, including when fewer results are available.
AP@k divides by min(number of relevant gallery items, k). Queries without any
relevant gallery item are excluded from averages and reported in coverage.
"""
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral


def _positive_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def _ranking_ids(ranking: Sequence[Hashable]) -> list[Hashable]:
    if isinstance(ranking, (str, bytes)):
        raise ValueError("A ranking must be a sequence of unique image IDs")
    ranking = list(ranking)
    try:
        if len(set(ranking)) != len(ranking):
            raise ValueError("A ranking must not contain duplicate image IDs")
    except TypeError as exc:
        raise ValueError("Image IDs must be hashable") from exc
    return ranking


def _relevant_ids(relevant: Iterable[Hashable]) -> set[Hashable]:
    if isinstance(relevant, (str, bytes)):
        raise ValueError("Relevance must be an iterable of image IDs")
    try:
        return set(relevant)
    except TypeError as exc:
        raise ValueError("Relevant image IDs must be hashable") from exc


def precision_at_k(ranking, relevant, k: int) -> float | None:
    """Relevant hits in the first k ranks, divided by k; None when R=0."""
    k = _positive_k(k)
    ranked, relevant = _ranking_ids(ranking), _relevant_ids(relevant)
    if not relevant:
        return None
    return sum(image_id in relevant for image_id in ranked[:k]) / k


def recall_at_k(ranking, relevant, k: int) -> float | None:
    """Relevant hits in the first k ranks, divided by all relevant gallery IDs."""
    k = _positive_k(k)
    ranked, relevant = _ranking_ids(ranking), _relevant_ids(relevant)
    if not relevant:
        return None
    return sum(image_id in relevant for image_id in ranked[:k]) / len(relevant)


def average_precision_at_k(ranking, relevant, k: int) -> float | None:
    """Sum precision at relevant ranks <= k, divided by min(R, k)."""
    k = _positive_k(k)
    ranked, relevant = _ranking_ids(ranking), _relevant_ids(relevant)
    if not relevant:
        return None
    hits, total = 0, 0.0
    for rank, image_id in enumerate(ranked[:k], start=1):
        if image_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def ranking_metrics(ranking, relevant) -> dict[str, float | None]:
    """Standard portfolio report for one query: P@5, P@10, R@10 and AP@10."""
    ranked, relevant = _ranking_ids(ranking), _relevant_ids(relevant)
    return {
        "precision@5": precision_at_k(ranked, relevant, 5),
        "precision@10": precision_at_k(ranked, relevant, 10),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "ap@10": average_precision_at_k(ranked, relevant, 10),
    }


@dataclass
class RetrievalEvaluation:
    summary: dict[str, float | int | None]
    per_query: list[dict]


def evaluate_rankings(
    rankings: Mapping[Hashable, Sequence[Hashable]],
    relevant_ids_by_query: Mapping[Hashable, Iterable[Hashable]],
    query_ids: Sequence[Hashable] | None = None,
) -> RetrievalEvaluation:
    """Macro-average across evaluable queries and retain every query record.

    By default the query universe is the union of both mappings, in insertion
    order. Thus a missing ranking is scored as an empty ranking, and missing
    relevance makes a query unevaluable rather than silently dropping it.
    An explicit query_ids sequence can also include queries absent from both.
    Neither mapping may contain IDs outside an explicit query universe.
    """
    if query_ids is None:
        query_ids = list(dict.fromkeys([*rankings, *relevant_ids_by_query]))
    else:
        query_ids = _ranking_ids(query_ids)
        if (set(rankings) | set(relevant_ids_by_query)) - set(query_ids):
            raise ValueError("A mapping contains a query outside query_ids")
    rows = []
    for query_id in query_ids:
        relevant = _relevant_ids(relevant_ids_by_query.get(query_id, ()))
        ranking = _ranking_ids(rankings.get(query_id, ()))
        rows.append({
            "query_id": query_id,
            "relevant_count": len(relevant),
            "returned_count": len(ranking),
            "evaluable": bool(relevant),
            **ranking_metrics(ranking, relevant),
        })
    evaluated = [row for row in rows if row["evaluable"]]
    summary = {}
    for measure in ("precision@5", "precision@10", "recall@10", "ap@10"):
        key = "map@10" if measure == "ap@10" else measure
        summary[key] = sum(row[measure] for row in evaluated) / len(evaluated) if evaluated else None
    summary.update({
        "queries_total": len(rows),
        "queries_evaluated": len(evaluated),
        "queries_without_relevant": len(rows) - len(evaluated),
        "coverage": len(evaluated) / len(rows) if rows else 0.0,
    })
    return RetrievalEvaluation(summary=summary, per_query=rows)
