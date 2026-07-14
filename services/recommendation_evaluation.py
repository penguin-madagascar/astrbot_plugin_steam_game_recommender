"""Pure offline metrics for evaluating recommendation rankings."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from math import log2


def ndcg_at_k(
    ranking: Sequence[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain for the first ``k`` results."""
    unique_ranking = _unique_ids(ranking)
    if not unique_ranking or k <= 0:
        return 0.0

    actual = _discounted_cumulative_gain(
        relevance_by_id.get(item_id, 0) for item_id in unique_ranking[:k]
    )
    ideal = _discounted_cumulative_gain(sorted(relevance_by_id.values(), reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def recall_at_k(
    ranking: Sequence[str],
    relevance_by_id: Mapping[str, int],
    k: int,
) -> float:
    """Return the share of positive-relevance candidates retrieved in the first ``k`` results."""
    unique_ranking = _unique_ids(ranking)
    relevant_ids = {item_id for item_id, relevance in relevance_by_id.items() if relevance > 0}
    if not relevant_ids or k <= 0:
        return 0.0

    retrieved_ids = set(unique_ranking[:k])
    return len(relevant_ids & retrieved_ids) / len(relevant_ids)


def constraint_violation_rate(
    ranking: Sequence[str],
    violating_ids: Collection[str],
) -> float:
    """Return the share of ranked results known to violate a hard constraint."""
    unique_ranking = _unique_ids(ranking)
    if not unique_ranking:
        return 0.0

    known_violations = set(violating_ids)
    return sum(item_id in known_violations for item_id in unique_ranking) / len(unique_ranking)


def fill_rate(ranking: Sequence[str], target_count: int) -> float:
    """Return how much of the requested result count was filled, capped at one."""
    unique_ranking = _unique_ids(ranking)
    if target_count <= 0:
        return 0.0
    return min(len(unique_ranking) / target_count, 1.0)


def hit_at_k(
    ranking: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Return one when at least one target is present in the first ``k`` results."""
    if k <= 0 or not relevant_ids:
        return 0.0
    return float(bool(set(_unique_ids(ranking)[:k]) & set(relevant_ids)))


def pairwise_accuracy(
    ranking: Sequence[str],
    preferred_pairs: Sequence[tuple[str, str]],
) -> float:
    """Measure how often a preferred item outranks its paired comparison item."""
    if not preferred_pairs:
        return 0.0
    positions = {item_id: index for index, item_id in enumerate(_unique_ids(ranking))}
    correct = 0
    for preferred_id, comparison_id in preferred_pairs:
        preferred_position = positions.get(preferred_id)
        comparison_position = positions.get(comparison_id)
        if (
            preferred_position is not None
            and comparison_position is not None
            and preferred_position < comparison_position
        ):
            correct += 1
    return correct / len(preferred_pairs)


def _unique_ids(ranking: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ranking))


def _discounted_cumulative_gain(relevances: Iterable[int]) -> float:
    return sum((2**relevance - 1) / log2(rank + 2) for rank, relevance in enumerate(relevances))
