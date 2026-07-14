from __future__ import annotations

from ..storage.models import RankedGame

_TIER_ORDER = {
    "A": 0,
    "broad": 0,
    "B": 1,
    "C": 2,
}
_MISSING_RETRIEVAL_RANK = 1_000_000_000


def effective_score(
    breakdown: object,
    *,
    fallback_score: float | None = None,
) -> float:
    """Return the single unrounded score used for ordering and display."""
    layer = float(getattr(breakdown, "layer_score", 0.0) or 0.0)
    retrieval_rank = int(getattr(breakdown, "retrieval_rank", 0) or 0)
    base = layer * 100.0
    if layer == 0.0 and retrieval_rank <= 0 and fallback_score is not None:
        return min(max(float(fallback_score), 0.0), 100.0)
    value = (
        base
        + float(getattr(breakdown, "language_adjustment", 0.0) or 0.0)
        + float(getattr(breakdown, "budget_adjustment", 0.0) or 0.0)
        + float(getattr(breakdown, "company_adjustment", 0.0) or 0.0)
    )
    return min(max(value, 0.0), 100.0)


def ranked_game_precedence_prefix(
    game: RankedGame,
) -> tuple[int, float, int]:
    """Return the shared tier and within-tier ordering prefix."""
    breakdown = game.score_breakdown
    retrieval_rank = int(breakdown.retrieval_rank)
    # broad belongs to queries without anchors, so it does not coexist with
    # A/B/C candidates produced by one ranking request.
    return (
        _TIER_ORDER.get(breakdown.relevance_tier, 3),
        -effective_score(breakdown, fallback_score=game.score),
        retrieval_rank if retrieval_rank > 0 else _MISSING_RETRIEVAL_RANK,
    )
