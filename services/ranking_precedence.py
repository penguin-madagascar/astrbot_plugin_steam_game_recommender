from __future__ import annotations

from ..storage.models import RankedGame

_TIER_ORDER = {
    "A": 0,
    "broad": 0,
    "B": 1,
    "C": 2,
}
_MISSING_RETRIEVAL_RANK = 1_000_000_000


def ranked_game_precedence_prefix(
    game: RankedGame,
) -> tuple[int, float, float, int]:
    """Return the shared tier and within-tier ordering prefix."""
    breakdown = game.score_breakdown
    raw_layer = float(breakdown.layer_score)
    retrieval_rank = int(breakdown.retrieval_rank)
    uses_layer_scoring = raw_layer != 0.0 or retrieval_rank > 0
    if not uses_layer_scoring and game.score:
        raw_layer = float(game.score) / 100.0

    effective_layer = raw_layer
    if uses_layer_scoring:
        effective_layer += (
            float(breakdown.budget_adjustment)
            + float(breakdown.language_adjustment)
        ) / 100.0

    # broad belongs to queries without anchors, so it does not coexist with
    # A/B/C candidates produced by one ranking request.
    return (
        _TIER_ORDER.get(breakdown.relevance_tier, 3),
        -effective_layer,
        -raw_layer,
        retrieval_rank if retrieval_rank > 0 else _MISSING_RETRIEVAL_RANK,
    )
