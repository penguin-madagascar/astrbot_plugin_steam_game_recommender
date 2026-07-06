from __future__ import annotations

from ..storage.models import RankedGame
from .similarity_ranker import select_diverse_results

DIVERSITY_STRICT = "strict"
DIVERSITY_BALANCED = "balanced"
DIVERSITY_HIGH = "high"
DIVERSITY_MODES = {DIVERSITY_STRICT, DIVERSITY_BALANCED, DIVERSITY_HIGH}


def select_results_by_diversity(
    games: list[RankedGame],
    limit: int,
    mode: str = DIVERSITY_STRICT,
) -> list[RankedGame]:
    if limit <= 0:
        return []
    if mode == DIVERSITY_BALANCED:
        return select_diverse_results(games, limit, group_by="primary", penalty_weight=15)
    if mode == DIVERSITY_HIGH:
        return select_diverse_results(games, limit, group_by="tier", penalty_weight=35)
    return games[:limit]
