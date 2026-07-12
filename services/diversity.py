from __future__ import annotations

from ..storage.models import RankedGame
from .similarity_ranker import TIER_ORDER, copy_facts, copy_ranked_game
from .tag_normalizer import candidate_canonical_tags

DIVERSITY_STRICT = "strict"
DIVERSITY_BALANCED = "balanced"
DIVERSITY_HIGH = "high"
DIVERSITY_MODES = {DIVERSITY_STRICT, DIVERSITY_BALANCED, DIVERSITY_HIGH}
DIVERSITY_REDUNDANCY_PENALTIES = {
    DIVERSITY_STRICT: 0.0,
    DIVERSITY_BALANCED: 0.15,
    DIVERSITY_HIGH: 0.30,
}


def select_results_by_diversity(
    games: list[RankedGame],
    limit: int,
    mode: str = DIVERSITY_STRICT,
) -> list[RankedGame]:
    if limit <= 0:
        return []
    penalty = DIVERSITY_REDUNDANCY_PENALTIES.get(mode, 0.0)
    if penalty <= 0:
        return games[:limit]

    selected: list[RankedGame] = []
    for tier in sorted({game.tier for game in games}, key=lambda item: TIER_ORDER.get(item, 9)):
        remaining = [game for game in games if game.tier == tier]
        tier_selected: list[RankedGame] = []
        while remaining and len(selected) < limit:
            ranked_choices = [
                (
                    relevance_for(game) - penalty * maximum_tag_similarity(game, tier_selected),
                    -index,
                    index,
                    game,
                )
                for index, game in enumerate(remaining)
            ]
            _mmr, _stable, best_index, best = max(ranked_choices)
            redundancy = maximum_tag_similarity(best, tier_selected)
            if redundancy > 0:
                best = copy_ranked_game(
                    best,
                    {
                        "facts": copy_facts(
                            best.facts,
                            {"diversity_penalty": redundancy * penalty},
                        )
                    },
                )
            selected.append(best)
            tier_selected.append(best)
            del remaining[best_index]
        if len(selected) >= limit:
            break
    return selected


def relevance_for(game: RankedGame) -> float:
    if game.facts.reranked_relevance_score > 0:
        return min(max(game.facts.reranked_relevance_score, 0.0), 1.0)
    if game.facts.base_relevance_score > 0:
        return min(max(game.facts.base_relevance_score, 0.0), 1.0)
    return min(max(float(game.score) / 100, 0.0), 1.0)


def maximum_tag_similarity(game: RankedGame, selected: list[RankedGame]) -> float:
    return max((tag_similarity(game, item) for item in selected), default=0.0)


def tag_similarity(left: RankedGame, right: RankedGame) -> float:
    left_tags = set(candidate_canonical_tags(left))
    right_tags = set(candidate_canonical_tags(right))
    union = left_tags | right_tags
    return len(left_tags & right_tags) / len(union) if union else 0.0
