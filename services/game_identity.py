from __future__ import annotations

import re
import unicodedata

from ..storage.models import GameCandidate, RankedGame

ENGLISH_EDITION_SUFFIXES = (
    "digital deluxe edition",
    "game of the year edition",
    "anniversary edition",
    "definitive edition",
    "complete edition",
    "ultimate edition",
    "special edition",
    "deluxe edition",
    "goty edition",
    "director s cut",
    "directors cut",
    "remastered",
    "remake",
    "redux",
    "vr",
)
CHINESE_EDITION_SUFFIXES = (
    "导演剪辑版",
    "特别版",
    "完全版",
    "完整版",
    "终极版",
    "豪华版",
    "重制版",
)


def is_confirmed_base_game(candidate: GameCandidate) -> bool:
    return candidate.app_type == "game"


def game_family_key(title: str) -> str:
    normalized = normalize_game_title(title)
    family = normalized
    while family:
        stripped = strip_one_edition_suffix(family)
        if stripped == family:
            break
        family = stripped
    return family or normalized


def is_edition_title(title: str) -> bool:
    normalized = normalize_game_title(title)
    return bool(normalized) and game_family_key(title) != normalized


def deduplicate_game_editions(
    games: list[RankedGame],
    preferred_appids: list[int] | None = None,
) -> list[RankedGame]:
    preferred = {int(appid) for appid in preferred_appids or []}
    families: dict[str, list[RankedGame]] = {}
    for game in games:
        families.setdefault(game_family_key(game.title), []).append(game)

    selected: list[RankedGame] = []
    for family in families.values():
        best_tier_order = min(ranked_game_precedence_key(game)[0] for game in family)
        tier_family = [
            game
            for game in family
            if ranked_game_precedence_key(game)[0] == best_tier_order
        ]
        preferred_games = [
            game
            for game in tier_family
            if game.appid is not None and int(game.appid) in preferred
        ]
        standard_games = [
            game for game in tier_family if not is_edition_title(game.title)
        ]
        pool = preferred_games or standard_games or tier_family
        selected.append(min(pool, key=ranked_game_precedence_key))
    return sorted(selected, key=ranked_game_precedence_key)


def ranked_game_precedence_key(game: RankedGame) -> tuple[float | int | str, ...]:
    tier_order = {"A": 0, "broad": 0, "B": 1, "C": 2}
    breakdown = game.score_breakdown
    raw_layer = float(breakdown.layer_score)
    has_scored_layer = raw_layer != 0.0
    if not has_scored_layer and game.score:
        raw_layer = float(game.score) / 100.0
    effective_layer = (
        raw_layer + float(breakdown.budget_adjustment) / 100.0
        if has_scored_layer
        else raw_layer
    )
    retrieval_rank = int(breakdown.retrieval_rank)
    return (
        tier_order.get(breakdown.relevance_tier, 3),
        -effective_layer,
        -raw_layer,
        retrieval_rank if retrieval_rank > 0 else 1_000_000_000,
        game.title.casefold(),
    )


def normalize_game_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(title or "")).casefold()
    return re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).strip()


def strip_one_edition_suffix(title: str) -> str:
    for suffix in ENGLISH_EDITION_SUFFIXES:
        marker = f" {suffix}"
        if title.endswith(marker):
            return title[: -len(marker)].strip()
    for suffix in CHINESE_EDITION_SUFFIXES:
        if title.endswith(suffix):
            return title[: -len(suffix)].strip()
    return title
