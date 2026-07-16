from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..storage.models import GameCandidate, SteamOwnedGame
from .game_identity import game_family_key, is_confirmed_base_game, is_edition_title


class UnplayedRecommendationError(ValueError):
    pass


class UnplayedSteamClient(Protocol):
    async def get_review_summary(self, appid: int) -> Any: ...

    async def get_game_detail(self, appid: int) -> GameCandidate: ...


@dataclass(frozen=True)
class UnplayedRecommendation:
    game: GameCandidate
    owned_game: SteamOwnedGame
    checked_count: int


async def pick_random_unplayed_game(
    owned_games: list[SteamOwnedGame],
    steam_client: UnplayedSteamClient,
    min_review_count: int = 50,
    min_positive_ratio: float = 0.65,
    rng: random.Random | None = None,
) -> UnplayedRecommendation:
    candidates = [
        game
        for game in deduplicate_owned_game_editions(owned_games)
        if game.appid
    ]
    if not candidates:
        raise UnplayedRecommendationError("Steam 游戏库中没有未游玩过的游戏。")

    shuffled = list(candidates)
    if rng is None:
        random.shuffle(shuffled)
    else:
        rng.shuffle(shuffled)

    min_count = max(int(min_review_count), 0)
    min_ratio = min(max(float(min_positive_ratio), 0.0), 1.0)
    checked_count = 0
    for owned_game in shuffled:
        summary = await steam_client.get_review_summary(owned_game.appid)
        checked_count += 1
        if not review_passes(summary, min_count, min_ratio):
            continue
        try:
            game = await steam_client.get_game_detail(owned_game.appid)
        except Exception:
            continue
        if not is_confirmed_base_game(game):
            continue
        return UnplayedRecommendation(
            game=attach_review_summary(game, owned_game, summary),
            owned_game=owned_game,
            checked_count=checked_count,
        )

    raise UnplayedRecommendationError(
        "没有找到未游玩且评价过线的游戏"
        f"（门槛：至少 {min_count} 条评测、好评率不低于 {min_ratio:.0%}）。"
    )


def deduplicate_owned_game_editions(
    owned_games: list[SteamOwnedGame],
) -> list[SteamOwnedGame]:
    families: dict[str, list[SteamOwnedGame]] = {}
    for owned_game in owned_games:
        title = owned_game.name or f"appid {owned_game.appid}"
        families.setdefault(game_family_key(title), []).append(owned_game)

    selected: list[SteamOwnedGame] = []
    for family in families.values():
        if any(game.playtime_forever > 0 for game in family):
            continue
        standard = next(
            (game for game in family if not is_edition_title(game.name or "")),
            None,
        )
        selected.append(standard or family[0])
    return selected


def review_passes(summary: Any, min_review_count: int, min_positive_ratio: float) -> bool:
    total_reviews = optional_int(getattr(summary, "total_reviews", None))
    positive_ratio = optional_float(getattr(summary, "positive_ratio", None))
    if total_reviews is None or total_reviews < 0 or total_reviews < min_review_count:
        return False
    return (
        positive_ratio is not None
        and 0.0 <= positive_ratio <= 1.0
        and positive_ratio >= min_positive_ratio
    )


def attach_review_summary(
    game: GameCandidate,
    owned_game: SteamOwnedGame,
    summary: Any,
) -> GameCandidate:
    data = dump_model(game)
    data["appid"] = owned_game.appid
    data["title"] = data.get("title") or owned_game.name or f"appid={owned_game.appid}"
    data["playtime"] = 0
    data["review_total"] = optional_int(getattr(summary, "total_reviews", None))
    data["review_positive_ratio"] = optional_float(getattr(summary, "positive_ratio", None))
    data["review_recent_ratio"] = optional_float(getattr(summary, "recent_positive_ratio", None))
    data["stores"] = data.get("stores") or ["Steam"]
    data["raw_url"] = data.get("raw_url") or (
        f"https://store.steampowered.com/app/{owned_game.appid}/"
    )
    return validate_candidate(data)


def format_unplayed_recommendation(
    recommendation: UnplayedRecommendation,
    reason: str,
) -> str:
    game = recommendation.game
    return f"《{game.title}》\n{reason.strip()}"


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else None
    return None


def optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_candidate(data: dict[str, Any]) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)
