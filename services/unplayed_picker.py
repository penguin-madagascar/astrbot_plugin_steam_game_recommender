from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from ..storage.models import GameCandidate, SteamOwnedGame


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
    candidates = [game for game in owned_games if game.appid and game.playtime_forever <= 0]
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
        game = await steam_client.get_game_detail(owned_game.appid)
        return UnplayedRecommendation(
            game=attach_review_summary(game, owned_game, summary),
            owned_game=owned_game,
            checked_count=checked_count,
        )

    raise UnplayedRecommendationError(
        "没有找到未游玩且评价过线的游戏"
        f"（门槛：至少 {min_count} 条评测、好评率不低于 {min_ratio:.0%}）。"
    )


def review_passes(summary: Any, min_review_count: int, min_positive_ratio: float) -> bool:
    total_reviews = optional_int(getattr(summary, "total_reviews", None)) or 0
    positive_ratio = optional_float(getattr(summary, "positive_ratio", None))
    if total_reviews < min_review_count:
        return False
    return positive_ratio is not None and positive_ratio >= min_positive_ratio


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
    min_review_count: int,
    min_positive_ratio: float,
) -> str:
    game = recommendation.game
    genres = "、".join(game.genres[:6]) if game.genres else "不确定"
    tags = "、".join(game.tags[:8]) if game.tags else "不确定"
    lines = [
        "随机从你的 Steam 未游玩游戏里挑了这款：",
        f"《{game.title}》",
        f"Steam 好评率：{format_ratio(game.review_positive_ratio)}",
        f"Steam 评测数：{game.review_total if game.review_total is not None else '不确定'}",
        f"类型：{genres}",
        f"标签：{tags}",
        (
            "筛选条件：游玩时长 0 分钟；"
            f"至少 {max(int(min_review_count), 0)} 条评测；"
            f"好评率不低于 {min(max(float(min_positive_ratio), 0.0), 1.0):.0%}"
        ),
    ]
    if game.raw_url:
        lines.append(f"数据来源：{game.raw_url}")
    return "\n".join(lines)


def format_ratio(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "不确定"


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_candidate(data: dict[str, Any]) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)
