from __future__ import annotations

import asyncio
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..clients.steam import SteamApiError
from ..storage.models import GameCandidate, SteamOwnedGame
from .game_identity import game_family_key, is_confirmed_base_game, is_edition_title

MAX_RANDOM_SAMPLE_SIZE = 50
MAX_RANDOM_CONCURRENCY = 5
RANDOM_RECOMMENDATION_TIMEOUT_SECONDS = 20.0


class UnplayedRecommendationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "unplayed_recommendation_unavailable",
    ) -> None:
        self.code = code
        super().__init__(message)


UNPLAYED_USER_MESSAGES = {
    "no_unplayed_games": "Steam 游戏库中没有未游玩过的游戏。",
    "random_scan_timeout": "随机推荐检查超时，请稍后再试。",
    "review_service_unavailable": "Steam 评测服务暂不可用，请稍后再试。",
    "no_qualified_games": "没有找到达到当前评测门槛的未玩游戏。",
}


def unplayed_user_message(error: UnplayedRecommendationError) -> str:
    return UNPLAYED_USER_MESSAGES.get(
        error.code,
        "随机推荐暂不可用，请稍后重试。",
    )


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
    *,
    sample_limit: int = MAX_RANDOM_SAMPLE_SIZE,
    concurrency: int = MAX_RANDOM_CONCURRENCY,
    timeout_seconds: float = RANDOM_RECOMMENDATION_TIMEOUT_SECONDS,
) -> UnplayedRecommendation:
    candidates = [
        game
        for game in deduplicate_owned_game_editions(owned_games)
        if game.appid
    ]
    if not candidates:
        raise UnplayedRecommendationError(
            "Steam 游戏库中没有未游玩过的游戏。",
            code="no_unplayed_games",
        )

    shuffled = list(candidates)
    if rng is None:
        random.shuffle(shuffled)
    else:
        rng.shuffle(shuffled)
    resolved_sample_limit = optional_int(sample_limit)
    if resolved_sample_limit is None:
        resolved_sample_limit = MAX_RANDOM_SAMPLE_SIZE
    resolved_sample_limit = min(
        max(resolved_sample_limit, 1),
        MAX_RANDOM_SAMPLE_SIZE,
    )
    shuffled = shuffled[:resolved_sample_limit]

    parsed_min_count = optional_int(min_review_count)
    min_count = max(parsed_min_count if parsed_min_count is not None else 50, 0)
    parsed_min_ratio = optional_float(min_positive_ratio)
    min_ratio = min(
        max(parsed_min_ratio if parsed_min_ratio is not None else 0.65, 0.0),
        1.0,
    )
    review_success_count = 0

    async def check_candidate(
        owned_game: SteamOwnedGame,
        checked_count: int,
    ) -> tuple[bool, UnplayedRecommendation | None]:
        try:
            summary = await steam_client.get_review_summary(owned_game.appid)
        except (SteamApiError, httpx.HTTPError):
            return False, None
        if not review_passes(summary, min_count, min_ratio):
            return True, None
        try:
            game = await steam_client.get_game_detail(owned_game.appid)
        except (SteamApiError, httpx.HTTPError):
            return True, None
        if not is_confirmed_base_game(game):
            return True, None
        return True, UnplayedRecommendation(
            game=attach_review_summary(game, owned_game, summary),
            owned_game=owned_game,
            checked_count=checked_count,
        )

    async def scan_candidates() -> UnplayedRecommendation | None:
        nonlocal review_success_count
        parsed_concurrency = optional_int(concurrency)
        batch_size = min(
            max(parsed_concurrency if parsed_concurrency is not None else 5, 1),
            MAX_RANDOM_CONCURRENCY,
        )
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            tasks = [
                asyncio.create_task(
                    check_candidate(owned_game, start + position + 1)
                )
                for position, owned_game in enumerate(batch)
            ]
            task_order = {task: index for index, task in enumerate(tasks)}
            pending = set(tasks)
            selected: UnplayedRecommendation | None = None
            try:
                while pending and selected is None:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in sorted(done, key=task_order.__getitem__):
                        succeeded, result = task.result()
                        review_success_count += int(succeeded)
                        if result is not None:
                            selected = result
                            break
            finally:
                for task in pending:
                    task.cancel()
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, Exception):
                        raise outcome
            if selected is not None:
                return selected
        return None

    parsed_timeout = optional_float(timeout_seconds)
    resolved_timeout = min(
        max(
            parsed_timeout
            if parsed_timeout is not None
            else RANDOM_RECOMMENDATION_TIMEOUT_SECONDS,
            0.01,
        ),
        RANDOM_RECOMMENDATION_TIMEOUT_SECONDS,
    )
    try:
        recommendation = await asyncio.wait_for(
            scan_candidates(),
            timeout=resolved_timeout,
        )
    except TimeoutError as exc:
        raise UnplayedRecommendationError(
            "随机推荐检查超时，请稍后再试。",
            code="random_scan_timeout",
        ) from exc
    if recommendation is not None:
        return recommendation

    if review_success_count == 0:
        raise UnplayedRecommendationError(
            "Steam 评测服务暂不可用，请稍后再试。",
            code="review_service_unavailable",
        )
    raise UnplayedRecommendationError(
        "没有找到未游玩且评价过线的游戏"
        f"（门槛：至少 {min_count} 条评测、好评率不低于 {min_ratio:.0%}）。",
        code="no_qualified_games",
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
