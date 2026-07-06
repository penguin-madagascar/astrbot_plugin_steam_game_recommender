from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any

from ..storage.models import GamePreference, RankedGame

MEMORY_TTL_MINUTES = 30


@dataclass(frozen=True)
class RecommendationMemory:
    chat_platform: str
    chat_user_id: str
    raw_query: str
    preference: GamePreference
    diversity_mode: str
    result_limit: int
    shown_appids: list[int]
    shown_titles: list[str]
    created_at: float


def build_recommendation_memory(
    chat_platform: str,
    chat_user_id: str,
    raw_query: str,
    preference: GamePreference,
    diversity_mode: str,
    result_limit: int,
    games: list[RankedGame],
    now: float | None = None,
) -> RecommendationMemory:
    memory = RecommendationMemory(
        chat_platform=chat_platform or "default",
        chat_user_id=chat_user_id,
        raw_query=raw_query,
        preference=preference,
        diversity_mode=diversity_mode,
        result_limit=result_limit,
        shown_appids=[],
        shown_titles=[],
        created_at=now if now is not None else time.time(),
    )
    return append_shown_games(memory, games)


async def save_recommendation_memory(cache: Any, memory: RecommendationMemory) -> None:
    await cache.set_json(
        recommendation_memory_key(memory.chat_platform, memory.chat_user_id),
        dump_memory(memory),
    )


async def load_recommendation_memory(
    chat_platform: str,
    chat_user_id: str,
    cache: Any,
    ttl_minutes: int = MEMORY_TTL_MINUTES,
    now: float | None = None,
) -> RecommendationMemory | None:
    payload = await cache.get_json(
        recommendation_memory_key(chat_platform, chat_user_id),
        24,
    )
    memory = parse_memory(payload)
    if memory is None:
        return None
    current = now if now is not None else time.time()
    if current - memory.created_at > max(ttl_minutes, 0) * 60:
        return None
    return memory


def append_shown_games(
    memory: RecommendationMemory,
    games: list[RankedGame],
) -> RecommendationMemory:
    appids = list(memory.shown_appids)
    titles = list(memory.shown_titles)
    for game in games:
        if game.appid is not None and int(game.appid) not in appids:
            appids.append(int(game.appid))
        title = normalize_title(game.title)
        if title and title not in titles:
            titles.append(title)
    return replace(memory, shown_appids=appids, shown_titles=titles)


def recommendation_memory_key(chat_platform: str, chat_user_id: str) -> str:
    return f"recommendation_memory:{chat_platform or 'default'}:{chat_user_id}"


def dump_memory(memory: RecommendationMemory) -> dict[str, Any]:
    return {
        "chat_platform": memory.chat_platform,
        "chat_user_id": memory.chat_user_id,
        "raw_query": memory.raw_query,
        "preference": dump_model(memory.preference),
        "diversity_mode": memory.diversity_mode,
        "result_limit": memory.result_limit,
        "shown_appids": memory.shown_appids,
        "shown_titles": memory.shown_titles,
        "created_at": memory.created_at,
    }


def parse_memory(payload: Any) -> RecommendationMemory | None:
    if not isinstance(payload, dict):
        return None
    preference_data = payload.get("preference")
    if not isinstance(preference_data, dict):
        return None
    validator = getattr(GamePreference, "model_validate", None)
    preference = validator(preference_data) if validator else GamePreference.parse_obj(preference_data)
    return RecommendationMemory(
        chat_platform=str(payload.get("chat_platform") or "default"),
        chat_user_id=str(payload.get("chat_user_id") or ""),
        raw_query=str(payload.get("raw_query") or ""),
        preference=preference,
        diversity_mode=str(payload.get("diversity_mode") or "strict"),
        result_limit=max(int(payload.get("result_limit") or 1), 1),
        shown_appids=[
            int(appid)
            for appid in payload.get("shown_appids") or []
            if safe_int(appid) is not None
        ],
        shown_titles=[
            title
            for title in (normalize_title(value) for value in payload.get("shown_titles") or [])
            if title
        ],
        created_at=float(payload.get("created_at") or 0),
    )


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()
