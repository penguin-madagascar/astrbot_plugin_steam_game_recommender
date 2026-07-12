from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import Any

from ..storage.models import GamePreference, RankedGame

MEMORY_TTL_MINUTES = 30
MAX_FEEDBACK_ENTRIES = 10


@dataclass(frozen=True)
class RecommendationResultSummary:
    appid: int | None
    title: str
    tags: list[str]


@dataclass(frozen=True)
class PreferencePatch:
    add_tags: list[str] = field(default_factory=list)
    remove_tags: list[str] = field(default_factory=list)
    condition_overrides: dict[str, Any] = field(default_factory=dict)
    clear_conditions: list[str] = field(default_factory=list)
    positive_reference_ordinals: list[int] = field(default_factory=list)
    negative_reference_ordinals: list[int] = field(default_factory=list)
    exclude_ordinals: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class RecommendationFeedback:
    patch: PreferencePatch
    created_at: float


@dataclass(frozen=True)
class RecommendationMemory:
    chat_platform: str
    chat_user_id: str
    raw_query: str
    preference: GamePreference
    result_limit: int
    shown_appids: list[int]
    shown_titles: list[str]
    created_at: float
    last_results: list[RecommendationResultSummary] = field(default_factory=list)
    feedback: list[RecommendationFeedback] = field(default_factory=list)


def build_recommendation_memory(
    chat_platform: str,
    chat_user_id: str,
    raw_query: str,
    preference: GamePreference,
    result_limit: int,
    games: list[RankedGame],
    now: float | None = None,
) -> RecommendationMemory:
    memory = RecommendationMemory(
        chat_platform=chat_platform or "default",
        chat_user_id=chat_user_id,
        raw_query=raw_query,
        preference=preference,
        result_limit=result_limit,
        shown_appids=[],
        shown_titles=[],
        created_at=now if now is not None else time.time(),
        last_results=summarize_games(games),
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
    cutoff = current - max(ttl_minutes, 0) * 60
    return replace(
        memory,
        feedback=[item for item in memory.feedback if item.created_at >= cutoff],
    )


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


def summarize_games(games: list[RankedGame]) -> list[RecommendationResultSummary]:
    return [
        RecommendationResultSummary(
            appid=int(game.appid) if game.appid is not None else None,
            title=game.title,
            tags=list(dict.fromkeys([*game.ordered_tags, *game.tags, *game.genres]))[:12],
        )
        for game in games
    ]


def append_feedback(
    memory: RecommendationMemory,
    patch: PreferencePatch,
    now: float | None = None,
) -> RecommendationMemory:
    feedback = [
        *memory.feedback,
        RecommendationFeedback(
            patch=patch,
            created_at=now if now is not None else time.time(),
        ),
    ][-MAX_FEEDBACK_ENTRIES:]
    return replace(memory, feedback=feedback)


def recommendation_memory_key(chat_platform: str, chat_user_id: str) -> str:
    return f"recommendation_memory:{chat_platform or 'default'}:{chat_user_id}"


def dump_memory(memory: RecommendationMemory) -> dict[str, Any]:
    return {
        "chat_platform": memory.chat_platform,
        "chat_user_id": memory.chat_user_id,
        "raw_query": memory.raw_query,
        "preference": dump_model(memory.preference),
        "result_limit": memory.result_limit,
        "shown_appids": memory.shown_appids,
        "shown_titles": memory.shown_titles,
        "created_at": memory.created_at,
        "last_results": [
            {
                "appid": item.appid,
                "title": item.title,
                "tags": item.tags,
            }
            for item in memory.last_results
        ],
        "feedback": [
            {
                "patch": {
                    "add_tags": item.patch.add_tags,
                    "remove_tags": item.patch.remove_tags,
                    "condition_overrides": item.patch.condition_overrides,
                    "clear_conditions": item.patch.clear_conditions,
                    "positive_reference_ordinals": item.patch.positive_reference_ordinals,
                    "negative_reference_ordinals": item.patch.negative_reference_ordinals,
                    "exclude_ordinals": item.patch.exclude_ordinals,
                },
                "created_at": item.created_at,
            }
            for item in memory.feedback[-MAX_FEEDBACK_ENTRIES:]
        ],
    }


def parse_memory(payload: Any) -> RecommendationMemory | None:
    if not isinstance(payload, dict):
        return None
    preference_data = payload.get("preference")
    if not isinstance(preference_data, dict):
        return None
    validator = getattr(GamePreference, "model_validate", None)
    preference = (
        validator(preference_data) if validator else GamePreference.parse_obj(preference_data)
    )
    return RecommendationMemory(
        chat_platform=str(payload.get("chat_platform") or "default"),
        chat_user_id=str(payload.get("chat_user_id") or ""),
        raw_query=str(payload.get("raw_query") or ""),
        preference=preference,
        result_limit=max(int(payload.get("result_limit") or 1), 1),
        shown_appids=[
            int(appid) for appid in payload.get("shown_appids") or [] if safe_int(appid) is not None
        ],
        shown_titles=[
            title
            for title in (normalize_title(value) for value in payload.get("shown_titles") or [])
            if title
        ],
        created_at=float(payload.get("created_at") or 0),
        last_results=parse_result_summaries(payload.get("last_results")),
        feedback=parse_feedback(payload.get("feedback")),
    )


def parse_result_summaries(value: Any) -> list[RecommendationResultSummary]:
    if not isinstance(value, list):
        return []
    results: list[RecommendationResultSummary] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        appid = safe_int(item.get("appid"))
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        results.append(
            RecommendationResultSummary(
                appid=appid,
                title=title,
                tags=[str(tag) for tag in item.get("tags") or [] if str(tag).strip()][:12],
            )
        )
    return results


def parse_feedback(value: Any) -> list[RecommendationFeedback]:
    if not isinstance(value, list):
        return []
    results: list[RecommendationFeedback] = []
    for item in value[-MAX_FEEDBACK_ENTRIES:]:
        if not isinstance(item, dict) or not isinstance(item.get("patch"), dict):
            continue
        patch = item["patch"]
        results.append(
            RecommendationFeedback(
                patch=PreferencePatch(
                    add_tags=text_list(patch.get("add_tags")),
                    remove_tags=text_list(patch.get("remove_tags")),
                    condition_overrides=dict(patch.get("condition_overrides") or {}),
                    clear_conditions=text_list(patch.get("clear_conditions")),
                    positive_reference_ordinals=int_list(patch.get("positive_reference_ordinals")),
                    negative_reference_ordinals=int_list(patch.get("negative_reference_ordinals")),
                    exclude_ordinals=int_list(patch.get("exclude_ordinals")),
                ),
                created_at=float(item.get("created_at") or 0),
            )
        )
    return results


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def text_list(value: Any) -> list[str]:
    return [str(item).strip() for item in value or [] if str(item).strip()]


def int_list(value: Any) -> list[int]:
    return [number for item in value or [] if (number := safe_int(item)) is not None]


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()
