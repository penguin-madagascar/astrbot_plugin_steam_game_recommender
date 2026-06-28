from __future__ import annotations

import re

from ..storage.models import GamePreference
from .reference_data import REFERENCE_PROFILES
from .reference_resolver import alias_for_title, extract_reference_titles


def infer_preference_from_text(text: str) -> GamePreference:
    lower = text.lower()
    platforms: list[str] = []
    if "steam" in lower:
        platforms.append("steam")
    if any(word in lower for word in ("pc", "电脑", "windows")):
        platforms.append("pc")
    if any(word in lower for word in ("switch", "任天堂", "ns")):
        platforms.append("nintendo switch")
    if any(word in lower for word in ("playstation", "ps5", "ps4", "psn")):
        platforms.append("playstation")
    if "xbox" in lower:
        platforms.append("xbox")

    genres_like = keyword_hits(
        lower,
        {
            "co-op": ("双人", "两人", "合作", "coop", "co-op"),
            "local co-op": ("本地合作", "同屏", "分屏", "双人"),
            "multiplayer": ("多人", "联机"),
            "puzzle": ("解谜", "谜题", "puzzle"),
            "adventure": ("冒险", "剧情", "adventure"),
            "casual": ("休闲", "轻松", "casual", "别太难", "不要太难"),
            "action": ("动作", "action"),
            "rpg": ("rpg", "角色扮演"),
            "party": ("聚会", "派对", "party"),
            "simulation": ("模拟", "simulation"),
            "farming": ("种田", "农场", "farming", "farm"),
            "management": ("经营", "management"),
            "crafting": ("制作", "crafting"),
            "building": ("建造", "building"),
            "racing": ("赛车", "竞速", "racing"),
        },
    )
    genres_dislike = keyword_hits(
        lower,
        {
            "horror": ("不要恐怖", "不恐怖", "恐怖", "horror"),
            "soulslike": ("魂like", "魂系", "soulslike", "souls-like"),
            "roguelike": ("肉鸽", "roguelike", "rogue-like"),
            "violent": ("血腥", "violent", "gore"),
        },
    )

    players = 2 if any(word in lower for word in ("双人", "两人", "合作", "co-op")) else None
    if players is None and "多人" in lower:
        players = 2

    budget = None
    budget_match = re.search(
        r"(?:预算|价格|价位)?\s*(\d+(?:\.\d+)?)\s*(?:以内|以下|元|块|rmb)",
        lower,
    )
    if budget_match:
        budget = float(budget_match.group(1))

    result_count = extract_result_count(lower) or 5

    difficulty = None
    if any(
        word in lower
        for word in (
            "别太难",
            "不要太难",
            "简单",
            "轻松",
            "休闲",
            "不要高难",
            "别高难",
            "不高难",
        )
    ):
        difficulty = "easy"
    elif any(word in lower for word in ("高难", "困难", "挑战")):
        difficulty = "hard"

    reference_like = extract_reference_games(text)
    for reference in reference_like:
        profile = reference_profile(reference)
        if profile:
            genres_like.extend(profile.genres_like)

    return GamePreference(
        platforms=platforms,
        genres_like=genres_like,
        genres_dislike=genres_dislike,
        reference_games_like=reference_like,
        players=players,
        budget=budget,
        language="中文" if "中文" in text or "汉化" in text else None,
        difficulty=difficulty,
        mood="轻松" if any(word in lower for word in ("轻松", "休闲", "治愈")) else None,
        result_count=result_count,
    )


def merge_text_preference(preference: GamePreference, text: str) -> GamePreference:
    inferred = infer_preference_from_text(text)
    data = dump_preference(preference)
    for field in (
        "genres_like",
        "genres_dislike",
        "reference_games_like",
        "reference_games_dislike",
        "parse_warnings",
    ):
        data[field] = merge_lists(getattr(preference, field), getattr(inferred, field))
    data["platforms"] = merge_platforms(preference.platforms, inferred.platforms)
    for field in ("players", "budget", "language", "difficulty", "mood"):
        if getattr(preference, field) in (None, "", []):
            data[field] = getattr(inferred, field)
    if explicit_count := extract_result_count(text):
        data["result_count"] = explicit_count
    elif not preference.result_count:
        data["result_count"] = inferred.result_count
    validator = getattr(GamePreference, "model_validate", None)
    return validator(data) if validator else GamePreference.parse_obj(data)


def extract_reference_games(text: str) -> list[str]:
    return merge_lists([], [normalize_reference_game(item) for item in extract_reference_titles(text)])


def normalize_reference_game(value: str) -> str:
    alias = alias_for_title(value)
    return alias.canonical_title if alias else value


def extract_result_count(text: str) -> int | None:
    count_match = re.search(r"(\d+)\s*(?:个|款|部)", text.lower())
    if not count_match:
        return None
    return min(max(int(count_match.group(1)), 1), 10)


def reference_profile(value: str):
    alias = alias_for_title(value)
    return REFERENCE_PROFILES.get(alias.rawg_slug) if alias else None


def keyword_hits(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, words in mapping.items() if any(word in text for word in words)]


def merge_lists(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in [*left, *right]:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def merge_platforms(llm_platforms: list[str], text_platforms: list[str]) -> list[str]:
    if text_platforms:
        return merge_lists([], text_platforms)
    return merge_lists([], llm_platforms)


def dump_preference(preference: GamePreference) -> dict:
    dumper = getattr(preference, "model_dump", None)
    return dumper() if dumper else preference.dict()
