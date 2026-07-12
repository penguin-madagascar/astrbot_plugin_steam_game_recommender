from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..storage.models import GamePreference
from .recommendation_memory import (
    PreferencePatch,
    RecommendationResultSummary,
)

RETRY_PREFIXES = (
    "重新推荐",
    "换一批",
    "再推荐",
    "gamerec_retry",
    "retry",
)


@dataclass(frozen=True)
class RetryRequest:
    is_retry: bool
    supplement: str = ""


@dataclass(frozen=True)
class ParsedPreferencePatch:
    patch: PreferencePatch
    residual_text: str
    warnings: list[str]


def parse_retry_request(text: str) -> RetryRequest:
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    for prefix in RETRY_PREFIXES:
        key = prefix.lower()
        if lowered == key:
            return RetryRequest(is_retry=True)
        if lowered.startswith(key):
            rest = stripped[len(prefix) :]
            if key.isascii() and rest and not rest[0].isspace():
                continue
            supplement = re.sub(r"^[\s,，、;；:：]+", "", rest).strip()
            return RetryRequest(is_retry=True, supplement=supplement)
    return RetryRequest(is_retry=False)


def parse_preference_patch(
    text: str,
    result_count: int,
) -> ParsedPreferencePatch:
    source = str(text or "").strip()
    positive: list[int] = []
    negative: list[int] = []
    excluded: list[int] = []
    warnings: list[str] = []
    spans: list[tuple[int, int]] = []
    ordinal_pattern = re.compile(r"第\s*(\d+)\s*款")
    for match in ordinal_pattern.finditer(source):
        ordinal = int(match.group(1))
        start = max(source.rfind("，", 0, match.start()), source.rfind(",", 0, match.start())) + 1
        punctuation = [
            position
            for mark in ("，", ",", "。", ";", "；")
            if (position := source.find(mark, match.end())) >= 0
        ]
        end = min(punctuation) if punctuation else len(source)
        clause = source[start:end]
        spans.append((start, end))
        if ordinal < 1 or ordinal > result_count:
            warnings.append(f"反馈序号第 {ordinal} 款超出上一批结果范围，已忽略。")
            continue
        if re.search(r"不喜欢|不想要.*这类", clause):
            negative.append(ordinal)
        elif re.search(r"喜欢|想要.*这类", clause) and "这类" in clause:
            positive.append(ordinal)
        elif re.search(r"不要|排除|去掉", clause):
            excluded.append(ordinal)

    residual = source
    for start, end in sorted(spans, reverse=True):
        residual = f"{residual[:start]} {residual[end:]}"
    residual = re.sub(r"^[\s,，、;；:：]+|[\s,，、;；:：]+$", "", residual)
    residual = re.sub(r"^(?:再)?换一批$", "", residual).strip()

    clear_conditions: list[str] = []
    for name, pattern in {
        "budget": r"(?:取消|清除|不要)(?:预算|价格)(?:限制|要求)?",
        "players": r"(?:取消|清除|不要)(?:人数|联机人数)(?:限制|要求)?",
        "language": r"(?:取消|清除|不要)(?:中文|语言)(?:限制|要求)?",
        "difficulty": r"(?:取消|清除|不要)(?:难度)(?:限制|要求)?",
        "mood": r"(?:取消|清除|不要)(?:氛围|心情)(?:限制|要求)?",
    }.items():
        if re.search(pattern, source):
            clear_conditions.append(name)

    overrides: dict[str, Any] = {}
    budget = re.search(r"预算(?:改为|改成|调整到|设为|到)?\s*[¥￥]?\s*(\d+(?:\.\d+)?)", source)
    if budget and "budget" not in clear_conditions:
        overrides["budget"] = float(budget.group(1))
    players = re.search(r"(?:改为|改成|调整为|要)?\s*(\d+)\s*人", source)
    if players and "players" not in clear_conditions:
        overrides["players"] = int(players.group(1))

    add_tags = explicit_tag_values(source, r"(?:增加|添加|加上)(?:标签)?")
    remove_tags = explicit_tag_values(source, r"(?:删除|移除|去掉)(?:标签)?")
    return ParsedPreferencePatch(
        patch=PreferencePatch(
            add_tags=add_tags,
            remove_tags=remove_tags,
            condition_overrides=overrides,
            clear_conditions=clear_conditions,
            positive_reference_ordinals=dedupe_ints(positive),
            negative_reference_ordinals=dedupe_ints(negative),
            exclude_ordinals=dedupe_ints(excluded),
        ),
        residual_text=residual,
        warnings=warnings,
    )


def apply_preference_patch(
    preference: GamePreference,
    patch: PreferencePatch,
    results: list[RecommendationResultSummary],
    warnings: list[str] | None = None,
) -> tuple[GamePreference, list[int], list[str]]:
    data = dump_model(preference)
    data["extra_tags"] = merge_text(data.get("extra_tags") or [], patch.add_tags)
    removed = {value.lower() for value in patch.remove_tags}
    for field_name in ("required_tags", "genres_like", "extra_tags"):
        data[field_name] = [
            value for value in data.get(field_name) or [] if str(value).lower() not in removed
        ]
    for field_name in patch.clear_conditions:
        if field_name == "language":
            data["preferred_languages"] = []
            data["required_languages"] = []
        elif field_name == "budget":
            data["budget"] = None
            data["budget_currency"] = None
        elif field_name in {"players", "difficulty", "mood"}:
            data[field_name] = None
    for field_name, value in patch.condition_overrides.items():
        if field_name in {"budget", "players", "difficulty", "mood"}:
            data[field_name] = value

    positive_titles = list(data.get("reference_games_like") or [])
    negative_titles = list(data.get("reference_games_dislike") or [])
    excluded_appids: list[int] = []
    excluded_titles: list[str] = []
    for ordinal in patch.positive_reference_ordinals:
        if 1 <= ordinal <= len(results):
            positive_titles = merge_text(positive_titles, [results[ordinal - 1].title])
    for ordinal in patch.negative_reference_ordinals:
        if 1 <= ordinal <= len(results):
            negative_titles = merge_text(negative_titles, [results[ordinal - 1].title])
    for ordinal in patch.exclude_ordinals:
        if 1 <= ordinal <= len(results):
            result = results[ordinal - 1]
            if result.appid is not None:
                excluded_appids.append(result.appid)
            excluded_titles.append(result.title.lower())
    data["reference_games_like"] = positive_titles
    data["reference_games_dislike"] = negative_titles
    data["parse_warnings"] = merge_text(
        data.get("parse_warnings") or [],
        warnings or [],
    )
    return validate_preference(data), excluded_appids, excluded_titles


def merge_retry_preferences(
    base: GamePreference,
    supplement: GamePreference,
) -> GamePreference:
    data = dump_model(base)
    for field_name in (
        "required_tags",
        "genres_like",
        "extra_tags",
        "genres_dislike",
        "reference_games_like",
        "reference_search_terms",
        "reference_games_dislike",
        "preferred_languages",
        "required_languages",
        "parse_warnings",
    ):
        data[field_name] = merge_text(
            data.get(field_name) or [],
            list(getattr(supplement, field_name)),
        )
    if supplement.platforms:
        data["platforms"] = list(supplement.platforms)
    for field_name in ("players", "budget", "region", "budget_currency", "difficulty", "mood"):
        value = getattr(supplement, field_name)
        if value is not None:
            data[field_name] = value
    if supplement.library_filter_mode:
        data["library_filter_mode"] = supplement.library_filter_mode
    return validate_preference(data)


def explicit_tag_values(text: str, prefix: str) -> list[str]:
    match = re.search(rf"{prefix}\s*([^，,。；;]+)", text)
    if not match:
        return []
    return [value.strip() for value in re.split(r"[、/|]", match.group(1)) if value.strip()]


def merge_text(left: list[str], right: list[str]) -> list[str]:
    result = list(left)
    seen = {value.lower() for value in result}
    for value in right:
        text = str(value or "").strip()
        if text and text.lower() not in seen:
            result.append(text)
            seen.add(text.lower())
    return result


def dedupe_ints(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_preference(data: dict[str, Any]) -> GamePreference:
    validator = getattr(GamePreference, "model_validate", None)
    return validator(data) if validator else GamePreference.parse_obj(data)
