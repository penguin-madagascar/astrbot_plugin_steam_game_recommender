from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from ..storage.models import MAX_REFERENCE_ENTITIES, GamePreference
from .company_preferences import merge_retry_company_preferences
from .preference_rules import extract_budget
from .reference_matching import title_key
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
    budget, budget_currency, budget_is_required = extract_budget(source.lower())
    if budget is not None and "budget" not in clear_conditions:
        overrides["budget"] = budget
        overrides["budget_currency"] = budget_currency
        overrides["budget_is_required"] = budget_is_required
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
            data["budget_is_required"] = False
        elif field_name in {"players", "difficulty", "mood"}:
            data[field_name] = None
    for field_name, value in patch.condition_overrides.items():
        if field_name in {
            "budget",
            "budget_currency",
            "budget_is_required",
            "players",
            "difficulty",
            "mood",
        }:
            data[field_name] = value

    excluded_appids: list[int] = []
    excluded_titles: list[str] = []
    for ordinal in patch.positive_reference_ordinals:
        if 1 <= ordinal <= len(results):
            result = results[ordinal - 1]
            set_reference_polarity(
                data,
                result.title,
                "positive",
                appid=result.appid,
            )
    for ordinal in patch.negative_reference_ordinals:
        if 1 <= ordinal <= len(results):
            result = results[ordinal - 1]
            set_reference_polarity(
                data,
                result.title,
                "negative",
                appid=result.appid,
            )
    for ordinal in patch.exclude_ordinals:
        if 1 <= ordinal <= len(results):
            result = results[ordinal - 1]
            if result.appid is not None:
                excluded_appids.append(result.appid)
            excluded_titles.append(result.title.lower())
    data["parse_warnings"] = merge_text(
        data.get("parse_warnings") or [],
        warnings or [],
    )
    return validate_preference(data), excluded_appids, excluded_titles


def set_reference_polarity(
    data: dict[str, Any],
    title: str,
    polarity: str,
    *,
    appid: int | None = None,
) -> None:
    normalized_title = str(title or "").strip()
    if not normalized_title:
        return
    target_field = (
        "reference_games_like"
        if polarity == "positive"
        else "reference_games_dislike"
    )
    opposite_field = (
        "reference_games_dislike"
        if polarity == "positive"
        else "reference_games_like"
    )
    title_key = normalize_reference_identity(normalized_title)
    related_keys = {title_key}
    resolved_values = [
        dict(value) if isinstance(value, dict) else dump_model(value)
        for value in data.get("resolved_reference_games") or []
    ]
    matching_resolved_positions: set[int] = set()
    for position, payload in enumerate(resolved_values):
        resolved_appid = payload.get("appid")
        appid_matches = (
            appid is not None
            and resolved_appid is not None
            and int(resolved_appid) == int(appid)
        )
        title_matches = title_key in {
            normalize_reference_identity(payload.get("raw_text")),
            normalize_reference_identity(payload.get("canonical_title")),
        }
        if appid_matches or title_matches:
            matching_resolved_positions.add(position)
            related_keys.update(
                filter(
                    None,
                    (
                        normalize_reference_identity(payload.get("raw_text")),
                        normalize_reference_identity(payload.get("canonical_title")),
                    ),
                )
            )

    entity_values = [
        dict(value) if isinstance(value, dict) else dump_model(value)
        for value in data.get("reference_entities") or []
    ]
    matching_entities = [
        payload
        for payload in entity_values
        if reference_payload_keys(payload).intersection(related_keys)
    ]
    active_titles = [
        str(payload.get("display_title") or "").strip()
        for payload in matching_entities
        if str(payload.get("display_title") or "").strip()
    ] or [normalized_title]
    active_title_keys = {
        normalize_reference_identity(value) for value in active_titles
    }
    for payload in matching_entities:
        payload["polarity"] = polarity

    for field_name in (target_field, opposite_field):
        data[field_name] = [
            value
            for value in data.get(field_name) or []
            if normalize_reference_identity(value) not in active_title_keys
            and normalize_reference_identity(value) not in related_keys
        ]
    data[target_field] = merge_text(
        list(data.get(target_field) or []),
        active_titles,
    )

    data["reference_entities"] = entity_values
    data["reference_search_terms"] = [
        alias
        for payload in entity_values
        if payload.get("polarity") == "positive"
        for alias in payload.get("aliases") or []
    ]
    resolved_polarity = "like" if polarity == "positive" else "dislike"
    for position in matching_resolved_positions:
        resolved_values[position]["polarity"] = resolved_polarity
    data["resolved_reference_games"] = resolved_values


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
        "preferred_languages",
        "required_languages",
        "parse_warnings",
    ):
        data[field_name] = merge_text(
            data.get(field_name) or [],
            list(getattr(supplement, field_name)),
        )
    merge_retry_reference_preferences(data, base, supplement)
    for field_name, identity_key in (
        (
            "derived_intent_tags",
            lambda payload: (str(payload.get("tag") or "").strip().casefold(),),
        ),
        (
            "soft_features",
            lambda payload: (
                str(payload.get("constraint_id") or "").strip().casefold(),
            ),
        ),
    ):
        data[field_name] = merge_structured_preferences(
            data.get(field_name) or [],
            list(getattr(supplement, field_name)),
            identity_key=identity_key,
            limit=3,
        )
    data["company_preferences"] = merge_retry_company_preferences(
        base.company_preferences,
        supplement.company_preferences,
        limit=3,
    )
    if supplement.platforms:
        data["platforms"] = list(supplement.platforms)
    if supplement.budget is not None:
        data["budget"] = supplement.budget
        data["budget_currency"] = supplement.budget_currency
        data["budget_is_required"] = supplement.budget_is_required
    for field_name in ("players", "region", "difficulty", "mood"):
        value = getattr(supplement, field_name)
        if value is not None:
            data[field_name] = value
    if supplement.library_filter_mode:
        data["library_filter_mode"] = supplement.library_filter_mode
    return validate_preference(data)


def merge_retry_reference_preferences(
    data: dict[str, Any],
    base: GamePreference,
    supplement: GamePreference,
) -> None:
    entries: list[tuple[dict[str, Any], bool]] = []

    def merge_entity(value: Any, *, supplement_owned: bool) -> None:
        incoming = dict(value) if isinstance(value, dict) else dump_model(value)
        incoming_keys = reference_payload_keys(incoming)
        if not incoming_keys:
            return
        matching_positions = [
            position
            for position, (payload, _owned) in enumerate(entries)
            if reference_payload_keys(payload).intersection(incoming_keys)
        ]
        if not matching_positions:
            entries.append((incoming, supplement_owned))
            return

        first_position = matching_positions[0]
        matched = [entries[position] for position in matching_positions]
        primary = dict(matched[0][0])
        aliases = list(primary.get("aliases") or [])
        primary_title = str(primary.get("display_title") or "").strip()
        for payload, _owned in matched[1:]:
            aliases = merge_text(
                aliases,
                [
                    str(payload.get("display_title") or ""),
                    *(payload.get("aliases") or []),
                ],
            )
        incoming_title = str(incoming.get("display_title") or "").strip()
        aliases = merge_text(
            aliases,
            [incoming_title, *(incoming.get("aliases") or [])],
        )
        primary["aliases"] = [
            alias
            for alias in aliases
            if str(alias or "").strip().casefold() != primary_title.casefold()
        ]
        primary["polarity"] = incoming.get("polarity", "positive")
        merged_owned = supplement_owned or any(owned for _payload, owned in matched)
        for position in reversed(matching_positions):
            entries.pop(position)
        entries.insert(first_position, (primary, merged_owned))

    for entity in base.reference_entities:
        merge_entity(entity, supplement_owned=False)
    for entity in supplement.reference_entities:
        merge_entity(entity, supplement_owned=True)

    while len(entries) > MAX_REFERENCE_ENTITIES:
        removable = next(
            (
                position
                for position, (_payload, supplement_owned) in enumerate(entries)
                if not supplement_owned
            ),
            0,
        )
        entries.pop(removable)

    merged = [payload for payload, _supplement_owned in entries]
    data["reference_entities"] = merged
    data["reference_games_like"] = [
        str(payload.get("display_title") or "")
        for payload in merged
        if payload.get("polarity") == "positive"
    ]
    data["reference_games_dislike"] = [
        str(payload.get("display_title") or "")
        for payload in merged
        if payload.get("polarity") == "negative"
    ]
    data["reference_search_terms"] = [
        alias
        for payload in merged
        if payload.get("polarity") == "positive"
        for alias in payload.get("aliases") or []
    ]


def reference_payload_keys(payload: dict[str, Any]) -> set[str]:
    return {
        normalized
        for value in [
            payload.get("display_title"),
            *(payload.get("aliases") or []),
        ]
        if (normalized := normalize_reference_identity(value))
    }


def normalize_reference_identity(value: Any) -> str:
    return title_key(str(value or ""))


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


def merge_structured_preferences(
    left: list[Any],
    right: list[Any],
    *,
    identity_key: Callable[[dict[str, Any]], tuple[str, ...]],
    limit: int,
) -> list[dict[str, Any]]:
    resolved_limit = max(int(limit), 0)
    if resolved_limit == 0:
        return []
    result: list[dict[str, Any]] = []
    identities: list[tuple[str, ...]] = []
    position_by_identity: dict[tuple[str, ...], int] = {}
    for value in left:
        payload = dict(value) if isinstance(value, dict) else dump_model(value)
        identity = identity_key(payload)
        if not identity or not all(identity) or identity in position_by_identity:
            continue
        position_by_identity[identity] = len(result)
        result.append(payload)
        identities.append(identity)
        if len(result) >= resolved_limit:
            break
    protected: set[tuple[str, ...]] = set()
    for value in right:
        payload = dict(value) if isinstance(value, dict) else dump_model(value)
        identity = identity_key(payload)
        if not identity or not all(identity):
            continue
        protected.add(identity)
        position = position_by_identity.get(identity)
        if position is not None:
            result[position] = payload
            continue
        position_by_identity[identity] = len(result)
        result.append(payload)
        identities.append(identity)
    for position in range(len(result) - 1, -1, -1):
        if len(result) <= resolved_limit:
            break
        if identities[position] in protected:
            continue
        result.pop(position)
        identities.pop(position)
    if len(result) > resolved_limit:
        result = result[-resolved_limit:]
    return result


def dedupe_ints(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_preference(data: dict[str, Any]) -> GamePreference:
    validator = getattr(GamePreference, "model_validate", None)
    return validator(data) if validator else GamePreference.parse_obj(data)
