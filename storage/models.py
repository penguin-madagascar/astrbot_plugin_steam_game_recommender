from __future__ import annotations

import math
import re
from typing import Any, Callable

from pydantic import BaseModel, Field, root_validator, validator

from ..services.recommendation_limits import (
    DEFAULT_RECOMMENDATION_COUNT,
    MAX_RECOMMENDATION_COUNT,
)

COMPANY_ALIAS_LIMIT = 5
MAX_REFERENCE_ENTITIES = 3
MAX_REFERENCE_ALIASES_PER_ENTITY = 3
REFERENCE_INPUT_LIMIT_WARNING = (
    "参考游戏或别名超过处理上限，已仅保留前 3 个参考实体，"
    "且每个实体最多使用 3 个标题候选。"
)
REFERENCE_ALIAS_MAPPING_WARNING = (
    "部分参考游戏别名无法可靠归属到原文提及的具体游戏，已忽略这些别名。"
)

LANGUAGE_ALIASES = {
    "chinese": "schinese",
    "simplified chinese": "schinese",
    "schinese": "schinese",
    "简体中文": "schinese",
    "简中": "schinese",
    "中文": "schinese",
    "traditional chinese": "tchinese",
    "tchinese": "tchinese",
    "繁体中文": "tchinese",
    "繁中": "tchinese",
    "english": "english",
    "英语": "english",
    "英文": "english",
    "japanese": "japanese",
    "日语": "japanese",
    "日文": "japanese",
    "korean": "koreana",
    "koreana": "koreana",
    "韩语": "koreana",
    "韩文": "koreana",
    "french": "french",
    "法语": "french",
    "german": "german",
    "德语": "german",
    "spanish": "spanish",
    "西班牙语": "spanish",
    "russian": "russian",
    "俄语": "russian",
    "portuguese": "portuguese",
    "葡萄牙语": "portuguese",
}


def normalize_language(value: Any) -> str:
    text = re.sub(r"\([^)]*\)|（[^）]*）|\*", "", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return LANGUAGE_ALIASES.get(text, text)


def split_language_list(value: Any) -> list[str]:
    values = split_text_list(value)
    normalized: list[str] = []
    for item in values:
        language = normalize_language(item)
        if language and language not in normalized:
            normalized.append(language)
    return normalized


def split_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，、/|;；\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.extend(re.split(r"[,，、/|;；\n]+", item))
            elif item is not None:
                parts.append(str(item))
    else:
        parts = [str(value)]

    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = re.sub(r"\s+", " ", str(part)).strip().lower()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def split_display_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，;；\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.extend(re.split(r"[,，;；\n]+", item))
            elif item is not None:
                parts.append(str(item))
    else:
        parts = [str(value)]

    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = re.sub(r"\s+", " ", str(part)).strip()
        key = text.lower()
        if text and key not in seen:
            normalized.append(text)
            seen.add(key)
    return normalized


def split_company_list(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = re.sub(r"\s+", " ", str(raw or "")).strip()
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _validated_model_list(
    value: Any,
    model_type: type[BaseModel],
    *,
    limit: int,
    key: str | None = None,
    identity_key: Callable[[BaseModel], Any] | None = None,
) -> list[BaseModel]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[BaseModel] = []
    seen: set[Any] = set()
    for raw in value:
        try:
            validator = getattr(model_type, "model_validate", None)
            item = validator(raw) if validator else model_type.parse_obj(raw)
        except (TypeError, ValueError):
            continue
        raw_identity = (
            identity_key(item)
            if identity_key is not None
            else getattr(item, str(key or ""), "")
        )
        if isinstance(raw_identity, (tuple, list)):
            identity = tuple(
                str(part or "").strip().casefold() for part in raw_identity
            )
            valid_identity = bool(identity) and all(identity)
        else:
            identity = str(raw_identity or "").strip().casefold()
            valid_identity = bool(identity)
        if not valid_identity or identity in seen:
            continue
        result.append(item)
        seen.add(identity)
        if len(result) >= max(int(limit), 0):
            break
    return result


def normalize_platform(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if "switch" in text or "任天堂" in text or text in {"ns", "nintendo"}:
        return "nintendo switch"
    if "steam" in text:
        return "steam"
    if text in {"pc", "电脑", "windows"} or "电脑" in text:
        return "pc"
    if "playstation" in text or text in {"ps", "ps4", "ps5", "psn"}:
        return "playstation"
    if "xbox" in text:
        return "xbox"
    return text


class ExplicitTagEvidence(BaseModel):
    target: str = ""
    tag: str = ""
    span: str = ""

    @validator("target", pre=True)
    def _normalize_target(cls, value: Any) -> str:
        target = str(value or "").strip().lower()
        return (
            target
            if target
            in {"required_tags", "genres_like", "extra_tags", "genres_dislike"}
            else ""
        )

    @validator("tag", pre=True)
    def _normalize_tag(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().lower()

    @validator("span", pre=True)
    def _normalize_span(cls, value: Any) -> str:
        return str(value or "").strip()

    class Config:
        extra = "ignore"


class DerivedIntentTag(BaseModel):
    tag: str
    source_span: str
    weight: float = Field(default=0.25, exclude=True)

    @validator("tag", pre=True)
    def _known_canonical_tag(cls, value: Any) -> str:
        from ..services.tag_normalizer import (
            ASCII_CANONICAL_TAG_PATTERN,
            canonical_steam_tag_name,
        )

        canonical = canonical_steam_tag_name(str(value or "").strip())
        if not ASCII_CANONICAL_TAG_PATTERN.fullmatch(canonical):
            raise ValueError("derived intent tag is not canonical")
        return canonical

    @validator("source_span", pre=True)
    def _required_source_span(cls, value: Any) -> str:
        span = str(value or "")
        if not span.strip():
            raise ValueError("derived intent tag requires a source span")
        return span

    @validator("weight", pre=True, always=True)
    def _fixed_weight(cls, _value: Any) -> float:
        return 0.25

    class Config:
        extra = "ignore"


class SoftFeature(BaseModel):
    constraint_id: str
    source_span: str
    normalized_text: str
    role: str = "optional"
    polarity: str = "positive"
    proxy_tags: list[str] = Field(default_factory=list)

    @validator("constraint_id", "normalized_text", pre=True)
    def _required_text(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            raise ValueError("soft feature text fields must not be empty")
        return text

    @validator("source_span", pre=True)
    def _verbatim_source_span(cls, value: Any) -> str:
        span = str(value or "")
        if not span.strip():
            raise ValueError("soft feature requires a source span")
        return span

    @validator("role", pre=True)
    def _valid_role(cls, value: Any) -> str:
        role = str(value or "").strip().lower()
        if role not in {"required", "core", "optional"}:
            raise ValueError("invalid soft feature role")
        return role

    @validator("polarity", pre=True)
    def _valid_polarity(cls, value: Any) -> str:
        polarity = str(value or "").strip().lower()
        if polarity not in {"positive", "negative"}:
            raise ValueError("invalid soft feature polarity")
        return polarity

    @validator("proxy_tags", pre=True)
    def _known_proxy_tags(cls, value: Any) -> list[str]:
        from ..services.tag_normalizer import (
            ASCII_CANONICAL_TAG_PATTERN,
            canonical_steam_tag_name,
        )

        result: list[str] = []
        for raw in split_text_list(value):
            canonical = canonical_steam_tag_name(raw)
            if (
                ASCII_CANONICAL_TAG_PATTERN.fullmatch(canonical)
                and canonical not in result
            ):
                result.append(canonical)
        return result

    class Config:
        extra = "ignore"


class CompanyPreference(BaseModel):
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    role: str = "either"
    strength: str = "preferred"
    source_span: str

    @validator("display_name", pre=True)
    def _required_display_text(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            raise ValueError("company preference text must not be empty")
        return text

    @validator("source_span", pre=True)
    def _verbatim_company_span(cls, value: Any) -> str:
        span = str(value or "")
        if not span.strip():
            raise ValueError("company preference requires a source span")
        return span

    @validator("aliases", pre=True)
    def _normalize_aliases(cls, value: Any) -> list[str]:
        return split_display_list(value)[:COMPANY_ALIAS_LIMIT]

    @validator("role", pre=True)
    def _valid_company_role(cls, value: Any) -> str:
        role = str(value or "").strip().lower()
        if role not in {"developer", "publisher", "either"}:
            raise ValueError("invalid company preference role")
        return role

    @validator("strength", pre=True)
    def _valid_company_strength(cls, value: Any) -> str:
        strength = str(value or "").strip().lower()
        if strength not in {"preferred", "strong"}:
            raise ValueError("invalid company preference strength")
        return strength

    class Config:
        extra = "ignore"


class ReferenceEntity(BaseModel):
    display_title: str
    aliases: list[str] = Field(default_factory=list)
    polarity: str = "positive"

    @validator("display_title", pre=True)
    def _required_display_title(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            raise ValueError("reference entity requires a display title")
        return text

    @validator("aliases", pre=True)
    def _normalize_reference_aliases(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("polarity", pre=True)
    def _normalize_reference_polarity(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("reference entity polarity must be a string")
        polarity = value.strip().lower()
        if polarity not in {"positive", "negative"}:
            raise ValueError("invalid reference entity polarity")
        return polarity

    class Config:
        extra = "ignore"


def _parse_reference_entities(value: Any) -> list[ReferenceEntity]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[ReferenceEntity] = []
    for raw in value:
        validator_fn = getattr(ReferenceEntity, "model_validate", None)
        item = validator_fn(raw) if validator_fn else ReferenceEntity.parse_obj(raw)
        result.append(item)
    return result


def _dedupe_display_values(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _legacy_reference_entities(
    positive_titles: list[str],
    search_terms: list[str],
    negative_titles: list[str],
) -> tuple[list[ReferenceEntity], list[ReferenceEntity]]:
    positive: list[ReferenceEntity] = []
    orphans: list[ReferenceEntity] = []
    if len(positive_titles) == 1:
        positive.append(
            ReferenceEntity(
                display_title=positive_titles[0],
                aliases=search_terms,
                polarity="positive",
            )
        )
    else:
        positive.extend(
            ReferenceEntity(display_title=title, polarity="positive")
            for title in positive_titles
        )
        title_keys = {title.casefold() for title in positive_titles}
        orphans.extend(
            ReferenceEntity(display_title=term, polarity="positive")
            for term in search_terms
            if term.casefold() not in title_keys
        )

    negative = [
        ReferenceEntity(display_title=title, polarity="negative")
        for title in negative_titles
    ]
    # Explicit titles retain priority.  Flat aliases for multiple references
    # have no trustworthy ownership information, so callers only warn about
    # them instead of binding them by list position.
    return [*positive, *negative], orphans


def _merge_reference_entity(
    result: list[ReferenceEntity],
    candidate: ReferenceEntity,
) -> None:
    key = (candidate.polarity, candidate.display_title.casefold())
    for position, current in enumerate(result):
        if (current.polarity, current.display_title.casefold()) != key:
            continue
        result[position] = ReferenceEntity(
            display_title=current.display_title,
            aliases=_dedupe_display_values([*current.aliases, *candidate.aliases]),
            polarity=current.polarity,
        )
        return
    result.append(candidate)


class GamePreference(BaseModel):
    platforms: list[str] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)
    genres_like: list[str] = Field(default_factory=list)
    extra_tags: list[str] = Field(default_factory=list)
    derived_intent_tags: list[DerivedIntentTag] = Field(default_factory=list)
    soft_features: list[SoftFeature] = Field(default_factory=list)
    company_preferences: list[CompanyPreference] = Field(default_factory=list)
    explicit_tag_evidence: list[ExplicitTagEvidence] = Field(
        default_factory=list,
        exclude=True,
    )
    genres_dislike: list[str] = Field(default_factory=list)
    reference_entities: list[ReferenceEntity] = Field(default_factory=list)
    reference_games_like: list[str] = Field(default_factory=list)
    reference_search_terms: list[str] = Field(default_factory=list)
    reference_games_dislike: list[str] = Field(default_factory=list)
    library_filter_mode: str | None = None
    resolved_reference_games: list["ResolvedReferenceGame"] = Field(default_factory=list)
    players: int | None = None
    budget: float | None = None
    budget_is_required: bool = False
    region: str | None = None
    budget_currency: str | None = None
    preferred_languages: list[str] = Field(default_factory=list)
    required_languages: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    mood: str | None = None
    quality_intent: str = "normal"
    allow_unreleased: bool = False
    result_count: int = DEFAULT_RECOMMENDATION_COUNT
    parse_warnings: list[str] = Field(default_factory=list)

    @root_validator(pre=True)
    def _group_and_bound_reference_entities(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        structured = _parse_reference_entities(data.get("reference_entities", []))
        positive_titles = split_display_list(data.get("reference_games_like", []))
        search_terms = split_display_list(data.get("reference_search_terms", []))
        negative_titles = split_display_list(data.get("reference_games_dislike", []))
        if structured and (positive_titles or negative_titles):
            active_legacy_keys = {
                ("positive", title.casefold()) for title in positive_titles
            } | {("negative", title.casefold()) for title in negative_titles}
            structured = [
                entity
                for entity in structured
                if (entity.polarity, entity.display_title.casefold())
                in active_legacy_keys
            ]
        explicit_legacy, orphan_terms = _legacy_reference_entities(
            positive_titles,
            search_terms,
            negative_titles,
        )

        grouped: list[ReferenceEntity] = []
        for entity in [*structured, *explicit_legacy]:
            _merge_reference_entity(grouped, entity)

        known_aliases = {
            alias.casefold()
            for entity in grouped
            for alias in [entity.display_title, *entity.aliases]
        }
        unmapped_orphan_terms = [
            entity
            for entity in orphan_terms
            if entity.display_title.casefold() not in known_aliases
        ]

        truncated = len(grouped) > MAX_REFERENCE_ENTITIES
        bounded: list[ReferenceEntity] = []
        for entity in grouped[:MAX_REFERENCE_ENTITIES]:
            aliases = _dedupe_display_values(
                [entity.display_title, *entity.aliases]
            )
            if len(aliases) > MAX_REFERENCE_ALIASES_PER_ENTITY:
                truncated = True
            aliases = aliases[:MAX_REFERENCE_ALIASES_PER_ENTITY]
            bounded.append(
                ReferenceEntity(
                    display_title=entity.display_title,
                    aliases=[
                        alias
                        for alias in aliases
                        if alias.casefold() != entity.display_title.casefold()
                    ],
                    polarity=entity.polarity,
                )
            )

        data["reference_entities"] = bounded
        data["reference_games_like"] = [
            entity.display_title
            for entity in bounded
            if entity.polarity == "positive"
        ]
        data["reference_games_dislike"] = [
            entity.display_title
            for entity in bounded
            if entity.polarity == "negative"
        ]
        data["reference_search_terms"] = _dedupe_display_values(
            [
                alias
                for entity in bounded
                if entity.polarity == "positive"
                for alias in entity.aliases
            ]
        )
        warnings = data.get("parse_warnings", [])
        warnings = (
            list(warnings)
            if isinstance(warnings, (list, tuple, set))
            else [warnings]
        )
        if truncated and REFERENCE_INPUT_LIMIT_WARNING not in warnings:
            warnings.append(REFERENCE_INPUT_LIMIT_WARNING)
        if (
            unmapped_orphan_terms
            and REFERENCE_ALIAS_MAPPING_WARNING not in warnings
        ):
            warnings.append(REFERENCE_ALIAS_MAPPING_WARNING)
        data["parse_warnings"] = warnings
        return data

    @validator("platforms", pre=True)
    def _normalize_platforms(cls, value: Any) -> list[str]:
        values = split_text_list(value)
        return [platform for item in values if (platform := normalize_platform(item))]

    @validator(
        "genres_like",
        "extra_tags",
        "genres_dislike",
        "required_tags",
        pre=True,
    )
    def _normalize_text_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator("parse_warnings", pre=True)
    def _normalize_parse_warnings(cls, value: Any) -> list[str]:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        result: list[str] = []
        seen: set[str] = set()
        for item in values:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            key = text.casefold()
            if text and key not in seen:
                result.append(text)
                seen.add(key)
        return result

    @validator("reference_search_terms", pre=True)
    def _normalize_reference_search_terms(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("reference_games_like", "reference_games_dislike", pre=True)
    def _normalize_reference_titles(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("reference_entities", pre=True)
    def _validated_reference_entities(cls, value: Any) -> list[ReferenceEntity]:
        return _validated_model_list(
            value,
            ReferenceEntity,
            limit=MAX_REFERENCE_ENTITIES,
            identity_key=lambda item: (item.polarity, item.display_title),
        )

    @validator("derived_intent_tags", pre=True)
    def _validated_derived_tags(cls, value: Any) -> list[DerivedIntentTag]:
        return _validated_model_list(value, DerivedIntentTag, limit=3, key="tag")

    @validator("soft_features", pre=True)
    def _validated_soft_features(cls, value: Any) -> list[SoftFeature]:
        return _validated_model_list(value, SoftFeature, limit=3, key="constraint_id")

    @validator("company_preferences", pre=True)
    def _validated_company_preferences(cls, value: Any) -> list[CompanyPreference]:
        from ..services.company_preferences import merge_company_preferences

        return merge_company_preferences(value or [], limit=3)

    @validator("library_filter_mode", pre=True)
    def _normalize_library_filter_mode(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        if text in {"exclude_owned", "排除已有"}:
            return "exclude_owned"
        if text in {"only_owned", "仅查看已有"}:
            return "only_owned"
        return None

    @validator("budget", pre=True)
    def _normalize_budget(cls, value: Any) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, dict):
            value = value.get("amount") or value.get("value") or value.get("max")
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None

    @validator("region", "budget_currency", pre=True)
    def _normalize_region_currency(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", "", str(value or "")).strip().upper()
        return text or None

    @validator("preferred_languages", "required_languages", pre=True)
    def _normalize_languages(cls, value: Any) -> list[str]:
        return split_language_list(value)

    @validator("players", pre=True)
    def _normalize_players(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        if isinstance(value, dict):
            value = value.get("count") or value.get("min") or value.get("value")
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None

    @validator("difficulty", "mood", pre=True)
    def _normalize_optional_text(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        return text or None

    @validator("quality_intent", pre=True, always=True)
    def _normalize_quality_intent(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        return "mainstream" if text == "mainstream" else "normal"

    @validator("result_count", pre=True, always=True)
    def _normalize_result_count(cls, value: Any) -> int:
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = DEFAULT_RECOMMENDATION_COUNT
        return min(max(count, 1), MAX_RECOMMENDATION_COUNT)

    class Config:
        extra = "ignore"


class SteamAccountBinding(BaseModel):
    chat_platform: str = "default"
    chat_user_id: str
    steam_id64: str
    account_kind: str
    display_value: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float | None = None
    updated_at: float | None = None

    @validator(
        "chat_platform",
        "chat_user_id",
        "steam_id64",
        "account_kind",
        "display_value",
        pre=True,
    )
    def _normalize_required_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @validator("chat_platform")
    def _default_platform(cls, value: str) -> str:
        return value or "default"

    class Config:
        extra = "ignore"


class ResolvedReferenceGame(BaseModel):
    raw_text: str
    normalized_title: str
    canonical_title: str
    appid: int | None = None
    store_url: str | None = None
    confidence: float = 0.0
    source: str = "text"
    polarity: str = "like"
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    platforms: list[str] = Field(default_factory=list)
    stores: list[str] = Field(default_factory=list)

    @validator("genres", "tags", "platforms", "stores", pre=True)
    def _normalize_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator("raw_text", "normalized_title", "canonical_title", "store_url", "source", pre=True)
    def _normalize_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @validator("confidence", pre=True)
    def _normalize_confidence(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, 0.0), 1.0)

    @validator("polarity", pre=True)
    def _normalize_polarity(cls, value: Any) -> str:
        return "dislike" if str(value or "").strip().lower() == "dislike" else "like"

    class Config:
        extra = "ignore"


class SteamSearchHit(BaseModel):
    appid: int
    title: str
    store_url: str | None = None
    tag_ids: list[int] = Field(default_factory=list)

    @validator("appid", pre=True)
    def _normalize_appid(cls, value: Any) -> int:
        return int(value)

    @validator("title", "store_url", pre=True)
    def _normalize_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @validator("tag_ids", pre=True)
    def _normalize_tag_ids(cls, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            return []
        result: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            try:
                tag_id = int(item)
            except (TypeError, ValueError):
                continue
            if tag_id > 0:
                result.append(tag_id)
        return result

    class Config:
        extra = "ignore"


class GameCandidate(BaseModel):
    title: str
    appid: int | None = None
    app_type: str | None = None
    platforms: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    genre_ids: list[int] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    category_ids: list[int] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    ordered_tags: list[str] = Field(default_factory=list)
    inferred_tags: list[str] = Field(default_factory=list)
    metacritic: int | None = None
    released: str | None = None
    release_date: str | None = None
    coming_soon: bool = False
    release_status_checked_at: float | None = None
    playtime: int | None = None
    stores: list[str] = Field(default_factory=list)
    raw_url: str | None = None
    review_total: int | None = None
    review_positive_ratio: float | None = None
    review_recent_ratio: float | None = None
    supported_languages: list[str] = Field(default_factory=list)
    language_data_available: bool = False
    internal_source_markers: list[str] = Field(default_factory=list)
    developers: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    developer_data_available: bool = False
    publisher_data_available: bool = False
    company_data_available: bool = False
    short_description: str | None = None
    detailed_description: str | None = None
    description: str | None = None

    @validator(
        "platforms",
        "genres",
        "categories",
        "tags",
        "ordered_tags",
        "inferred_tags",
        "stores",
        pre=True,
    )
    def _normalize_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator("genre_ids", "category_ids", pre=True)
    def _normalize_metadata_ids(cls, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            return []
        result: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            try:
                genre_id = int(item)
            except (TypeError, ValueError):
                continue
            if genre_id > 0 and genre_id not in result:
                result.append(genre_id)
        return result

    @validator("internal_source_markers", pre=True)
    def _normalize_internal_markers(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("developers", "publishers", pre=True)
    def _normalize_companies(cls, value: Any) -> list[str]:
        return split_company_list(value)

    @validator("short_description", "detailed_description", "description", pre=True)
    def _normalize_descriptions(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text or None

    @validator("review_total", pre=True)
    def _normalize_review_total(cls, value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not number.is_integer() or number < 0:
            return None
        return int(number)

    @validator("release_status_checked_at", pre=True)
    def _normalize_release_status_checked_at(cls, value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @validator("review_positive_ratio", "review_recent_ratio", pre=True)
    def _normalize_review_ratio(cls, value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None
        return number

    @validator("supported_languages", pre=True)
    def _normalize_supported_languages(cls, value: Any) -> list[str]:
        return split_language_list(value)

    @validator("title", pre=True)
    def _normalize_title(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @validator("app_type", pre=True)
    def _normalize_app_type(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        return text or None

    class Config:
        extra = "ignore"


class SteamOwnedGame(BaseModel):
    appid: int
    name: str | None = None
    playtime_forever: int = 0

    @validator("appid", "playtime_forever", pre=True)
    def _normalize_int(cls, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @validator("name", pre=True)
    def _normalize_name(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text or None

    class Config:
        extra = "ignore"


class ScoreBreakdown(BaseModel):
    relevance_tier: str = "broad"
    anchor_coverage: float = 0.0
    supporting_similarity: float = 0.0
    negative_reference_similarity: float = 0.0
    semantic_score: float = 0.0
    wilson_lower_bound: float = 0.0
    quality_score: float = 0.0
    layer_score: float = 0.0
    retrieval_rank: int = 0
    tag_coverage: float = 0.0
    positive_reference: float | None = None
    library_profile: float | None = None
    review_reputation: float = 0.0
    popularity: float = 0.0
    positive_score: float = 0.0
    negative_reference_penalty: float = 0.0
    unknown_constraints_penalty: float = 0.0
    language_adjustment: float = 0.0
    budget_adjustment: float = 0.0
    company_adjustment: float = 0.0
    quality_source: str = "none"

    @validator(
        "anchor_coverage",
        "supporting_similarity",
        "negative_reference_similarity",
        "semantic_score",
        "wilson_lower_bound",
        "quality_score",
        "layer_score",
        "tag_coverage",
        "positive_reference",
        "library_profile",
        "review_reputation",
        "popularity",
        pre=True,
    )
    def _normalize_ratio(cls, value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, 0.0), 1.0)

    @validator("relevance_tier", pre=True)
    def _normalize_relevance_tier(cls, value: Any) -> str:
        tier = str(value or "").strip()
        return tier if tier in {"A", "B", "C", "broad"} else "broad"

    @validator("quality_source", pre=True, always=True)
    def _normalize_quality_source(cls, value: Any) -> str:
        source = str(value or "none").strip().lower()
        if source not in {"actual_reviews", "unreleased_prior", "none"}:
            raise ValueError("invalid quality source")
        return source

    @validator("retrieval_rank", pre=True)
    def _normalize_retrieval_rank(cls, value: Any) -> int:
        try:
            rank = int(value)
        except (TypeError, ValueError):
            rank = 0
        return max(rank, 0)

    @validator("positive_score", pre=True)
    def _normalize_positive_score(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, 0.0), 100.0)

    @validator("negative_reference_penalty", pre=True)
    def _normalize_negative_penalty(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, 0.0), 20.0)

    @validator("unknown_constraints_penalty", pre=True)
    def _normalize_unknown_penalty(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, 0.0), 15.0)

    @validator("language_adjustment", pre=True)
    def _normalize_language_adjustment(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, -10.0), 0.0)

    @validator("budget_adjustment", pre=True)
    def _normalize_budget_adjustment(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, -10.0), 5.0)

    @validator("company_adjustment", pre=True)
    def _normalize_company_adjustment(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, -10.0), 0.0)

    class Config:
        extra = "ignore"


class RecommendationEvidence(BaseModel):
    evidence_id: str
    category: str
    sentiment: str
    text: str
    important: bool = False

    @validator("evidence_id", "category", "text", pre=True)
    def _normalize_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @validator("sentiment", pre=True)
    def _normalize_sentiment(cls, value: Any) -> str:
        sentiment = str(value or "").strip().lower()
        return sentiment if sentiment in {"positive", "negative", "uncertain"} else "positive"

    class Config:
        extra = "ignore"


class GamePriceSummary(BaseModel):
    region: str = "CN"
    currency: str | None = None
    current_price: str | None = None
    current_amount: float | None = None
    historic_low: str | None = None
    historic_low_amount: float | None = None
    recent_sale_price: str | None = None
    recent_sale_amount: float | None = None
    sale_time_status: str | None = None

    @validator("region", "currency", pre=True)
    def _normalize_price_codes(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", "", str(value or "")).strip().upper()
        return text or None

    class Config:
        extra = "ignore"


class RankedGame(GameCandidate):
    score: int = 0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    core_feature_verification: str = "not_applicable"
    recommendation_evidence: list[RecommendationEvidence] = Field(default_factory=list)
    recommendation_reason: str = ""
    caution_reason: str | None = None
    price_summary: GamePriceSummary | None = None

    @validator("score", pre=True)
    def _normalize_score(cls, value: Any) -> int:
        try:
            number = round(float(value))
        except (TypeError, ValueError):
            number = 0
        return min(max(number, 0), 100)

    @validator("core_feature_verification", pre=True, always=True)
    def _normalize_core_feature_verification(cls, value: Any) -> str:
        status = str(value or "not_applicable").strip().lower()
        if status not in {
            "not_applicable",
            "verified",
            "unknown",
            "technical_failure",
        }:
            raise ValueError("invalid core feature verification status")
        return status

    @validator("recommendation_reason", pre=True)
    def _normalize_recommendation_reason(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @validator("caution_reason", pre=True)
    def _normalize_caution_reason(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text or None

    @classmethod
    def from_candidate(
        cls,
        candidate: GameCandidate,
        score: int,
        score_breakdown: ScoreBreakdown,
        recommendation_evidence: list[RecommendationEvidence],
    ) -> "RankedGame":
        dumper = getattr(candidate, "model_dump", None)
        data = dumper() if dumper else candidate.dict()
        data["score"] = score
        data["score_breakdown"] = score_breakdown
        data["recommendation_evidence"] = recommendation_evidence
        validator = getattr(cls, "model_validate", None)
        return validator(data) if validator else cls.parse_obj(data)


try:
    GamePreference.model_rebuild()
    GameCandidate.model_rebuild()
    RankedGame.model_rebuild()
except AttributeError:  # pydantic v1
    GamePreference.update_forward_refs(ResolvedReferenceGame=ResolvedReferenceGame)
