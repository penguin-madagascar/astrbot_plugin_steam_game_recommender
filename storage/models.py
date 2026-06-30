from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, validator


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


class GamePreference(BaseModel):
    platforms: list[str] = Field(default_factory=list)
    genres_like: list[str] = Field(default_factory=list)
    extra_tags: list[str] = Field(default_factory=list)
    genres_dislike: list[str] = Field(default_factory=list)
    reference_games_like: list[str] = Field(default_factory=list)
    reference_search_terms: list[str] = Field(default_factory=list)
    reference_games_dislike: list[str] = Field(default_factory=list)
    library_filter_mode: str | None = None
    resolved_reference_games: list["ResolvedReferenceGame"] = Field(default_factory=list)
    players: int | None = None
    budget: float | None = None
    language: str | None = None
    difficulty: str | None = None
    mood: str | None = None
    result_count: int = 5
    parse_warnings: list[str] = Field(default_factory=list)

    @validator("platforms", pre=True)
    def _normalize_platforms(cls, value: Any) -> list[str]:
        values = split_text_list(value)
        return [platform for item in values if (platform := normalize_platform(item))]

    @validator(
        "genres_like",
        "extra_tags",
        "genres_dislike",
        "reference_games_like",
        "reference_games_dislike",
        "parse_warnings",
        pre=True,
    )
    def _normalize_text_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator("reference_search_terms", pre=True)
    def _normalize_reference_search_terms(cls, value: Any) -> list[str]:
        return split_display_list(value)

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

    @validator("players", pre=True)
    def _normalize_players(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        if isinstance(value, dict):
            value = value.get("count") or value.get("min") or value.get("value")
        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None

    @validator("language", "difficulty", "mood", pre=True)
    def _normalize_optional_text(cls, value: Any) -> str | None:
        text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        return text or None

    @validator("result_count", pre=True, always=True)
    def _normalize_result_count(cls, value: Any) -> int:
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 5
        return min(max(count, 1), 10)

    class Config:
        extra = "ignore"


class AccountBinding(BaseModel):
    chat_platform: str = "default"
    chat_user_id: str
    provider: str
    account_id: str
    account_kind: str
    display_value: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float | None = None
    updated_at: float | None = None

    @validator(
        "chat_platform",
        "chat_user_id",
        "provider",
        "account_id",
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

    class Config:
        extra = "ignore"


class GameCandidate(BaseModel):
    title: str
    appid: int | None = None
    platforms: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metacritic: int | None = None
    released: str | None = None
    release_date: str | None = None
    playtime: int | None = None
    stores: list[str] = Field(default_factory=list)
    raw_url: str | None = None
    review_total: int | None = None
    review_positive_ratio: float | None = None
    review_recent_ratio: float | None = None
    index_source: str | None = None
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_reasons: list[str] = Field(default_factory=list)
    source_warnings: list[str] = Field(default_factory=list)
    tier: str = ""
    fit_points: list[str] = Field(default_factory=list)
    risk_points: list[str] = Field(default_factory=list)
    facts: "GameFacts" = Field(default_factory=lambda: GameFacts())
    description: str | None = None

    @validator("platforms", "genres", "tags", "stores", pre=True)
    def _normalize_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator(
        "reasons",
        "warnings",
        "source_reasons",
        "source_warnings",
        "fit_points",
        "risk_points",
        pre=True,
    )
    def _normalize_display_lists(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("title", pre=True)
    def _normalize_title(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

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


class GameFacts(BaseModel):
    platform_families: list[str] = Field(default_factory=list)
    matched_platforms: list[str] = Field(default_factory=list)
    missing_platforms: list[str] = Field(default_factory=list)
    coop_modes: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    hard_blocks: list[str] = Field(default_factory=list)
    matched_like_terms: list[str] = Field(default_factory=list)
    missing_like_terms: list[str] = Field(default_factory=list)
    required_hits: list[str] = Field(default_factory=list)
    required_misses: list[str] = Field(default_factory=list)
    has_coop: bool = False
    has_local_coop: bool = False
    has_online_coop: bool = False
    has_split_screen: bool = False
    has_remote_play: bool = False
    ordinary_multiplayer: bool = False
    singleplayer_only: bool = False
    horror: bool = False
    chinese: bool = False
    switch2_only: bool = False
    reference_similarity: float = 0.0
    match_coverage: float = 0.0
    match_score: float = 0.0
    confidence: float = 0.0

    @validator(
        "platform_families",
        "matched_platforms",
        "missing_platforms",
        "coop_modes",
        "data_sources",
        "hard_blocks",
        "matched_like_terms",
        "missing_like_terms",
        "required_hits",
        "required_misses",
        pre=True,
    )
    def _normalize_lists(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("reference_similarity", "match_coverage", "match_score", "confidence", pre=True)
    def _normalize_float(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return min(max(number, 0.0), 1.0)

    class Config:
        extra = "ignore"


class GamePriceSummary(BaseModel):
    source: str = "steam_price_heybox"
    appid: int | None = None
    country: str = "CN"
    current_price: str | None = None
    lowest_price: str | None = None
    lowest_date: str | None = None
    lowest_discount: int | None = None
    sale_status: str | None = None
    region_summary: str | None = None
    store_url: str | None = None
    heybox_url: str | None = None
    current_cny: float | None = None
    lowest_cny: float | None = None

    class Config:
        extra = "ignore"


class RankedGame(GameCandidate):
    price_summary: GamePriceSummary | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: GameCandidate,
        score: float,
        reasons: list[str],
        warnings: list[str],
    ) -> "RankedGame":
        dumper = getattr(candidate, "model_dump", None)
        data = dumper() if dumper else candidate.dict()
        data["score"] = round(score, 2)
        data["reasons"] = reasons
        data["warnings"] = warnings
        validator = getattr(cls, "model_validate", None)
        return validator(data) if validator else cls.parse_obj(data)


try:
    GamePreference.model_rebuild()
    GameCandidate.model_rebuild()
    RankedGame.model_rebuild()
except AttributeError:  # pydantic v1
    GamePreference.update_forward_refs(ResolvedReferenceGame=ResolvedReferenceGame)
    GameCandidate.update_forward_refs(GameFacts=GameFacts)
    RankedGame.update_forward_refs(GameFacts=GameFacts)
