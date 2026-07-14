from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, validator

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


class GamePreference(BaseModel):
    platforms: list[str] = Field(default_factory=list)
    required_tags: list[str] = Field(default_factory=list)
    genres_like: list[str] = Field(default_factory=list)
    extra_tags: list[str] = Field(default_factory=list)
    explicit_tag_evidence: list[ExplicitTagEvidence] = Field(
        default_factory=list,
        exclude=True,
    )
    genres_dislike: list[str] = Field(default_factory=list)
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
        "required_tags",
        "parse_warnings",
        pre=True,
    )
    def _normalize_text_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator("reference_search_terms", pre=True)
    def _normalize_reference_search_terms(cls, value: Any) -> list[str]:
        return split_display_list(value)

    @validator("reference_games_like", "reference_games_dislike", pre=True)
    def _normalize_reference_titles(cls, value: Any) -> list[str]:
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
            count = 5
        return min(max(count, 1), 10)

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
    tags: list[str] = Field(default_factory=list)
    ordered_tags: list[str] = Field(default_factory=list)
    inferred_tags: list[str] = Field(default_factory=list)
    metacritic: int | None = None
    released: str | None = None
    release_date: str | None = None
    coming_soon: bool = False
    playtime: int | None = None
    stores: list[str] = Field(default_factory=list)
    raw_url: str | None = None
    review_total: int | None = None
    review_positive_ratio: float | None = None
    review_recent_ratio: float | None = None
    supported_languages: list[str] = Field(default_factory=list)
    language_data_available: bool = False
    internal_source_markers: list[str] = Field(default_factory=list)
    description: str | None = None

    @validator(
        "platforms",
        "genres",
        "tags",
        "ordered_tags",
        "inferred_tags",
        "stores",
        pre=True,
    )
    def _normalize_lists(cls, value: Any) -> list[str]:
        return split_text_list(value)

    @validator("internal_source_markers", pre=True)
    def _normalize_internal_markers(cls, value: Any) -> list[str]:
        return split_display_list(value)

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
    recommendation_evidence: list[RecommendationEvidence] = Field(default_factory=list)
    recommendation_reason: str = ""
    price_summary: GamePriceSummary | None = None

    @validator("score", pre=True)
    def _normalize_score(cls, value: Any) -> int:
        try:
            number = round(float(value))
        except (TypeError, ValueError):
            number = 0
        return min(max(number, 0), 100)

    @validator("recommendation_reason", pre=True)
    def _normalize_reason(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

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
