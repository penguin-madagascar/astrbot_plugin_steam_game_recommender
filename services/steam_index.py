from __future__ import annotations

from typing import Any, Protocol

from ..storage.models import GameCandidate, GamePreference, RankedGame
from .similarity_ranker import (
    SteamTagProfile,
    build_profile_from_preference,
    rank_steam_candidates,
)

STEAM_INDEX_CACHE_KEY = "steam_index:entries"
STEAM_INDEX_FALLBACK_WARNING = (
    "Steam 索引推荐暂不可用，已降级为实时 Steam/RAWG 查询。"
)
STEAM_INDEX_SCOPE_WARNING = (
    "Steam 索引推荐仅覆盖 Steam/PC；其他平台继续使用 RAWG 数据源。"
)
STEAM_INDEX_PLATFORMS = {"steam", "pc"}


class SteamIndexCache(Protocol):
    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        ...

    async def set_json(self, key: str, payload: Any) -> None:
        ...


class SteamIndexClient(Protocol):
    async def search_games(self, **kwargs: Any) -> list[GameCandidate]:
        ...


class SteamGameIndexService:
    def __init__(
        self,
        steam_client: SteamIndexClient,
        cache: SteamIndexCache,
        ttl_hours: int = 168,
        min_review_count: int = 50,
        min_positive_ratio: float = 0.65,
        page_size: int = 20,
    ) -> None:
        self.steam_client = steam_client
        self.cache = cache
        self.ttl_hours = max(int(ttl_hours), 1)
        self.min_review_count = max(int(min_review_count), 0)
        self.min_positive_ratio = min(max(float(min_positive_ratio), 0.0), 1.0)
        self.page_size = min(max(int(page_size), 1), 40)

    async def recommend(
        self,
        preference: GamePreference,
        limit: int,
    ) -> list[RankedGame]:
        entries = await self.load_entries()
        ranked = rank_entries(entries, preference, self.min_review_count, self.min_positive_ratio)
        if ranked:
            return ranked[:limit]

        refreshed = await self.refresh_entries(preference, entries)
        ranked = rank_entries(refreshed, preference, self.min_review_count, self.min_positive_ratio)
        return ranked[:limit]

    async def load_entries(self) -> list[GameCandidate]:
        payload = await self.cache.get_json(STEAM_INDEX_CACHE_KEY, self.ttl_hours)
        return parse_entries(payload)

    async def refresh_entries(
        self,
        preference: GamePreference,
        existing: list[GameCandidate] | None = None,
    ) -> list[GameCandidate]:
        entries = list(existing or [])
        profile = build_profile_from_preference(
            preference,
            reference_candidates=reference_candidates(preference, entries),
        )
        for query in search_terms_for(preference, profile):
            try:
                results = await self.steam_client.search_games(
                    search=query,
                    page_size=self.page_size,
                    ordering="-relevance",
                )
            except Exception:
                continue
            for candidate in results:
                entries.append(await self.enrich_candidate(candidate))

        entries = dedupe_entries(entries)
        if entries:
            await self.cache.set_json(
                STEAM_INDEX_CACHE_KEY,
                [dump_model(candidate) for candidate in entries],
            )
        return entries

    async def enrich_candidate(self, candidate: GameCandidate) -> GameCandidate:
        data = dump_model(candidate)
        data["index_source"] = "steam_index"
        appid = data.get("appid")
        if appid is not None and hasattr(self.steam_client, "get_review_summary"):
            try:
                summary = await self.steam_client.get_review_summary(int(appid))
            except Exception:
                summary = None
            if summary is not None:
                data["review_total"] = getattr(summary, "total_reviews", None)
                data["review_positive_ratio"] = getattr(summary, "positive_ratio", None)
                data["review_recent_ratio"] = getattr(summary, "recent_positive_ratio", None)
        return validate_candidate(data)


def should_use_steam_index(preference: GamePreference) -> bool:
    return not preference.platforms or all(
        platform in STEAM_INDEX_PLATFORMS for platform in preference.platforms
    )


def rank_entries(
    entries: list[GameCandidate],
    preference: GamePreference,
    min_review_count: int,
    min_positive_ratio: float,
) -> list[RankedGame]:
    profile = build_profile_from_preference(
        preference,
        reference_candidates=reference_candidates(preference, entries),
    )
    return rank_steam_candidates(entries, profile, min_review_count, min_positive_ratio)


def reference_candidates(
    preference: GamePreference,
    entries: list[GameCandidate],
) -> list[GameCandidate]:
    references = {title.lower() for title in preference.reference_games_like if title}
    if not references:
        return []
    return [
        entry for entry in entries
        if entry.title.lower() in references
        or any(reference in entry.title.lower() for reference in references)
    ]


def search_terms_for(preference: GamePreference, profile: SteamTagProfile) -> list[str]:
    terms: list[str] = []
    terms.extend(preference.reference_games_like[:3])
    include = [tag.replace("_", " ") for tag in profile.include_tags[:6]]
    if include:
        terms.append(" ".join(include[:3]))
        terms.extend(include[:4])
    if preference.players and preference.players >= 2:
        terms.extend(["co-op", "local co-op"])
    if not terms:
        terms.append("popular co-op")
    return dedupe_texts(terms)


def parse_entries(payload: Any) -> list[GameCandidate]:
    if not isinstance(payload, list):
        return []
    entries: list[GameCandidate] = []
    for item in payload:
        if isinstance(item, GameCandidate):
            entries.append(item)
        elif isinstance(item, dict):
            entries.append(validate_candidate(item))
    return entries


def dedupe_entries(entries: list[GameCandidate]) -> list[GameCandidate]:
    result: list[GameCandidate] = []
    seen: set[str] = set()
    for entry in entries:
        key = str(entry.appid or entry.title.lower())
        if key and key not in seen:
            result.append(entry)
            seen.add(key)
    return result


def dedupe_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_candidate(data: dict[str, Any]) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)
