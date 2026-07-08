from __future__ import annotations

from typing import Any, Protocol

from ..storage.models import GameCandidate, GamePreference, RankedGame
from .similarity_ranker import (
    SteamTagProfile,
    build_profile_from_preference,
    rank_steam_candidates,
)
from .diversity import DIVERSITY_STRICT, select_results_by_diversity
from .tag_normalizer import candidate_canonical_tags

STEAM_INDEX_CACHE_KEY = "steam_index:entries"
STEAM_INDEX_FALLBACK_WARNING = (
    "Steam 索引暂不可用，已尝试通过 Steam 公共搜索刷新候选；如果仍为空，请换更明确的标签或参考游戏。"
)
STEAM_ONLY_SCOPE_WARNING = (
    "当前版本仅覆盖 Steam/PC 推荐，暂不支持 Switch、PlayStation、Xbox 等跨平台候选。"
)
STEAM_INDEX_PLATFORMS = {"steam", "pc"}
AAA_SEARCH_TERMS = ["popular", "action adventure", "open world", "story rich", "rpg"]
AAA_INTENT_MARKERS = {"aaa", "3a", "triple-a", "triple a", "大作", "单机大作"}


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
        profile_tag_weights: dict[str, float] | None = None,
        diversity_mode: str = DIVERSITY_STRICT,
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
    ) -> list[RankedGame]:
        if preference.platforms and not has_supported_steam_platform(preference):
            return []
        entries = await self.load_entries()
        ranked = rank_entries(
            entries,
            preference,
            self.min_review_count,
            self.min_positive_ratio,
            profile_tag_weights=profile_tag_weights,
        )
        ranked = exclude_previously_shown(ranked, excluded_appids, excluded_titles)
        if ranked:
            return select_results_by_diversity(ranked, limit, diversity_mode)

        refreshed = await self.refresh_entries(preference, entries)
        ranked = rank_entries(
            refreshed,
            preference,
            self.min_review_count,
            self.min_positive_ratio,
            profile_tag_weights=profile_tag_weights,
        )
        ranked = exclude_previously_shown(ranked, excluded_appids, excluded_titles)
        return select_results_by_diversity(ranked, limit, diversity_mode)

    async def load_entries(self) -> list[GameCandidate]:
        payload = await self.cache.get_json(STEAM_INDEX_CACHE_KEY, self.ttl_hours)
        return parse_entries(payload)

    async def refresh_entries(
        self,
        preference: GamePreference,
        existing: list[GameCandidate] | None = None,
    ) -> list[GameCandidate]:
        entries = list(existing or [])
        searched: set[str] = set()
        profile = build_profile_from_preference(preference)
        entries = await self.search_and_append(
            entries,
            search_terms_for(preference, profile),
            searched,
            reference_terms=reference_terms_for(preference),
        )

        expanded_profile = build_profile_from_preference(
            preference,
            reference_candidates=reference_candidates(preference, entries),
        )
        entries = await self.search_and_append(
            entries,
            search_terms_for(preference, expanded_profile),
            searched,
            reference_terms=reference_terms_for(preference),
        )
        entries = dedupe_entries(entries)
        if entries:
            await self.cache.set_json(
                STEAM_INDEX_CACHE_KEY,
                [dump_model(candidate) for candidate in entries],
            )
        return entries

    async def search_and_append(
        self,
        entries: list[GameCandidate],
        queries: list[str],
        searched: set[str],
        reference_terms: list[str],
    ) -> list[GameCandidate]:
        reference_keys = {normalize_text(title) for title in reference_terms}
        for query in queries:
            key = query.lower()
            if key in searched:
                continue
            searched.add(key)
            try:
                results = await self.steam_client.search_games(
                    search=query,
                    page_size=self.page_size,
                    ordering="-relevance",
                )
            except Exception:
                continue
            is_reference_query = normalize_text(query) in reference_keys
            for index, candidate in enumerate(results):
                enriched = await self.enrich_candidate(candidate)
                if is_reference_query and index == 0:
                    enriched = mark_reference_query(enriched, query)
                entries.append(enriched)
        return entries

    async def enrich_candidate(self, candidate: GameCandidate) -> GameCandidate:
        data = dump_model(candidate)
        data["index_source"] = "steam_index"
        canonical_tags = candidate_canonical_tags(candidate)
        if canonical_tags:
            data["tags"] = dedupe_texts([*(data.get("tags") or []), *canonical_tags])
            reasons = list(data.get("source_reasons") or [])
            if "tag_enrichment:steam_detail" not in reasons:
                reasons.append("tag_enrichment:steam_detail")
            data["source_reasons"] = reasons
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


def steam_only_scope_warning_for(preference: GamePreference) -> str | None:
    if unsupported_platforms(preference):
        return STEAM_ONLY_SCOPE_WARNING
    return None


def has_supported_steam_platform(preference: GamePreference) -> bool:
    return not preference.platforms or any(
        platform in STEAM_INDEX_PLATFORMS for platform in preference.platforms
    )


def unsupported_platforms(preference: GamePreference) -> list[str]:
    return [
        platform
        for platform in preference.platforms
        if platform not in STEAM_INDEX_PLATFORMS
    ]


def rank_entries(
    entries: list[GameCandidate],
    preference: GamePreference,
    min_review_count: int,
    min_positive_ratio: float,
    profile_tag_weights: dict[str, float] | None = None,
) -> list[RankedGame]:
    profile = build_profile_from_preference(
        preference,
        reference_candidates=reference_candidates(preference, entries),
    )
    return rank_steam_candidates(
        entries,
        profile,
        min_review_count,
        min_positive_ratio,
        profile_tag_weights=profile_tag_weights,
    )


def exclude_previously_shown(
    games: list[RankedGame],
    excluded_appids: list[int] | None,
    excluded_titles: list[str] | None,
) -> list[RankedGame]:
    appids = {int(appid) for appid in excluded_appids or []}
    titles = {normalize_text(title) for title in excluded_titles or [] if title}
    if not appids and not titles:
        return games
    return [
        game
        for game in games
        if not (
            (game.appid is not None and int(game.appid) in appids)
            or normalize_text(game.title) in titles
        )
    ]


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
        or any(reason.startswith("reference_query:") for reason in entry.source_reasons)
    ]


def search_terms_for(preference: GamePreference, profile: SteamTagProfile) -> list[str]:
    terms: list[str] = []
    if has_aaa_intent(preference):
        terms.extend(AAA_SEARCH_TERMS)
    terms.extend(preference.reference_games_like[:3])
    terms.extend(preference.reference_search_terms[:3])
    include = [tag.replace("_", " ") for tag in profile.include_tags[:6]]
    if include:
        terms.append(" ".join(include[:3]))
        terms.extend(include[:4])
    if preference.players and preference.players >= 2:
        terms.extend(["co-op", "local co-op"])
    if not terms:
        terms.append("popular co-op")
    return dedupe_texts(terms)


def has_aaa_intent(preference: GamePreference) -> bool:
    values = [*preference.genres_like, *preference.extra_tags]
    normalized = {normalize_text(value) for value in values}
    return bool(normalized & AAA_INTENT_MARKERS)


def reference_terms_for(preference: GamePreference) -> list[str]:
    return dedupe_texts([
        *preference.reference_games_like,
        *preference.reference_search_terms,
    ])


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


def mark_reference_query(candidate: GameCandidate, query: str) -> GameCandidate:
    data = dump_model(candidate)
    reasons = list(data.get("source_reasons") or [])
    marker = f"reference_query:{query}"
    if marker not in reasons:
        reasons.append(marker)
    data["source_reasons"] = reasons
    return validate_candidate(data)


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())
