from __future__ import annotations

from typing import Protocol

from ..storage.models import GameCandidate, GamePreference, RankedGame
from .game_facts import build_game_facts
from .ranker import (
    game_has_disliked_term,
    game_matches_any_platform,
    has_singleplayer_only_signal,
    has_multiplayer_signal,
)
from .reference_data import ReferenceProfile
from .reference_resolver import (
    ReferenceGameResolver,
    normalize_reference_title,
    reference_profile_for,
    title_similarity,
)
from .search_plan import SearchQuery, build_search_plan
from .tiered_ranker import build_ranked_game, sort_ranked_games

STEAM_FALLBACK_WARNING = (
    "未配置 RAWG API Key，当前使用 Steam 公开数据源，主要覆盖 Steam/PC；"
    "Switch/PlayStation/Xbox 覆盖有限。"
)
STEAM_SOURCE_PLATFORMS = {"steam", "pc"}


class GameSource(Protocol):
    async def search_games(
        self,
        search: str | None = None,
        platforms: list[str] | None = None,
        genres: list[str] | None = None,
        tags: list[str] | None = None,
        page_size: int = 20,
        ordering: str = "-rating",
    ) -> list[GameCandidate]:
        ...


class GameRecommender:
    def __init__(
        self,
        game_source: GameSource,
        max_results: int = 5,
        steam_source: GameSource | None = None,
    ) -> None:
        self.game_source = game_source
        self.steam_source = steam_source
        self.max_results = min(max(max_results, 1), 10)

    async def recommend(
        self,
        preference: GamePreference,
        candidate_pool_size: int | None = None,
    ) -> list[RankedGame]:
        await self._enrich_reference_preferences(preference)
        candidates = await self._recall_candidates(preference)
        filtered = self._filter_candidates(candidates, preference)
        ranked: list[RankedGame] = []
        for candidate in filtered:
            steam_candidate = await self._find_steam_candidate(candidate)
            facts = build_game_facts(candidate, preference, steam_candidate)
            game = build_ranked_game(candidate, preference, facts)
            if game is not None:
                ranked.append(game)
        ranked = sort_ranked_games(ranked)
        limit = candidate_pool_size or preference.result_count or self.max_results
        return ranked[: min(max(limit, 1), 30)]

    async def _enrich_reference_preferences(self, preference: GamePreference) -> None:
        resolved = await ReferenceGameResolver(self.game_source).resolve_reference_games(
            "",
            preference,
        )
        preference.resolved_reference_games = resolved
        for entity in resolved:
            append_unique(preference.reference_games_like, entity.canonical_title)
            profile = reference_profile_for(entity)
            if profile:
                for term in profile.genres_like:
                    append_unique(preference.genres_like, term)
                for term in profile.excluded_tags:
                    append_unique(preference.genres_dislike, term)
                continue
            if entity.confidence >= 0.70:
                for term in [*entity.genres[:4], *entity.tags[:6]]:
                    if term not in {"singleplayer", "single player"}:
                        append_unique(preference.genres_like, term)

    async def _recall_candidates(self, preference: GamePreference) -> list[GameCandidate]:
        candidates: list[GameCandidate] = []
        page_size = max(self.max_results * 4, 20)
        for query in build_search_plan(
            preference,
            preference.resolved_reference_games,
            page_size=page_size,
        ):
            results = await self._execute_search_query(query)
            if query.source == "seed":
                results = annotate_seed_query_results(
                    results,
                    query.search or "",
                    preference,
                )
            candidates.extend(results)
        return dedupe_candidates(candidates)

    async def _execute_search_query(self, query: SearchQuery) -> list[GameCandidate]:
        source = (
            self.steam_source
            if query.source == "steam" and self.steam_source is not None
            else self.game_source
        )
        return await source.search_games(
            search=query.search,
            platforms=query.platforms,
            genres=query.genres,
            tags=query.tags,
            page_size=query.page_size,
            ordering=query.ordering,
        )

    async def _find_steam_candidate(
        self,
        candidate: GameCandidate,
    ) -> GameCandidate | None:
        if self.steam_source is None:
            return None
        if not has_steam_signal(candidate):
            return None
        results = await self.steam_source.search_games(
            search=candidate.title,
            page_size=3,
            ordering="-relevance",
        )
        for result in results:
            if title_similarity(candidate.title, result.title) >= 0.82:
                return result
        return None

    def _filter_candidates(
        self,
        candidates: list[GameCandidate],
        preference: GamePreference,
    ) -> list[GameCandidate]:
        filtered = []
        for candidate in candidates:
            if not candidate.title:
                continue
            if is_reference_game(candidate, preference.reference_games_like):
                continue
            if is_downloadable_content(candidate):
                continue
            if not game_matches_any_platform(candidate, preference.platforms):
                continue
            if game_has_disliked_term(candidate, preference.genres_dislike):
                continue
            if (
                preference.players
                and preference.players >= 2
                and not has_multiplayer_signal(candidate)
                and not has_reference_seed_signal(candidate)
            ):
                continue
            if (
                preference.players
                and preference.players >= 2
                and has_singleplayer_only_signal(candidate)
            ):
                continue
            title = candidate.title.lower()
            if any(reference.lower() in title for reference in preference.reference_games_dislike):
                continue
            filtered.append(candidate)
        return filtered


def adapt_preference_for_steam_source(preference: GamePreference) -> None:
    if STEAM_FALLBACK_WARNING not in preference.parse_warnings:
        preference.parse_warnings.append(STEAM_FALLBACK_WARNING)

    if not preference.platforms:
        return

    steam_platforms = [
        platform for platform in preference.platforms if platform in STEAM_SOURCE_PLATFORMS
    ]
    preference.platforms = steam_platforms or ["steam"]


def dedupe_candidates(candidates: list[GameCandidate]) -> list[GameCandidate]:
    result: list[GameCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if is_downloadable_content(candidate):
            continue
        key = str(candidate.rawg_id or normalized_title_key(candidate.title))
        if key and key not in seen:
            result.append(candidate)
            seen.add(key)
    return result


def fallback_search_query(preference: GamePreference) -> str:
    if preference.players and preference.players >= 2:
        return "co-op multiplayer"
    if preference.genres_like:
        return " ".join(preference.genres_like[:2])
    if preference.platforms:
        return " ".join(preference.platforms)
    if preference.budget is not None:
        return "discounted games"
    if preference.language:
        return "chinese games"
    if preference.difficulty:
        return "casual games"
    if preference.mood:
        return preference.mood
    return "popular games"


def is_reference_game(candidate: GameCandidate, references: list[str]) -> bool:
    title_key = normalized_title_key(candidate.title)
    return any(
        title_key == normalized_title_key(reference)
        for reference in references
        if reference
    )


def is_downloadable_content(candidate: GameCandidate) -> bool:
    haystack = " | ".join(
        [candidate.title, *candidate.genres, *candidate.tags]
    ).lower()
    if any(
        term in haystack
        for term in (
            " dlc",
            "expansion",
            "expansion pack",
            "downloadable content",
            "friend's pass",
            "friends pass",
        )
    ):
        return True
    title = candidate.title.lower()
    return any(
        term in title
        for term in (
            "blood and wine",
            "hearts of stone",
            "episode ",
            "season pass",
            "soundtrack",
            "friend's pass",
            "friends pass",
        )
    )


def normalized_title_key(title: str) -> str:
    lowered = title.lower().replace("–", "-").replace("—", "-")
    lowered = lowered.replace("game of the year", "").replace("complete edition", "")
    lowered = lowered.replace("definitive edition", "").replace("special edition", "")
    return "".join(ch for ch in lowered if ch.isalnum())


def annotate_seed_candidates(
    candidates: list[GameCandidate],
    seed_title: str,
    profile: ReferenceProfile,
    reference_title: str,
) -> list[GameCandidate]:
    result: list[GameCandidate] = []
    seed_key = normalize_reference_title(seed_title)
    for candidate in candidates:
        if title_similarity(seed_title, candidate.title) < 0.70:
            continue
        data = dump_candidate(candidate)
        reasons = list(data.get("source_reasons") or [])
        warnings = list(data.get("source_warnings") or [])
        seed_notes = profile.seed_notes.get(seed_key) or (
            f"参考画像种子：与 {reference_title} 的核心玩法接近",
        )
        seed_warnings = profile.seed_warnings.get(seed_key) or ()
        for note in seed_notes:
            append_unique(reasons, note)
        for warning in seed_warnings:
            append_unique(warnings, warning)
        data["source_reasons"] = reasons
        data["source_warnings"] = warnings
        result.append(validate_candidate(data))
    return result


def annotate_seed_query_results(
    candidates: list[GameCandidate],
    seed_title: str,
    preference: GamePreference,
) -> list[GameCandidate]:
    for entity in preference.resolved_reference_games:
        if entity.confidence < 0.70:
            continue
        profile = reference_profile_for(entity)
        if not profile or seed_title not in profile.seed_titles:
            continue
        return annotate_seed_candidates(
            candidates,
            seed_title,
            profile,
            entity.canonical_title,
        )
    return candidates


def has_reference_seed_signal(candidate: GameCandidate) -> bool:
    return any("参考画像种子" in reason for reason in candidate.source_reasons)


def has_steam_signal(candidate: GameCandidate) -> bool:
    haystack = " | ".join([*candidate.platforms, *candidate.stores]).lower()
    return any(term in haystack for term in ("steam", "pc", "windows"))


def append_unique(values: list[str], text: str) -> None:
    if text and text not in values:
        values.append(text)


def dump_candidate(candidate: GameCandidate) -> dict:
    dumper = getattr(candidate, "model_dump", None)
    return dumper() if dumper else candidate.dict()


def validate_candidate(data: dict) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)
