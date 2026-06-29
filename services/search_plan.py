from __future__ import annotations

from dataclasses import dataclass, field

from ..clients.rawg import RAWG_GENRE_SLUGS, RAWG_TAG_SLUGS
from ..storage.models import GamePreference, ResolvedReferenceGame
from .reference_resolver import reference_profile_for


@dataclass(frozen=True)
class SearchQuery:
    search: str | None = None
    platforms: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    page_size: int = 20
    ordering: str = "-rating"
    source: str = "rawg"


def build_search_plan(
    preference: GamePreference,
    resolved_references: list[ResolvedReferenceGame],
    page_size: int,
) -> list[SearchQuery]:
    queries: list[SearchQuery] = []

    for entity in resolved_references:
        profile = reference_profile_for(entity)
        if not profile:
            continue
        for seed in profile.seed_titles:
            queries.append(
                SearchQuery(
                    search=seed,
                    page_size=5,
                    ordering="-relevance",
                    source="seed",
                )
            )

    if preference.players and preference.players >= 2:
        for query in ("co-op", "local co-op", "split screen co-op"):
            queries.append(SearchQuery(search=query, page_size=page_size, source="steam"))

    genre_terms = [term for term in preference.genres_like if term in RAWG_GENRE_SLUGS]
    tag_terms = [term for term in preference.genres_like if term in RAWG_TAG_SLUGS]
    if preference.players and preference.players >= 2:
        tag_terms.extend(["co-op", "multiplayer"])
    if genre_terms or tag_terms or preference.platforms:
        structured_genres = genre_terms[:3]
        structured_tags = dedupe(tag_terms)[:4]
        queries.append(
            SearchQuery(
                platforms=preference.platforms,
                genres=structured_genres,
                tags=structured_tags,
                page_size=page_size,
                ordering="-relevance",
                source="rawg",
            )
        )
        queries.append(
            SearchQuery(
                platforms=preference.platforms,
                genres=structured_genres,
                tags=structured_tags,
                page_size=page_size,
                ordering="-rating",
                source="rawg",
            )
        )

    if not queries and has_explicit_preference(preference):
        queries.append(
            SearchQuery(
                search=fallback_search_query(preference),
                platforms=preference.platforms,
                page_size=page_size,
                source="rawg",
            )
        )
    elif not queries:
        queries.append(SearchQuery(search="popular games", page_size=page_size, source="rawg"))
    elif not any(query.search or query.platforms or query.genres or query.tags for query in queries):
        queries.append(SearchQuery(search=fallback_search_query(preference), page_size=page_size))

    return queries


def has_explicit_preference(preference: GamePreference) -> bool:
    return any(
        (
            preference.platforms,
            preference.genres_like,
            preference.genres_dislike,
            preference.reference_games_like,
            preference.reference_games_dislike,
            preference.players,
            preference.budget,
            preference.language,
            preference.difficulty,
            preference.mood,
        )
    )


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


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result
