from __future__ import annotations

from ..clients.rawg import RAWG_GENRE_SLUGS, RAWG_TAG_SLUGS, RawgClient
from ..storage.models import GameCandidate, GamePreference, RankedGame
from .ranker import game_has_disliked_term, game_matches_any_platform, score_game


class GameRecommender:
    def __init__(self, rawg_client: RawgClient, max_results: int = 5) -> None:
        self.rawg_client = rawg_client
        self.max_results = min(max(max_results, 1), 10)

    async def recommend(self, preference: GamePreference) -> list[RankedGame]:
        candidates = await self._recall_candidates(preference)
        filtered = self._filter_candidates(candidates, preference)
        ranked: list[RankedGame] = []
        for candidate in filtered:
            score, reasons, warnings = score_game(candidate, preference)
            ranked.append(RankedGame.from_candidate(candidate, score, reasons, warnings))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: preference.result_count or self.max_results]

    async def _recall_candidates(self, preference: GamePreference) -> list[GameCandidate]:
        candidates: list[GameCandidate] = []
        page_size = max(self.max_results * 4, 20)

        for reference in preference.reference_games_like[:3]:
            candidates.extend(
                await self.rawg_client.search_games(
                    search=reference,
                    platforms=preference.platforms,
                    page_size=page_size,
                    ordering="-rating",
                )
            )

        genre_terms = [term for term in preference.genres_like if term in RAWG_GENRE_SLUGS]
        tag_terms = [term for term in preference.genres_like if term in RAWG_TAG_SLUGS]
        if preference.players and preference.players >= 2:
            tag_terms.extend(["co-op", "multiplayer"])
        if genre_terms or tag_terms:
            candidates.extend(
                await self.rawg_client.search_games(
                    platforms=preference.platforms,
                    genres=genre_terms[:3],
                    tags=tag_terms[:4],
                    page_size=page_size,
                    ordering="-rating",
                )
            )

        if not candidates:
            query = " ".join(preference.genres_like[:2]) or preference.mood
            candidates.extend(
                await self.rawg_client.search_games(
                    search=query,
                    platforms=preference.platforms,
                    page_size=page_size,
                    ordering="-rating",
                )
            )

        return dedupe_candidates(candidates)

    def _filter_candidates(
        self,
        candidates: list[GameCandidate],
        preference: GamePreference,
    ) -> list[GameCandidate]:
        filtered = []
        for candidate in candidates:
            if not candidate.title:
                continue
            if not game_matches_any_platform(candidate, preference.platforms):
                continue
            if game_has_disliked_term(candidate, preference.genres_dislike):
                continue
            title = candidate.title.lower()
            if any(reference.lower() in title for reference in preference.reference_games_dislike):
                continue
            filtered.append(candidate)
        return filtered


def dedupe_candidates(candidates: list[GameCandidate]) -> list[GameCandidate]:
    result: list[GameCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.rawg_id or candidate.title.lower())
        if key and key not in seen:
            result.append(candidate)
            seen.add(key)
    return result

