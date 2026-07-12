from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from typing import Any, Protocol

from ..storage.models import GameFacts, GamePreference, RankedGame

EMBEDDING_TOP_K = 20
EMBEDDING_TIMEOUT_SECONDS = 10.0
EMBEDDING_CACHE_TTL_HOURS = 720


class EmbeddingProviderProtocol(Protocol):
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingCacheProtocol(Protocol):
    async def get_json(self, key: str, ttl_hours: int) -> Any | None: ...

    async def set_json(self, key: str, payload: Any) -> None: ...


class EmbeddingRerankerProtocol(Protocol):
    async def rerank(
        self,
        preference: GamePreference,
        raw_query: str,
        games: list[RankedGame],
    ) -> list[RankedGame]: ...


class EmbeddingReranker:
    def __init__(
        self,
        context: Any,
        cache: EmbeddingCacheProtocol,
        provider_id: str = "",
        timeout_seconds: float = EMBEDDING_TIMEOUT_SECONDS,
    ) -> None:
        self.context = context
        self.cache = cache
        self.provider_id = str(provider_id or "").strip()
        self.timeout_seconds = max(float(timeout_seconds), 0.001)
        self.last_degradation_reason: str | None = None

    async def rerank(
        self,
        preference: GamePreference,
        raw_query: str,
        games: list[RankedGame],
    ) -> list[RankedGame]:
        self.last_degradation_reason = None
        eligible = [
            (index, game)
            for index, game in enumerate(games[:EMBEDDING_TOP_K])
            if game.facts.constraint_status != "violated"
        ]
        query_text = positive_query_text(preference, raw_query)
        if not eligible or not query_text:
            self.last_degradation_reason = "empty_input"
            return list(games)

        try:
            provider = await self._resolve_provider()
        except Exception:
            provider = None
        if provider is None or not hasattr(provider, "get_embeddings"):
            self.last_degradation_reason = "provider_unavailable"
            return list(games)

        identity = provider_identity(provider, self.provider_id)
        candidate_texts = [candidate_embedding_text(game) for _index, game in eligible]
        cache_keys = [
            candidate_cache_key(identity, game, text)
            for (_index, game), text in zip(eligible, candidate_texts, strict=True)
        ]
        cached_vectors = await asyncio.gather(
            *(self._load_cached_vector(key) for key in cache_keys)
        )
        missing_indices = [index for index, vector in enumerate(cached_vectors) if vector is None]
        request_texts = [query_text, *(candidate_texts[index] for index in missing_indices)]

        try:
            response = await asyncio.wait_for(
                provider.get_embeddings(request_texts),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            self.last_degradation_reason = "timeout"
            return list(games)
        except Exception:
            self.last_degradation_reason = "provider_error"
            return list(games)

        if not isinstance(response, list) or len(response) != len(request_texts):
            self.last_degradation_reason = "invalid_vectors"
            return list(games)
        vectors = [normalize_vector(vector) for vector in response]
        if any(vector is None for vector in vectors):
            self.last_degradation_reason = "invalid_vectors"
            return list(games)
        query_vector = vectors[0]
        assert query_vector is not None

        candidate_vectors = list(cached_vectors)
        for request_index, candidate_index in enumerate(missing_indices, start=1):
            vector = vectors[request_index]
            assert vector is not None
            candidate_vectors[candidate_index] = vector
            await self._store_cached_vector(cache_keys[candidate_index], vector)

        normalized_candidates = [normalize_vector(vector) for vector in candidate_vectors]
        if any(vector is None for vector in normalized_candidates):
            self.last_degradation_reason = "invalid_vectors"
            return list(games)
        if any(len(vector or []) != len(query_vector) for vector in normalized_candidates):
            self.last_degradation_reason = "invalid_vectors"
            return list(games)

        scored: dict[int, RankedGame] = {}
        for (game_index, game), vector in zip(
            eligible,
            normalized_candidates,
            strict=True,
        ):
            assert vector is not None
            semantic = cosine_similarity(query_vector, vector)
            base = game.facts.base_relevance_score
            fused = min(max(base * 0.75 + semantic * 0.25, 0.0), 1.0)
            facts = copy_facts(
                game.facts,
                {
                    "embedding_similarity_score": semantic,
                    "reranked_relevance_score": fused,
                },
            )
            scored[game_index] = copy_game(
                game,
                {
                    "score": round(fused * 100, 2),
                    "facts": facts,
                },
            )

        result = list(games)
        tiers = {game.tier for _index, game in eligible}
        for tier in tiers:
            positions = [index for index, game in eligible if game.tier == tier]
            tier_games = sorted(
                (scored[index] for index in positions),
                key=lambda game: (
                    -game.facts.reranked_relevance_score,
                    game.title,
                ),
            )
            for position, game in zip(positions, tier_games, strict=True):
                result[position] = game
        return result

    async def _resolve_provider(self) -> EmbeddingProviderProtocol | None:
        if self.provider_id:
            getter = getattr(self.context, "get_provider_by_id", None)
            if not getter:
                return None
            provider = getter(self.provider_id)
            return await provider if inspect.isawaitable(provider) else provider

        getter = getattr(self.context, "get_all_embedding_providers", None)
        if not getter:
            return None
        providers = getter()
        providers = await providers if inspect.isawaitable(providers) else providers
        return providers[0] if providers else None

    async def _load_cached_vector(self, key: str) -> list[float] | None:
        try:
            payload = await self.cache.get_json(key, EMBEDDING_CACHE_TTL_HOURS)
        except Exception:
            return None
        vector = payload.get("vector") if isinstance(payload, dict) else payload
        return normalize_vector(vector)

    async def _store_cached_vector(self, key: str, vector: list[float]) -> None:
        try:
            await self.cache.set_json(key, {"vector": vector})
        except Exception:
            return


def positive_query_text(preference: GamePreference, raw_query: str) -> str:
    del raw_query
    parts = [
        *preference.required_tags,
        *preference.genres_like,
        *preference.extra_tags,
        *preference.reference_games_like,
        *preference.platforms,
    ]
    if preference.players:
        parts.append(f"{preference.players} players")
    for value in (preference.language, preference.difficulty, preference.mood):
        if value:
            parts.append(value)
    values = dedupe_text(parts)
    return f"推荐意图：{'；'.join(values)}" if values else ""


def candidate_embedding_text(game: RankedGame) -> str:
    parts = [
        f"标题：{game.title}",
        f"有序标签：{'，'.join(game.ordered_tags)}",
        f"类型：{'，'.join([*game.tags, *game.genres])}",
    ]
    if game.description:
        parts.append(f"描述：{game.description[:500]}")
    return "\n".join(parts)


def provider_identity(provider: Any, configured_id: str) -> str:
    provider_id = configured_id or str(
        getattr(provider, "provider_id", None)
        or getattr(provider, "id", None)
        or type(provider).__name__
    )
    model = str(
        getattr(provider, "model_name", None) or getattr(provider, "model", None) or "default"
    )
    return f"{provider_id}:{model}"


def candidate_cache_key(identity: str, game: RankedGame, text: str) -> str:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    item_id = str(game.appid) if game.appid is not None else game.title.lower()
    digest = hashlib.sha256(f"{identity}|{item_id}|{content_hash}".encode("utf-8")).hexdigest()
    return f"embedding_rerank:v1:{digest}"


def normalize_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    if not any(item != 0 for item in vector):
        return None
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return min(max(dot / (left_norm * right_norm), 0.0), 1.0)


def copy_facts(facts: GameFacts, update: dict[str, Any]) -> GameFacts:
    copier = getattr(facts, "model_copy", None)
    return copier(update=update) if copier else facts.copy(update=update)


def copy_game(game: RankedGame, update: dict[str, Any]) -> RankedGame:
    copier = getattr(game, "model_copy", None)
    return copier(update=update) if copier else game.copy(update=update)


def dedupe_text(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result
