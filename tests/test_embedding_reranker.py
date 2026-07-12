from __future__ import annotations

import asyncio
import unittest
from typing import Any, Callable

from astrbot_plugin_game_recommender.services.embedding_reranker import EmbeddingReranker
from astrbot_plugin_game_recommender.storage.models import (
    GameFacts,
    GamePreference,
    RankedGame,
)


class EmbeddingRerankerTest(unittest.IsolatedAsyncioTestCase):
    async def test_batches_top_twenty_and_uses_seventy_five_twenty_five_fusion(self) -> None:
        provider = FakeEmbeddingProvider(vector_for_text)
        reranker = EmbeddingReranker(
            FakeContext([provider]),
            MemoryCache(),
            timeout_seconds=1,
        )
        games = [
            ranked("Lexical Match", base=0.90, tier="strong"),
            ranked("Semantic Match", base=0.80, tier="strong"),
            *[
                ranked(f"Tail {index}", base=0.70 - index * 0.01, tier="recommended")
                for index in range(19)
            ],
        ]

        result = await reranker.rerank(
            GamePreference(genres_like=["co-op", "puzzle"]),
            "合作解谜，不要恐怖",
            games,
        )

        self.assertEqual(result[0].title, "Semantic Match")
        self.assertAlmostEqual(result[0].facts.embedding_similarity_score, 1.0)
        self.assertAlmostEqual(result[0].facts.reranked_relevance_score, 0.85)
        self.assertEqual(result[-1].title, "Tail 18")
        self.assertEqual(len(provider.calls[0]), 21)
        self.assertNotIn("恐怖", provider.calls[0][0])

    async def test_reranking_never_moves_games_across_tiers(self) -> None:
        provider = FakeEmbeddingProvider(vector_for_text)
        reranker = EmbeddingReranker(FakeContext([provider]), MemoryCache())
        games = [
            ranked("Lexical Match", base=0.75, tier="strong"),
            ranked("Semantic Match", base=0.95, tier="recommended"),
        ]

        result = await reranker.rerank(
            GamePreference(genres_like=["puzzle"]),
            "puzzle",
            games,
        )

        self.assertEqual([game.tier for game in result], ["strong", "recommended"])
        self.assertEqual([game.title for game in result], ["Lexical Match", "Semantic Match"])

    async def test_candidate_vectors_are_cached_but_query_vector_is_not(self) -> None:
        provider = FakeEmbeddingProvider(vector_for_text)
        cache = MemoryCache()
        reranker = EmbeddingReranker(
            FakeContext([provider]),
            cache,
            provider_id="fake-provider",
        )
        games = [ranked("Semantic Match", base=0.8, tier="strong")]

        await reranker.rerank(GamePreference(genres_like=["puzzle"]), "puzzle", games)
        await reranker.rerank(GamePreference(genres_like=["puzzle"]), "puzzle", games)

        self.assertEqual([len(call) for call in provider.calls], [2, 1])
        self.assertEqual(len(cache.payloads), 1)
        self.assertTrue(all(ttl == 720 for ttl in cache.read_ttls))

    async def test_timeout_and_invalid_vectors_fall_back_to_original_order(self) -> None:
        games = [
            ranked("First", base=0.8, tier="strong"),
            ranked("Second", base=0.7, tier="strong"),
        ]
        slow = EmbeddingReranker(
            FakeContext([SlowEmbeddingProvider()]),
            MemoryCache(),
            timeout_seconds=0.01,
        )
        invalid = EmbeddingReranker(
            FakeContext([FakeEmbeddingProvider(lambda _text: [float("nan")])]),
            MemoryCache(),
        )

        slow_result = await slow.rerank(GamePreference(genres_like=["puzzle"]), "puzzle", games)
        invalid_result = await invalid.rerank(
            GamePreference(genres_like=["puzzle"]),
            "puzzle",
            games,
        )

        self.assertEqual([game.title for game in slow_result], ["First", "Second"])
        self.assertEqual([game.title for game in invalid_result], ["First", "Second"])
        self.assertEqual(slow.last_degradation_reason, "timeout")
        self.assertEqual(invalid.last_degradation_reason, "invalid_vectors")

    async def test_explicit_provider_id_and_first_available_provider_resolution(self) -> None:
        first = FakeEmbeddingProvider(vector_for_text, provider_id="first")
        selected = FakeEmbeddingProvider(vector_for_text, provider_id="selected")
        context = FakeContext([first, selected])
        games = [ranked("Semantic Match", base=0.8, tier="strong")]

        await EmbeddingReranker(context, MemoryCache()).rerank(
            GamePreference(genres_like=["puzzle"]),
            "puzzle",
            games,
        )
        await EmbeddingReranker(
            context,
            MemoryCache(),
            provider_id="selected",
        ).rerank(
            GamePreference(genres_like=["puzzle"]),
            "puzzle",
            games,
        )

        self.assertTrue(first.calls)
        self.assertTrue(selected.calls)
        self.assertEqual(context.requested_ids, ["selected"])


class FakeEmbeddingProvider:
    def __init__(
        self,
        vector_factory: Callable[[str], list[float]],
        provider_id: str = "fake-provider",
    ) -> None:
        self.vector_factory = vector_factory
        self.provider_id = provider_id
        self.model_name = "fake-model"
        self.calls: list[list[str]] = []

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector_factory(text) for text in texts]


class SlowEmbeddingProvider:
    provider_id = "slow"
    model_name = "slow-model"

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(1)
        return [[1.0, 0.0] for _text in texts]


class FakeContext:
    def __init__(self, providers: list[Any]) -> None:
        self.providers = providers
        self.requested_ids: list[str] = []

    def get_all_embedding_providers(self) -> list[Any]:
        return self.providers

    def get_provider_by_id(self, provider_id: str) -> Any | None:
        self.requested_ids.append(provider_id)
        return next(
            (
                provider
                for provider in self.providers
                if getattr(provider, "provider_id", "") == provider_id
            ),
            None,
        )


class MemoryCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.read_ttls: list[int] = []

    async def get_json(self, key: str, ttl_hours: int) -> Any | None:
        self.read_ttls.append(ttl_hours)
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


def vector_for_text(text: str) -> list[float]:
    if text.startswith("推荐意图") or "Semantic Match" in text:
        return [1.0, 0.0]
    return [0.0, 1.0]


def ranked(title: str, base: float, tier: str) -> RankedGame:
    return RankedGame(
        title=title,
        appid=abs(hash(title)) % 1_000_000,
        tags=["Puzzle", "Co-op"],
        genres=["Adventure"],
        description="A semantic puzzle adventure.",
        tier=tier,
        score=base * 100,
        facts=GameFacts(base_relevance_score=base),
    )


if __name__ == "__main__":
    unittest.main()
