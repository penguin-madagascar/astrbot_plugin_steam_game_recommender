from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamReviewSummary,
    SteamStorefrontPage,
)
from astrbot_plugin_steam_game_recommender.services.preference_parser import (
    PreferenceParser,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    SteamGameIndexService,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    RankedGame,
    SteamSearchHit,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "e2e_anchor_recommendation_scenarios.json"


@dataclass(frozen=True)
class E2ERun:
    scenario: dict[str, Any]
    preference: GamePreference
    ranked: tuple[RankedGame, ...]
    client: "FrozenSteamClient"
    llm_prompts: tuple[str, ...]

    @property
    def ranking(self) -> list[int]:
        return [int(game.appid) for game in self.ranked if game.appid is not None]


class MemoryIndexCache:
    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}

    async def get_json(self, key: str, _ttl_hours: int) -> Any | None:
        return self.payloads.get(key)

    async def set_json(self, key: str, payload: Any) -> None:
        self.payloads[key] = payload


class FrozenLLMContext:
    """LLM fake that verifies the production parser receives the raw query."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.query = str(scenario["query"])
        self.response = dict(scenario.get("frozen_llm", {}))
        self.prompts: list[str] = []

    async def llm_generate(self, **kwargs: Any) -> SimpleNamespace:
        prompt = str(kwargs.get("prompt") or "")
        if self.query not in prompt:
            raise AssertionError("frozen LLM prompt omitted the original query")
        self.prompts.append(prompt)
        return SimpleNamespace(
            completion_text=json.dumps(self.response, ensure_ascii=False)
        )


class FrozenSteamClient:
    """Steam contract fake backed only by a checked-in storefront snapshot."""

    language = "schinese"

    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.games = {int(item["appid"]): item for item in fixture["games"]}
        self.tag_by_id = {
            int(item["tagid"]): str(item["name"]) for item in fixture["tags"]
        }
        self.detail_calls: list[int] = []
        self.store_tag_calls: list[int] = []
        self.review_calls: list[int] = []
        self.reference_calls: list[tuple[str, str]] = []
        self.storefront_tag_calls: list[int] = []
        self.top_seller_calls = 0

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.fixture["tags"]]

    async def search_game_refs(
        self,
        *,
        search: str,
        page_size: int,
        ordering: str,
        language: str | None = None,
        **_kwargs: Any,
    ) -> list[SteamSearchHit]:
        del ordering
        self.reference_calls.append((search, str(language or self.language)))
        results = self.fixture["reference_search"].get(normalized_query(search), [])
        return [SteamSearchHit(**item) for item in results[:page_size]]

    async def search_storefront_tag(
        self,
        tag_id: int,
        page_size: int = 20,
        start: int = 0,
    ) -> SteamStorefrontPage:
        resolved = int(tag_id)
        if resolved not in self.tag_by_id:
            raise LookupError(f"unknown fixture tag id: {resolved}")
        self.storefront_tag_calls.append(resolved)
        response = self.fixture["storefront_tag_results"].get(str(resolved))
        if response is None:
            appids: list[int] = []
            total_count = 0
        else:
            appids = [int(appid) for appid in response.get("appids", [])]
            total_count = int(response["total_count"])
        hits = [self._hit(appid) for appid in appids]
        if hits:
            hits.append(hits[0])
        return SteamStorefrontPage(
            hits=tuple(hits[start : start + page_size]),
            total_count=total_count,
            start=max(int(start), 0),
        )

    async def browse_top_sellers(
        self,
        page_size: int = 60,
        start: int = 0,
    ) -> SteamStorefrontPage:
        self.top_seller_calls += 1
        appids = [int(appid) for appid in self.fixture["top_seller_appids"]]
        hits = [self._hit(appid) for appid in appids]
        return SteamStorefrontPage(
            hits=tuple(hits[start : start + page_size]),
            total_count=len(appids),
            start=max(int(start), 0),
        )

    async def get_game_detail(self, appid: int) -> GameCandidate:
        resolved = int(appid)
        self.detail_calls.append(resolved)
        item = self.games[resolved]
        return GameCandidate(
            appid=resolved,
            title=item["title"],
            app_type=item.get("app_type", "game"),
            platforms=["PC"],
            stores=["Steam"],
            raw_url=f"https://store.steampowered.com/app/{resolved}/",
            coming_soon=bool(item.get("coming_soon", False)),
        )

    async def get_store_page_tags(self, appid: int) -> list[str]:
        resolved = int(appid)
        self.store_tag_calls.append(resolved)
        return list(self.games[resolved].get("ordered_tags", []))

    async def get_review_summary(self, appid: int) -> SteamReviewSummary:
        resolved = int(appid)
        self.review_calls.append(resolved)
        item = self.games[resolved]
        total = item.get("review_total")
        if total is None:
            raise LookupError("fixture intentionally has no review summary")
        ratio = item.get("review_positive_ratio")
        return SteamReviewSummary(
            total_reviews=int(total),
            positive_ratio=float(ratio) if ratio is not None else None,
            recent_positive_ratio=float(ratio) if ratio is not None else None,
        )

    def _hit(self, appid: int) -> SteamSearchHit:
        item = self.games[int(appid)]
        return SteamSearchHit(
            appid=int(appid),
            title=item["title"],
            store_url=f"https://store.steampowered.com/app/{int(appid)}/",
        )


def load_e2e_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


async def run_e2e_scenario(
    fixture: dict[str, Any],
    scenario: dict[str, Any],
    *,
    limit: int = 50,
) -> E2ERun:
    llm_context = FrozenLLMContext(scenario)
    parser = PreferenceParser(llm_context, provider_id="frozen-e2e")
    preference = await parser.parse_preference(object(), str(scenario["query"]))
    client = FrozenSteamClient(fixture)
    index = SteamGameIndexService(
        client,
        MemoryIndexCache(),
        clock=lambda: 1_700_000_000.0,
    )
    ranked = await index.recommend(preference, limit=limit)
    return E2ERun(
        scenario,
        preference,
        tuple(ranked),
        client,
        tuple(llm_context.prompts),
    )


def normalized_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()
