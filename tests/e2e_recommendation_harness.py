from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from astrbot_plugin_steam_game_recommender.clients.steam import (
    SteamMoreLikeResult,
    SteamReviewSummary,
    SteamStorefrontPage,
)
from astrbot_plugin_steam_game_recommender.services.preference_parser import (
    PreferenceParser,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import (
    RecallValidationBatch,
    SteamGameIndexService,
    SteamIndexEntry,
)
from astrbot_plugin_steam_game_recommender.services.steam_recall import CandidateHit
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
    validation_appids: tuple[int, ...]

    @property
    def ranking(self) -> list[int]:
        return [int(game.appid) for game in self.ranked if game.appid is not None]

    @property
    def retrieved_appids(self) -> list[int]:
        """Return the exact first-60/optional-second-40 validation input order."""
        return list(self.validation_appids)


class RecordingSteamGameIndexService(SteamGameIndexService):
    """Test-only index service that records the real recall quality boundary."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.validation_appids: list[int] = []

    async def _validate_recall_hits(
        self,
        hits: tuple[CandidateHit, ...],
        records: dict[str, SteamIndexEntry],
        prefetched: dict[int, GameCandidate],
    ) -> RecallValidationBatch:
        self.validation_appids.extend(
            int(hit.candidate.appid)
            for hit in hits
            if hit.candidate.appid is not None
        )
        return await super()._validate_recall_hits(hits, records, prefetched)


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
        self.tag_id_by_name = {
            name.casefold(): tag_id for tag_id, name in self.tag_by_id.items()
        }
        self.detail_calls: list[int] = []
        self.store_tag_calls: list[int] = []
        self.review_calls: list[int] = []
        self.reference_calls: list[tuple[str, str]] = []
        self.storefront_tag_calls: list[int] = []
        self.more_like_calls: list[tuple[int, bool]] = []
        self.storefront_intersection_calls: list[tuple[tuple[int, int], int]] = []
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

    async def search_storefront_tags(
        self,
        tag_ids: list[int] | tuple[int, ...],
        page_size: int = 40,
        start: int = 0,
    ) -> SteamStorefrontPage:
        resolved = tuple(int(tag_id) for tag_id in tag_ids)
        if len(resolved) != 2 or len(set(resolved)) != 2 or any(
            tag_id <= 0 for tag_id in resolved
        ):
            raise ValueError("Steam tag intersection requires two distinct IDs.")
        self.storefront_intersection_calls.append((resolved, int(page_size)))
        response = self.fixture["storefront_intersection_results"].get(
            ",".join(str(tag_id) for tag_id in resolved)
        )
        if response is None:
            hits: list[SteamSearchHit] = []
            total_count = 0
        else:
            hits = [self._fixture_hit(item) for item in response.get("hits", [])]
            total_count = int(response["total_count"])
        resolved_start = max(int(start), 0)
        return SteamStorefrontPage(
            hits=tuple(hits[resolved_start : resolved_start + page_size]),
            total_count=total_count,
            start=resolved_start,
        )

    async def get_more_like(
        self,
        appid: int,
        *,
        allow_unreleased: bool = False,
    ) -> SteamMoreLikeResult:
        resolved = int(appid)
        self.more_like_calls.append((resolved, bool(allow_unreleased)))
        response = self.fixture["more_like_results"].get(str(resolved), {})
        items = list(response.get("released", []))[:20]
        if allow_unreleased:
            items.extend(response.get("upcoming", [])[:20])
        hits: list[SteamSearchHit] = []
        seen: set[int] = {resolved}
        for item in items:
            hit = self._fixture_hit(item)
            if hit.appid in seen:
                continue
            seen.add(hit.appid)
            hits.append(hit)
        return SteamMoreLikeResult(hits=tuple(hits), stale=False)

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
            genres=list(item.get("genres", [])),
            categories=list(item.get("categories", [])),
            stores=["Steam"],
            raw_url=f"https://store.steampowered.com/app/{resolved}/",
            coming_soon=bool(item.get("coming_soon", False)),
            short_description=item.get("short_description"),
            detailed_description=item.get("detailed_description"),
            description=item.get("description"),
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
            tag_ids=tuple(
                tag_id
                for tag in item.get("ordered_tags", [])
                if (tag_id := self.tag_id_by_name.get(str(tag).casefold())) is not None
            ),
        )

    def _fixture_hit(self, item: dict[str, Any]) -> SteamSearchHit:
        appid = int(item["appid"])
        tag_ids = tuple(int(tag_id) for tag_id in item["tag_ids"])
        if not str(item.get("title") or "").strip() or not tag_ids:
            raise AssertionError("frozen storefront hit requires title and ordered tag_ids")
        return SteamSearchHit(
            appid=appid,
            title=str(item["title"]),
            store_url=f"https://store.steampowered.com/app/{appid}/",
            tag_ids=tag_ids,
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
    preference = (
        await parser.parse_preference(object(), str(scenario["query"]))
    ).preference
    client = FrozenSteamClient(fixture)
    index = RecordingSteamGameIndexService(
        client,
        MemoryIndexCache(),
        clock=lambda: 1_700_000_000.0,
    )
    ranked = await index.recommend(preference, limit=limit)
    return E2ERun(
        scenario=scenario,
        preference=preference,
        ranked=tuple(ranked),
        client=client,
        llm_prompts=tuple(llm_context.prompts),
        validation_appids=tuple(index.validation_appids),
    )


def normalized_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()
