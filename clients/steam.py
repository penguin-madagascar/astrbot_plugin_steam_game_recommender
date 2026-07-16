from __future__ import annotations

import asyncio
import hashlib
import html
import json
import math
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from ..storage.models import GameCandidate, SteamOwnedGame, SteamSearchHit
from ..storage.repository import SQLiteCacheRepository

STEAM_STORE_SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
STEAM_APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAM_APP_REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
STEAM_OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
STEAM_STORE_BASE_URL = "https://store.steampowered.com/app"
STEAM_POPULAR_TAGS_URL = "https://store.steampowered.com/tagdata/populartags/english"
STEAM_STOREFRONT_SEARCH_URL = "https://store.steampowered.com/search/results"
STEAM_MORE_LIKE_URL = "https://store.steampowered.com/recommended/morelike/app/{appid}/"
STOREFRONT_FRESH_TTL_HOURS = 24
STOREFRONT_STALE_TTL_HOURS = 24 * 7
APPDETAILS_RELEASE_FRESH_TTL_HOURS = 1
APPDETAILS_RELEASE_STALE_TTL_HOURS = 6
APPDETAILS_CACHE_VERSION = 1
STORE_PAGE_TAG_CACHE_VERSION = 1
MAX_RETRY_AFTER_SECONDS = 5.0
STEAM_AGE_COOKIES = {
    "birthtime": "0",
    "lastagecheckage": "1-January-1970",
    "wants_mature_content": "1",
}

STEAM_GENRE_TERMS = {
    "action": "action",
    "动作": "action",
    "adventure": "adventure",
    "冒险": "adventure",
    "rpg": "rpg",
    "角色扮演": "rpg",
    "puzzle": "puzzle",
    "解谜": "puzzle",
    "strategy": "strategy",
    "策略": "strategy",
    "casual": "casual",
    "休闲": "casual",
    "simulation": "simulation",
    "模拟": "simulation",
    "racing": "racing",
    "竞速": "racing",
    "sports": "sports",
    "体育": "sports",
    "platformer": "platformer",
    "平台跳跃": "platformer",
    "shooter": "shooter",
    "射击": "shooter",
    "indie": "indie",
    "独立": "indie",
}

STEAM_TAG_TERMS = {
    "co-op": "co-op",
    "coop": "co-op",
    "合作": "co-op",
    "双人": "co-op",
    "multiplayer": "multiplayer",
    "多人": "multiplayer",
    "local co-op": "local co-op",
    "本地合作": "local co-op",
    "family": "family",
    "家庭": "family",
    "party": "party",
    "聚会": "party",
    "relaxing": "relaxing",
    "轻松": "relaxing",
}


class SteamApiError(RuntimeError):
    pass


class SteamTransientError(SteamApiError):
    transient = True


@dataclass(frozen=True)
class SteamReviewSummary:
    total_reviews: int
    positive_ratio: float | None = None
    recent_positive_ratio: float | None = None


@dataclass(frozen=True)
class SteamStorefrontPage:
    hits: tuple[SteamSearchHit, ...]
    total_count: int
    start: int
    stale: bool = False


@dataclass(frozen=True)
class SteamMoreLikeSections:
    released: tuple[SteamSearchHit, ...]
    upcoming: tuple[SteamSearchHit, ...]


@dataclass(frozen=True)
class SteamMoreLikeResult:
    hits: tuple[SteamSearchHit, ...]
    stale: bool = False


@dataclass(frozen=True)
class SteamTagVocabularySnapshot:
    tags: tuple[dict[str, Any], ...]
    fetched_at: float


class SteamClient:
    """Steam Store public API data source for Steam-only recommendations."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        cache: SQLiteCacheRepository,
        cache_ttl_hours: int = 24,
        default_country: str = "CN",
        language: str = "schinese",
        steam_api_key: str = "",
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self.cache = cache
        self.cache_ttl_hours = cache_ttl_hours
        self.default_country = default_country.strip().upper() or "CN"
        self.language = language.strip() or "schinese"
        self.steam_api_key = steam_api_key.strip()
        self.clock = clock
        self._sleeper = sleeper

    async def search_games(
        self,
        search: str | None = None,
        platforms: list[str] | None = None,
        genres: list[str] | None = None,
        tags: list[str] | None = None,
        page_size: int = 20,
        ordering: str = "-relevance",
        language: str | None = None,
        reuse_cache: bool = True,
    ) -> list[GameCandidate]:
        hits = await self.search_game_refs(
            search=search,
            platforms=platforms,
            genres=genres,
            tags=tags,
            page_size=page_size,
            ordering=ordering,
            language=language,
            reuse_cache=reuse_cache,
        )
        games: list[GameCandidate] = []
        for hit in hits:
            try:
                candidate = await self.get_game_detail(hit.appid)
            except SteamApiError:
                continue
            if candidate.app_type == "game":
                games.append(candidate)
        return games

    async def search_game_refs(
        self,
        search: str | None = None,
        platforms: list[str] | None = None,
        genres: list[str] | None = None,
        tags: list[str] | None = None,
        page_size: int = 10,
        ordering: str = "-relevance",
        language: str | None = None,
        reuse_cache: bool = True,
    ) -> list[SteamSearchHit]:
        del ordering, platforms
        query = build_search_query(search, genres or [], tags or [])
        data = await self._get_store_search_payload(
            {
                "term": query,
                "cc": self.default_country,
                "l": str(language or self.language).strip() or self.language,
            },
            reuse_cache=reuse_cache,
        )
        items = data["items"]
        hits: list[SteamSearchHit] = []
        result_limit = min(max(page_size, 1), 40)
        for item in items or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "app":
                continue
            appid = optional_int(item.get("id"))
            if appid is None:
                appid = optional_int(item.get("appid"))
            title = str(item.get("name") or "").strip()
            if appid is None or appid <= 0 or not title:
                continue
            hits.append(
                SteamSearchHit(
                    appid=appid,
                    title=title,
                    store_url=f"{STEAM_STORE_BASE_URL}/{appid}/",
                )
            )
            if len(hits) >= result_limit:
                break
        return hits

    async def search_storefront_tag(
        self,
        tag_id: int,
        page_size: int = 20,
        start: int = 0,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        resolved_tag_id = optional_int(tag_id)
        if resolved_tag_id is None or resolved_tag_id <= 0:
            raise ValueError("Steam tag ID must be positive.")
        page = await self._get_storefront_page(
            {
                "ignore_preferences": 1,
                "tags": resolved_tag_id,
                "ndl": 1,
                "l": "english",
                "cc": self.default_country,
                "start": max(int(start), 0),
                "count": min(max(int(page_size), 1), 60),
                "infinite": 1,
            },
            reuse_cache=reuse_cache,
        )
        return page

    async def search_storefront_tags(
        self,
        tag_ids: list[int] | tuple[int, ...],
        page_size: int = 40,
        start: int = 0,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        resolved = tuple(optional_int(tag_id) for tag_id in tag_ids)
        if len(resolved) != 2 or None in resolved or len(set(resolved)) != 2 or any(
            tag_id is None or tag_id <= 0 for tag_id in resolved
        ):
            raise ValueError("Steam tag intersection requires two distinct IDs.")
        return await self._get_storefront_page(
            {
                "ignore_preferences": 1,
                "tags": ",".join(str(tag_id) for tag_id in resolved),
                "ndl": 1,
                "l": "english",
                "cc": self.default_country,
                "start": max(int(start), 0),
                "count": min(max(int(page_size), 1), 40),
                "infinite": 1,
            },
            reuse_cache=reuse_cache,
        )

    async def search_storefront_term(
        self,
        term: str,
        page_size: int = 20,
        start: int = 0,
        language: str | None = None,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        resolved_term = str(term or "").strip()
        if not resolved_term:
            raise ValueError("Steam storefront term must not be empty.")
        return await self._get_storefront_page(
            {
                "ignore_preferences": 1,
                "term": resolved_term,
                "ndl": 1,
                "l": str(language or self.language).strip() or self.language,
                "cc": self.default_country,
                "start": max(int(start), 0),
                "count": min(max(int(page_size), 1), 60),
                "infinite": 1,
            },
            reuse_cache=reuse_cache,
        )

    async def search_storefront_company(
        self,
        term: str,
        role: str,
        page_size: int = 20,
        start: int = 0,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        resolved_term = str(term or "").strip()
        resolved_role = str(role or "").strip().lower()
        if not resolved_term:
            raise ValueError("Steam storefront company must not be empty.")
        if resolved_role not in {"developer", "publisher"}:
            raise ValueError("Steam storefront company role is invalid.")
        return await self._get_storefront_page(
            {
                "ignore_preferences": 1,
                resolved_role: resolved_term,
                "ndl": 1,
                "l": "english",
                "cc": self.default_country,
                "start": max(int(start), 0),
                "count": min(max(int(page_size), 1), 20),
                "infinite": 1,
            },
            reuse_cache=reuse_cache,
        )

    async def browse_top_sellers(
        self,
        page_size: int = 60,
        start: int = 0,
        reuse_cache: bool = True,
    ) -> SteamStorefrontPage:
        return await self._get_storefront_page(
            {
                "ignore_preferences": 1,
                "filter": "topsellers",
                "ndl": 1,
                "l": "english",
                "cc": self.default_country,
                "start": max(int(start), 0),
                "count": min(max(int(page_size), 1), 60),
                "infinite": 1,
            },
            reuse_cache=reuse_cache,
        )

    async def get_game_detail(
        self,
        appid: int,
        bypass_cache: bool = False,
    ) -> GameCandidate:
        resolved_appid = positive_int(appid, "Steam AppID")
        params = {
            "appids": resolved_appid,
            "cc": self.default_country,
            "l": self.language,
        }
        payload, fetched_at = await self._get_game_detail_payload(
            resolved_appid,
            params,
            bypass_cache=bypass_cache,
        )
        data = validate_appdetails_payload(resolved_appid, payload)
        return parse_steam_game(
            resolved_appid,
            data,
            release_status_checked_at=fetched_at,
        )

    async def get_review_summary(self, appid: int) -> SteamReviewSummary:
        resolved_appid = positive_int(appid, "Steam AppID")
        data = await self._get_json(
            STEAM_APP_REVIEWS_URL.format(appid=resolved_appid),
            {
                "json": 1,
                "language": "all",
                "purchase_type": "all",
                "num_per_page": 0,
            },
        )
        summary = data.get("query_summary") if isinstance(data, dict) else None
        if not isinstance(summary, dict):
            raise SteamApiError(
                f"Steam 评测摘要返回了无效数据：appid={resolved_appid}"
            )
        total = optional_int(summary.get("total_reviews"))
        positive = optional_int(summary.get("total_positive"))
        if (
            total is None
            or positive is None
            or total < 0
            or positive < 0
            or positive > total
        ):
            raise SteamApiError(
                f"Steam 评测摘要包含无效计数：appid={resolved_appid}"
            )
        positive_ratio = positive / total if total > 0 and positive is not None else None
        return SteamReviewSummary(
            total_reviews=total,
            positive_ratio=positive_ratio,
            recent_positive_ratio=positive_ratio,
        )

    async def get_popular_tags(
        self,
        language: str = "english",
    ) -> list[dict[str, Any]]:
        snapshot = await self.get_popular_tags_snapshot(language=language)
        return [dict(tag) for tag in snapshot.tags]

    async def get_popular_tags_snapshot(
        self,
        language: str = "english",
    ) -> SteamTagVocabularySnapshot:
        resolved_language = str(language or "english").strip() or "english"
        url = popular_tags_url(resolved_language)
        cache_key = f"{self._cache_key(url, {})}:v2"
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        fresh = await self.cache.get_json(fresh_key, STOREFRONT_FRESH_TTL_HOURS)
        if fresh is not None:
            try:
                snapshot = parse_popular_tag_snapshot(fresh)
                if self._tag_snapshot_age(snapshot) <= STOREFRONT_FRESH_TTL_HOURS * 3600:
                    return snapshot
            except SteamApiError:
                pass

        error: SteamApiError
        try:
            response = await self._request_get(url, {})
            tags = parse_popular_tags(response.json())
        except ValueError:
            error = SteamApiError("Steam 热门标签返回了无法解析的 JSON。")
        except SteamApiError as exc:
            error = exc
        else:
            snapshot = SteamTagVocabularySnapshot(
                tags=tuple(tags),
                fetched_at=float(self.clock()),
            )
            payload = popular_tag_snapshot_payload(snapshot)
            await self.cache.set_json(fresh_key, payload)
            await self.cache.set_json(stale_key, payload)
            return snapshot

        stale = await self.cache.get_json(stale_key, STOREFRONT_STALE_TTL_HOURS)
        if stale is not None:
            try:
                snapshot = parse_popular_tag_snapshot(stale)
                if self._tag_snapshot_age(snapshot) <= STOREFRONT_STALE_TTL_HOURS * 3600:
                    return snapshot
            except SteamApiError:
                pass
        raise error

    def _tag_snapshot_age(self, snapshot: SteamTagVocabularySnapshot) -> float:
        return max(float(self.clock()) - snapshot.fetched_at, 0.0)

    async def get_more_like(
        self,
        appid: int,
        *,
        allow_unreleased: bool = False,
        reuse_cache: bool = True,
    ) -> SteamMoreLikeResult:
        resolved_appid = optional_int(appid)
        if resolved_appid is None or resolved_appid <= 0:
            raise ValueError("Steam AppID must be positive.")
        url = STEAM_MORE_LIKE_URL.format(appid=resolved_appid)
        cache_key = f"{self._cache_key(url, {'l': 'english'})}:v1"
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        if reuse_cache:
            fresh = await self.cache.get_json(fresh_key, STOREFRONT_FRESH_TTL_HOURS)
            if fresh is not None:
                try:
                    sections, fetched_at = parse_more_like_snapshot(fresh)
                    if self._snapshot_age(fetched_at) <= STOREFRONT_FRESH_TTL_HOURS * 3600:
                        return select_more_like_hits(
                            sections,
                            resolved_appid,
                            allow_unreleased=allow_unreleased,
                        )
                except SteamApiError:
                    pass

        error: SteamApiError
        try:
            response = await self._request_get(
                url,
                {"l": "english"},
                cookies=STEAM_AGE_COOKIES,
            )
            html_text = response.text
            sections = parse_more_like_html(html_text)
        except SteamApiError as exc:
            error = exc
        except (AttributeError, TypeError):
            error = SteamApiError("Steam 相似游戏页面返回了无效内容。")
        else:
            snapshot = {
                "html": html_text,
                "fetched_at": float(self.clock()),
            }
            await self.cache.set_json(fresh_key, snapshot)
            await self.cache.set_json(stale_key, snapshot)
            return select_more_like_hits(
                sections,
                resolved_appid,
                allow_unreleased=allow_unreleased,
            )

        stale = await self.cache.get_json(stale_key, STOREFRONT_STALE_TTL_HOURS)
        if stale is not None:
            try:
                sections, fetched_at = parse_more_like_snapshot(stale)
                if self._snapshot_age(fetched_at) <= STOREFRONT_STALE_TTL_HOURS * 3600:
                    return replace(
                        select_more_like_hits(
                            sections,
                            resolved_appid,
                            allow_unreleased=allow_unreleased,
                        ),
                        stale=True,
                    )
            except SteamApiError:
                pass
        raise error

    def _snapshot_age(self, fetched_at: float) -> float:
        return max(float(self.clock()) - fetched_at, 0.0)

    async def get_store_page_tags(self, appid: int) -> list[str]:
        resolved_appid = optional_int(appid)
        if resolved_appid is None or resolved_appid <= 0:
            raise ValueError("Steam AppID must be positive.")
        cache_key = (
            f"steam:store-page-tags:v{STORE_PAGE_TAG_CACHE_VERSION}:"
            f"{resolved_appid}:english"
        )
        cached = await self.cache.get_json(cache_key, self.cache_ttl_hours)
        if cached is not None:
            return parse_cached_store_page_tags(cached)

        expected_url = f"{STEAM_STORE_BASE_URL}/{resolved_appid}/"
        response = await self._request_get(
            expected_url,
            {"l": "english"},
            cookies=STEAM_AGE_COOKIES,
        )
        tags = validated_store_page_tags(resolved_appid, response)
        await self.cache.set_json(cache_key, tags)
        return tags

    def has_web_api_key(self) -> bool:
        return bool(self.steam_api_key)

    async def get_owned_games(self, steam_id64: str) -> list[SteamOwnedGame]:
        if not self.steam_api_key:
            raise SteamApiError("未配置 steam_api_key，无法查询 Steam 游戏库。")

        data = await self._get_json(
            STEAM_OWNED_GAMES_URL,
            {
                "key": self.steam_api_key,
                "steamid": steam_id64,
                "include_appinfo": 1,
                "include_played_free_games": 1,
                "format": "json",
            },
        )
        response = data.get("response") if isinstance(data, dict) else None
        games = response.get("games") if isinstance(response, dict) else None
        if not isinstance(games, list):
            return []

        owned_games: list[SteamOwnedGame] = []
        for item in games:
            if not isinstance(item, dict):
                continue
            appid = optional_int(item.get("appid"))
            playtime = optional_int(item.get("playtime_forever"))
            if appid is None or appid <= 0 or playtime is None or playtime < 0:
                continue
            owned_games.append(
                SteamOwnedGame(
                    appid=appid,
                    name=optional_text(item.get("name")),
                    playtime_forever=playtime,
                )
            )
        return owned_games

    async def _get_json(
        self,
        url: str,
        params: dict[str, Any],
        validator: Callable[[Any], None] | None = None,
        bypass_cache: bool = False,
    ) -> Any:
        cache_key = self._cache_key(url, params)
        cached = (
            None
            if bypass_cache
            else await self.cache.get_json(cache_key, self.cache_ttl_hours)
        )
        if cached is not None:
            try:
                if validator is not None:
                    validator(cached)
            except SteamApiError:
                pass
            else:
                return cached

        try:
            response = await self._request_get(url, params)
            data = response.json()
        except ValueError as exc:
            raise SteamApiError("Steam 返回了无法解析的 JSON。") from exc

        if validator is not None:
            validator(data)
        await self.cache.set_json(cache_key, data)
        return data

    async def _get_store_search_payload(
        self,
        params: dict[str, Any],
        *,
        reuse_cache: bool,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(STEAM_STORE_SEARCH_URL, params)
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        if reuse_cache:
            fresh = await self.cache.get_json(fresh_key, STOREFRONT_FRESH_TTL_HOURS)
            if fresh is not None:
                try:
                    validate_store_search_payload(fresh)
                except SteamApiError:
                    pass
                else:
                    return fresh

        error: SteamApiError
        try:
            response = await self._request_get(STEAM_STORE_SEARCH_URL, params)
            payload = response.json()
            validate_store_search_payload(payload)
        except ValueError:
            error = SteamApiError("Steam 返回了无法解析的 JSON。")
        except SteamApiError as exc:
            error = exc
        else:
            await self.cache.set_json(fresh_key, payload)
            await self.cache.set_json(stale_key, payload)
            return payload

        stale = await self.cache.get_json(stale_key, STOREFRONT_STALE_TTL_HOURS)
        if stale is not None:
            try:
                validate_store_search_payload(stale)
            except SteamApiError:
                pass
            else:
                return stale
        raise error

    async def _get_game_detail_payload(
        self,
        appid: int,
        params: dict[str, Any],
        *,
        bypass_cache: bool,
    ) -> tuple[dict[str, Any], float]:
        cache_key = (
            f"{self._cache_key(STEAM_APP_DETAILS_URL, params)}:"
            f"appdetails:v{APPDETAILS_CACHE_VERSION}"
        )
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        if not bypass_cache:
            fresh = await self.cache.get_json(
                fresh_key,
                APPDETAILS_RELEASE_FRESH_TTL_HOURS,
            )
            if fresh is not None:
                try:
                    payload, fetched_at = parse_appdetails_snapshot(fresh)
                    validate_appdetails_payload(appid, payload)
                except SteamApiError:
                    pass
                else:
                    return payload, fetched_at

        try:
            response = await self._request_get(STEAM_APP_DETAILS_URL, params)
        except SteamTransientError as error:
            stale = await self.cache.get_json(
                stale_key,
                APPDETAILS_RELEASE_STALE_TTL_HOURS,
            )
            if stale is not None:
                try:
                    payload, fetched_at = parse_appdetails_snapshot(stale)
                    validate_appdetails_payload(appid, payload)
                except SteamApiError:
                    pass
                else:
                    return payload, fetched_at
            raise error

        try:
            payload = response.json()
        except ValueError as exc:
            raise SteamApiError("Steam 返回了无法解析的 JSON。") from exc
        validate_appdetails_payload(appid, payload)
        fetched_at = max(float(self.clock()), 0.0)
        snapshot = appdetails_snapshot_payload(payload, fetched_at)
        await self.cache.set_json(fresh_key, snapshot)
        await self.cache.set_json(stale_key, snapshot)
        return payload, fetched_at

    async def _get_storefront_page(
        self,
        params: dict[str, Any],
        *,
        reuse_cache: bool,
    ) -> SteamStorefrontPage:
        cache_key = self._cache_key(STEAM_STOREFRONT_SEARCH_URL, params)
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        if reuse_cache:
            fresh = await self.cache.get_json(fresh_key, STOREFRONT_FRESH_TTL_HOURS)
            if fresh is not None:
                try:
                    return parse_storefront_page(fresh)
                except SteamApiError:
                    pass

        error: SteamApiError
        try:
            response = await self._request_get(STEAM_STOREFRONT_SEARCH_URL, params)
            payload = response.json()
            page = parse_storefront_page(payload)
        except ValueError as exc:
            error = SteamApiError("Steam 商店筛选返回了无法解析的 JSON。")
        except SteamApiError as exc:
            error = exc
        else:
            await self.cache.set_json(fresh_key, payload)
            await self.cache.set_json(stale_key, payload)
            return page

        stale = await self.cache.get_json(stale_key, STOREFRONT_STALE_TTL_HOURS)
        if stale is not None:
            try:
                return replace(parse_storefront_page(stale), stale=True)
            except SteamApiError:
                pass
        raise error

    async def _request_get(
        self,
        url: str,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        for attempt in range(2):
            try:
                response = await self.client.get(url, params=params, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                retryable = is_retryable_steam_read_error(exc)
                if attempt == 0 and retryable:
                    retry_after = retry_after_seconds(exc)
                    if retry_after is not None:
                        await self._sleeper(retry_after)
                    continue
                error_type = SteamTransientError if retryable else SteamApiError
                raise error_type(f"Steam 请求失败：{exc}") from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _cache_key(url: str, params: dict[str, Any]) -> str:
        raw = json.dumps({"url": url, "params": params}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"steam:{digest}"


def build_search_query(search: str | None, genres: list[str], tags: list[str]) -> str:
    if search and search.strip():
        return search.strip()

    terms = [
        *(STEAM_GENRE_TERMS.get(item.lower(), item) for item in genres[:3]),
        *(STEAM_TAG_TERMS.get(item.lower(), item) for item in tags[:4]),
    ]
    query = " ".join(unique_texts(terms)).strip()
    return query or "popular"


def validate_store_search_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise SteamApiError("Steam 搜索返回了无效数据。")


def popular_tags_url(language: str) -> str:
    resolved = str(language or "english").strip() or "english"
    return f"https://store.steampowered.com/tagdata/populartags/{resolved}"


def is_retryable_steam_read_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {408, 429} or (
        isinstance(status_code, int) and 500 <= status_code <= 599
    )


def retry_after_seconds(exc: httpx.HTTPError) -> float | None:
    response = getattr(exc, "response", None)
    value = str(getattr(response, "headers", {}).get("Retry-After", "")).strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(delay):
        return None
    return min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS)


def parse_storefront_page(payload: Any) -> SteamStorefrontPage:
    if not isinstance(payload, dict) or not payload.get("success"):
        raise SteamApiError("Steam 商店筛选返回了无效状态。")
    results_html = payload.get("results_html")
    if not isinstance(results_html, str):
        raise SteamApiError("Steam 商店筛选缺少结果 HTML。")
    total_count = storefront_non_negative_int(payload.get("total_count"), "total_count")
    start = storefront_non_negative_int(payload.get("start"), "start")
    return SteamStorefrontPage(
        hits=tuple(parse_storefront_results_html(results_html)),
        total_count=total_count,
        start=start,
    )


def parse_popular_tags(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise SteamApiError("Steam 热门标签返回了无效数据。")

    tags: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise SteamApiError("Steam 热门标签返回了无效数据。")
        tagid = item.get("tagid")
        name = item.get("name")
        if type(tagid) is not int or tagid <= 0:
            raise SteamApiError("Steam 热门标签包含无效 tagid。")
        if not isinstance(name, str) or not name.strip():
            raise SteamApiError("Steam 热门标签包含无效名称。")
        tag = {"tagid": tagid, "name": name.strip()}
        raw_count = item.get("total_count", item.get("count"))
        if raw_count is not None:
            tag["count"] = storefront_non_negative_int(raw_count, "count")
        tags.append(tag)
    return tags


def parse_popular_tag_snapshot(payload: Any) -> SteamTagVocabularySnapshot:
    if not isinstance(payload, dict):
        raise SteamApiError("Steam 热门标签缓存缺少快照元数据。")
    fetched_at = payload.get("fetched_at")
    if isinstance(fetched_at, bool) or not isinstance(fetched_at, (int, float)):
        raise SteamApiError("Steam 热门标签缓存时间无效。")
    fetched_at = float(fetched_at)
    if not math.isfinite(fetched_at) or fetched_at < 0:
        raise SteamApiError("Steam 热门标签缓存时间无效。")
    return SteamTagVocabularySnapshot(
        tags=tuple(parse_popular_tags(payload.get("tags"))),
        fetched_at=fetched_at,
    )


def popular_tag_snapshot_payload(
    snapshot: SteamTagVocabularySnapshot,
) -> dict[str, Any]:
    return {
        "fetched_at": snapshot.fetched_at,
        "tags": [dict(tag) for tag in snapshot.tags],
    }


def storefront_non_negative_int(value: Any, field_name: str) -> int:
    invalid = SteamApiError(f"Steam 商店筛选字段 {field_name} 无效。")
    if isinstance(value, bool):
        raise invalid
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise invalid
        number = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        number = int(value)
    else:
        raise invalid
    if number < 0:
        raise invalid
    return number


def parse_storefront_results_html(value: str) -> list[SteamSearchHit]:
    parser = _StorefrontResultParser()
    parser.feed(value)
    parser.close()
    return parser.hits


def parse_more_like_snapshot(
    payload: Any,
) -> tuple[SteamMoreLikeSections, float]:
    if not isinstance(payload, dict) or not isinstance(payload.get("html"), str):
        raise SteamApiError("Steam 相似游戏缓存无效。")
    fetched_at = payload.get("fetched_at")
    if (
        isinstance(fetched_at, bool)
        or not isinstance(fetched_at, (int, float))
        or not math.isfinite(float(fetched_at))
        or float(fetched_at) < 0
    ):
        raise SteamApiError("Steam 相似游戏缓存时间无效。")
    return parse_more_like_html(payload["html"]), float(fetched_at)


def parse_more_like_html(value: str) -> SteamMoreLikeSections:
    if not isinstance(value, str):
        raise SteamApiError("Steam 相似游戏页面返回了无效内容。")
    parser = _MoreLikeParser()
    parser.feed(value)
    parser.close()
    if "released" not in parser.seen_sections:
        raise SteamApiError("Steam 相似游戏页面缺少 released 区段。")
    return SteamMoreLikeSections(
        released=tuple(parser.hits["released"]),
        upcoming=tuple(parser.hits["upcoming"]),
    )


def select_more_like_hits(
    sections: SteamMoreLikeSections,
    reference_appid: int,
    *,
    allow_unreleased: bool,
) -> SteamMoreLikeResult:
    selected = [*sections.released[:20]]
    if allow_unreleased:
        selected.extend(sections.upcoming[:20])
    hits: list[SteamSearchHit] = []
    seen: set[int] = set()
    for hit in selected:
        if hit.appid == reference_appid or hit.appid in seen:
            continue
        hits.append(hit)
        seen.add(hit.appid)
    return SteamMoreLikeResult(hits=tuple(hits))


def parse_storefront_tag_ids(value: Any) -> list[int]:
    if not isinstance(value, str):
        return []
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    result: list[int] = []
    for item in payload:
        if isinstance(item, bool):
            continue
        tag_id = optional_int(item)
        if tag_id is not None and tag_id > 0:
            result.append(tag_id)
    return result


class _StorefrontResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[SteamSearchHit] = []
        self._seen_appids: set[int] = set()
        self._appid: int | None = None
        self._tag_ids: list[int] = []
        self._collect_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a":
            classes = set(str(attributes.get("class") or "").split())
            if "search_result_row" not in classes:
                return
            appid = optional_int(attributes.get("data-ds-appid"))
            self._appid = appid if appid is not None and appid > 0 else None
            self._tag_ids = parse_storefront_tag_ids(
                attributes.get("data-ds-tagids")
            )
            self._collect_title = False
            self._title_parts = []
            return
        if tag == "span" and self._appid is not None:
            classes = set(str(attributes.get("class") or "").split())
            if "title" in classes:
                self._collect_title = True

    def handle_data(self, data: str) -> None:
        if self._collect_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._collect_title:
            self._collect_title = False
            return
        if tag != "a" or self._appid is None:
            return
        title = "".join(self._title_parts).strip()
        if title and self._appid not in self._seen_appids:
            self.hits.append(
                SteamSearchHit(
                    appid=self._appid,
                    title=title,
                    store_url=f"{STEAM_STORE_BASE_URL}/{self._appid}/",
                    tag_ids=self._tag_ids,
                )
            )
            self._seen_appids.add(self._appid)
        self._appid = None
        self._tag_ids = []
        self._collect_title = False
        self._title_parts = []


class _MoreLikeParser(HTMLParser):
    _TITLE_CLASSES = {
        "title",
        "tab_item_name",
        "similar_grid_item_name",
        "similar_grid_item_title",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: dict[str, list[SteamSearchHit]] = {
            "released": [],
            "upcoming": [],
        }
        self.seen_sections: set[str] = set()
        self._stack: list[tuple[str, str | None]] = []
        self._appid: int | None = None
        self._tag_ids: list[int] = []
        self._title_parts: list[str] = []
        self._collect_title = False
        self._anchor_section: str | None = None
        self._seen_appids: dict[str, set[int]] = {
            "released": set(),
            "upcoming": set(),
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        inherited = self._stack[-1][1] if self._stack else None
        detected = section_name(attributes)
        section = inherited or detected
        self._stack.append((tag, section))
        if detected is not None and inherited is None:
            self.seen_sections.add(detected)

        if tag == "a" and section in self.hits:
            appid = optional_int(attributes.get("data-ds-appid"))
            if appid is None or appid <= 0:
                return
            self._appid = appid
            self._tag_ids = parse_storefront_tag_ids(attributes.get("data-ds-tagids"))
            self._title_parts = []
            self._collect_title = False
            self._anchor_section = section
            return

        if self._appid is not None:
            classes = set(str(attributes.get("class") or "").split())
            if classes & self._TITLE_CLASSES:
                self._collect_title = True

    def handle_data(self, data: str) -> None:
        if self._collect_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._collect_title and tag in {"span", "div"}:
            self._collect_title = False
        if tag == "a" and self._appid is not None and self._anchor_section is not None:
            title = " ".join("".join(self._title_parts).split())
            seen = self._seen_appids[self._anchor_section]
            if title and self._appid not in seen:
                self.hits[self._anchor_section].append(
                    SteamSearchHit(
                        appid=self._appid,
                        title=title,
                        store_url=f"{STEAM_STORE_BASE_URL}/{self._appid}/",
                        tag_ids=self._tag_ids,
                    )
                )
                seen.add(self._appid)
            self._appid = None
            self._tag_ids = []
            self._title_parts = []
            self._collect_title = False
            self._anchor_section = None

        matching_index = next(
            (
                index
                for index in range(len(self._stack) - 1, -1, -1)
                if self._stack[index][0] == tag
            ),
            None,
        )
        if matching_index is not None:
            del self._stack[matching_index:]


def section_name(attributes: dict[str, str | None]) -> str | None:
    marker = " ".join(
        str(attributes.get(name) or "").casefold()
        for name in ("id", "class", "data-section")
    )
    tokens = set(re.split(r"[^a-z0-9]+", marker))
    compact = re.sub(r"[^a-z]", "", marker)
    if "upcoming" in tokens or "comingsoon" in compact:
        return "upcoming"
    if "released" in tokens:
        return "released"
    return None


def validate_appdetails_payload(appid: int, payload: Any) -> dict[str, Any]:
    entry = payload.get(str(appid)) if isinstance(payload, dict) else None
    if not isinstance(entry, dict) or not entry.get("success"):
        raise SteamApiError(f"Steam 商店没有返回 appid={appid} 的游戏资料。")
    data = entry.get("data")
    if not isinstance(data, dict):
        raise SteamApiError(f"Steam 商店返回了无效的游戏资料：appid={appid}")
    return data


def appdetails_snapshot_payload(
    payload: dict[str, Any],
    fetched_at: float,
) -> dict[str, Any]:
    return {
        "fetched_at": fetched_at,
        "payload": payload,
    }


def parse_appdetails_snapshot(payload: Any) -> tuple[dict[str, Any], float]:
    if not isinstance(payload, dict) or not isinstance(payload.get("payload"), dict):
        raise SteamApiError("Steam 游戏资料缓存无效。")
    fetched_at = payload.get("fetched_at")
    if (
        isinstance(fetched_at, bool)
        or not isinstance(fetched_at, (int, float))
        or not math.isfinite(float(fetched_at))
        or float(fetched_at) < 0
    ):
        raise SteamApiError("Steam 游戏资料缓存时间无效。")
    return payload["payload"], float(fetched_at)


def parse_steam_game(
    appid: int,
    data: dict[str, Any],
    *,
    release_status_checked_at: float | None = None,
) -> GameCandidate:
    genres = description_list(data.get("genres"))
    genre_ids = id_list(data.get("genres"))
    categories = description_list(data.get("categories"))
    category_ids = id_list(data.get("categories"))
    developer_data_available = isinstance(data.get("developers"), list)
    publisher_data_available = isinstance(data.get("publishers"), list)
    developers = text_list(data.get("developers"))
    publishers = text_list(data.get("publishers"))
    short_description = clean_html_text(data.get("short_description"))
    detailed_description = clean_html_text(
        data.get("detailed_description") or data.get("about_the_game")
    )
    languages = parse_languages(data.get("supported_languages"))
    metacritic = data.get("metacritic") if isinstance(data.get("metacritic"), dict) else {}
    release = data.get("release_date") if isinstance(data.get("release_date"), dict) else {}
    release_date = optional_text(release.get("date"))
    metacritic_score = optional_int(metacritic.get("score"))
    if metacritic_score is not None and not 0 <= metacritic_score <= 100:
        metacritic_score = None
    return GameCandidate(
        appid=appid,
        title=str(data.get("name") or f"appid={appid}").strip(),
        app_type=optional_text(data.get("type")),
        platforms=parse_platforms(data.get("platforms")),
        genres=genres,
        genre_ids=genre_ids,
        categories=categories,
        category_ids=category_ids,
        tags=categories,
        metacritic=metacritic_score,
        released=release_date,
        release_date=release_date,
        coming_soon=release.get("coming_soon") is True,
        release_status_checked_at=release_status_checked_at,
        stores=["Steam"],
        raw_url=f"{STEAM_STORE_BASE_URL}/{appid}/",
        supported_languages=languages,
        language_data_available=bool(languages),
        internal_source_markers=["steam_appdetails"],
        developers=developers,
        publishers=publishers,
        developer_data_available=developer_data_available,
        publisher_data_available=publisher_data_available,
        company_data_available=(developer_data_available or publisher_data_available),
        short_description=short_description or None,
        detailed_description=detailed_description or None,
        description=short_description or detailed_description or None,
    )


def parse_platforms(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["PC"]
    platforms = []
    if value.get("windows"):
        platforms.append("PC")
    if value.get("mac"):
        platforms.append("macOS")
    if value.get("linux"):
        platforms.append("Linux")
    return platforms or ["PC"]


def description_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    descriptions = []
    for item in value:
        if isinstance(item, dict) and item.get("description"):
            descriptions.append(str(item["description"]))
    return unique_texts(descriptions)


def text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return unique_texts([str(item) for item in value if isinstance(item, str)])


def id_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        identifier = optional_int(item.get("id")) if isinstance(item, dict) else None
        if identifier is not None and identifier > 0 and identifier not in result:
            result.append(identifier)
    return result


def parse_languages(value: Any) -> list[str]:
    text = clean_html_text(value)
    if not text:
        return []
    return unique_texts(re.split(r"[,，、/]+", text))


def parse_store_page_tags(text: str) -> list[str]:
    tags = []
    for match in re.finditer(
        r"<a\b[^>]*class=\"[^\"]*\bapp_tag\b[^\"]*\"[^>]*>(.*?)</a>",
        text,
        flags=re.I | re.S,
    ):
        tag = clean_html_text(match.group(1))
        if tag:
            tags.append(tag)
    return unique_texts(tags)


def parse_cached_store_page_tags(payload: Any) -> list[str]:
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise SteamApiError("Steam 商店标签缓存无效。")
    return unique_texts(payload)


def validated_store_page_tags(appid: int, response: Any) -> list[str]:
    expected_path = f"/app/{int(appid)}"
    final_path = urlparse(str(getattr(response, "url", "") or "")).path.rstrip("/")
    if getattr(response, "history", None) or (
        final_path and final_path != expected_path
    ):
        raise SteamApiError(f"Steam 商店页发生了重定向：appid={appid}")

    text = str(getattr(response, "text", "") or "")
    lower = text.casefold()
    age_markers = (
        "agecheck",
        "agegate",
        "please enter your birth date",
        "请输入您的出生日期",
    )
    redirect_markers = (
        "http-equiv=\"refresh\"",
        "http-equiv='refresh'",
        "window.location",
        "location.href",
    )
    if any(marker in lower for marker in (*age_markers, *redirect_markers)):
        raise SteamApiError(f"Steam 商店页不是可用的游戏主页：appid={appid}")

    main_markers = (
        "game_page_background",
        "game_area_description",
        "apphub_appname",
        "popular user-defined tags for this product",
        "glance_tags popular_tags",
    )
    if not any(marker in lower for marker in main_markers):
        raise SteamApiError(f"Steam 商店页缺少游戏主内容：appid={appid}")

    tags = parse_store_page_tags(text)
    descriptor_only = {tag.casefold() for tag in tags} <= {"violent", "gore"}
    if tags and descriptor_only:
        raise SteamApiError(f"Steam 商店页只包含内容描述符：appid={appid}")
    return tags


def clean_html_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def unique_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value).split()).strip()
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else None
    return None


def positive_int(value: Any, field_name: str) -> int:
    number = optional_int(value)
    if number is None or number <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return number


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
