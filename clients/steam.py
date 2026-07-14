from __future__ import annotations

import hashlib
import html
import json
import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

import httpx

from ..services.tag_normalizer import record_steam_tag_result_count
from ..storage.models import GameCandidate, SteamOwnedGame, SteamSearchHit
from ..storage.repository import SQLiteCacheRepository

STEAM_STORE_SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
STEAM_APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAM_APP_REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
STEAM_OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
STEAM_STORE_BASE_URL = "https://store.steampowered.com/app"
STEAM_POPULAR_TAGS_URL = "https://store.steampowered.com/tagdata/populartags/english"
STEAM_STOREFRONT_SEARCH_URL = "https://store.steampowered.com/search/results"
STOREFRONT_FRESH_TTL_HOURS = 24
STOREFRONT_STALE_TTL_HOURS = 24 * 7

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
    ) -> None:
        self.client = client
        self.cache = cache
        self.cache_ttl_hours = cache_ttl_hours
        self.default_country = default_country.strip().upper() or "CN"
        self.language = language.strip() or "schinese"
        self.steam_api_key = steam_api_key.strip()

    async def search_games(
        self,
        search: str | None = None,
        platforms: list[str] | None = None,
        genres: list[str] | None = None,
        tags: list[str] | None = None,
        page_size: int = 20,
        ordering: str = "-relevance",
        language: str | None = None,
    ) -> list[GameCandidate]:
        hits = await self.search_game_refs(
            search=search,
            platforms=platforms,
            genres=genres,
            tags=tags,
            page_size=page_size,
            ordering=ordering,
            language=language,
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
    ) -> list[SteamSearchHit]:
        del ordering, platforms
        query = build_search_query(search, genres or [], tags or [])
        data = await self._get_json(
            STEAM_STORE_SEARCH_URL,
            {
                "term": query,
                "cc": self.default_country,
                "l": str(language or self.language).strip() or self.language,
            },
        )
        items = data.get("items") if isinstance(data, dict) else []
        hits: list[SteamSearchHit] = []
        result_limit = min(max(page_size, 1), 40)
        for item in items or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "app":
                continue
            appid = optional_int(item.get("id") or item.get("appid"))
            title = str(item.get("name") or "").strip()
            if not appid or not title:
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
    ) -> SteamStorefrontPage:
        resolved_tag_id = int(tag_id)
        if resolved_tag_id <= 0:
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
            }
        )
        record_steam_tag_result_count(resolved_tag_id, page.total_count)
        return page

    async def browse_top_sellers(
        self,
        page_size: int = 60,
        start: int = 0,
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
            }
        )

    async def get_game_detail(self, appid: int) -> GameCandidate:
        payload = await self._get_json(
            STEAM_APP_DETAILS_URL,
            {
                "appids": appid,
                "cc": self.default_country,
                "l": self.language,
            },
        )
        entry = payload.get(str(appid)) if isinstance(payload, dict) else None
        if not isinstance(entry, dict) or not entry.get("success"):
            raise SteamApiError(f"Steam 商店没有返回 appid={appid} 的游戏资料。")
        data = entry.get("data")
        if not isinstance(data, dict):
            raise SteamApiError(f"Steam 商店返回了无效的游戏资料：appid={appid}")
        return parse_steam_game(appid, data)

    async def get_review_summary(self, appid: int) -> SteamReviewSummary:
        data = await self._get_json(
            STEAM_APP_REVIEWS_URL.format(appid=appid),
            {
                "json": 1,
                "language": "all",
                "purchase_type": "all",
                "num_per_page": 0,
            },
        )
        summary = data.get("query_summary") if isinstance(data, dict) else None
        if not isinstance(summary, dict):
            raise SteamApiError(f"Steam 评测摘要返回了无效数据：appid={appid}")
        total = optional_int(summary.get("total_reviews")) or 0
        positive = optional_int(summary.get("total_positive"))
        positive_ratio = positive / total if total > 0 and positive is not None else None
        return SteamReviewSummary(
            total_reviews=total,
            positive_ratio=positive_ratio,
            recent_positive_ratio=positive_ratio,
        )

    async def get_popular_tags(self) -> list[dict[str, Any]]:
        cache_key = self._cache_key(STEAM_POPULAR_TAGS_URL, {})
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        fresh = await self.cache.get_json(fresh_key, STOREFRONT_FRESH_TTL_HOURS)
        if fresh is not None:
            try:
                return parse_popular_tags(fresh)
            except SteamApiError:
                pass

        error: SteamApiError
        try:
            response = await self.client.get(STEAM_POPULAR_TAGS_URL, params={})
            response.raise_for_status()
            tags = parse_popular_tags(response.json())
        except httpx.HTTPError as exc:
            error = SteamApiError(f"Steam 热门标签请求失败：{exc}")
        except ValueError:
            error = SteamApiError("Steam 热门标签返回了无法解析的 JSON。")
        except SteamApiError as exc:
            error = exc
        else:
            await self.cache.set_json(fresh_key, tags)
            await self.cache.set_json(stale_key, tags)
            return tags

        stale = await self.cache.get_json(stale_key, STOREFRONT_STALE_TTL_HOURS)
        if stale is not None:
            try:
                return parse_popular_tags(stale)
            except SteamApiError:
                pass
        raise error

    async def get_store_page_tags(self, appid: int) -> list[str]:
        text = await self._get_text(
            f"{STEAM_STORE_BASE_URL}/{int(appid)}/",
            {"l": "english"},
        )
        return parse_store_page_tags(text)

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
            if not appid:
                continue
            owned_games.append(
                SteamOwnedGame(
                    appid=appid,
                    name=optional_text(item.get("name")),
                    playtime_forever=optional_int(item.get("playtime_forever")) or 0,
                )
            )
        return owned_games

    async def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        cache_key = self._cache_key(url, params)
        cached = await self.cache.get_json(cache_key, self.cache_ttl_hours)
        if cached is not None:
            return cached

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise SteamApiError(f"Steam 请求失败：{exc}") from exc
        except ValueError as exc:
            raise SteamApiError("Steam 返回了无法解析的 JSON。") from exc

        await self.cache.set_json(cache_key, data)
        return data

    async def _get_storefront_page(self, params: dict[str, Any]) -> SteamStorefrontPage:
        cache_key = self._cache_key(STEAM_STOREFRONT_SEARCH_URL, params)
        fresh_key = f"{cache_key}:fresh"
        stale_key = f"{cache_key}:stale"
        fresh = await self.cache.get_json(fresh_key, STOREFRONT_FRESH_TTL_HOURS)
        if fresh is not None:
            try:
                return parse_storefront_page(fresh)
            except SteamApiError:
                pass

        error: SteamApiError
        try:
            response = await self.client.get(STEAM_STOREFRONT_SEARCH_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            page = parse_storefront_page(payload)
        except httpx.HTTPError as exc:
            error = SteamApiError(f"Steam 商店筛选请求失败：{exc}")
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
                return parse_storefront_page(stale)
            except SteamApiError:
                pass
        raise error

    async def _get_text(self, url: str, params: dict[str, Any]) -> str:
        cache_key = self._cache_key(url, params)
        cached = await self.cache.get_json(cache_key, self.cache_ttl_hours)
        if cached is not None:
            return str(cached)

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SteamApiError(f"Steam 请求失败：{exc}") from exc

        text = str(getattr(response, "text", "") or "")
        await self.cache.set_json(cache_key, text)
        return text

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
        tags.append({"tagid": tagid, "name": name.strip()})
    return tags


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


class _StorefrontResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[SteamSearchHit] = []
        self._seen_appids: set[int] = set()
        self._appid: int | None = None
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
                )
            )
            self._seen_appids.add(self._appid)
        self._appid = None
        self._collect_title = False
        self._title_parts = []


def parse_steam_game(appid: int, data: dict[str, Any]) -> GameCandidate:
    genres = description_list(data.get("genres"))
    categories = description_list(data.get("categories"))
    languages = parse_languages(data.get("supported_languages"))
    metacritic = data.get("metacritic") if isinstance(data.get("metacritic"), dict) else {}
    release = data.get("release_date") if isinstance(data.get("release_date"), dict) else {}
    release_date = optional_text(release.get("date"))
    return GameCandidate(
        appid=appid,
        title=str(data.get("name") or f"appid={appid}").strip(),
        app_type=optional_text(data.get("type")),
        platforms=parse_platforms(data.get("platforms")),
        genres=genres,
        tags=categories,
        metacritic=optional_int(metacritic.get("score")),
        released=release_date,
        release_date=release_date,
        stores=["Steam"],
        raw_url=f"{STEAM_STORE_BASE_URL}/{appid}/",
        supported_languages=languages,
        language_data_available=bool(languages),
        internal_source_markers=["steam_appdetails"],
        description=clean_html_text(data.get("short_description") or data.get("about_the_game")),
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
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
