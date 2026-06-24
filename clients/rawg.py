from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from ..storage.models import GameCandidate
from ..storage.repository import SQLiteCacheRepository

RAWG_BASE_URL = "https://api.rawg.io/api"
RAWG_WEB_BASE_URL = "https://rawg.io/games"

RAWG_PLATFORM_IDS = {
    "pc": [4],
    "steam": [4],
    "nintendo switch": [7],
    "playstation": [187, 18, 16, 15, 27],
    "xbox": [186, 1, 14, 80],
}

RAWG_GENRE_SLUGS = {
    "action": "action",
    "动作": "action",
    "adventure": "adventure",
    "冒险": "adventure",
    "rpg": "role-playing-games-rpg",
    "角色扮演": "role-playing-games-rpg",
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

RAWG_TAG_SLUGS = {
    "co-op": "co-op",
    "coop": "co-op",
    "合作": "co-op",
    "双人": "co-op",
    "multiplayer": "multiplayer",
    "多人": "multiplayer",
    "local co-op": "local-co-op",
    "本地合作": "local-co-op",
    "family": "family-friendly",
    "家庭": "family-friendly",
    "party": "party",
    "聚会": "party",
    "relaxing": "relaxing",
    "轻松": "relaxing",
}


class RawgConfigurationError(RuntimeError):
    pass


class RawgApiError(RuntimeError):
    pass


class RawgClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        cache: SQLiteCacheRepository,
        cache_ttl_hours: int = 24,
    ) -> None:
        self.client = client
        self.api_key = api_key.strip()
        self.cache = cache
        self.cache_ttl_hours = cache_ttl_hours

    async def search_games(
        self,
        search: str | None = None,
        platforms: list[str] | None = None,
        genres: list[str] | None = None,
        tags: list[str] | None = None,
        page_size: int = 20,
        ordering: str = "-rating",
    ) -> list[GameCandidate]:
        params: dict[str, Any] = {
            "page_size": min(max(page_size, 1), 40),
            "ordering": ordering,
        }
        if search:
            params["search"] = search
        platform_ids = map_rawg_platform_ids(platforms or [])
        if platform_ids:
            params["platforms"] = ",".join(str(item) for item in platform_ids)
        genre_slugs = map_slugs(genres or [], RAWG_GENRE_SLUGS)
        if genre_slugs:
            params["genres"] = ",".join(genre_slugs)
        tag_slugs = map_slugs(tags or [], RAWG_TAG_SLUGS)
        if tag_slugs:
            params["tags"] = ",".join(tag_slugs)

        data = await self._get_json("/games", params)
        results = data.get("results") if isinstance(data, dict) else []
        return [parse_rawg_game(item) for item in results or [] if isinstance(item, dict)]

    async def get_game_detail(self, rawg_id: int) -> GameCandidate:
        data = await self._get_json(f"/games/{rawg_id}", {})
        return parse_rawg_game(data if isinstance(data, dict) else {})

    async def _get_json(self, path: str, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RawgConfigurationError(
                "请先在插件配置中填写 rawg_api_key。MVP 需要 RAWG API Key 才能查询游戏事实数据。"
            )
        request_params = dict(params)
        request_params["key"] = self.api_key
        cache_key = self._cache_key(path, params)
        cached = await self.cache.get_json(cache_key, self.cache_ttl_hours)
        if cached is not None:
            return cached

        try:
            response = await self.client.get(f"{RAWG_BASE_URL}{path}", params=request_params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise RawgApiError(f"RAWG 请求失败：{exc}") from exc
        except ValueError as exc:
            raise RawgApiError("RAWG 返回了无法解析的 JSON。") from exc

        await self.cache.set_json(cache_key, data)
        return data

    @staticmethod
    def _cache_key(path: str, params: dict[str, Any]) -> str:
        raw = json.dumps({"path": path, "params": params}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"rawg:{digest}"


def map_rawg_platform_ids(platforms: list[str]) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for platform in platforms:
        for item in RAWG_PLATFORM_IDS.get(platform.lower(), []):
            if item not in seen:
                ids.append(item)
                seen.add(item)
    return ids


def map_slugs(values: list[str], mapping: dict[str, str]) -> list[str]:
    slugs: list[str] = []
    seen: set[str] = set()
    for value in values:
        slug = mapping.get(value.lower())
        if slug and slug not in seen:
            slugs.append(slug)
            seen.add(slug)
    return slugs


def parse_rawg_game(data: dict[str, Any]) -> GameCandidate:
    rawg_id = optional_int(data.get("id"))
    slug = str(data.get("slug") or "").strip()
    return GameCandidate(
        rawg_id=rawg_id,
        title=str(data.get("name") or data.get("name_original") or "").strip(),
        platforms=extract_platforms(data.get("platforms")),
        genres=extract_names(data.get("genres")),
        tags=extract_names(data.get("tags"))[:30],
        rating=optional_float(data.get("rating")),
        metacritic=optional_int(data.get("metacritic")),
        released=optional_text(data.get("released")),
        playtime=optional_int(data.get("playtime")),
        stores=extract_stores(data.get("stores")),
        raw_url=f"{RAWG_WEB_BASE_URL}/{slug}" if slug else None,
        description=optional_text(data.get("description_raw")),
    )


def extract_platforms(value: Any) -> list[str]:
    platforms: list[str] = []
    if not isinstance(value, list):
        return platforms
    for item in value:
        platform = item.get("platform") if isinstance(item, dict) else None
        name = platform.get("name") if isinstance(platform, dict) else None
        if name:
            platforms.append(str(name))
    return unique_texts(platforms)


def extract_stores(value: Any) -> list[str]:
    stores: list[str] = []
    if not isinstance(value, list):
        return stores
    for item in value:
        store = item.get("store") if isinstance(item, dict) else None
        name = store.get("name") if isinstance(store, dict) else None
        if name:
            stores.append(str(name))
    return unique_texts(stores)


def extract_names(value: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(value, list):
        return names
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return unique_texts(names)


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


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None

