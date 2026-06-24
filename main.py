from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.star.filter.command import GreedyStr

from .clients.rawg import RawgApiError, RawgClient, RawgConfigurationError
from .services.formatter import (
    format_game_detail,
    format_recommendations_with_llm,
)
from .services.preference_parser import PreferenceParser
from .services.recommender import GameRecommender
from .storage.repository import SQLiteCacheRepository

PLUGIN_NAME = "astrbot_plugin_game_recommender"
PLUGIN_VERSION = "0.1.0"
PLUGIN_DESCRIPTION = "基于 RAWG 数据和规则排序的自然语言多平台游戏推荐插件。"


@register(
    PLUGIN_NAME,
    "jiangxingda",
    PLUGIN_DESCRIPTION,
    PLUGIN_VERSION,
)
class GameRecommenderPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        timeout = safe_int(self.config.get("timeout_seconds"), 15)
        self.max_results = min(max(safe_int(self.config.get("max_results"), 5), 1), 10)
        self.provider_id = str(self.config.get("llm_provider_id", "") or "").strip()

        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": f"{PLUGIN_NAME}/{PLUGIN_VERSION}",
                "Accept": "application/json",
            },
        )
        data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self.cache = SQLiteCacheRepository(data_dir / "rawg_cache.sqlite3")
        self.rawg_client = RawgClient(
            client=self.http_client,
            api_key=str(self.config.get("rawg_api_key", "") or ""),
            cache=self.cache,
            cache_ttl_hours=safe_int(self.config.get("cache_ttl_hours"), 24),
        )
        self.preference_parser = PreferenceParser(context, self.provider_id)
        self.recommender = GameRecommender(self.rawg_client, max_results=self.max_results)

    async def terminate(self) -> None:
        await self.http_client.aclose()
        logger.info("Game recommender plugin stopped.")

    @filter.command("游戏推荐", desc="根据自然语言需求推荐游戏。")
    async def recommend_games(self, event: AstrMessageEvent, query: GreedyStr):
        text = str(query).strip()
        if not text:
            yield event.plain_result("请输入需求，例如：/游戏推荐 Switch 和 Steam 双人合作，不要恐怖，预算 100 以内")
            return

        try:
            preference = await self.preference_parser.parse_preference(event, text)
            ranked_games = await self.recommender.recommend(preference)
            message = await format_recommendations_with_llm(
                self.context,
                event,
                self.provider_id,
                preference,
                ranked_games,
                limit=self.max_results,
            )
        except RawgConfigurationError as exc:
            yield event.plain_result(str(exc))
            return
        except RawgApiError as exc:
            logger.warning(f"RAWG game recommendation failed: {exc}")
            yield event.plain_result(f"RAWG 查询失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Game recommendation failed")
            yield event.plain_result(f"游戏推荐失败：{exc}")
            return

        yield event.plain_result(message)

    @filter.command("游戏详情", desc="查询 RAWG 游戏基础资料。")
    async def game_detail(self, event: AstrMessageEvent, query: GreedyStr):
        title = str(query).strip()
        if not title:
            yield event.plain_result("请输入游戏名，例如：/游戏详情 It Takes Two")
            return

        try:
            candidates = await self.rawg_client.search_games(search=title, page_size=1)
            if not candidates:
                yield event.plain_result(f"没有在 RAWG 查询到游戏：{title}")
                return
            candidate = candidates[0]
            game = (
                await self.rawg_client.get_game_detail(candidate.rawg_id)
                if candidate.rawg_id is not None
                else candidate
            )
        except RawgConfigurationError as exc:
            yield event.plain_result(str(exc))
            return
        except RawgApiError as exc:
            logger.warning(f"RAWG game detail failed: {exc}")
            yield event.plain_result(f"RAWG 查询失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Game detail lookup failed")
            yield event.plain_result(f"游戏详情查询失败：{exc}")
            return

        yield event.plain_result(format_game_detail(game))


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

