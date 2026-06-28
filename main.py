from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.star.filter.command import GreedyStr

from .clients.rawg import RawgApiError, RawgClient, RawgConfigurationError
from .clients.steam import SteamApiError, SteamClient
from .services.formatter import (
    format_game_detail,
    format_recommendation_messages_with_llm,
)
from .services.message_delivery import build_forward_message_chain
from .services.preference_parser import PreferenceParser
from .services.recommender import GameRecommender, adapt_preference_for_steam_source
from .services.steam_price_bridge import SteamPriceBridge
from .storage.repository import SQLiteCacheRepository

PLUGIN_NAME = "astrbot_plugin_game_recommender"
PLUGIN_VERSION = "0.3.1"
PLUGIN_DESCRIPTION = (
    "默认无需 API Key 即可基于 Steam/PC 公开数据推荐游戏；"
    "填写 RAWG API Key 后支持 PlayStation、Xbox、Nintendo Switch 候选召回与筛选。"
)


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
        self.steam_client = SteamClient(
            client=self.http_client,
            cache=self.cache,
            cache_ttl_hours=safe_int(self.config.get("cache_ttl_hours"), 24),
            default_country=str(self.config.get("default_region") or "CN"),
            language="schinese",
        )
        self.game_source = (
            self.rawg_client if self.rawg_client.is_configured() else self.steam_client
        )
        self.preference_parser = PreferenceParser(context, self.provider_id)
        self.recommender = GameRecommender(
            self.game_source,
            max_results=self.max_results,
            steam_source=self.steam_client,
        )
        self.price_bridge = SteamPriceBridge(self.http_client, self.config)
        if self.price_bridge.is_available():
            logger.info(
                "Detected astrbot_plugin_steam_price_heybox; Steam price enrichment enabled."
            )
        else:
            logger.info(
                "astrbot_plugin_steam_price_heybox is not available; "
                "game recommendations continue without price enrichment."
            )

    async def terminate(self) -> None:
        await self.http_client.aclose()
        logger.info("Game recommender plugin stopped.")

    @filter.command(
        "gamerec",
        alias={"游戏推荐"},
        desc="根据自然语言需求推荐游戏。",
    )
    async def recommend_games(self, event: AstrMessageEvent, query: GreedyStr):
        text = str(query).strip()
        if not text:
            yield event.plain_result(
                "请输入需求，例如：/gamerec Switch 和 Steam 双人合作，"
                "不要恐怖，预算 100 以内"
            )
            return

        try:
            preference = await self.preference_parser.parse_preference(event, text)
            if not self.rawg_client.is_configured():
                adapt_preference_for_steam_source(preference)
            candidate_pool_size = (
                max(self.max_results * 3, preference.result_count or self.max_results)
                if preference.budget is not None or self.price_bridge.is_available()
                else None
            )
            ranked_games = await self.recommender.recommend(
                preference,
                candidate_pool_size=candidate_pool_size,
            )
            ranked_games = await self.price_bridge.enrich_ranked_games(ranked_games, preference)
            messages = await format_recommendation_messages_with_llm(
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
        except SteamApiError as exc:
            logger.warning(f"Steam game recommendation failed: {exc}")
            yield event.plain_result(f"Steam 查询失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Game recommendation failed")
            yield event.plain_result(f"游戏推荐失败：{exc}")
            return

        forward_chain = build_forward_message_chain(messages)
        if forward_chain and hasattr(event, "chain_result"):
            yield event.chain_result(forward_chain)
        else:
            yield event.plain_result("\n\n".join(messages))

    @filter.command(
        "gamedesc",
        alias={"游戏详情"},
        desc="查询游戏基础资料和 Steam 价格。",
    )
    async def game_detail(self, event: AstrMessageEvent, query: GreedyStr):
        title = str(query).strip()
        if not title:
            yield event.plain_result("请输入游戏名，例如：/gamedesc It Takes Two")
            return

        try:
            candidates = await self.game_source.search_games(search=title, page_size=1)
            if not candidates:
                yield event.plain_result(f"没有查询到游戏：{title}")
                return
            candidate = candidates[0]
            game = (
                await self.rawg_client.get_game_detail(candidate.rawg_id)
                if self.rawg_client.is_configured() and candidate.rawg_id is not None
                else candidate
            )
            price_summary = await self.price_bridge.lookup(game.title)
        except RawgConfigurationError as exc:
            yield event.plain_result(str(exc))
            return
        except RawgApiError as exc:
            logger.warning(f"RAWG game detail failed: {exc}")
            yield event.plain_result(f"RAWG 查询失败：{exc}")
            return
        except SteamApiError as exc:
            logger.warning(f"Steam game detail failed: {exc}")
            yield event.plain_result(f"Steam 查询失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Game detail lookup failed")
            yield event.plain_result(f"游戏详情查询失败：{exc}")
            return

        yield event.plain_result(format_game_detail(game, price_summary))


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
