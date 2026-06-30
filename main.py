from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.star.filter.command import GreedyStr

from .clients.steam import SteamApiError, SteamClient
from .services.account_binding import (
    AccountBindingError,
    chat_identity_from_event,
    parse_account_binding_command,
)
from .services.formatter import (
    format_game_detail,
    format_recommendation_messages_with_llm,
)
from .services.message_delivery import build_forward_message_chain
from .services.played_filter import (
    LIBRARY_FILTER_EXCLUDE_OWNED,
    LIBRARY_FILTER_ONLY_OWNED,
    LibraryFilterModeError,
    detect_library_filter_mode,
    filter_games_by_library_mode,
    parse_library_filter_command,
    resolve_library_filter_mode,
)
from .services.preference_parser import PreferenceParser
from .services.recommendation_limits import effective_result_limit
from .services.steam_index import (
    STEAM_INDEX_FALLBACK_WARNING,
    SteamGameIndexService,
    has_supported_steam_platform,
    steam_only_scope_warning_for,
)
from .services.steam_price_bridge import SteamPriceBridge
from .services.unplayed_picker import (
    UnplayedRecommendationError,
    format_unplayed_recommendation,
    pick_random_unplayed_game,
)
from .services.user_profile import load_bound_user_tag_weights
from .storage.models import AccountBinding
from .storage.repository import SQLiteCacheRepository

PLUGIN_NAME = "astrbot_plugin_game_recommender"
PLUGIN_VERSION = "0.4.0"
PLUGIN_DESCRIPTION = (
    "基于 Steam/PC 公开数据、本地索引和标签相似度推荐游戏；"
    "当前版本暂不做跨平台候选召回。"
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
        self.cache = SQLiteCacheRepository(data_dir / "steam_cache.sqlite3")
        self.steam_client = SteamClient(
            client=self.http_client,
            cache=self.cache,
            cache_ttl_hours=safe_int(self.config.get("cache_ttl_hours"), 24),
            default_country=str(self.config.get("default_region") or "CN"),
            language="schinese",
            steam_api_key=str(self.config.get("steam_api_key") or ""),
        )
        self.preference_parser = PreferenceParser(context, self.provider_id)
        self.steam_index = SteamGameIndexService(
            steam_client=self.steam_client,
            cache=self.cache,
            ttl_hours=safe_int(self.config.get("steam_index_ttl_hours"), 168),
            min_review_count=safe_int(self.config.get("steam_min_review_count"), 50),
            min_positive_ratio=safe_float(self.config.get("steam_min_positive_ratio"), 0.65),
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
        raw_text = str(query).strip()
        if not raw_text:
            yield event.plain_result(
                "请输入需求，例如：/gamerec Switch 和 Steam 双人合作，"
                "不要恐怖，预算 100 以内（当前仅推荐 Steam/PC 候选）"
            )
            return

        try:
            command_filter = parse_library_filter_command(raw_text)
            text = command_filter.query
            if not text:
                yield event.plain_result(
                    "请输入游戏需求，例如：/gamerec 排除已有 Steam 双人合作解谜"
                )
                return
            text_filter_mode = detect_library_filter_mode(text)
            preference = await self.preference_parser.parse_preference(event, text)
            library_filter_mode = resolve_library_filter_mode(
                command_filter.mode,
                text_filter_mode,
                preference.library_filter_mode,
            )
            preference.library_filter_mode = library_filter_mode
            if warning := steam_only_scope_warning_for(preference):
                preference.parse_warnings.append(warning)
            if not has_supported_steam_platform(preference):
                yield event.plain_result(preference.parse_warnings[-1])
                return
            result_limit = effective_result_limit(self.max_results, preference.result_count)
            candidate_pool_size = None
            if preference.budget is not None or self.price_bridge.is_available():
                candidate_pool_size = max(result_limit * 3, result_limit)
            if library_filter_mode:
                candidate_pool_size = max(
                    candidate_pool_size or result_limit,
                    result_limit * 6,
                    result_limit + 20,
                )
            profile_tag_weights = await self._user_profile_tag_weights(event)
            ranked_games = await self._recommend_with_steam_index(
                preference,
                limit=candidate_pool_size or result_limit,
                profile_tag_weights=profile_tag_weights,
            )
            if library_filter_mode:
                ranked_games = await self._filter_library_games(
                    event,
                    preference,
                    ranked_games,
                    library_filter_mode,
                )
            ranked_games = await self.price_bridge.enrich_ranked_games(ranked_games, preference)
            messages = await format_recommendation_messages_with_llm(
                self.context,
                event,
                self.provider_id,
                preference,
                ranked_games,
                limit=result_limit,
            )
        except SteamApiError as exc:
            logger.warning(f"Steam game recommendation failed: {exc}")
            yield event.plain_result(f"Steam 查询失败：{exc}")
            return
        except LibraryFilterModeError as exc:
            yield event.plain_result(f"游戏库过滤参数错误：{exc}")
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
            candidates = await self.steam_client.search_games(search=title, page_size=1)
            if not candidates:
                yield event.plain_result(f"没有查询到游戏：{title}")
                return
            game = candidates[0]
            price_summary = await self.price_bridge.lookup(game.title)
        except SteamApiError as exc:
            logger.warning(f"Steam game detail failed: {exc}")
            yield event.plain_result(f"Steam 查询失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Game detail lookup failed")
            yield event.plain_result(f"游戏详情查询失败：{exc}")
            return

        yield event.plain_result(format_game_detail(game, price_summary))

    @filter.command(
        "accountbind",
        alias={"账号绑定"},
        desc="绑定当前聊天用户的游戏平台账号。",
    )
    async def account_bind(self, event: AstrMessageEvent, query: GreedyStr):
        text = str(query).strip()
        try:
            chat_platform, chat_user_id = chat_identity_from_event(event)
            if not text:
                bindings = await self.cache.list_account_bindings(chat_platform, chat_user_id)
                if not bindings:
                    yield event.plain_result(
                        "还没有绑定账号。请使用 /accountbind steam <SteamID64 或好友码>。"
                    )
                    return
                lines = ["当前绑定账号："]
                for binding in bindings:
                    lines.append(
                        f"- {binding.provider}: {binding.account_id}（{binding.account_kind}）"
                    )
                yield event.plain_result("\n".join(lines))
                return

            parsed = parse_account_binding_command(text)
            saved = await self.cache.upsert_account_binding(
                AccountBinding(
                    chat_platform=chat_platform,
                    chat_user_id=chat_user_id,
                    provider=parsed.provider,
                    account_id=parsed.account_id,
                    account_kind=parsed.account_kind,
                    display_value=parsed.display_value,
                    metadata=parsed.metadata,
                )
            )
        except AccountBindingError as exc:
            yield event.plain_result(f"账号绑定失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Account binding failed")
            yield event.plain_result(f"账号绑定失败：{exc}")
            return

        yield event.plain_result(
            f"账号绑定成功：Steam ID {saved.account_id}（来源：{saved.account_kind}）。"
        )

    @filter.command(
        "unplayedrec",
        alias={"未玩推荐"},
        desc="从已绑定 Steam 库中随机推荐一款未玩且评价过线的游戏。",
    )
    async def recommend_unplayed_game(self, event: AstrMessageEvent):
        try:
            chat_platform, chat_user_id = chat_identity_from_event(event)
            binding = await self.cache.get_account_binding(chat_platform, chat_user_id, "steam")
            if binding is None:
                yield event.plain_result(
                    "当前用户未绑定 Steam 账号；请先使用 /accountbind steam <SteamID64 或好友码>。"
                )
                return
            if not self.steam_client.has_web_api_key():
                yield event.plain_result("未配置 steam_api_key，无法读取 Steam 游戏库。")
                return

            owned_games = await self.steam_client.get_owned_games(binding.account_id)
            if not owned_games:
                yield event.plain_result("Steam 游戏库为空或不可见，无法推荐未玩游戏。")
                return

            min_review_count = safe_int(self.config.get("steam_min_review_count"), 50)
            min_positive_ratio = safe_float(self.config.get("steam_min_positive_ratio"), 0.65)
            recommendation = await pick_random_unplayed_game(
                owned_games,
                self.steam_client,
                min_review_count=min_review_count,
                min_positive_ratio=min_positive_ratio,
            )
        except AccountBindingError as exc:
            yield event.plain_result(f"未玩游戏推荐失败：{exc}")
            return
        except UnplayedRecommendationError as exc:
            yield event.plain_result(f"未玩游戏推荐失败：{exc}")
            return
        except SteamApiError as exc:
            logger.warning(f"Steam unplayed game recommendation failed: {exc}")
            yield event.plain_result(f"Steam 查询失败：{exc}")
            return
        except Exception as exc:
            logger.exception("Unplayed game recommendation failed")
            yield event.plain_result(f"未玩游戏推荐失败：{exc}")
            return

        yield event.plain_result(
            format_unplayed_recommendation(
                recommendation,
                min_review_count=min_review_count,
                min_positive_ratio=min_positive_ratio,
            )
        )

    async def _recommend_with_steam_index(
        self,
        preference,
        limit: int,
        profile_tag_weights: dict[str, float] | None = None,
    ):
        ranked_games = await self.steam_index.recommend(
            preference,
            limit=limit,
            profile_tag_weights=profile_tag_weights,
        )
        if ranked_games:
            return ranked_games
        if STEAM_INDEX_FALLBACK_WARNING not in preference.parse_warnings:
            preference.parse_warnings.append(STEAM_INDEX_FALLBACK_WARNING)
        return []

    async def _user_profile_tag_weights(self, event: AstrMessageEvent) -> dict[str, float]:
        try:
            chat_platform, chat_user_id = chat_identity_from_event(event)
            entries = await self.steam_index.load_entries()
            if not entries:
                return {}
            return await load_bound_user_tag_weights(
                chat_platform,
                chat_user_id,
                self.cache,
                self.steam_client,
                entries,
            )
        except Exception as exc:
            logger.debug(f"Steam user profile weights skipped: {exc}")
            return {}

    async def _filter_library_games(
        self,
        event: AstrMessageEvent,
        preference,
        ranked_games,
        mode: str,
    ):
        try:
            chat_platform, chat_user_id = chat_identity_from_event(event)
        except AccountBindingError as exc:
            raise LibraryFilterModeError(f"无法识别当前用户，不能执行游戏库过滤：{exc}") from exc

        binding = await self.cache.get_account_binding(chat_platform, chat_user_id, "steam")
        if binding is None:
            raise LibraryFilterModeError(
                "当前用户未绑定 Steam 账号；请先使用 /accountbind steam <SteamID64 或好友码>。"
            )

        if not self.steam_client.has_web_api_key():
            raise LibraryFilterModeError(
                "未配置 steam_api_key，无法读取 Steam 游戏库。"
            )

        try:
            owned_games = await self.steam_client.get_owned_games(binding.account_id)
        except SteamApiError as exc:
            logger.warning(f"Steam owned games lookup failed: {exc}")
            raise LibraryFilterModeError(f"Steam 游戏库不可读，无法执行游戏库过滤：{exc}") from exc

        if not owned_games:
            raise LibraryFilterModeError("Steam 游戏库为空或不可见，无法执行游戏库过滤。")

        filtered, removed_count = filter_games_by_library_mode(ranked_games, owned_games, mode)
        if mode == LIBRARY_FILTER_EXCLUDE_OWNED:
            preference.parse_warnings.append(
                f"已排除 Steam 游戏库中已有的 {removed_count} 款候选。"
            )
        elif mode == LIBRARY_FILTER_ONLY_OWNED:
            preference.parse_warnings.append(
                f"已仅保留 Steam 游戏库中已有的 {len(filtered)} 款候选。"
            )
        return filtered


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
