from __future__ import annotations

import re
import time
from dataclasses import dataclass
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
from .services.diversity import DIVERSITY_STRICT, select_results_by_diversity
from .services.embedding_reranker import EmbeddingReranker
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
from .services.recommendation_memory import (
    PreferencePatch,
    RecommendationMemory,
    append_feedback,
    append_shown_games,
    build_recommendation_memory,
    load_recommendation_memory,
    save_recommendation_memory,
    summarize_games,
)
from .services.retry_command import (
    apply_preference_patch,
    merge_retry_preferences,
    parse_preference_patch,
    parse_retry_request,
)
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
from .services.user_profile import build_user_tag_weights
from .storage.models import AccountBinding, GamePreference, RankedGame, SteamOwnedGame
from .storage.repository import SQLiteCacheRepository

PLUGIN_NAME = "astrbot_plugin_game_recommender"
PLUGIN_VERSION = "0.5.0"
PLUGIN_DESCRIPTION = (
    "基于 Steam/PC 公开数据、本地索引和标签相似度推荐游戏；当前版本暂不做跨平台候选召回。"
)


@dataclass(frozen=True)
class PreparedRecommendation:
    raw_query: str
    preference: GamePreference
    diversity_mode: str
    result_limit: int


@dataclass(frozen=True)
class RecommendationRun:
    messages: list[str]
    ranked_games: list[RankedGame]
    preference: GamePreference
    diversity_mode: str
    result_limit: int
    raw_query: str


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
        self.enable_llm_fallback = bool(self.config.get("enable_llm_fallback"))
        self.enable_embedding_rerank = bool(self.config.get("enable_embedding_rerank", False))
        self.embedding_provider_id = str(self.config.get("embedding_provider_id", "") or "").strip()

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
        self.embedding_reranker = (
            EmbeddingReranker(
                context,
                self.cache,
                provider_id=self.embedding_provider_id,
            )
            if self.enable_embedding_rerank
            else None
        )
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
            retry_request = parse_retry_request(raw_text)
            if retry_request.is_retry:
                messages = await self._retry_recommendation_messages(
                    event,
                    retry_request.supplement,
                )
            else:
                prepared = await self._prepare_recommendation(event, raw_text)
                run = await self._run_recommendation(event, prepared)
                await self._save_recent_recommendation(event, run)
                messages = run.messages
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

        yield self._recommendation_result(event, messages)

    @filter.command(
        "gamerec_retry",
        alias={"重新推荐", "换一批"},
        desc="基于最近一次游戏推荐换一批候选。",
    )
    async def retry_recommend_games(self, event: AstrMessageEvent, query: GreedyStr):
        try:
            messages = await self._retry_recommendation_messages(event, str(query).strip())
        except SteamApiError as exc:
            logger.warning(f"Steam retry recommendation failed: {exc}")
            yield event.plain_result(f"Steam 查询失败：{exc}")
            return
        except LibraryFilterModeError as exc:
            yield event.plain_result(f"游戏库过滤参数错误：{exc}")
            return
        except Exception as exc:
            logger.exception("Retry game recommendation failed")
            yield event.plain_result(f"重新推荐失败：{exc}")
            return

        yield self._recommendation_result(event, messages)

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
        diversity_mode: str = "strict",
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
    ):
        ranked_games = await self.steam_index.recommend(
            preference,
            limit=limit,
            profile_tag_weights=profile_tag_weights,
            diversity_mode=diversity_mode,
            excluded_appids=excluded_appids,
            excluded_titles=excluded_titles,
        )
        if ranked_games:
            return ranked_games
        if STEAM_INDEX_FALLBACK_WARNING not in preference.parse_warnings:
            preference.parse_warnings.append(STEAM_INDEX_FALLBACK_WARNING)
        return []

    async def _user_profile_tag_weights(
        self,
        event: AstrMessageEvent,
        owned_games: list[SteamOwnedGame] | None = None,
    ) -> dict[str, float]:
        try:
            entries = await self.steam_index.load_entries()
            if not entries:
                return {}
            games = owned_games
            if games is None:
                games = await self._owned_games_for_recommendation(event, required=False)
            return build_user_tag_weights(games, entries)
        except Exception as exc:
            logger.debug(f"Steam user profile weights skipped: {exc}")
            return {}

    async def _owned_games_for_recommendation(
        self,
        event: AstrMessageEvent,
        required: bool,
    ) -> list[SteamOwnedGame]:
        try:
            chat_platform, chat_user_id = chat_identity_from_event(event)
        except AccountBindingError as exc:
            if required:
                raise LibraryFilterModeError(
                    f"无法识别当前用户，不能执行游戏库过滤：{exc}"
                ) from exc
            return []
        binding = await self.cache.get_account_binding(chat_platform, chat_user_id, "steam")
        if binding is None:
            if required:
                raise LibraryFilterModeError(
                    "当前用户未绑定 Steam 账号；请先使用 /accountbind steam <SteamID64 或好友码>。"
                )
            return []
        if not self.steam_client.has_web_api_key():
            if required:
                raise LibraryFilterModeError("未配置 steam_api_key，无法读取 Steam 游戏库。")
            return []
        try:
            owned_games = await self.steam_client.get_owned_games(binding.account_id)
        except SteamApiError as exc:
            if required:
                logger.warning(f"Steam owned games lookup failed: {exc}")
                raise LibraryFilterModeError(
                    f"Steam 游戏库不可读，无法执行游戏库过滤：{exc}"
                ) from exc
            return []
        if required and not owned_games:
            raise LibraryFilterModeError("Steam 游戏库为空或不可见，无法执行游戏库过滤。")
        return owned_games

    async def _prepare_recommendation(
        self,
        event: AstrMessageEvent,
        raw_text: str,
    ) -> PreparedRecommendation:
        command_filter = parse_library_filter_command(raw_text)
        text = command_filter.query
        if not text:
            raise LibraryFilterModeError(
                "请输入游戏需求，例如：/gamerec 排除已有 Steam 双人合作解谜"
            )
        text_filter_mode = detect_library_filter_mode(text)
        preference = await self.preference_parser.parse_preference(event, text)
        diversity_mode = getattr(preference, "diversity_mode", DIVERSITY_STRICT)
        library_filter_mode = resolve_library_filter_mode(
            command_filter.mode,
            text_filter_mode,
            preference.library_filter_mode,
        )
        preference.library_filter_mode = library_filter_mode
        if warning := steam_only_scope_warning_for(preference):
            preference.parse_warnings.append(warning)
        if not has_supported_steam_platform(preference) and not self.enable_llm_fallback:
            raise LibraryFilterModeError(preference.parse_warnings[-1])
        result_limit = effective_result_limit(self.max_results, preference.result_count)
        return PreparedRecommendation(
            raw_query=raw_text,
            preference=preference,
            diversity_mode=diversity_mode,
            result_limit=result_limit,
        )

    async def _run_recommendation(
        self,
        event: AstrMessageEvent,
        prepared: PreparedRecommendation,
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
    ) -> RecommendationRun:
        started_at = time.perf_counter()
        checkpoint = started_at
        phase_times: dict[str, float] = {}

        def finish_phase(name: str) -> None:
            nonlocal checkpoint
            now = time.perf_counter()
            phase_times[name] = (now - checkpoint) * 1000
            checkpoint = now

        preference = prepared.preference
        result_limit = prepared.result_limit
        ranked_games: list[RankedGame] = []
        retrieved_count = 0
        filtered_count = 0
        degradation_reason = "none"
        if has_supported_steam_platform(preference):
            candidate_pool_size = min(60, max(30, result_limit * 6))
            owned_games = await self._owned_games_for_recommendation(
                event,
                required=bool(preference.library_filter_mode),
            )
            profile_tag_weights = await self._user_profile_tag_weights(event, owned_games)
            finish_phase("profile")
            ranked_games = await self._recommend_with_steam_index(
                preference,
                limit=candidate_pool_size,
                profile_tag_weights=profile_tag_weights,
                diversity_mode=DIVERSITY_STRICT,
                excluded_appids=excluded_appids,
                excluded_titles=excluded_titles,
            )
            retrieved_count = len(ranked_games)
            finish_phase("recall_rank")
            embedding_reranker = getattr(self, "embedding_reranker", None)
            if embedding_reranker is not None:
                ranked_games = await embedding_reranker.rerank(
                    preference,
                    prepared.raw_query,
                    ranked_games,
                )
                degradation_reason = (
                    getattr(embedding_reranker, "last_degradation_reason", None) or "none"
                )
            finish_phase("embedding")
            if preference.library_filter_mode:
                ranked_games = await self._filter_library_games(
                    preference,
                    ranked_games,
                    preference.library_filter_mode,
                    owned_games,
                )
            filtered_count = max(retrieved_count - len(ranked_games), 0)
            finish_phase("library_filter")
            if preference.budget is not None:
                ranked_games = await self.price_bridge.enrich_ranked_games(
                    ranked_games,
                    preference,
                )
                ranked_games = select_results_by_diversity(
                    ranked_games,
                    result_limit,
                    prepared.diversity_mode,
                )
            else:
                ranked_games = select_results_by_diversity(
                    ranked_games,
                    result_limit,
                    prepared.diversity_mode,
                )
                ranked_games = await self.price_bridge.enrich_ranked_games(
                    ranked_games,
                    preference,
                )
            finish_phase("final_selection")
        logger.debug(
            "Game recommendation pipeline: elapsed_ms=%.1f candidates=%d "
            "filtered=%d selected=%d refill_pool=%d degradation=%s "
            "profile_ms=%.1f recall_rank_ms=%.1f embedding_ms=%.1f "
            "library_filter_ms=%.1f final_selection_ms=%.1f",
            (time.perf_counter() - started_at) * 1000,
            retrieved_count,
            filtered_count,
            len(ranked_games),
            max(retrieved_count - result_limit, 0),
            degradation_reason,
            phase_times.get("profile", 0.0),
            phase_times.get("recall_rank", 0.0),
            phase_times.get("embedding", 0.0),
            phase_times.get("library_filter", 0.0),
            phase_times.get("final_selection", 0.0),
        )
        messages = await format_recommendation_messages_with_llm(
            self.context,
            event,
            self.provider_id,
            preference,
            ranked_games,
            limit=result_limit,
            enable_empty_fallback=self.enable_llm_fallback,
            raw_query=prepared.raw_query,
        )
        return RecommendationRun(
            messages=messages,
            ranked_games=ranked_games,
            preference=preference,
            diversity_mode=prepared.diversity_mode,
            result_limit=result_limit,
            raw_query=prepared.raw_query,
        )

    async def _retry_recommendation_messages(
        self,
        event: AstrMessageEvent,
        supplement: str = "",
    ) -> list[str]:
        chat_platform, chat_user_id = chat_identity_from_event(event)
        memory = await load_recommendation_memory(
            chat_platform,
            chat_user_id,
            self.cache,
        )
        if memory is None:
            return ["没有可用于重新推荐的近期记录。请先使用 /gamerec 提出一次游戏推荐需求。"]

        patch = PreferencePatch()
        patch_excluded_appids: list[int] = []
        patch_excluded_titles: list[str] = []
        if supplement:
            parsed_patch = parse_preference_patch(
                supplement,
                len(memory.last_results),
            )
            patch = parsed_patch.patch
            preference = memory.preference
            diversity_mode = memory.diversity_mode
            result_limit = memory.result_limit
            if parsed_patch.residual_text:
                supplemental = await self._prepare_recommendation(
                    event,
                    parsed_patch.residual_text,
                )
                preference = merge_retry_preferences(
                    preference,
                    supplemental.preference,
                )
                if explicitly_changes_diversity(parsed_patch.residual_text):
                    diversity_mode = supplemental.diversity_mode
                if explicitly_changes_result_count(parsed_patch.residual_text):
                    result_limit = supplemental.result_limit
            preference, patch_excluded_appids, patch_excluded_titles = apply_preference_patch(
                preference,
                patch,
                memory.last_results,
                parsed_patch.warnings,
            )
            prepared = PreparedRecommendation(
                raw_query=f"{memory.raw_query} {supplement}".strip(),
                preference=preference,
                diversity_mode=diversity_mode,
                result_limit=result_limit,
            )
        else:
            prepared = PreparedRecommendation(
                raw_query=memory.raw_query,
                preference=memory.preference,
                diversity_mode=memory.diversity_mode,
                result_limit=memory.result_limit,
            )
        run = await self._run_recommendation(
            event,
            prepared,
            excluded_appids=list(dict.fromkeys([*memory.shown_appids, *patch_excluded_appids])),
            excluded_titles=list(dict.fromkeys([*memory.shown_titles, *patch_excluded_titles])),
        )
        await self._save_retry_memory(
            chat_platform,
            chat_user_id,
            memory,
            run,
            patch if supplement else None,
        )
        return run.messages

    async def _save_recent_recommendation(
        self,
        event: AstrMessageEvent,
        run: RecommendationRun,
    ) -> None:
        if not run.ranked_games:
            return
        chat_platform, chat_user_id = chat_identity_from_event(event)
        memory = build_recommendation_memory(
            chat_platform=chat_platform,
            chat_user_id=chat_user_id,
            raw_query=run.raw_query,
            preference=run.preference,
            diversity_mode=run.diversity_mode,
            result_limit=run.result_limit,
            games=run.ranked_games[: run.result_limit],
        )
        await save_recommendation_memory(self.cache, memory)

    async def _save_retry_memory(
        self,
        chat_platform: str,
        chat_user_id: str,
        memory: RecommendationMemory,
        run: RecommendationRun,
        patch: PreferencePatch | None = None,
    ) -> None:
        updated = RecommendationMemory(
            chat_platform=chat_platform,
            chat_user_id=chat_user_id,
            raw_query=run.raw_query,
            preference=run.preference,
            diversity_mode=run.diversity_mode,
            result_limit=run.result_limit,
            shown_appids=list(memory.shown_appids),
            shown_titles=list(memory.shown_titles),
            created_at=time.time(),
            last_results=(
                summarize_games(run.ranked_games[: run.result_limit])
                if run.ranked_games
                else list(memory.last_results)
            ),
            feedback=list(memory.feedback),
        )
        if patch is not None:
            updated = append_feedback(updated, patch)
        if run.ranked_games:
            updated = append_shown_games(updated, run.ranked_games[: run.result_limit])
        await save_recommendation_memory(self.cache, updated)

    def _recommendation_result(self, event: AstrMessageEvent, messages: list[str]):
        forward_chain = build_forward_message_chain(messages)
        if forward_chain and hasattr(event, "chain_result"):
            return event.chain_result(forward_chain)
        return event.plain_result("\n\n".join(messages))

    async def _filter_library_games(
        self,
        preference,
        ranked_games,
        mode: str,
        owned_games: list[SteamOwnedGame],
    ):
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


def explicitly_changes_diversity(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered for marker in ("多样", "重复", "strict", "balanced", "high", "严格")
    )


def explicitly_changes_result_count(text: str) -> bool:
    return bool(re.search(r"\d+\s*(?:款|个|部)", str(text or "")))
