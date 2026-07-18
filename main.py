from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
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
from .services.explanation_builder import (
    generate_recommendation_reasons,
    generate_unplayed_reason,
    resolve_provider_id,
)
from .services.formatter import format_recommendation_messages
from .services.game_identity import is_confirmed_base_game
from .services.llm_fallback import (
    LlmFallbackContractError,
    LlmFallbackProviderError,
    LlmFallbackVerificationError,
    UnverifiedGameSuggestion,
    generate_unverified_game_suggestions,
    verify_fallback_suggestion_titles,
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
from .services.preference_rules import extract_result_count
from .services.recommendation_limits import (
    DEFAULT_RECOMMENDATION_COUNT,
    MAX_RECOMMENDATION_COUNT,
    effective_result_limit,
)
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
from .services.region_query import normalize_region, parse_region_query, region_currency
from .services.retry_command import (
    apply_preference_patch,
    merge_retry_preferences,
    parse_preference_patch,
    parse_retry_request,
)
from .services.semantic_feature_verifier import (
    SemanticFeatureVerifier,
    verify_ranked_features,
)
from .services.run_notices import RunNotice, dedupe_run_notices
from .services.safe_errors import log_external_failure
from .services.steam_index import (
    STEAM_TAG_RECALL_DEGRADED_WARNING,
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
from .storage.models import GamePreference, RankedGame, SteamAccountBinding, SteamOwnedGame
from .storage.repository import SQLiteCacheRepository

PLUGIN_NAME = "astrbot_plugin_steam_game_recommender"
PLUGIN_VERSION = "0.7.0"
PLUGIN_DESCRIPTION = "基于 Steam 公开数据、连续评分和可信证据生成精简游戏推荐。"
MEMORY_SAVE_WARNING = (
    "⚠️ 本次结果无法用于“换一批”，但上面的推荐内容仍然有效。"
)


@dataclass(frozen=True)
class PreparedRecommendation:
    raw_query: str
    preference: GamePreference
    result_limit: int
    run_notices: tuple[RunNotice, ...] = ()


@dataclass(frozen=True)
class RecommendationRun:
    messages: list[str]
    ranked_games: list[RankedGame]
    preference: GamePreference
    result_limit: int
    raw_query: str
    run_notices: tuple[RunNotice, ...] = ()
    unverified_suggestions: tuple[UnverifiedGameSuggestion, ...] = ()

    @property
    def used_unverified_fallback(self) -> bool:
        return bool(self.unverified_suggestions)


@register(
    PLUGIN_NAME,
    "jiangxingda",
    PLUGIN_DESCRIPTION,
    PLUGIN_VERSION,
)
class SteamGameRecommenderPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.config = config or {}
        model_config = config_section(self.config, "model_and_access")
        price_config = config_section(self.config, "price_and_region")
        self.recommendation_config = config_section(
            self.config,
            "recommendation_and_scoring",
        )
        cache_config = config_section(self.config, "cache_and_network")

        timeout = safe_bounded_int(
            cache_config.get("timeout_seconds"),
            15,
            minimum=1,
            maximum=120,
        )
        self.reuse_identical_query_cache = safe_bool(
            cache_config.get("reuse_identical_query_cache"),
            False,
        )
        self.max_results = min(
            max(
                safe_int(
                    self.recommendation_config.get("max_results"),
                    DEFAULT_RECOMMENDATION_COUNT,
                ),
                1,
            ),
            MAX_RECOMMENDATION_COUNT,
        )
        self.provider_id = str(model_config.get("llm_provider_id", "") or "").strip()
        self.fallback_provider_id = str(
            model_config.get("llm_fallback_provider_id", "") or ""
        ).strip()
        self.semantic_verification_batch_size = min(
            max(
                safe_int(
                    model_config.get("semantic_verification_batch_size"),
                    5,
                ),
                1,
            ),
            10,
        )
        self.default_region = normalize_region(str(price_config.get("default_region") or "CN"))
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
            cache_ttl_hours=safe_bounded_int(
                cache_config.get("cache_ttl_hours"),
                24,
                minimum=1,
                maximum=8760,
            ),
            default_country=self.default_region,
            language="schinese",
            steam_api_key=str(model_config.get("steam_api_key") or ""),
        )
        self.preference_parser = PreferenceParser(context, self.provider_id)
        self.steam_index = SteamGameIndexService(
            steam_client=self.steam_client,
            cache=self.cache,
            ttl_hours=safe_bounded_int(
                self.recommendation_config.get("steam_index_ttl_hours"),
                168,
                minimum=1,
                maximum=8760,
            ),
            reuse_cache=self.reuse_identical_query_cache,
        )
        self.price_bridge = SteamPriceBridge(
            self.http_client,
            {"default_region": self.default_region},
        )
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
                "请输入需求，例如：/gamerec 双人合作解谜，不要恐怖，预算 100 以内"
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
                messages = list(run.messages)
                try:
                    await self._save_recent_recommendation(event, run)
                except Exception as exc:
                    log_external_failure(
                        logger,
                        "recommendation_memory_save_failed",
                        stage="recommendation_memory_save",
                        exc=exc,
                    )
                    messages.append(MEMORY_SAVE_WARNING)
        except SteamApiError as exc:
            log_external_failure(
                logger,
                "recommendation_steam_failed",
                stage="recommendation_steam",
                exc=exc,
            )
            yield event.plain_result("Steam 查询暂时不可用，请稍后重试。")
            return
        except LibraryFilterModeError as exc:
            yield event.plain_result(f"游戏库过滤参数错误：{exc}")
            return
        except Exception as exc:
            log_external_failure(
                logger,
                "recommendation_failed",
                stage="recommendation",
                exc=exc,
            )
            yield event.plain_result("游戏推荐暂时失败，请稍后重试。")
            return

        yield self._recommendation_result(event, messages)

    @filter.command(
        "gamerec_retry",
        alias={"重新推荐", "换一批"},
        desc="基于最近一次游戏推荐换一批候选。",
    )
    async def retry_recommend_games(self, event: AstrMessageEvent, query: GreedyStr):
        try:
            messages = await self._retry_recommendation_messages(
                event,
                str(query).strip(),
            )
        except SteamApiError as exc:
            log_external_failure(
                logger,
                "retry_steam_failed",
                stage="retry_steam",
                exc=exc,
            )
            yield event.plain_result("Steam 查询暂时不可用，请稍后重试。")
            return
        except LibraryFilterModeError as exc:
            yield event.plain_result(f"游戏库过滤参数错误：{exc}")
            return
        except Exception as exc:
            log_external_failure(
                logger,
                "retry_failed",
                stage="retry_recommendation",
                exc=exc,
            )
            yield event.plain_result("重新推荐暂时失败，请稍后重试。")
            return

        yield self._recommendation_result(event, messages)

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
                binding = await self.cache.get_steam_account_binding(chat_platform, chat_user_id)
                if binding is None:
                    yield event.plain_result(
                        "还没有绑定账号。请使用 /accountbind <SteamID64 或好友码>。"
                    )
                    return
                yield event.plain_result(
                    f"当前绑定 Steam ID：{binding.steam_id64}（{binding.account_kind}）。"
                )
                return

            parsed = parse_account_binding_command(text)
            saved = await self.cache.upsert_steam_account_binding(
                SteamAccountBinding(
                    chat_platform=chat_platform,
                    chat_user_id=chat_user_id,
                    steam_id64=parsed.steam_id64,
                    account_kind=parsed.account_kind,
                    display_value=parsed.display_value,
                    metadata=parsed.metadata,
                )
            )
        except AccountBindingError as exc:
            yield event.plain_result(f"账号绑定失败：{exc}")
            return
        except Exception as exc:
            log_external_failure(
                logger,
                "account_binding_failed",
                stage="account_binding_storage",
                exc=exc,
            )
            yield event.plain_result("账号绑定暂时失败，请稍后重试。")
            return

        yield event.plain_result(
            f"账号绑定成功：Steam ID {saved.steam_id64}（来源：{saved.account_kind}）。"
        )

    @filter.command(
        "randomrec",
        alias={"随机推荐"},
        desc="从已绑定 Steam 库中随机推荐一款未玩且评价过线的游戏。",
    )
    async def recommend_random_game(self, event: AstrMessageEvent):
        try:
            chat_platform, chat_user_id = chat_identity_from_event(event)
            binding = await self.cache.get_steam_account_binding(chat_platform, chat_user_id)
            if binding is None:
                yield event.plain_result(
                    "当前用户未绑定 Steam 账号；请先使用 /accountbind <SteamID64 或好友码>。"
                )
                return
            if not self.steam_client.has_web_api_key():
                yield event.plain_result("未配置 steam_api_key，无法读取 Steam 游戏库。")
                return

            owned_games = await self.steam_client.get_owned_games(binding.steam_id64)
            if not owned_games:
                yield event.plain_result("Steam 游戏库为空或不可见，无法进行随机推荐。")
                return

            min_review_count = safe_bounded_int(
                self.recommendation_config.get("steam_min_review_count"),
                50,
                minimum=0,
                maximum=1_000_000,
            )
            min_positive_ratio = safe_bounded_float(
                self.recommendation_config.get("steam_min_positive_ratio"),
                0.65,
                minimum=0.0,
                maximum=1.0,
            )
            recommendation = await pick_random_unplayed_game(
                owned_games,
                self.steam_client,
                min_review_count=min_review_count,
                min_positive_ratio=min_positive_ratio,
            )
            reason = await generate_unplayed_reason(
                self.context,
                event,
                self.provider_id,
                recommendation.game,
            )
        except AccountBindingError as exc:
            yield event.plain_result(f"随机推荐失败：{exc}")
            return
        except UnplayedRecommendationError as exc:
            yield event.plain_result(f"随机推荐失败：{exc}")
            return
        except SteamApiError as exc:
            log_external_failure(
                logger,
                "random_recommendation_steam_failed",
                stage="random_recommendation_steam",
                exc=exc,
            )
            yield event.plain_result("Steam 查询暂时不可用，请稍后重试。")
            return
        except Exception as exc:
            log_external_failure(
                logger,
                "random_recommendation_failed",
                stage="random_recommendation",
                exc=exc,
            )
            yield event.plain_result("随机推荐暂时失败，请稍后重试。")
            return

        yield event.plain_result(
            format_unplayed_recommendation(
                recommendation,
                reason,
            )
        )

    async def _recommend_with_steam_index(
        self,
        preference,
        limit: int,
        requested_limit: int | None = None,
        profile_tag_weights: dict[str, float] | None = None,
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
        preferred_appids: list[int] | None = None,
    ):
        return await self.steam_index.recommend(
            preference,
            limit=limit,
            profile_tag_weights=profile_tag_weights,
            excluded_appids=excluded_appids,
            excluded_titles=excluded_titles,
            preferred_appids=preferred_appids,
            requested_limit=requested_limit,
        )

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
            log_external_failure(
                logger,
                "user_profile_skipped",
                stage="user_profile",
                exc=exc,
                level=logging.DEBUG,
            )
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
        binding = await self.cache.get_steam_account_binding(chat_platform, chat_user_id)
        if binding is None:
            if required:
                raise LibraryFilterModeError(
                    "当前用户未绑定 Steam 账号；请先使用 /accountbind <SteamID64 或好友码>。"
                )
            return []
        if not self.steam_client.has_web_api_key():
            if required:
                raise LibraryFilterModeError("未配置 steam_api_key，无法读取 Steam 游戏库。")
            return []
        try:
            owned_games = await self.steam_client.get_owned_games(binding.steam_id64)
        except SteamApiError as exc:
            if required:
                log_external_failure(
                    logger,
                    "owned_games_lookup_failed",
                    stage="owned_games",
                    exc=exc,
                )
                raise LibraryFilterModeError(
                    "Steam 游戏库暂时不可读，无法执行游戏库过滤。"
                ) from exc
            return []
        if required and not owned_games:
            raise LibraryFilterModeError("Steam 游戏库为空或不可见，无法执行游戏库过滤。")
        return owned_games

    async def _prepare_recommendation(
        self,
        event: AstrMessageEvent,
        raw_text: str,
        default_region: str | None = None,
    ) -> PreparedRecommendation:
        command_filter = parse_library_filter_command(raw_text)
        region_query = parse_region_query(
            command_filter.query,
            default_region=default_region or getattr(self, "default_region", "CN"),
        )
        text = region_query.query
        if not text:
            raise LibraryFilterModeError(
                "请输入游戏需求，例如：/gamerec 排除已有 Steam 双人合作解谜"
            )
        text_filter_mode = detect_library_filter_mode(text)
        parse_outcome = await self.preference_parser.parse_preference(event, text)
        preference = parse_outcome.preference
        preference.region = region_query.region
        if preference.budget is not None and not preference.budget_currency:
            preference.budget_currency = region_currency(region_query.region)
        library_filter_mode = resolve_library_filter_mode(
            command_filter.mode,
            text_filter_mode,
            preference.library_filter_mode,
        )
        preference.library_filter_mode = library_filter_mode
        if warning := steam_only_scope_warning_for(preference):
            preference.parse_warnings.append(warning)
        result_limit = effective_result_limit(self.max_results, preference.result_count)
        return PreparedRecommendation(
            raw_query=raw_text,
            preference=preference,
            result_limit=result_limit,
            run_notices=parse_outcome.prelude_messages,
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
        semantic_feature_notices = ()
        semantic_feature_candidate_count = 0
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
                requested_limit=result_limit,
                profile_tag_weights=profile_tag_weights,
                excluded_appids=excluded_appids,
                excluded_titles=excluded_titles,
                preferred_appids=(
                    [owned.appid for owned in owned_games if owned.appid]
                    if preference.library_filter_mode == LIBRARY_FILTER_ONLY_OWNED
                    else None
                ),
            )
            if STEAM_TAG_RECALL_DEGRADED_WARNING in preference.parse_warnings:
                degradation_reason = "steam_tag_recall"
            ranked_games = [
                game for game in ranked_games if is_confirmed_base_game(game)
            ]
            retrieved_count = len(ranked_games)
            finish_phase("recall_rank")
            if preference.library_filter_mode:
                ranked_games = await self._filter_library_games(
                    preference,
                    ranked_games,
                    preference.library_filter_mode,
                    owned_games,
                )
            filtered_count = max(retrieved_count - len(ranked_games), 0)
            finish_phase("library_filter")
            if preference.soft_features:
                resolved_provider_id = await resolve_provider_id(
                    self.context,
                    event,
                    self.provider_id,
                )
                verifier = (
                    SemanticFeatureVerifier(
                        self.context,
                        self.cache,
                        provider_id=resolved_provider_id,
                        batch_size=self.semantic_verification_batch_size,
                        locale=str(
                            getattr(self.steam_client, "language", "schinese")
                            or "schinese"
                        ),
                        reuse_cache=getattr(
                            self,
                            "reuse_identical_query_cache",
                            True,
                        ),
                    )
                    if resolved_provider_id
                    else None
                )
                semantic_outcome = await verify_ranked_features(
                    ranked_games,
                    preference.soft_features,
                    verifier,
                    result_limit=result_limit,
                    quality_intent=preference.quality_intent,
                )
                ranked_games = list(semantic_outcome.games)
                semantic_feature_notices = semantic_outcome.notices
                semantic_feature_candidate_count = semantic_outcome.candidate_count
                finish_phase("semantic_features")
            if preference.budget is None:
                ranked_games = ranked_games[:result_limit]
            ranked_games = await self.price_bridge.enrich_ranked_games(
                ranked_games,
                preference,
            )
            ranked_games = ranked_games[:result_limit]
            finish_phase("final_selection")
            ranked_games = await generate_recommendation_reasons(
                self.context,
                event,
                self.provider_id,
                ranked_games,
            )
            finish_phase("reasons")
        logger.debug(
            "Game recommendation pipeline: elapsed_ms=%.1f candidates=%d "
            "filtered=%d selected=%d refill_pool=%d degradation=%s "
            "semantic_candidates=%d semantic_notices=%s "
            "profile_ms=%.1f recall_rank_ms=%.1f "
            "library_filter_ms=%.1f semantic_features_ms=%.1f "
            "final_selection_ms=%.1f reasons_ms=%.1f",
            (time.perf_counter() - started_at) * 1000,
            retrieved_count,
            filtered_count,
            len(ranked_games),
            max(retrieved_count - result_limit, 0),
            degradation_reason,
            semantic_feature_candidate_count,
            ",".join(notice.code for notice in semantic_feature_notices) or "none",
            phase_times.get("profile", 0.0),
            phase_times.get("recall_rank", 0.0),
            phase_times.get("library_filter", 0.0),
            phase_times.get("semantic_features", 0.0),
            phase_times.get("final_selection", 0.0),
            phase_times.get("reasons", 0.0),
        )
        run_notices = dedupe_run_notices(
            [
                *prepared.run_notices,
                *(
                    RunNotice(notice.code, "warning", notice.message)
                    for notice in semantic_feature_notices
                ),
            ]
        )
        unverified_suggestions: tuple[UnverifiedGameSuggestion, ...] = ()
        fallback_provider_id = str(
            getattr(self, "fallback_provider_id", "") or ""
        ).strip()
        if not ranked_games and fallback_provider_id:
            try:
                generated_suggestions = await generate_unverified_game_suggestions(
                    self.context,
                    fallback_provider_id,
                    raw_query=prepared.raw_query,
                    preference=preference,
                    result_limit=result_limit,
                )
                unverified_suggestions = await verify_fallback_suggestion_titles(
                    self.steam_client,
                    generated_suggestions,
                    result_limit=result_limit,
                    reuse_cache=bool(
                        getattr(self, "reuse_identical_query_cache", False)
                    ),
                )
                if not unverified_suggestions:
                    run_notices = dedupe_run_notices(
                        [
                            *run_notices,
                            RunNotice(
                                "llm_fallback_titles_unresolved",
                                "warning",
                                "LLM 兜底未找到可由 Steam 目录确认的候选名称，"
                                "本次未展示模型文本。",
                            ),
                        ]
                    )
            except (
                LlmFallbackContractError,
                LlmFallbackProviderError,
                LlmFallbackVerificationError,
            ) as exc:
                log_external_failure(
                    logger,
                    "empty_result_fallback_failed",
                    stage="empty_result_fallback",
                    exc=exc,
                )
                run_notices = dedupe_run_notices(
                    [
                        *run_notices,
                        RunNotice(
                            "llm_fallback_unavailable",
                            "warning",
                            "LLM 兜底服务暂时不可用，本次仅保留规则空结果。",
                        ),
                    ]
                )
        messages = format_recommendation_messages(
            preference,
            ranked_games,
            limit=result_limit,
            run_notices=run_notices,
            unverified_suggestions=unverified_suggestions,
        )
        return RecommendationRun(
            messages=messages,
            ranked_games=ranked_games,
            preference=preference,
            result_limit=result_limit,
            raw_query=prepared.raw_query,
            run_notices=run_notices,
            unverified_suggestions=unverified_suggestions,
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
            result_limit = memory.result_limit
            run_notices: tuple[RunNotice, ...] = ()
            if parsed_patch.residual_text:
                supplemental = await self._prepare_recommendation(
                    event,
                    parsed_patch.residual_text,
                    default_region=preference.region,
                )
                preference = merge_retry_preferences(
                    preference,
                    supplemental.preference,
                )
                if explicitly_changes_result_count(parsed_patch.residual_text):
                    result_limit = supplemental.result_limit
                run_notices = supplemental.run_notices
            preference, patch_excluded_appids, patch_excluded_titles = apply_preference_patch(
                preference,
                patch,
                memory.last_results,
                parsed_patch.warnings,
            )
            prepared = PreparedRecommendation(
                raw_query=f"{memory.raw_query} {supplement}".strip(),
                preference=preference,
                result_limit=result_limit,
                run_notices=run_notices,
            )
        else:
            prepared = PreparedRecommendation(
                raw_query=memory.raw_query,
                preference=memory.preference,
                result_limit=memory.result_limit,
            )
        run = await self._run_recommendation(
            event,
            prepared,
            excluded_appids=list(dict.fromkeys([*memory.shown_appids, *patch_excluded_appids])),
            excluded_titles=list(dict.fromkeys([*memory.shown_titles, *patch_excluded_titles])),
        )
        if not run.used_unverified_fallback:
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
    except (OverflowError, TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return default


def safe_bounded_int(
    value: Any,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parsed = safe_int(value, default)
    return min(max(parsed, minimum), maximum)


def safe_bounded_float(
    value: Any,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    parsed = safe_float(value, default)
    if not math.isfinite(parsed):
        parsed = default
    return min(max(parsed, minimum), maximum)


def config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    return section if isinstance(section, Mapping) else {}


def explicitly_changes_result_count(text: str) -> bool:
    return extract_result_count(str(text or "")) is not None
