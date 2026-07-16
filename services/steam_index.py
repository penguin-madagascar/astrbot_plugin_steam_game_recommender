from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Protocol

import httpx

from ..clients.steam import SteamApiError, is_retryable_steam_read_error
from ..storage.models import (
    COMPANY_ALIAS_LIMIT,
    CompanyPreference,
    GameCandidate,
    GamePreference,
    RankedGame,
    ResolvedReferenceGame,
    SteamSearchHit,
)
from .company_preferences import (
    CompanyMatchStatus,
    company_match_status,
    matches_company_preference,
)
from .candidate_tag_evidence import build_candidate_tag_evidence
from .game_identity import (
    deduplicate_game_editions,
    game_family_key,
    is_confirmed_base_game,
)
from .recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    ReferencePolarity,
    ReferenceQuery,
    build_recommendation_intent,
    expand_intent_with_reference_tags,
)
from .reference_matching import ReferenceMatch, match_reference_query, title_key
from .similarity_ranker import (
    SteamTagProfile,
    build_profile_from_preference,
    rank_steam_candidates,
)
from .steam_recall import (
    CandidateHit,
    CandidateRecallResult,
    RecallHealth,
    RecallSource,
    RecallSourceHealth,
    RecallSourceStatus,
    RecallUnavailableError,
    RRF_K,
    merge_candidate_sources,
    select_recall_seeds,
)
from .tag_normalizer import (
    canonical_steam_tag_name,
    canonical_tags_from_terms,
    extract_description_terms,
    normalize_key,
    normalize_tag,
    register_steam_tag_aliases,
    static_canonical_tags,
)
from .tag_presentation import build_tag_presentations

logger = logging.getLogger(__name__)

STEAM_INDEX_CACHE_KEY = "steam_index"
STEAM_INDEX_SCHEMA_VERSION = 3
STEAM_INDEX_LEGACY_CACHE_KEYS = ("steam_index:v4", "steam_index:v3")
STEAM_INDEX_MAX_ENTRIES = 3_000
STEAM_INDEX_MAX_SEARCH_TERMS = 256
STEAM_INDEX_MAX_SEARCHES_PER_ROUND = 8
STEAM_INDEX_SEARCH_RESULTS_PER_TERM = 10
STEAM_INDEX_MAX_NEW_APPIDS_PER_ROUND = 60
STEAM_HTTP_CONCURRENCY = 6
REFERENCE_TAG_COUNT_PREFETCH_PER_GAME = 5
MAX_REFERENCE_DETAIL_ATTEMPTS_PER_ENTITY = 3
STEAM_TAG_ALIASES_FRESH_SECONDS = 24 * 60 * 60
STEAM_TAG_ALIASES_STALE_SECONDS = 7 * STEAM_TAG_ALIASES_FRESH_SECONDS
RELEASE_STATUS_FRESH_SECONDS = 60 * 60
RELEASE_STATUS_STALE_SECONDS = 6 * 60 * 60
SNAPSHOT_STORAGE_TTL_HOURS = 24 * 3650
REFERENCE_MATCH_THRESHOLD = 0.75

STEAM_INDEX_FALLBACK_WARNING = (
    "Steam 索引暂不可用，已尝试通过 Steam 公共搜索刷新候选；"
    "如果仍为空，请换更明确的标签或参考游戏。"
)
STEAM_TAG_RECALL_DEGRADED_WARNING = (
    "Steam 标签召回能力暂时下降，已使用独立标签文本搜索和本地索引补位。"
)
STEAM_ONLY_SCOPE_WARNING = (
    "当前版本仅支持 Steam 商店游戏，无法验证 Switch、PlayStation 或 Xbox 候选。"
)
STEAM_INDEX_PLATFORMS = {"steam", "pc"}


@dataclass(frozen=True)
class SteamIndexEntry:
    candidate: GameCandidate
    refreshed_at: float
    needs_revalidation: bool = False


@dataclass(frozen=True)
class SteamIndexSnapshot:
    entries: list[SteamIndexEntry] = field(default_factory=list)
    search_coverage: dict[str, float] = field(default_factory=dict)
    schema_version: int = STEAM_INDEX_SCHEMA_VERSION


class ReferenceResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    NO_HIT = "no_hit"
    AMBIGUOUS = "ambiguous"
    TRANSIENT_FAILURE = "transient_failure"
    CONTRACT_FAILURE = "contract_failure"


@dataclass(frozen=True)
class ReferenceSearchFailure:
    alias: str
    language: str
    source: str
    kind: str
    error: str


@dataclass(frozen=True)
class ReferenceSearchOutcome:
    hits: tuple[SteamSearchHit, ...] = ()
    attempted: int = 0
    succeeded: int = 0
    failures: tuple[ReferenceSearchFailure, ...] = ()
    validation_failures: tuple[ReferenceSearchFailure, ...] = ()


@dataclass(frozen=True)
class ReferenceResolutionBatch:
    enriched_count: int = 0
    statuses: tuple[tuple[ReferenceQuery, ReferenceResolutionStatus], ...] = ()


@dataclass(frozen=True)
class RecallSourceFetch:
    source: RecallSource
    health: RecallSourceHealth
    total_count: int = 0


@dataclass(frozen=True)
class RecallValidationBatch:
    hits: tuple[CandidateHit, ...]
    attempts: int
    transient_failures: int
    contract_failures: int


class SteamIndexCache(Protocol):
    async def get_json(self, key: str, ttl_hours: int) -> Any | None: ...

    async def set_json(self, key: str, payload: Any) -> None: ...


class SteamIndexClient(Protocol):
    async def search_game_refs(self, **kwargs: Any) -> list[SteamSearchHit]: ...


class SteamGameIndexService:
    def __init__(
        self,
        steam_client: SteamIndexClient,
        cache: SteamIndexCache,
        ttl_hours: int = 168,
        page_size: int = STEAM_INDEX_SEARCH_RESULTS_PER_TERM,
        clock: Callable[[], float] = time.time,
        reuse_cache: bool = True,
    ) -> None:
        self.steam_client = steam_client
        self.cache = cache
        self.ttl_hours = max(int(ttl_hours), 1)
        self.page_size = min(max(int(page_size), 1), STEAM_INDEX_SEARCH_RESULTS_PER_TERM)
        self.clock = clock
        self.reuse_cache = reuse_cache
        self._tag_vocabulary_languages: tuple[str, ...] = ("english", "schinese")
        self._tag_vocabulary_payloads: dict[
            str,
            tuple[tuple[dict[str, Any], ...], float],
        ] = {}
        self._steam_tag_ids: dict[str, int] = {}
        self._canonical_tag_by_id: dict[int, str] = {}
        self._steam_tag_aliases: dict[str, str] = {}
        self._presentation_tag_names: dict[str, str] = {}
        self._tag_aliases_lock = asyncio.Lock()
        self._tag_aliases_task: asyncio.Task[bool] | None = None
        self._snapshot_lock = asyncio.Lock()
        self._bootstrap_lock = asyncio.Lock()
        self._bootstrap_task: asyncio.Task[None] | None = None
        self._steam_semaphore = asyncio.Semaphore(STEAM_HTTP_CONCURRENCY)
        self._storefront_tag_tasks: dict[tuple[int, int], asyncio.Task[Any]] = {}
        self._storefront_tag_tasks_lock = asyncio.Lock()
        self._intersection_tasks: dict[tuple[tuple[int, int], int], asyncio.Task[Any]] = {}
        self._intersection_tasks_lock = asyncio.Lock()
        self._more_like_tasks: dict[tuple[int, bool], asyncio.Task[Any]] = {}
        self._more_like_tasks_lock = asyncio.Lock()

    def _discovery_cache_kwargs(
        self,
        function: Callable[..., Any],
    ) -> dict[str, bool]:
        if callable_accepts_keyword(function, "reuse_cache"):
            return {"reuse_cache": self.reuse_cache}
        return {}

    async def recommend(
        self,
        preference: GamePreference,
        limit: int,
        profile_tag_weights: dict[str, float] | None = None,
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
        preferred_appids: list[int] | None = None,
        requested_limit: int | None = None,
    ) -> list[RankedGame]:
        if preference.platforms and not has_supported_steam_platform(preference):
            return []

        await self.bootstrap()
        await self.ensure_steam_tag_aliases()
        request_intent = self._build_request_intent(preference)
        request_tags = [tag.tag for tag in request_intent.tags]
        filtered_explicit_tag = self._has_filtered_explicit_tag(
            preference,
            request_intent,
        )
        if filtered_explicit_tag or (
            request_tags
            and any(self._usable_steam_tag_id_for(tag) is None for tag in request_tags)
        ):
            if STEAM_TAG_RECALL_DEGRADED_WARNING not in preference.parse_warnings:
                preference.parse_warnings.append(STEAM_TAG_RECALL_DEGRADED_WARNING)
        snapshot = await self.load_snapshot()
        recall, reference_entries, recall_intent = (
            await self._recall_specific_candidates(
                preference,
                snapshot,
                excluded_appids=excluded_appids,
                excluded_titles=excluded_titles,
                requested_limit=(
                    max(int(requested_limit), 0)
                    if requested_limit is not None
                    else max(int(limit), 0)
                ),
            )
        )
        ranking_entries = dedupe_entries(
            [*reference_entries, *recall.candidates]
        )
        ranked = rank_entries(
            ranking_entries,
            preference,
            profile_tag_weights=profile_tag_weights,
            retrieval_ranks={
                int(hit.candidate.appid): hit.retrieval_rank
                for hit in recall.hits
                if hit.candidate.appid is not None
            },
            intent=recall_intent,
            presentation_tags=self._presentation_tag_names,
        )
        ranked = exclude_previously_shown(ranked, excluded_appids, excluded_titles)
        ranked = [
            game
            for game in ranked
            if game.score_breakdown.relevance_tier != "C"
        ]
        ranked = deduplicate_game_editions(ranked, preferred_appids)
        return ranked[:limit]

    async def _recall_specific_candidates(
        self,
        preference: GamePreference,
        snapshot: SteamIndexSnapshot,
        *,
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
        requested_limit: int,
    ) -> tuple[
        CandidateRecallResult,
        list[GameCandidate],
        RecommendationIntent,
    ]:
        now = float(self.clock())
        prefetched: dict[int, GameCandidate] = {}
        records = {
            entry_key(record.candidate): record
            for record in snapshot.entries
            if entry_is_recall_usable(record)
        }
        reference_resolution = await self._resolve_reference_groups(
            preference,
            records,
            now,
            new_appid_budget=100,
            prefetched=prefetched,
        )

        record_candidates = [record.candidate for record in records.values()]
        positive_references = reference_candidates(preference, record_candidates)
        initial_intent = self._build_request_intent(preference)
        has_positive_reference_query = any(
            reference.polarity is ReferencePolarity.POSITIVE
            for reference in initial_intent.references
        )
        has_non_reference_positive_signal = bool(
            preference.company_preferences
        ) or any(
            intent_tag.role
            in {
                IntentTagRole.REQUIRED,
                IntentTagRole.ANCHOR,
                IntentTagRole.SUPPORTING,
                IntentTagRole.RECALL_ONLY,
            }
            for intent_tag in initial_intent.tags
        )
        if (
            has_positive_reference_query
            and not positive_references
            and not has_non_reference_positive_signal
        ):
            failed_references = [
                (reference, status)
                for reference, status in reference_resolution.statuses
                if reference.polarity is ReferencePolarity.POSITIVE
                and status
                in {
                    ReferenceResolutionStatus.TRANSIENT_FAILURE,
                    ReferenceResolutionStatus.CONTRACT_FAILURE,
                }
            ]
            if failed_references:
                health = RecallHealth(
                    sources=tuple(
                        RecallSourceHealth(
                            source_id=f"reference:{index}",
                            critical=True,
                            status=(
                                RecallSourceStatus.TRANSIENT_FAILURE
                                if status
                                is ReferenceResolutionStatus.TRANSIENT_FAILURE
                                else RecallSourceStatus.CONTRACT_FAILURE
                            ),
                        )
                        for index, (_reference, status) in enumerate(
                            failed_references,
                            start=1,
                        )
                    ),
                )
                raise RecallUnavailableError(health)
            return CandidateRecallResult(hits=()), [], initial_intent

        prefetched_tag_sources, reference_tag_counts, count_prefetch_degraded = (
            await self._prefetch_reference_tag_sources(
                positive_references,
                prefetched,
            )
        )
        intent = expand_intent_with_reference_tags(
            initial_intent,
            positive_references,
            tag_result_counts=reference_tag_counts,
        )
        ranked_index_candidates = rank_entries(
            record_candidates,
            preference,
            intent=intent,
            presentation_tags=self._presentation_tag_names,
        )
        seeds = select_recall_seeds(intent)
        tag_fetches = list(
            await asyncio.gather(
                *(
                    self._fetch_tag_source(
                        seed.tag,
                        prefetched,
                        prefetched_tag_sources,
                    )
                    for seed in seeds
                )
            )
        )
        weighted_tag_fetches = [
            replace(
                fetch,
                source=replace(fetch.source, weight=seed.weight),
            )
            for seed, fetch in zip(seeds, tag_fetches)
        ]
        more_like_fetches = list(
            await asyncio.gather(
                *(
                    self._fetch_more_like_source(
                        reference,
                        allow_unreleased=intent.allow_unreleased,
                    )
                    for reference in positive_references
                    if reference.appid is not None
                )
            )
        )
        intersection_fetch = await self._fetch_intersection_source(intent)
        source_fetches = [*more_like_fetches]
        if intersection_fetch is not None:
            source_fetches.append(intersection_fetch)
        source_fetches.extend(weighted_tag_fetches)
        company_fetches = list(
            await asyncio.gather(
                *(
                    self._fetch_company_source(company, source_index=index)
                    for index, company in enumerate(preference.company_preferences)
                )
            )
        )
        source_fetches.extend(company_fetches)
        company_preferences_by_source = {
            fetch.source.source_id: company
            for fetch, company in zip(company_fetches, preference.company_preferences)
        }

        has_anchor = any(
            intent_tag.role is IntentTagRole.ANCHOR
            for intent_tag in intent.tags
        )
        if (
            intent.quality_intent is QualityIntent.MAINSTREAM
            or not has_anchor
        ):
            source_fetches.append(await self._fetch_top_sellers())

        seed_appids = {
            int(item.appid)
            for item in preference.resolved_reference_games
            if item.appid is not None
            and item.confidence >= REFERENCE_MATCH_THRESHOLD
        }
        excluded = seed_appids | {int(appid) for appid in excluded_appids or []}
        filtered_fetches = [
            replace(
                fetch,
                source=filter_recall_source(fetch.source, excluded),
            )
            for fetch in source_fetches
        ]
        index_candidates = tuple(
            candidate
            for candidate in ranked_index_candidates
            if candidate.appid is not None
            and int(candidate.appid) not in excluded
        )
        index_source = RecallSource(
            source_id="index",
            source_kind="index",
            source_tag=None,
            candidates=index_candidates,
            weight=0.35,
        )
        index_health = RecallSourceHealth(
            source_id=index_source.source_id,
            critical=False,
            status=(
                RecallSourceStatus.SUCCESS
                if index_candidates
                else RecallSourceStatus.EMPTY
            ),
            candidate_count=len(index_candidates),
        )
        filtered_fetches.append(RecallSourceFetch(index_source, index_health))
        degraded = count_prefetch_degraded or any(
            fetch.health.status
            in {
                RecallSourceStatus.STALE,
                RecallSourceStatus.TRANSIENT_FAILURE,
                RecallSourceStatus.CONTRACT_FAILURE,
            }
            for fetch in filtered_fetches
        ) or any(
            fetch.source.source_kind == "tag_text"
            for fetch in filtered_fetches
        )
        source_health = tuple(fetch.health for fetch in filtered_fetches)
        merged = merge_candidate_sources(
            [fetch.source for fetch in filtered_fetches],
            seeds=seeds,
            warnings=(STEAM_TAG_RECALL_DEGRADED_WARNING,) if degraded else (),
            degraded=degraded,
            health=RecallHealth(sources=source_health),
        )
        company_validation_kwargs = (
            {"company_preferences_by_source": company_preferences_by_source}
            if company_preferences_by_source
            else {}
        )
        first_batch = await self._validate_recall_hits(
            merged.hits[:60],
            records,
            prefetched,
            **company_validation_kwargs,
        )
        verified_hits = list(first_batch.hits)
        first_eligible = self._eligible_recall_count(
            verified_hits,
            preference,
            intent,
            excluded_appids,
            excluded_titles,
        )
        validation_attempts = first_batch.attempts
        transient_failures = first_batch.transient_failures
        contract_failures = first_batch.contract_failures
        company_source_ids = set(company_preferences_by_source)
        available_company_hits = sum(
            any(source.source_id in company_source_ids for source in hit.source_hits)
            for hit in merged.hits
        )
        company_target = min(
            max(int(requested_limit), 0),
            available_company_hits,
        )
        verified_company_hits = sum(
            any(source.source_id in company_source_ids for source in hit.source_hits)
            for hit in verified_hits
        )
        if (
            (
                first_eligible < 2 * max(int(requested_limit), 0)
                or verified_company_hits < company_target
            )
            and len(merged.hits) > 60
        ):
            second_batch = await self._validate_recall_hits(
                merged.hits[60:100],
                records,
                prefetched,
                **company_validation_kwargs,
            )
            verified_hits.extend(second_batch.hits)
            validation_attempts += second_batch.attempts
            transient_failures += second_batch.transient_failures
            contract_failures += second_batch.contract_failures

        verified_hits.sort(key=lambda hit: (-hit.rrf_score, hit.retrieval_rank))
        verified_hits = [
            replace(hit, retrieval_rank=rank)
            for rank, hit in enumerate(verified_hits, start=1)
        ]
        eligible = self._eligible_recall_count(
            verified_hits,
            preference,
            intent,
            excluded_appids,
            excluded_titles,
        )
        health = RecallHealth(
            sources=source_health,
            validation_attempts=validation_attempts,
            validation_transient_failures=transient_failures,
            validation_contract_failures=contract_failures,
            verified=len(verified_hits),
            eligible=eligible,
        )
        if health.unavailable(limit=requested_limit):
            raise RecallUnavailableError(health)
        verified = CandidateRecallResult(
            hits=tuple(verified_hits),
            seeds=merged.seeds,
            warnings=merged.warnings,
            degraded=merged.degraded,
            health=health,
        )

        if degraded and STEAM_TAG_RECALL_DEGRADED_WARNING not in preference.parse_warnings:
            preference.parse_warnings.append(STEAM_TAG_RECALL_DEGRADED_WARNING)

        for hit in verified.hits:
            records[entry_key(hit.candidate)] = SteamIndexEntry(hit.candidate, now)
        persisted = await self._merge_and_persist_snapshot(
            records.values(),
            snapshot.search_coverage,
        )
        reference_entries = [
            record.candidate
            for record in persisted.entries
            if record.candidate.appid in seed_appids
            and entry_is_validated(record)
        ]
        logger.debug(
            "recommendation_recall event=recall_complete tags=%s tag_ids=%s "
            "sources=%s verified_count=%d degraded=%s",
            [seed.tag for seed in seeds],
            [self._usable_steam_tag_id_for(seed.tag) for seed in seeds],
            {
                f"{kind}:{tag or '-'}": len(candidates)
                for kind, tag, candidates in (
                    (
                        fetch.source.source_kind,
                        fetch.source.source_tag,
                        fetch.source.candidates,
                    )
                    for fetch in filtered_fetches
                )
            },
            len(verified.hits),
            degraded,
        )
        return verified, reference_entries, intent

    async def _fetch_tag_source(
        self,
        tag: str,
        prefetched: dict[int, GameCandidate],
        prefetched_tag_sources: dict[str, RecallSourceFetch] | None = None,
    ) -> RecallSourceFetch:
        if prefetched_tag_sources is not None and tag in prefetched_tag_sources:
            return prefetched_tag_sources[tag]

        tag_id = self._usable_steam_tag_id_for(tag)
        source_id = f"tag:{tag}"
        if tag_id is not None:
            search_storefront = getattr(
                self.steam_client,
                "search_storefront_tag",
                None,
            )
            if search_storefront is None:
                return failed_source_fetch(
                    source_id,
                    "tag",
                    tag,
                    1.0,
                    SteamApiError("Steam storefront tag search is unavailable."),
                )
            try:
                page = await self._shared_storefront_tag_page(
                    tag_id,
                    search_storefront,
                    page_size=40,
                )
            except Exception as exc:
                # A known tag ID must never degrade into a title query.
                return failed_source_fetch(
                    source_id,
                    "tag",
                    tag,
                    1.0,
                    exc,
                )
            candidates = tuple(
                self._candidate_from_storefront_hit(hit)
                for hit in page.hits[:40]
            )
            return successful_source_fetch(
                RecallSource(
                    source_id=source_id,
                    source_kind="tag",
                    source_tag=tag,
                    candidates=candidates,
                    weight=1.0,
                ),
                stale=bool(getattr(page, "stale", False)),
                total_count=int(page.total_count),
            )

        query = tag.replace("_", " ")
        try:
            hits = await self._search_refs(
                query,
                page_size=40,
                prefetched=prefetched,
            )
        except Exception as exc:
            return failed_source_fetch(
                source_id,
                "tag_text",
                tag,
                1.0,
                exc,
            )
        candidates = tuple(candidate_from_search_hit(hit) for hit in hits[:40])
        return successful_source_fetch(
            RecallSource(
                source_id=source_id,
                source_kind="tag_text",
                source_tag=tag,
                candidates=candidates,
                weight=1.0,
            ),
            total_count=len(candidates),
        )

    def _build_request_intent(self, preference: GamePreference) -> RecommendationIntent:
        if not self._steam_tag_aliases_are_usable_stale():
            return build_recommendation_intent(preference)
        vocabulary = frozenset({*static_canonical_tags(), *self._steam_tag_ids})
        return build_recommendation_intent(
            preference,
            known_tags=vocabulary,
            known_tag_aliases=self._steam_tag_aliases,
        )

    def _has_filtered_explicit_tag(
        self,
        preference: GamePreference,
        request_intent: RecommendationIntent,
    ) -> bool:
        if not self._steam_tag_aliases_are_usable_stale():
            return False
        accepted = {item.tag for item in request_intent.tags}
        cold_intent = build_recommendation_intent(preference)
        return any(
            item.source == IntentTagSource.EXPLICIT and item.tag not in accepted
            for item in cold_intent.tags
        )

    async def _prefetch_reference_tag_sources(
        self,
        references: list[GameCandidate],
        prefetched: dict[int, GameCandidate],
    ) -> tuple[
        dict[str, RecallSourceFetch],
        dict[str, int],
        bool,
    ]:
        tags: list[str] = []
        seen: set[str] = set()
        degraded = False
        for reference in references:
            ordered = canonical_tags_from_terms(
                reference.ordered_tags[:REFERENCE_TAG_COUNT_PREFETCH_PER_GAME]
            )
            if not ordered:
                degraded = True
            for tag in ordered:
                if tag in seen:
                    continue
                seen.add(tag)
                tags.append(tag)
        if not tags:
            return {}, {}, degraded

        fetches = await asyncio.gather(
            *(self._fetch_tag_source(tag, prefetched) for tag in tags)
        )
        source_by_tag = dict(zip(tags, fetches))
        counts_by_tag = {
            tag: fetch.total_count
            for tag, fetch in zip(tags, fetches)
            if fetch.health.status
            not in {
                RecallSourceStatus.TRANSIENT_FAILURE,
                RecallSourceStatus.CONTRACT_FAILURE,
            }
        }
        return (
            source_by_tag,
            counts_by_tag,
            degraded
            or any(
                fetch.health.status
                in {
                    RecallSourceStatus.STALE,
                    RecallSourceStatus.TRANSIENT_FAILURE,
                    RecallSourceStatus.CONTRACT_FAILURE,
                }
                for fetch in fetches
            ),
        )

    async def _shared_storefront_tag_page(
        self,
        tag_id: int,
        search_storefront: Callable[..., Any],
        page_size: int = 20,
    ) -> Any:
        key = (int(tag_id), int(page_size))
        async with self._storefront_tag_tasks_lock:
            task = self._storefront_tag_tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._run_storefront_tag_task(
                        tag_id,
                        search_storefront,
                        page_size,
                    )
                )
                self._storefront_tag_tasks[key] = task
        return await asyncio.shield(task)

    async def _run_storefront_tag_task(
        self,
        tag_id: int,
        search_storefront: Callable[..., Any],
        page_size: int,
    ) -> Any:
        task = asyncio.current_task()
        key = (int(tag_id), int(page_size))
        try:
            return await self._request_storefront_tag_page(
                tag_id,
                search_storefront,
                page_size,
            )
        finally:
            async with self._storefront_tag_tasks_lock:
                if self._storefront_tag_tasks.get(key) is task:
                    self._storefront_tag_tasks.pop(key, None)

    async def _request_storefront_tag_page(
        self,
        tag_id: int,
        search_storefront: Callable[..., Any],
        page_size: int,
    ) -> Any:
        async with self._steam_semaphore:
            return await search_storefront(
                tag_id,
                page_size=page_size,
                **self._discovery_cache_kwargs(search_storefront),
            )

    async def _fetch_company_source(
        self,
        preference: CompanyPreference,
        *,
        source_index: int,
    ) -> RecallSourceFetch:
        source_id = f"company:{int(source_index)}"
        search = getattr(self.steam_client, "search_storefront_company", None)
        if search is None:
            return failed_source_fetch(
                source_id,
                "company",
                None,
                1.0,
                SteamApiError("Steam storefront company search is unavailable."),
            )
        aliases = dedupe_texts(
            [preference.display_name, *preference.aliases]
        )[:COMPANY_ALIAS_LIMIT]
        roles = (
            ("developer", "publisher")
            if preference.role == "either"
            else (preference.role,)
        )
        requests = [(alias, role) for alias in aliases for role in roles]

        async def search_one(alias: str, role: str) -> Any:
            async with self._steam_semaphore:
                return await search(
                    alias,
                    role,
                    page_size=20,
                    **self._discovery_cache_kwargs(search),
                )

        results = await asyncio.gather(
            *(search_one(alias, role) for alias, role in requests),
            return_exceptions=True,
        )
        pages = [item for item in results if not isinstance(item, BaseException)]
        failures = [item for item in results if isinstance(item, BaseException)]
        for failure in failures:
            if not isinstance(failure, Exception):
                raise failure
            failure_status(failure)
        if not pages:
            error = (
                failures[0]
                if failures
                else SteamApiError("company search returned no result")
            )
            return failed_source_fetch(source_id, "company", None, 1.0, error)

        by_appid: dict[int, tuple[GameCandidate, int, tuple[int, int]]] = {}
        for request_index, page in enumerate(pages):
            for rank, hit in enumerate(page.hits[:20], start=1):
                candidate = self._candidate_from_storefront_hit(hit)
                appid = int(candidate.appid or 0)
                if appid <= 0:
                    continue
                current = by_appid.get(appid)
                if current is None:
                    by_appid[appid] = (candidate, rank, (request_index, rank))
                else:
                    merged = merge_recall_candidate_evidence(current[0], candidate)
                    best_rank = min(current[1], rank)
                    best_seen = (
                        (request_index, rank)
                        if rank < current[1]
                        else min(current[2], (request_index, rank))
                        if rank == current[1]
                        else current[2]
                    )
                    by_appid[appid] = (
                        merged,
                        best_rank,
                        best_seen,
                    )
        ordered = sorted(
            by_appid.values(),
            key=lambda item: (item[1], item[2], int(item[0].appid or 0)),
        )
        source = RecallSource(
            source_id=source_id,
            source_kind="company",
            source_tag=None,
            candidates=tuple(item[0] for item in ordered),
            weight=1.0,
            candidate_ranks=tuple(item[1] for item in ordered),
        )
        return successful_source_fetch(
            source,
            stale=bool(failures) or any(bool(getattr(page, "stale", False)) for page in pages),
            total_count=max((int(getattr(page, "total_count", 0)) for page in pages), default=0),
        )

    async def _fetch_top_sellers(
        self,
    ) -> RecallSourceFetch:
        source_id = "top_seller"
        browser = getattr(self.steam_client, "browse_top_sellers", None)
        if browser is None:
            return failed_source_fetch(
                source_id,
                "top_seller",
                None,
                0.5,
                SteamApiError("Steam top-seller browsing is unavailable."),
            )
        try:
            async with self._steam_semaphore:
                page = await browser(
                    page_size=60,
                    **self._discovery_cache_kwargs(browser),
                )
            candidates = tuple(
                self._candidate_from_storefront_hit(hit)
                for hit in page.hits[:60]
            )
        except Exception as exc:
            return failed_source_fetch(
                source_id,
                "top_seller",
                None,
                0.5,
                exc,
            )
        return successful_source_fetch(
            RecallSource(
                source_id=source_id,
                source_kind="top_seller",
                source_tag=None,
                candidates=candidates,
                weight=0.5,
            ),
            stale=bool(getattr(page, "stale", False)),
            total_count=int(page.total_count),
        )

    async def _fetch_intersection_source(
        self,
        intent: RecommendationIntent,
    ) -> RecallSourceFetch | None:
        components: list[tuple[str, int]] = []
        seen_ids: set[int] = set()
        for role in (IntentTagRole.REQUIRED, IntentTagRole.ANCHOR):
            for intent_tag in intent.tags:
                if intent_tag.role is not role:
                    continue
                tag_id = self._usable_steam_tag_id_for(intent_tag.tag)
                if tag_id is None or tag_id in seen_ids:
                    continue
                components.append((intent_tag.tag, tag_id))
                seen_ids.add(tag_id)
                if len(components) == 2:
                    break
            if len(components) == 2:
                break
        if len(components) < 2:
            return None

        component_tags = tuple(tag for tag, _tag_id in components)
        tag_ids = tuple(tag_id for _tag, tag_id in components)
        source_id = f"intersection:{tag_ids[0]},{tag_ids[1]}"
        search = getattr(self.steam_client, "search_storefront_tags", None)
        if search is None:
            return failed_source_fetch(
                source_id,
                "intersection",
                None,
                1.3,
                SteamApiError("Steam tag intersection search is unavailable."),
                component_tags=component_tags,
            )
        try:
            page = await self._shared_intersection_page(
                tag_ids,
                search,
                page_size=40,
            )
            candidates = tuple(
                self._candidate_from_storefront_hit(hit)
                for hit in page.hits[:40]
            )
        except Exception as exc:
            return failed_source_fetch(
                source_id,
                "intersection",
                None,
                1.3,
                exc,
                component_tags=component_tags,
            )
        return successful_source_fetch(
            RecallSource(
                source_id=source_id,
                source_kind="intersection",
                source_tag=None,
                candidates=candidates,
                weight=1.3,
                component_tags=component_tags,
            ),
            stale=bool(getattr(page, "stale", False)),
            total_count=int(page.total_count),
        )

    async def _shared_intersection_page(
        self,
        tag_ids: tuple[int, int],
        search: Callable[..., Any],
        *,
        page_size: int,
    ) -> Any:
        key = (tag_ids, int(page_size))
        async with self._intersection_tasks_lock:
            task = self._intersection_tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._run_intersection_task(key, search)
                )
                self._intersection_tasks[key] = task
        return await asyncio.shield(task)

    async def _run_intersection_task(
        self,
        key: tuple[tuple[int, int], int],
        search: Callable[..., Any],
    ) -> Any:
        task = asyncio.current_task()
        tag_ids, page_size = key
        try:
            async with self._steam_semaphore:
                return await search(
                    list(tag_ids),
                    page_size=page_size,
                    **self._discovery_cache_kwargs(search),
                )
        finally:
            async with self._intersection_tasks_lock:
                if self._intersection_tasks.get(key) is task:
                    self._intersection_tasks.pop(key, None)

    async def _fetch_more_like_source(
        self,
        reference: GameCandidate,
        *,
        allow_unreleased: bool,
    ) -> RecallSourceFetch:
        appid = int(reference.appid or 0)
        source_id = f"more_like:{appid}"
        getter = getattr(self.steam_client, "get_more_like", None)
        if getter is None:
            return failed_source_fetch(
                source_id,
                "more_like",
                None,
                1.2,
                SteamApiError("Steam More Like This is unavailable."),
            )
        try:
            page = await self._shared_more_like_page(
                appid,
                allow_unreleased,
                getter,
            )
            candidates = tuple(
                self._candidate_from_storefront_hit(hit)
                for hit in page.hits
                if hit.appid != appid
            )
        except Exception as exc:
            return failed_source_fetch(
                source_id,
                "more_like",
                None,
                1.2,
                exc,
            )
        return successful_source_fetch(
            RecallSource(
                source_id=source_id,
                source_kind="more_like",
                source_tag=None,
                candidates=candidates,
                weight=1.2,
            ),
            stale=bool(getattr(page, "stale", False)),
            total_count=len(candidates),
        )

    async def _shared_more_like_page(
        self,
        appid: int,
        allow_unreleased: bool,
        getter: Callable[..., Any],
    ) -> Any:
        key = (int(appid), bool(allow_unreleased))
        async with self._more_like_tasks_lock:
            task = self._more_like_tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._run_more_like_task(key, getter)
                )
                self._more_like_tasks[key] = task
        return await asyncio.shield(task)

    async def _run_more_like_task(
        self,
        key: tuple[int, bool],
        getter: Callable[..., Any],
    ) -> Any:
        task = asyncio.current_task()
        appid, allow_unreleased = key
        try:
            async with self._steam_semaphore:
                return await getter(
                    appid,
                    allow_unreleased=allow_unreleased,
                    **self._discovery_cache_kwargs(getter),
                )
        finally:
            async with self._more_like_tasks_lock:
                if self._more_like_tasks.get(key) is task:
                    self._more_like_tasks.pop(key, None)

    def _candidate_from_storefront_hit(
        self,
        hit: SteamSearchHit,
    ) -> GameCandidate:
        candidate = candidate_from_search_hit(hit)
        ordered = dedupe_texts(
            [
                canonical
                for tag_id in hit.tag_ids
                if (canonical := self._canonical_tag_by_id.get(int(tag_id)))
            ]
        )
        if not ordered:
            return candidate
        data = dump_model(candidate)
        data["ordered_tags"] = ordered
        return validate_candidate(data)

    async def _validate_recall_hits(
        self,
        hits: tuple[CandidateHit, ...],
        records: dict[str, SteamIndexEntry],
        prefetched: dict[int, GameCandidate],
        *,
        company_preferences_by_source: Mapping[str, CompanyPreference] | None = None,
    ) -> RecallValidationBatch:
        company_preferences_by_source = company_preferences_by_source or {}
        now = float(self.clock())
        records_by_appid = {
            int(record.candidate.appid): record
            for record in records.values()
            if record.candidate.appid is not None
        }
        existing = {
            int(record.candidate.appid): record.candidate
            for record in records.values()
            if record.candidate.appid is not None
            and entry_is_current(record, now)
        }
        legacy_appids = {
            int(record.candidate.appid)
            for record in records.values()
            if record.candidate.appid is not None and record.needs_revalidation
        }

        async def validate(
            hit: CandidateHit,
        ) -> tuple[CandidateHit | None, RecallSourceStatus | None]:
            appid = int(hit.candidate.appid or 0)
            local_record = records_by_appid.get(appid)
            candidate = existing.get(appid)
            validation_status: RecallSourceStatus | None = None
            force_release_refresh = (
                local_record is not None
                and entry_is_validated(local_record)
                and not entry_release_status_is_fresh(local_record, now)
            )
            company_hits = [
                source
                for source in hit.source_hits
                if source.source_id in company_preferences_by_source
            ]
            force_company_refresh = bool(company_hits) and (
                candidate is None
                or "steam_appdetails" not in candidate.internal_source_markers
                or any(
                    company_match_status(
                        candidate,
                        company_preferences_by_source[source.source_id],
                    )
                    is CompanyMatchStatus.UNKNOWN
                    for source in company_hits
                )
            )
            if force_company_refresh:
                candidate = None
            if candidate is None:
                try:
                    candidate = await self._fetch_candidate(
                        appid,
                        prefetched,
                        bypass_cache=(
                            appid in legacy_appids
                            or force_company_refresh
                            or force_release_refresh
                        ),
                    )
                except Exception as exc:
                    status = failure_status(exc)
                    if (
                        status is RecallSourceStatus.TRANSIENT_FAILURE
                        and local_record is not None
                        and entry_release_status_is_usable_stale(local_record, now)
                    ):
                        candidate = local_record.candidate
                        validation_status = status
                    else:
                        return None, status
                else:
                    if candidate is not None:
                        refreshed_record = SteamIndexEntry(candidate, now)
                        if (
                            not entry_release_status_is_fresh(refreshed_record, now)
                            and not entry_release_status_is_usable_stale(
                                refreshed_record,
                                now,
                            )
                        ):
                            return None, RecallSourceStatus.TRANSIENT_FAILURE
                        if not entry_release_status_is_fresh(refreshed_record, now):
                            validation_status = RecallSourceStatus.TRANSIENT_FAILURE
                        records[entry_key(candidate)] = refreshed_record
            if candidate is None or not is_confirmed_base_game(candidate):
                return None, None
            retained_sources = tuple(
                source
                for source in hit.source_hits
                if source.source_id not in company_preferences_by_source
                or matches_company_preference(
                    candidate,
                    company_preferences_by_source[source.source_id],
                )
            )
            if not retained_sources:
                return None, None
            removed_company_contribution = sum(
                source.source_weight / (RRF_K + source.source_rank)
                for source in hit.source_hits
                if source not in retained_sources and source.source_kind == "company"
            )
            candidate = merge_recall_candidate_evidence(candidate, hit.candidate)
            return (
                replace(
                    hit,
                    candidate=candidate,
                    source_hits=retained_sources,
                    rrf_score=max(hit.rrf_score - removed_company_contribution, 0.0),
                ),
                validation_status,
            )

        validated = await asyncio.gather(*(validate(hit) for hit in hits))
        valid_hits = [hit for hit, _failure in validated if hit is not None]
        return RecallValidationBatch(
            hits=tuple(valid_hits),
            attempts=len(hits),
            transient_failures=sum(
                failure is RecallSourceStatus.TRANSIENT_FAILURE
                for _hit, failure in validated
            ),
            contract_failures=sum(
                failure is RecallSourceStatus.CONTRACT_FAILURE
                for _hit, failure in validated
            ),
        )

    def _eligible_recall_count(
        self,
        hits: list[CandidateHit],
        preference: GamePreference,
        intent: RecommendationIntent,
        excluded_appids: list[int] | None,
        excluded_titles: list[str] | None,
    ) -> int:
        ranked = rank_entries(
            [hit.candidate for hit in hits],
            preference,
            retrieval_ranks={
                int(hit.candidate.appid): hit.retrieval_rank
                for hit in hits
                if hit.candidate.appid is not None
            },
            intent=intent,
            presentation_tags=self._presentation_tag_names,
        )
        ranked = exclude_previously_shown(
            ranked,
            excluded_appids,
            excluded_titles,
        )
        return sum(
            game.score_breakdown.relevance_tier != "C"
            for game in ranked
        )

    async def ensure_steam_tag_aliases(self) -> bool:
        getter = getattr(self.steam_client, "get_popular_tags_snapshot", None)
        if getter is None:
            getter = getattr(self.steam_client, "get_popular_tags", None)
        if getter is not None:
            self._tag_vocabulary_languages = (
                ("english", "schinese")
                if callable_accepts_keyword(getter, "language")
                else ("english",)
            )
        if self._steam_tag_aliases_are_fresh():
            return self._steam_tag_aliases_are_usable_stale()
        if not getter:
            usable = self._steam_tag_aliases_are_usable_stale()
            if not usable:
                self._expire_tag_vocabulary()
            return usable

        async with self._tag_aliases_lock:
            if self._steam_tag_aliases_are_fresh():
                return True
            if self._tag_aliases_task is None:
                self._tag_aliases_task = asyncio.create_task(
                    self._run_tag_alias_load(getter)
                )
            task = self._tag_aliases_task
        return await asyncio.shield(task)

    async def _run_tag_alias_load(self, getter: Callable[..., Any]) -> bool:
        task = asyncio.current_task()
        try:
            return await self._load_steam_tag_aliases(getter)
        finally:
            async with self._tag_aliases_lock:
                if self._tag_aliases_task is task:
                    self._tag_aliases_task = None

    async def _load_steam_tag_aliases(self, getter: Callable[..., Any]) -> bool:
        languages = self._tag_vocabulary_languages_needing_refresh()
        if not languages:
            return self._steam_tag_aliases_are_usable_stale()
        try:
            payloads = await self._request_tag_vocabularies(getter, languages)
        except Exception as exc:
            failure_status(exc)
            self._expire_tag_vocabulary()
            usable = self._steam_tag_aliases_are_usable_stale()
            return usable

        for language, payload in payloads:
            tags = tuple(
                item
                for item in list(getattr(payload, "tags", payload) or [])
                if isinstance(item, dict)
            )
            fetched_at = float(getattr(payload, "fetched_at", self.clock()))
            if tags and max(
                float(self.clock())
                - fetched_at,
                0.0,
            ) <= STEAM_TAG_ALIASES_STALE_SECONDS:
                self._tag_vocabulary_payloads[language] = (tags, fetched_at)

        self._expire_tag_vocabulary()
        return self._steam_tag_aliases_are_usable_stale()

    def _rebuild_tag_vocabulary(self) -> None:
        english = self._tag_vocabulary_payloads.get("english")
        if english is None:
            self._steam_tag_ids.clear()
            self._canonical_tag_by_id.clear()
            self._steam_tag_aliases.clear()
            self._presentation_tag_names.clear()
            return

        canonical_by_id: dict[int, str] = {}
        for item in english[0]:
            tag_id = item.get("tagid", item.get("id"))
            if tag_id is None:
                continue
            canonical_by_id[int(tag_id)] = canonical_steam_tag_name(
                str(item.get("name") or "")
            )

        tags: list[dict[str, Any]] = []
        for language in ("english", "schinese"):
            vocabulary = self._tag_vocabulary_payloads.get(language)
            if vocabulary is None:
                continue
            for item in vocabulary[0]:
                tag_id = item.get("tagid", item.get("id"))
                if tag_id is None:
                    continue
                resolved_id = int(tag_id)
                canonical = canonical_by_id.get(resolved_id)
                if canonical is None:
                    continue
                tags.append({**item, "canonical": canonical})

        register_steam_tag_aliases(tags, register_ids=False)
        self._steam_tag_aliases = {
            normalize_key(str(item.get("name") or "")): str(item["canonical"])
            for item in tags
            if normalize_key(str(item.get("name") or ""))
        }
        self._steam_tag_ids = {
            canonical: tag_id for tag_id, canonical in canonical_by_id.items()
        }
        self._canonical_tag_by_id = dict(canonical_by_id)
        schinese = self._tag_vocabulary_payloads.get("schinese")
        self._presentation_tag_names = build_tag_presentations(
            english[0],
            schinese[0] if schinese is not None else (),
        )

    async def _request_tag_vocabularies(
        self,
        getter: Callable[..., Any],
        languages: tuple[str, ...],
    ) -> list[tuple[str, Any]]:
        if not callable_accepts_keyword(getter, "language"):
            async with self._steam_semaphore:
                return [("english", await getter())]

        async def fetch(language: str) -> tuple[str, Any]:
            async with self._steam_semaphore:
                return language, await getter(language=language)

        results = await asyncio.gather(
            *(fetch(language) for language in languages),
            return_exceptions=True,
        )
        payloads = [item for item in results if not isinstance(item, BaseException)]
        if not payloads:
            raise RuntimeError("Steam tag vocabularies are unavailable.")
        return payloads

    def _usable_steam_tag_id_for(self, tag: str) -> int | None:
        if not self._steam_tag_aliases_are_usable_stale():
            return None
        canonical = (
            self._steam_tag_aliases.get(normalize_key(tag))
            or normalize_tag(tag)
            or canonical_steam_tag_name(tag)
        )
        return self._steam_tag_ids.get(canonical)

    def _expire_tag_vocabulary(self) -> None:
        expired = [
            language
            for language in self._tag_vocabulary_payloads
            if self._tag_vocabulary_age(language) > STEAM_TAG_ALIASES_STALE_SECONDS
        ]
        for language in expired:
            self._tag_vocabulary_payloads.pop(language, None)
        self._rebuild_tag_vocabulary()

    def _steam_tag_aliases_are_fresh(self) -> bool:
        return bool(self._tag_vocabulary_languages) and all(
            self._tag_vocabulary_age(language) < STEAM_TAG_ALIASES_FRESH_SECONDS
            for language in self._tag_vocabulary_languages
        )

    def _steam_tag_aliases_are_usable_stale(self) -> bool:
        return (
            self._tag_vocabulary_age("english") <= STEAM_TAG_ALIASES_STALE_SECONDS
            and bool(self._canonical_tag_by_id)
        )

    def _tag_vocabulary_languages_needing_refresh(self) -> tuple[str, ...]:
        return tuple(
            language
            for language in self._tag_vocabulary_languages
            if self._tag_vocabulary_age(language) >= STEAM_TAG_ALIASES_FRESH_SECONDS
        )

    def _tag_vocabulary_age(self, language: str) -> float:
        vocabulary = self._tag_vocabulary_payloads.get(language)
        if vocabulary is None:
            return float("inf")
        return max(float(self.clock()) - vocabulary[1], 0.0)

    async def bootstrap(self) -> None:
        async with self._bootstrap_lock:
            if self._bootstrap_task is None:
                self._bootstrap_task = asyncio.create_task(self._run_bootstrap())
            task = self._bootstrap_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._bootstrap_lock:
                if self._bootstrap_task is task and task.done():
                    self._bootstrap_task = None
            raise

    async def _run_bootstrap(self) -> None:
        await asyncio.gather(
            self.load_snapshot(),
            self.ensure_steam_tag_aliases(),
        )

    async def load_snapshot(self) -> SteamIndexSnapshot:
        payload = await self.cache.get_json(
            STEAM_INDEX_CACHE_KEY,
            SNAPSHOT_STORAGE_TTL_HOURS,
        )
        if is_current_snapshot_payload(payload):
            return parse_snapshot(payload)
        if is_previous_snapshot_payload(payload):
            migrated = parse_snapshot_records(
                payload,
                needs_revalidation=True,
                preserve_coverage=False,
            )
            await self.cache.set_json(
                STEAM_INDEX_CACHE_KEY,
                snapshot_payload(migrated),
            )
            return migrated

        for legacy_key in STEAM_INDEX_LEGACY_CACHE_KEYS:
            legacy_payload = await self.cache.get_json(
                legacy_key,
                SNAPSHOT_STORAGE_TTL_HOURS,
            )
            if not is_legacy_snapshot_payload(legacy_payload):
                continue
            migrated = parse_legacy_snapshot(legacy_payload)
            await self.cache.set_json(
                STEAM_INDEX_CACHE_KEY,
                snapshot_payload(migrated),
            )
            return migrated
        return SteamIndexSnapshot()

    async def load_entries(self) -> list[GameCandidate]:
        await self.bootstrap()
        snapshot = await self.load_snapshot()
        return [record.candidate for record in snapshot.entries]

    async def refresh_entries(
        self,
        preference: GamePreference,
        existing: list[GameCandidate] | None = None,
        target_pool: int = STEAM_INDEX_MAX_NEW_APPIDS_PER_ROUND,
        snapshot: SteamIndexSnapshot | None = None,
    ) -> list[GameCandidate]:
        refresh_started = time.perf_counter()
        await self.bootstrap()
        await self.ensure_steam_tag_aliases()
        now = float(self.clock())
        current = snapshot or await self.load_snapshot()
        records = {
            entry_key(record.candidate): record
            for record in current.entries
            if entry_is_recall_usable(record)
        }
        for candidate in existing or []:
            if not is_confirmed_base_game(candidate):
                continue
            key = entry_key(candidate)
            if key not in records:
                records[key] = SteamIndexEntry(candidate=candidate, refreshed_at=now)

        prefetched: dict[int, GameCandidate] = {}
        coverage = dict(current.search_coverage)
        seen_hits: set[int] = set()
        enrichment_limit = min(
            STEAM_INDEX_MAX_NEW_APPIDS_PER_ROUND,
            max(int(target_pool), 0),
        )
        reference_resolution = await self._resolve_reference_groups(
            preference,
            records,
            now,
            new_appid_budget=enrichment_limit,
            prefetched=prefetched,
        )
        enriched_count = reference_resolution.enriched_count
        searched_queries: list[str] = []

        candidates = [record.candidate for record in records.values()]
        initial_profile = build_profile_from_preference(
            preference,
            reference_candidates=reference_candidates(preference, candidates),
            negative_reference_candidates=negative_reference_candidates(
                preference,
                candidates,
            ),
        )
        if enriched_count < enrichment_limit:
            initial_queries = [
                query
                for query in search_terms_for(preference, initial_profile)
                if not self.reuse_cache
                or not query_is_covered(query, coverage, now, self.ttl_hours)
            ][:STEAM_INDEX_MAX_SEARCHES_PER_ROUND]
        else:
            initial_queries = []

        async def process_searches(queries: list[str]) -> None:
            nonlocal enriched_count
            if not queries:
                return
            searched_queries.extend(queries)
            results = await self._search_queries(queries, prefetched)
            observed_hits: list[SteamSearchHit] = []
            for query, hits, succeeded in results:
                if succeeded:
                    coverage[normalize_text(query)] = now
                observed_hits.extend(hits)

            existing_appids = {
                int(record.candidate.appid)
                for record in records.values()
                if record.candidate.appid is not None
                and entry_is_current(record, now)
            }
            revalidation_appids = {
                int(record.candidate.appid)
                for record in records.values()
                if record.candidate.appid is not None and (
                    record.needs_revalidation
                    or not entry_release_status_is_fresh(record, now)
                )
            }
            new_hits: list[SteamSearchHit] = []
            for hit in observed_hits:
                if hit.appid in seen_hits or hit.appid in existing_appids:
                    continue
                seen_hits.add(hit.appid)
                new_hits.append(hit)
            remaining = max(enrichment_limit - enriched_count, 0)
            enriched = await self._enrich_hits(
                new_hits[:remaining],
                prefetched,
                bypass_cache_appids=revalidation_appids,
            )
            enriched_count += len(enriched)
            for candidate in enriched:
                records[entry_key(candidate)] = SteamIndexEntry(candidate, now)

        await process_searches(initial_queries)

        remaining_searches = STEAM_INDEX_MAX_SEARCHES_PER_ROUND - len(searched_queries)
        if remaining_searches > 0 and enriched_count < enrichment_limit:
            candidates = [record.candidate for record in records.values()]
            expanded_profile = build_profile_from_preference(
                preference,
                reference_candidates=reference_candidates(preference, candidates),
                negative_reference_candidates=negative_reference_candidates(
                    preference,
                    candidates,
                ),
            )
            searched_keys = {normalize_text(query) for query in searched_queries}
            supplemental_queries = [
                query
                for query in search_terms_for(preference, expanded_profile)
                if normalize_text(query) not in searched_keys
                and (
                    not self.reuse_cache
                    or not query_is_covered(query, coverage, now, self.ttl_hours)
                )
            ][:remaining_searches]
            await process_searches(supplemental_queries)

        refreshed_snapshot = await self._merge_and_persist_snapshot(
            records.values(),
            coverage,
        )
        logger.debug(
            "Steam index refresh: elapsed_ms=%.1f searches=%d enriched=%d "
            "candidates=%d coverage=%d",
            (time.perf_counter() - refresh_started) * 1000,
            len(searched_queries),
            enriched_count,
            len(refreshed_snapshot.entries),
            len(refreshed_snapshot.search_coverage),
        )
        return [
            record.candidate
            for record in refreshed_snapshot.entries
            if entry_is_validated(record)
        ]

    async def _resolve_reference_groups(
        self,
        preference: GamePreference,
        records: dict[str, SteamIndexEntry],
        now: float,
        new_appid_budget: int,
        prefetched: dict[int, GameCandidate],
    ) -> ReferenceResolutionBatch:
        references = self._build_request_intent(preference).references
        available_appids = {
            int(record.candidate.appid)
            for record in records.values()
            if record.candidate.appid is not None
        }
        prune_reference_resolution_state(
            preference,
            references,
            available_appids,
        )
        enriched_count = 0
        statuses: list[tuple[ReferenceQuery, ReferenceResolutionStatus]] = []
        for reference in references:
            search_outcome = ReferenceSearchOutcome()
            records_by_appid = {
                int(record.candidate.appid): record
                for record in records.values()
                if record.candidate.appid is not None
            }
            existing_hits = [
                SteamSearchHit(
                    appid=appid,
                    title=record.candidate.title,
                    store_url=record.candidate.raw_url,
                )
                for appid, record in records_by_appid.items()
            ]
            match = match_reference_query(reference, existing_hits)
            matched_record = (
                records_by_appid.get(match.hit.appid) if match is not None else None
            )
            candidate = (
                matched_record.candidate
                if matched_record is not None and entry_is_current(matched_record, now)
                else None
            )
            refreshed_at = (
                matched_record.refreshed_at
                if matched_record is not None
                else now
            )
            if (
                not self.reuse_cache
                or match is None
                or match.match_kind != "exact"
                or candidate is None
            ):
                search_outcome = await self._search_reference_group(
                    reference,
                    prefetched,
                )
                observed_hits = list(search_outcome.hits)
                observed_match = match_reference_query(
                    reference,
                    [*existing_hits, *observed_hits],
                )
                existing_record = (
                    records_by_appid.get(observed_match.hit.appid)
                    if observed_match is not None
                    else None
                )
                if existing_record is not None and entry_is_current(existing_record, now):
                    match = observed_match
                    evidence_hit = max(
                        (
                            hit
                            for hit in observed_hits
                            if hit.appid == observed_match.hit.appid
                        ),
                        key=lambda hit: len(hit.tag_ids),
                        default=observed_match.hit,
                    )
                    candidate = self._apply_search_hit_tags(
                        existing_record.candidate,
                        evidence_hit,
                    )
                    refreshed_at = existing_record.refreshed_at
                elif observed_match is None:
                    match = None
                    candidate = None
                elif enriched_count >= new_appid_budget:
                    log_deferred_reference_group(reference, observed_match)
                    if match is None or candidate is None:
                        status = classify_reference_resolution(
                            search_outcome,
                            None,
                            None,
                        )
                        statuses.append((reference, status))
                        continue
                else:
                    selected_match, selected_candidate, validation_failures = (
                        await self._select_reference_candidate(
                            reference,
                            [*existing_hits, *observed_hits],
                            prefetched,
                            known_candidates={
                                appid: record.candidate
                                for appid, record in records_by_appid.items()
                                if entry_is_current(record, now)
                            },
                            bypass_cache_appids={
                                appid
                                for appid, record in records_by_appid.items()
                                if record.needs_revalidation
                                or not entry_release_status_is_fresh(record, now)
                            },
                        )
                    )
                    if validation_failures:
                        search_outcome = replace(
                            search_outcome,
                            validation_failures=(
                                *search_outcome.validation_failures,
                                *validation_failures,
                            ),
                        )
                    if selected_match is not None and selected_candidate is not None:
                        match = selected_match
                        candidate = selected_candidate
                        selected_record = records_by_appid.get(
                            selected_match.hit.appid
                        )
                        if selected_record is None:
                            enriched_count += 1
                            refreshed_at = now
                        elif (
                            selected_record.needs_revalidation
                            or not entry_release_status_is_fresh(selected_record, now)
                        ):
                            refreshed_at = now
                        else:
                            refreshed_at = selected_record.refreshed_at
                    else:
                        match = None
                        candidate = None

            polarity = reference_polarity(reference)
            if match is not None and candidate is not None:
                candidate = mark_reference_query(
                    candidate,
                    reference.display_title,
                    polarity,
                )
                records[entry_key(candidate)] = SteamIndexEntry(candidate, refreshed_at)
            status = classify_reference_resolution(
                search_outcome,
                match,
                candidate,
            )
            record_reference_group_resolution(
                preference,
                reference,
                match,
                candidate,
                status,
            )
            statuses.append((reference, status))
        return ReferenceResolutionBatch(
            enriched_count=enriched_count,
            statuses=tuple(statuses),
        )

    async def _search_reference_group(
        self,
        reference: ReferenceQuery,
        prefetched: dict[int, GameCandidate],
    ) -> ReferenceSearchOutcome:
        locale = str(getattr(self.steam_client, "language", "english") or "english")
        languages = dedupe_texts(["english", locale])
        requests = [
            (alias, language)
            for alias in dedupe_texts(list(reference.aliases))
            for language in languages
        ]
        search_storefront = getattr(
            self.steam_client,
            "search_storefront_term",
            None,
        )

        async def search_one(
            alias: str,
            language: str,
        ) -> tuple[list[SteamSearchHit], int, int, list[ReferenceSearchFailure]]:
            attempted = 0
            succeeded = 0
            failures: list[ReferenceSearchFailure] = []
            storefront_hits: list[SteamSearchHit] = []
            if search_storefront is not None:
                attempted += 1
                try:
                    async with self._steam_semaphore:
                        page = await search_storefront(
                            alias,
                            page_size=20,
                            start=0,
                            language=language,
                            **self._discovery_cache_kwargs(search_storefront),
                        )
                    hits = [validate_search_hit(hit) for hit in page.hits]
                except Exception as exc:
                    failures.append(
                        reference_search_failure(
                            alias,
                            language,
                            "storefront",
                            exc,
                        )
                    )
                else:
                    succeeded += 1
                    if hits and match_reference_query(reference, hits) is not None:
                        return hits, attempted, succeeded, failures
                    storefront_hits = hits
            attempted += 1
            try:
                hits = await self._search_refs(
                    alias,
                    page_size=20,
                    language=language,
                    prefetched=prefetched,
                )
            except Exception as exc:
                failures.append(
                    reference_search_failure(
                        alias,
                        language,
                        "storesearch",
                        exc,
                    )
                )
                hits = []
            else:
                succeeded += 1
            return [*storefront_hits, *hits], attempted, succeeded, failures

        results = await asyncio.gather(
            *(search_one(alias, language) for alias, language in requests)
        )
        return ReferenceSearchOutcome(
            hits=tuple(hit for hits, _attempted, _succeeded, _failures in results for hit in hits),
            attempted=sum(item[1] for item in results),
            succeeded=sum(item[2] for item in results),
            failures=tuple(failure for item in results for failure in item[3]),
        )

    async def _select_reference_candidate(
        self,
        reference: ReferenceQuery,
        hits: list[SteamSearchHit],
        prefetched: dict[int, GameCandidate],
        known_candidates: Mapping[int, GameCandidate] | None = None,
        bypass_cache_appids: set[int] | None = None,
    ) -> tuple[
        ReferenceMatch | None,
        GameCandidate | None,
        tuple[ReferenceSearchFailure, ...],
    ]:
        remaining = list(hits)
        failures: list[ReferenceSearchFailure] = []
        detail_attempts = 0
        while match := match_reference_query(reference, remaining):
            candidate = (known_candidates or {}).get(match.hit.appid)
            if candidate is None:
                if detail_attempts >= MAX_REFERENCE_DETAIL_ATTEMPTS_PER_ENTITY:
                    break
                detail_attempts += 1
                try:
                    candidate = await self._load_reference_candidate(
                        match.hit.appid,
                        prefetched,
                        bypass_cache=match.hit.appid in (bypass_cache_appids or set()),
                    )
                except Exception as exc:
                    failures.append(
                        reference_search_failure(
                            reference.display_title,
                            str(getattr(self.steam_client, "language", "") or ""),
                            "detail",
                            exc,
                        )
                    )
                    candidate = None
            if candidate is not None:
                evidence_hit = max(
                    (hit for hit in hits if hit.appid == match.hit.appid),
                    key=lambda hit: len(hit.tag_ids),
                    default=match.hit,
                )
                return (
                    match,
                    self._apply_search_hit_tags(candidate, evidence_hit),
                    tuple(failures),
                )
            remaining = [hit for hit in remaining if hit.appid != match.hit.appid]
        return None, None, tuple(failures)

    def _apply_search_hit_tags(
        self,
        candidate: GameCandidate,
        hit: SteamSearchHit,
    ) -> GameCandidate:
        mapped = dedupe_texts(
            [
                canonical
                for tag_id in hit.tag_ids
                if (canonical := self._canonical_tag_by_id.get(int(tag_id)))
            ]
        )
        if not mapped:
            return candidate
        data = dump_model(candidate)
        data["ordered_tags"] = dedupe_texts(
            [*mapped, *(data.get("ordered_tags") or [])]
        )
        markers = list(data.get("internal_source_markers") or [])
        marker = "tag_enrichment:steam_storefront_result"
        if marker not in markers:
            markers.append(marker)
        data["internal_source_markers"] = markers
        return validate_candidate(data)

    async def _load_reference_candidate(
        self,
        appid: int,
        prefetched: dict[int, GameCandidate],
        *,
        bypass_cache: bool = False,
    ) -> GameCandidate | None:
        return await self._fetch_candidate(
            appid,
            prefetched,
            bypass_cache=bypass_cache,
        )

    async def _search_queries(
        self,
        queries: list[str],
        prefetched: dict[int, GameCandidate],
    ) -> list[tuple[str, list[SteamSearchHit], bool]]:
        async def search_one(query: str) -> tuple[str, list[SteamSearchHit], bool]:
            try:
                return query, await self._search_refs(
                    query,
                    prefetched=prefetched,
                ), True
            except Exception as exc:
                failure_status(exc)
                return query, [], False

        return list(await asyncio.gather(*(search_one(query) for query in queries)))

    async def _search_refs(
        self,
        query: str,
        page_size: int | None = None,
        language: str | None = None,
        prefetched: dict[int, GameCandidate] | None = None,
    ) -> list[SteamSearchHit]:
        resolved_page_size = self.page_size if page_size is None else int(page_size)
        search_refs = getattr(self.steam_client, "search_game_refs", None)
        if search_refs:
            kwargs: dict[str, Any] = {
                "search": query,
                "page_size": resolved_page_size,
                "ordering": "-relevance",
            }
            if language is not None:
                kwargs["language"] = language
            kwargs.update(self._discovery_cache_kwargs(search_refs))
            async with self._steam_semaphore:
                results = await search_refs(**kwargs)
            return [validate_search_hit(hit) for hit in results]

        search_games = getattr(self.steam_client, "search_games", None)
        if not search_games:
            return []
        async with self._steam_semaphore:
            candidates = await search_games(
                search=query,
                page_size=resolved_page_size,
                ordering="-relevance",
                language=language,
                **self._discovery_cache_kwargs(search_games),
            )
        hits: list[SteamSearchHit] = []
        for candidate in candidates:
            if candidate.appid is None:
                continue
            if prefetched is not None:
                prefetched[int(candidate.appid)] = candidate
            hits.append(
                SteamSearchHit(
                    appid=int(candidate.appid),
                    title=candidate.title,
                    store_url=candidate.raw_url,
                )
            )
        return hits

    async def _enrich_hits(
        self,
        hits: list[SteamSearchHit],
        prefetched: dict[int, GameCandidate],
        *,
        bypass_cache_appids: set[int] | None = None,
    ) -> list[GameCandidate]:
        async def enrich_one(hit: SteamSearchHit) -> GameCandidate | None:
            return await self._load_candidate(
                hit.appid,
                prefetched,
                bypass_cache=hit.appid in (bypass_cache_appids or set()),
            )

        candidates = await asyncio.gather(*(enrich_one(hit) for hit in hits))
        return [candidate for candidate in candidates if candidate is not None]

    async def _load_candidate(
        self,
        appid: int,
        prefetched: dict[int, GameCandidate],
        *,
        bypass_cache: bool = False,
    ) -> GameCandidate | None:
        try:
            return await self._fetch_candidate(
                appid,
                prefetched,
                bypass_cache=bypass_cache,
            )
        except Exception as exc:
            failure_status(exc)
            return None

    async def _fetch_candidate(
        self,
        appid: int,
        prefetched: dict[int, GameCandidate],
        *,
        bypass_cache: bool = False,
    ) -> GameCandidate | None:
        if appid <= 0:
            return None
        candidate = prefetched.get(appid)
        detail_getter = getattr(self.steam_client, "get_game_detail", None)
        if detail_getter is not None:
            kwargs = (
                {"bypass_cache": True}
                if bypass_cache
                and callable_accepts_keyword(detail_getter, "bypass_cache")
                else {}
            )
            async with self._steam_semaphore:
                candidate = await detail_getter(appid, **kwargs)
            if (
                candidate is not None
                and candidate.release_status_checked_at is None
            ):
                data = dump_model(candidate)
                data["release_status_checked_at"] = float(self.clock())
                candidate = validate_candidate(data)
        if (
            candidate is None
            or candidate.appid is None
            or int(candidate.appid) != appid
            or not is_confirmed_base_game(candidate)
        ):
            return None
        return await self.enrich_candidate(candidate, ensure_aliases=False)

    async def _merge_and_persist_snapshot(
        self,
        additions: Any,
        search_coverage: dict[str, float],
    ) -> SteamIndexSnapshot:
        additions = list(additions)
        async with self._snapshot_lock:
            latest = await self.load_snapshot()
            records = {
                entry_key(record.candidate): record
                for record in latest.entries
                if entry_is_recall_usable(record)
            }
            for record in additions:
                if not entry_is_recall_usable(record):
                    continue
                key = entry_key(record.candidate)
                current = records.get(key)
                if current is None or entry_should_replace(current, record):
                    records[key] = record
            coverage = dict(latest.search_coverage)
            for query, covered_at in search_coverage.items():
                coverage[query] = max(
                    float(coverage.get(query, 0.0) or 0.0),
                    float(covered_at),
                )
            snapshot = prune_snapshot(
                SteamIndexSnapshot(
                    entries=list(records.values()),
                    search_coverage=coverage,
                )
            )
            await self.cache.set_json(
                STEAM_INDEX_CACHE_KEY,
                snapshot_payload(snapshot),
            )
            return snapshot

    async def enrich_candidate(
        self,
        candidate: GameCandidate,
        *,
        ensure_aliases: bool = True,
    ) -> GameCandidate:
        has_steam_tags = (
            await self.ensure_steam_tag_aliases()
            if ensure_aliases
            else self._steam_tag_aliases_are_usable_stale()
        )
        data = dump_model(candidate)
        markers = list(data.get("internal_source_markers") or [])
        if "steam_index" not in markers:
            markers.append("steam_index")
        if has_steam_tags and "tag_enrichment:steam_popular_tags" not in markers:
            markers.append("tag_enrichment:steam_popular_tags")
        appid = data.get("appid")
        if appid is not None and hasattr(self.steam_client, "get_store_page_tags"):
            try:
                async with self._steam_semaphore:
                    store_tags = await self.steam_client.get_store_page_tags(int(appid))
            except Exception as exc:
                failure_status(exc)
                store_tags = []
            if store_tags:
                data["ordered_tags"] = dedupe_texts(store_tags)
                if "tag_enrichment:steam_store_page_tags" not in markers:
                    markers.append("tag_enrichment:steam_store_page_tags")
        data["internal_source_markers"] = markers

        enriched_candidate = validate_candidate(data)
        direct_tags = canonical_tags_from_terms(
            [*enriched_candidate.tags, *enriched_candidate.genres]
        )
        if direct_tags:
            data["tags"] = dedupe_texts([*(data.get("tags") or []), *direct_tags])
            if "tag_enrichment:steam_detail" not in markers:
                markers.append("tag_enrichment:steam_detail")
            data["internal_source_markers"] = markers
        if enriched_candidate.description:
            inferred_tags = canonical_tags_from_terms(
                extract_description_terms(enriched_candidate.description)
            )
            data["inferred_tags"] = dedupe_texts(
                [*(data.get("inferred_tags") or []), *inferred_tags]
            )
        if appid is not None and hasattr(self.steam_client, "get_review_summary"):
            try:
                async with self._steam_semaphore:
                    summary = await self.steam_client.get_review_summary(int(appid))
            except Exception as exc:
                failure_status(exc)
                summary = None
            if summary is not None:
                data["review_total"] = getattr(summary, "total_reviews", None)
                data["review_positive_ratio"] = getattr(summary, "positive_ratio", None)
                data["review_recent_ratio"] = getattr(summary, "recent_positive_ratio", None)
        return validate_candidate(data)


def steam_only_scope_warning_for(preference: GamePreference) -> str | None:
    if unsupported_platforms(preference):
        return STEAM_ONLY_SCOPE_WARNING
    return None


def has_supported_steam_platform(preference: GamePreference) -> bool:
    return not preference.platforms or any(
        platform in STEAM_INDEX_PLATFORMS for platform in preference.platforms
    )


def unsupported_platforms(preference: GamePreference) -> list[str]:
    return [platform for platform in preference.platforms if platform not in STEAM_INDEX_PLATFORMS]


def rank_entries(
    entries: list[GameCandidate],
    preference: GamePreference,
    profile_tag_weights: dict[str, float] | None = None,
    retrieval_ranks: Mapping[int, int] | None = None,
    intent: RecommendationIntent | None = None,
    presentation_tags: Mapping[str, str] | None = None,
) -> list[RankedGame]:
    positives = reference_candidates(preference, entries)
    negatives = negative_reference_candidates(preference, entries)
    resolved_intent = intent or expand_intent_with_reference_tags(
        build_recommendation_intent(preference),
        positives,
    )
    profile = build_profile_from_preference(
        preference,
        reference_candidates=positives,
        negative_reference_candidates=negatives,
    )
    ranked = rank_steam_candidates(
        entries,
        resolved_intent,
        profile_tag_weights=profile_tag_weights,
        positive_reference_candidates=positives,
        negative_reference_candidates=negatives,
        retrieval_ranks=retrieval_ranks,
        language_profile=profile,
        presentation_tag_names=presentation_tags,
    )
    if logger.isEnabledFor(logging.DEBUG):
        _log_ranking_diagnostics(resolved_intent, entries, ranked)
    return ranked


def _log_ranking_diagnostics(
    intent: RecommendationIntent,
    entries: list[GameCandidate],
    ranked: list[RankedGame],
) -> None:
    required_tags = [
        tag.tag for tag in intent.tags if tag.role is IntentTagRole.REQUIRED
    ]
    anchor_weights = {
        tag.tag: tag.weight
        for tag in intent.tags
        if tag.role is IntentTagRole.ANCHOR
    }
    total_anchor_weight = sum(anchor_weights.values())
    diagnostics = []
    for game in ranked[:20]:
        candidate_evidence = build_candidate_tag_evidence(game)
        anchor_contributions = {
            tag: {
                "evidence": round(candidate_evidence.direct.get(tag, 0.0), 4),
                "contribution": round(
                    (
                        weight
                        * candidate_evidence.direct.get(tag, 0.0)
                        / total_anchor_weight
                    )
                    if total_anchor_weight > 0.0
                    else 0.0,
                    4,
                ),
            }
            for tag, weight in anchor_weights.items()
        }
        diagnostics.append(
            {
                "appid": game.appid,
                "tier": game.score_breakdown.relevance_tier,
                "anchor_coverage": round(game.score_breakdown.anchor_coverage, 4),
                "anchor_contributions": anchor_contributions,
                "supporting_similarity": round(
                    game.score_breakdown.supporting_similarity,
                    4,
                ),
                "semantic_score": round(game.score_breakdown.semantic_score, 4),
                "quality_score": round(game.score_breakdown.quality_score, 4),
                "layer_score": round(game.score_breakdown.layer_score, 4),
                "retrieval_rank": game.score_breakdown.retrieval_rank,
            }
        )
    logger.debug(
        "recommendation_rank event=rank_complete candidate_count=%d "
        "ranked_count=%d quality_intent=%s required_tags=%s anchors=%s results=%s",
        len(entries),
        len(ranked),
        intent.quality_intent.value,
        required_tags,
        list(anchor_weights),
        diagnostics,
    )


def exclude_previously_shown(
    games: list[RankedGame],
    excluded_appids: list[int] | None,
    excluded_titles: list[str] | None,
) -> list[RankedGame]:
    appids = {int(appid) for appid in excluded_appids or []}
    families = {game_family_key(title) for title in excluded_titles or [] if title}
    if not appids and not families:
        return games
    return [
        game
        for game in games
        if not (
            (game.appid is not None and int(game.appid) in appids)
            or game_family_key(game.title) in families
        )
    ]


def reference_candidates(
    preference: GamePreference,
    entries: list[GameCandidate],
) -> list[GameCandidate]:
    return matching_reference_candidates(
        preference.reference_games_like,
        entries,
        polarity="like",
        resolved=preference.resolved_reference_games,
    )


def negative_reference_candidates(
    preference: GamePreference,
    entries: list[GameCandidate],
) -> list[GameCandidate]:
    return matching_reference_candidates(
        preference.reference_games_dislike,
        entries,
        polarity="dislike",
        resolved=preference.resolved_reference_games,
    )


def matching_reference_candidates(
    titles: list[str],
    entries: list[GameCandidate],
    polarity: str,
    resolved: list[ResolvedReferenceGame] | None = None,
) -> list[GameCandidate]:
    title_keys = {normalize_text(title) for title in titles if title}
    resolved_appids = {
        int(item.appid)
        for item in resolved or []
        if item.polarity == polarity
        if normalize_text(item.raw_text) in title_keys
        if item.appid is not None and item.confidence >= REFERENCE_MATCH_THRESHOLD
    }
    return [
        entry
        for entry in entries
        if entry.appid is not None and int(entry.appid) in resolved_appids
    ]


def search_terms_for(preference: GamePreference, profile: SteamTagProfile) -> list[str]:
    terms: list[str] = []
    include = [tag.replace("_", " ") for tag in profile.include_tags[:6]]
    if include:
        terms.append(" ".join(include[:3]))
        terms.extend(include[:4])
    if preference.players and preference.players >= 2:
        terms.extend(["co-op", "local co-op"])
    if not terms:
        terms.append("popular co-op")
    return dedupe_texts(terms)


def reference_polarity(reference: ReferenceQuery) -> str:
    return "dislike" if reference.polarity is ReferencePolarity.NEGATIVE else "like"


def reference_warning(display_title: str) -> str:
    return f"参考游戏“{display_title}”未能可靠解析，未扩展其标签。"


def reference_transient_warning(display_title: str) -> str:
    return f"参考游戏“{display_title}”暂时无法搜索，未扩展其标签。"


def reference_contract_warning(display_title: str) -> str:
    return f"参考游戏“{display_title}”搜索响应无效，未扩展其标签。"


def reference_warning_for_status(
    display_title: str,
    status: ReferenceResolutionStatus,
) -> str | None:
    if status is ReferenceResolutionStatus.RESOLVED:
        return None
    if status is ReferenceResolutionStatus.TRANSIENT_FAILURE:
        return reference_transient_warning(display_title)
    if status is ReferenceResolutionStatus.CONTRACT_FAILURE:
        return reference_contract_warning(display_title)
    return reference_warning(display_title)


def prune_reference_resolution_state(
    preference: GamePreference,
    references: tuple[ReferenceQuery, ...],
    available_appids: set[int],
) -> None:
    active = {
        (normalize_text(alias), reference_polarity(reference))
        for reference in references
        for alias in reference.aliases
    }
    preference.resolved_reference_games = [
        item
        for item in preference.resolved_reference_games
        if item.appid is not None
        and int(item.appid) in available_appids
        and (normalize_text(item.raw_text), item.polarity) in active
    ]
    preference.parse_warnings = [
        warning
        for warning in preference.parse_warnings
        if not is_reference_resolution_warning(warning)
    ]


def is_reference_resolution_warning(value: str) -> bool:
    text = " ".join(str(value or "").split())
    return text.startswith("参考游戏“") and any(
        text.endswith(suffix)
        for suffix in (
            "”未能可靠解析，未扩展其标签。",
            "”暂时无法搜索，未扩展其标签。",
            "”搜索响应无效，未扩展其标签。",
        )
    )


def log_deferred_reference_group(
    reference: ReferenceQuery,
    match: ReferenceMatch,
) -> None:
    logger.debug(
        "recommendation_reference event=resolution polarity=%s status=deferred "
        "alias_count=%d appid=%s confidence=%.3f",
        reference_polarity(reference),
        len(reference.aliases),
        match.hit.appid,
        match.confidence,
    )


def record_reference_group_resolution(
    preference: GamePreference,
    reference: ReferenceQuery,
    match: ReferenceMatch | None,
    candidate: GameCandidate | None,
    status: ReferenceResolutionStatus = ReferenceResolutionStatus.AMBIGUOUS,
) -> None:
    polarity = reference_polarity(reference)
    alias_keys = {normalize_text(alias) for alias in reference.aliases}
    preference.resolved_reference_games = [
        item
        for item in preference.resolved_reference_games
        if not (
            item.polarity == polarity and normalize_text(item.raw_text) in alias_keys
        )
    ]
    warning_keys = {
        normalize_text(warning)
        for alias in reference.aliases
        for warning in (
            reference_warning(alias),
            reference_transient_warning(alias),
            reference_contract_warning(alias),
        )
    }
    preference.parse_warnings = [
        warning
        for warning in preference.parse_warnings
        if normalize_text(warning) not in warning_keys
    ]

    succeeded = match is not None and candidate is not None
    resolved = ResolvedReferenceGame(
        raw_text=reference.display_title,
        normalized_title=title_key(reference.display_title),
        canonical_title=candidate.title if succeeded else "",
        appid=(
            int(candidate.appid)
            if succeeded and candidate.appid is not None
            else None
        ),
        store_url=(candidate.raw_url or match.hit.store_url) if succeeded else None,
        confidence=match.confidence if succeeded else 0.0,
        source="steam_alias_group",
        polarity=polarity,
        genres=candidate.genres if succeeded else [],
        tags=(
            dedupe_texts([*candidate.ordered_tags, *candidate.tags])
            if succeeded
            else []
        ),
        platforms=candidate.platforms if succeeded else [],
        stores=candidate.stores if succeeded else [],
    )
    preference.resolved_reference_games.append(resolved)
    warning = reference_warning_for_status(reference.display_title, status)
    if not succeeded and warning is not None:
        preference.parse_warnings.append(warning)
    logger.debug(
        "recommendation_reference event=resolution polarity=%s status=%s "
        "resolution_state=%s "
        "alias_count=%d appid=%s confidence=%.3f",
        polarity,
        status.value,
        "resolved" if succeeded else "unresolved",
        len(reference.aliases),
        resolved.appid,
        resolved.confidence,
    )


def reference_search_failure(
    alias: str,
    language: str,
    source: str,
    exc: Exception,
) -> ReferenceSearchFailure:
    status = failure_status(exc)
    return ReferenceSearchFailure(
        alias=alias,
        language=language,
        source=source,
        kind=(
            "transient"
            if status is RecallSourceStatus.TRANSIENT_FAILURE
            else "contract"
        ),
        error=str(exc),
    )


def classify_reference_resolution(
    outcome: ReferenceSearchOutcome,
    match: ReferenceMatch | None,
    candidate: GameCandidate | None,
) -> ReferenceResolutionStatus:
    if match is not None and candidate is not None:
        return ReferenceResolutionStatus.RESOLVED
    if outcome.validation_failures:
        if all(
            failure.kind == "transient"
            for failure in outcome.validation_failures
        ):
            return ReferenceResolutionStatus.TRANSIENT_FAILURE
        return ReferenceResolutionStatus.CONTRACT_FAILURE
    if outcome.hits:
        return ReferenceResolutionStatus.AMBIGUOUS
    if outcome.succeeded:
        return ReferenceResolutionStatus.NO_HIT
    if outcome.failures and all(
        failure.kind == "transient" for failure in outcome.failures
    ):
        return ReferenceResolutionStatus.TRANSIENT_FAILURE
    return ReferenceResolutionStatus.CONTRACT_FAILURE


def references_are_resolved(preference: GamePreference) -> bool:
    expected = {
        (normalize_text(reference.display_title), reference_polarity(reference))
        for reference in build_recommendation_intent(preference).references
    }
    if not expected:
        return True
    resolved = {
        (normalize_text(item.raw_text), item.polarity)
        for item in preference.resolved_reference_games
        if item.appid is not None and item.confidence >= REFERENCE_MATCH_THRESHOLD
    }
    return expected <= resolved


def query_is_covered(
    query: str,
    coverage: dict[str, float],
    now: float,
    ttl_hours: int,
) -> bool:
    covered_at = float(coverage.get(normalize_text(query), 0.0) or 0.0)
    return covered_at > 0 and now - covered_at < max(ttl_hours, 1) * 3600


def parse_snapshot(payload: Any) -> SteamIndexSnapshot:
    if not is_current_snapshot_payload(payload):
        return SteamIndexSnapshot()
    return parse_snapshot_records(payload, needs_revalidation=False)


def parse_legacy_snapshot(payload: Any) -> SteamIndexSnapshot:
    if not is_legacy_snapshot_payload(payload):
        return SteamIndexSnapshot()
    return parse_snapshot_records(
        payload,
        needs_revalidation=True,
        preserve_coverage=False,
    )


def parse_snapshot_records(
    payload: dict[str, Any],
    *,
    needs_revalidation: bool,
    preserve_coverage: bool = True,
) -> SteamIndexSnapshot:
    entries: list[SteamIndexEntry] = []
    for item in payload.get("entries") or []:
        if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
            continue
        try:
            refreshed_at = float(item.get("refreshed_at") or 0.0)
        except (TypeError, ValueError):
            refreshed_at = 0.0
        candidate = validate_candidate(item["candidate"])
        record_needs_revalidation = (
            needs_revalidation or item.get("needs_revalidation") is True
        )
        if is_confirmed_base_game(candidate) or (
            record_needs_revalidation
            and candidate.appid is not None
            and bool(candidate.title)
        ):
            entries.append(
                SteamIndexEntry(
                    candidate=candidate,
                    refreshed_at=refreshed_at,
                    needs_revalidation=record_needs_revalidation,
                )
            )
    raw_coverage = payload.get("search_coverage")
    coverage: dict[str, float] = {}
    if preserve_coverage and isinstance(raw_coverage, dict):
        for query, covered_at in raw_coverage.items():
            try:
                coverage[normalize_text(query)] = float(covered_at)
            except (TypeError, ValueError):
                continue
    return prune_snapshot(SteamIndexSnapshot(entries=entries, search_coverage=coverage))


def prune_snapshot(snapshot: SteamIndexSnapshot) -> SteamIndexSnapshot:
    newest_by_key: dict[str, SteamIndexEntry] = {}
    for record in snapshot.entries:
        if not entry_is_recall_usable(record):
            continue
        key = entry_key(record.candidate)
        current = newest_by_key.get(key)
        if current is None or entry_should_replace(current, record):
            newest_by_key[key] = record
    entries = sorted(
        newest_by_key.values(),
        key=lambda record: (-record.refreshed_at, entry_key(record.candidate)),
    )[:STEAM_INDEX_MAX_ENTRIES]
    coverage = dict(
        sorted(
            snapshot.search_coverage.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )[:STEAM_INDEX_MAX_SEARCH_TERMS]
    )
    return SteamIndexSnapshot(entries=entries, search_coverage=coverage)


def snapshot_payload(snapshot: SteamIndexSnapshot) -> dict[str, Any]:
    return {
        "schema_version": STEAM_INDEX_SCHEMA_VERSION,
        "entries": [
            {
                "candidate": dump_model(record.candidate),
                "refreshed_at": record.refreshed_at,
                "needs_revalidation": record.needs_revalidation,
            }
            for record in snapshot.entries
        ],
        "search_coverage": snapshot.search_coverage,
    }


def is_current_snapshot_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and type(payload.get("schema_version")) is int
        and payload.get("schema_version") == STEAM_INDEX_SCHEMA_VERSION
    )


def is_previous_snapshot_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and type(payload.get("schema_version")) is int
        and payload.get("schema_version") in {1, 2}
    )


def is_legacy_snapshot_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and type(payload.get("version")) is int
        and payload.get("version") in {3, 4}
    )


def entry_is_validated(record: SteamIndexEntry) -> bool:
    return not record.needs_revalidation and is_confirmed_base_game(record.candidate)


def release_status_age(candidate: GameCandidate, now: float) -> float | None:
    checked_at = candidate.release_status_checked_at
    if checked_at is None:
        return None
    return max(float(now) - checked_at, 0.0)


def entry_release_status_is_fresh(
    record: SteamIndexEntry,
    now: float,
) -> bool:
    if not record.candidate.coming_soon:
        return True
    age = release_status_age(record.candidate, now)
    return age is not None and age <= RELEASE_STATUS_FRESH_SECONDS


def entry_release_status_is_usable_stale(
    record: SteamIndexEntry,
    now: float,
) -> bool:
    if not entry_is_validated(record) or not record.candidate.coming_soon:
        return False
    age = release_status_age(record.candidate, now)
    return age is not None and age <= RELEASE_STATUS_STALE_SECONDS


def entry_is_current(record: SteamIndexEntry, now: float) -> bool:
    return entry_is_validated(record) and entry_release_status_is_fresh(record, now)


def entry_is_recall_usable(record: SteamIndexEntry) -> bool:
    return entry_is_validated(record) or (
        record.needs_revalidation
        and record.candidate.appid is not None
        and bool(record.candidate.title)
    )


def entry_should_replace(
    current: SteamIndexEntry,
    incoming: SteamIndexEntry,
) -> bool:
    if current.needs_revalidation != incoming.needs_revalidation:
        return current.needs_revalidation and not incoming.needs_revalidation
    return incoming.refreshed_at >= current.refreshed_at


def callable_accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == keyword
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        )
        for parameter in parameters
    )


def entry_key(candidate: GameCandidate) -> str:
    if candidate.appid is not None:
        return f"appid:{int(candidate.appid)}"
    return f"title:{normalize_text(candidate.title)}"


def dedupe_entries(entries: list[GameCandidate]) -> list[GameCandidate]:
    result: list[GameCandidate] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry_key(entry)
        if key not in seen:
            result.append(entry)
            seen.add(key)
    return result


def dedupe_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        key = text.lower()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_candidate(data: dict[str, Any]) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)


def validate_search_hit(value: Any) -> SteamSearchHit:
    if isinstance(value, SteamSearchHit):
        return value
    validator = getattr(SteamSearchHit, "model_validate", None)
    return validator(value) if validator else SteamSearchHit.parse_obj(value)


def candidate_from_search_hit(hit: SteamSearchHit) -> GameCandidate:
    return GameCandidate(
        title=hit.title,
        appid=hit.appid,
        raw_url=hit.store_url,
    )


def filter_recall_source(
    source: RecallSource,
    excluded_appids: set[int],
) -> RecallSource:
    ranks = source.candidate_ranks or tuple(range(1, len(source.candidates) + 1))
    retained = [
        (candidate, rank)
        for candidate, rank in zip(source.candidates, ranks)
        if candidate.appid is not None and int(candidate.appid) not in excluded_appids
    ]
    return replace(
        source,
        candidates=tuple(candidate for candidate, _rank in retained),
        candidate_ranks=(
            tuple(rank for _candidate, rank in retained)
            if source.candidate_ranks
            else ()
        ),
    )


def successful_source_fetch(
    source: RecallSource,
    *,
    stale: bool = False,
    total_count: int = 0,
) -> RecallSourceFetch:
    status = (
        RecallSourceStatus.STALE
        if stale
        else RecallSourceStatus.SUCCESS
        if source.candidates
        else RecallSourceStatus.EMPTY
    )
    return RecallSourceFetch(
        source=source,
        health=RecallSourceHealth(
            source_id=source.source_id,
            critical=source.source_kind != "index",
            status=status,
            candidate_count=len(source.candidates),
        ),
        total_count=max(int(total_count), 0),
    )


def failed_source_fetch(
    source_id: str,
    source_kind: str,
    source_tag: str | None,
    weight: float,
    exc: Exception,
    *,
    component_tags: tuple[str, ...] = (),
) -> RecallSourceFetch:
    source = RecallSource(
        source_id=source_id,
        source_kind=source_kind,
        source_tag=source_tag,
        candidates=(),
        weight=weight,
        component_tags=component_tags,
    )
    return RecallSourceFetch(
        source=source,
        health=RecallSourceHealth(
            source_id=source_id,
            critical=True,
            status=failure_status(exc),
        ),
    )


def failure_status(exc: Exception) -> RecallSourceStatus:
    if not isinstance(exc, (SteamApiError, httpx.HTTPError)):
        raise exc
    transient = bool(getattr(exc, "transient", False)) or (
        isinstance(exc, httpx.HTTPError)
        and is_retryable_steam_read_error(exc)
    )
    return (
        RecallSourceStatus.TRANSIENT_FAILURE
        if transient
        else RecallSourceStatus.CONTRACT_FAILURE
    )


def merge_recall_candidate_evidence(
    candidate: GameCandidate,
    recalled: GameCandidate,
) -> GameCandidate:
    if not recalled.ordered_tags:
        return candidate
    data = dump_model(candidate)
    data["ordered_tags"] = dedupe_texts(
        [*recalled.ordered_tags, *(data.get("ordered_tags") or [])]
    )
    return validate_candidate(data)


def mark_reference_query(
    candidate: GameCandidate,
    query: str,
    polarity: str = "like",
) -> GameCandidate:
    data = dump_model(candidate)
    markers = list(data.get("internal_source_markers") or [])
    marker = f"reference_query:{polarity}:{query}"
    if marker not in markers:
        markers.append(marker)
    data["internal_source_markers"] = markers
    return validate_candidate(data)


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())
