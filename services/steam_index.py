from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Protocol

from ..storage.models import (
    GameCandidate,
    GamePreference,
    RankedGame,
    ResolvedReferenceGame,
    SteamSearchHit,
)
from .game_identity import (
    deduplicate_game_editions,
    game_family_key,
    is_confirmed_base_game,
)
from .similarity_ranker import (
    SteamTagProfile,
    build_profile_from_preference,
    rank_steam_candidates,
    resolve_positive_component_weights,
)
from .tag_normalizer import (
    canonical_tags_from_terms,
    extract_description_terms,
    register_steam_tag_aliases,
)

logger = logging.getLogger(__name__)

STEAM_INDEX_CACHE_KEY = "steam_index:v3"
STEAM_INDEX_VERSION = 3
STEAM_INDEX_MAX_ENTRIES = 3_000
STEAM_INDEX_MAX_SEARCH_TERMS = 256
STEAM_INDEX_MAX_SEARCHES_PER_ROUND = 8
STEAM_INDEX_SEARCH_RESULTS_PER_TERM = 10
STEAM_INDEX_MAX_NEW_APPIDS_PER_ROUND = 60
STEAM_HTTP_CONCURRENCY = 6
USABLE_SCORE_THRESHOLD = 38
SNAPSHOT_STORAGE_TTL_HOURS = 24 * 3650
REFERENCE_MATCH_THRESHOLD = 0.75

STEAM_INDEX_FALLBACK_WARNING = (
    "Steam 索引暂不可用，已尝试通过 Steam 公共搜索刷新候选；"
    "如果仍为空，请换更明确的标签或参考游戏。"
)
STEAM_ONLY_SCOPE_WARNING = (
    "当前版本仅支持 Steam 商店游戏，无法验证 Switch、PlayStation 或 Xbox 候选。"
)
STEAM_INDEX_PLATFORMS = {"steam", "pc"}
AAA_SEARCH_TERMS = ["popular", "action adventure", "open world", "story rich", "rpg"]
AAA_INTENT_MARKERS = {"aaa", "3a", "triple-a", "triple a", "大作", "单机大作"}


@dataclass(frozen=True)
class SteamIndexEntry:
    candidate: GameCandidate
    refreshed_at: float


@dataclass(frozen=True)
class SteamIndexSnapshot:
    entries: list[SteamIndexEntry] = field(default_factory=list)
    search_coverage: dict[str, float] = field(default_factory=dict)
    version: int = STEAM_INDEX_VERSION


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
        min_review_count: int = 50,
        min_positive_ratio: float = 0.65,
        page_size: int = STEAM_INDEX_SEARCH_RESULTS_PER_TERM,
        clock: Callable[[], float] = time.time,
        positive_component_weights: Mapping[str, Any] | None = None,
    ) -> None:
        self.steam_client = steam_client
        self.cache = cache
        self.ttl_hours = max(int(ttl_hours), 1)
        self.min_review_count = max(int(min_review_count), 0)
        self.min_positive_ratio = min(max(float(min_positive_ratio), 0.0), 1.0)
        self.page_size = min(max(int(page_size), 1), STEAM_INDEX_SEARCH_RESULTS_PER_TERM)
        self.clock = clock
        self.positive_component_weights = resolve_positive_component_weights(
            positive_component_weights
        )
        self._tag_aliases_attempted = False
        self._tag_aliases_loaded = False
        self._round_prefetched: dict[int, GameCandidate] = {}

    async def recommend(
        self,
        preference: GamePreference,
        limit: int,
        profile_tag_weights: dict[str, float] | None = None,
        excluded_appids: list[int] | None = None,
        excluded_titles: list[str] | None = None,
        preferred_appids: list[int] | None = None,
    ) -> list[RankedGame]:
        if preference.platforms and not has_supported_steam_platform(preference):
            return []

        await self.ensure_steam_tag_aliases()
        snapshot = await self.load_snapshot()
        entries = [record.candidate for record in snapshot.entries]
        ranked = rank_entries(
            entries,
            preference,
            self.min_review_count,
            self.min_positive_ratio,
            profile_tag_weights=profile_tag_weights,
            positive_component_weights=self.positive_component_weights,
        )
        ranked = exclude_previously_shown(ranked, excluded_appids, excluded_titles)
        ranked = deduplicate_game_editions(ranked, preferred_appids)
        quality_target = max(10, max(int(limit), 0) * 2)
        quality_count = sum(game.score >= USABLE_SCORE_THRESHOLD for game in ranked)
        if quality_count >= quality_target and references_are_resolved(preference):
            return ranked[:limit]

        target_pool = min(60, max(30, max(int(limit), 0) * 6))
        refreshed = await self.refresh_entries(
            preference,
            entries,
            target_pool=target_pool,
            snapshot=snapshot,
        )
        ranked = rank_entries(
            refreshed,
            preference,
            self.min_review_count,
            self.min_positive_ratio,
            profile_tag_weights=profile_tag_weights,
            positive_component_weights=self.positive_component_weights,
        )
        ranked = exclude_previously_shown(ranked, excluded_appids, excluded_titles)
        ranked = deduplicate_game_editions(ranked, preferred_appids)
        return ranked[:limit]

    async def ensure_steam_tag_aliases(self) -> bool:
        if self._tag_aliases_attempted:
            return self._tag_aliases_loaded
        self._tag_aliases_attempted = True
        getter = getattr(self.steam_client, "get_popular_tags", None)
        if not getter:
            return False
        try:
            tags = await getter()
        except Exception:
            return False
        register_steam_tag_aliases(tags)
        self._tag_aliases_loaded = bool(tags)
        return self._tag_aliases_loaded

    async def load_snapshot(self) -> SteamIndexSnapshot:
        payload = await self.cache.get_json(
            STEAM_INDEX_CACHE_KEY,
            SNAPSHOT_STORAGE_TTL_HOURS,
        )
        return parse_snapshot(payload)

    async def load_entries(self) -> list[GameCandidate]:
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
        await self.ensure_steam_tag_aliases()
        now = float(self.clock())
        current = snapshot or await self.load_snapshot()
        records = {
            entry_key(record.candidate): record
            for record in current.entries
            if is_confirmed_base_game(record.candidate)
        }
        for candidate in existing or []:
            if not is_confirmed_base_game(candidate):
                continue
            key = entry_key(candidate)
            if key not in records:
                records[key] = SteamIndexEntry(candidate=candidate, refreshed_at=now)

        self._round_prefetched = {}
        coverage = dict(current.search_coverage)
        markers: dict[int, list[tuple[str, str]]] = {}
        seen_hits: set[int] = set()
        enrichment_limit = min(
            STEAM_INDEX_MAX_NEW_APPIDS_PER_ROUND,
            max(int(target_pool), 0),
        )
        enriched_count = 0
        searched_queries: list[str] = []

        pending_references: list[str] = []
        for query in reference_terms_for(preference):
            polarity = reference_polarity_for(query, preference) or "like"
            existing_hit = best_reference_hit(
                query,
                [
                    SteamSearchHit(
                        appid=int(record.candidate.appid),
                        title=record.candidate.title,
                        store_url=record.candidate.raw_url,
                    )
                    for record in records.values()
                    if record.candidate.appid is not None
                ],
            )
            if existing_hit is not None:
                record_reference_resolution(preference, query, [existing_hit], polarity)
                markers.setdefault(existing_hit.appid, []).append((query, polarity))
            else:
                pending_references.append(query)

        initial_profile = build_profile_from_preference(preference)
        if pending_references:
            initial_queries = pending_references[:STEAM_INDEX_MAX_SEARCHES_PER_ROUND]
        else:
            initial_queries = [
                query
                for query in search_terms_for(preference, initial_profile)
                if not query_is_covered(query, coverage, now, self.ttl_hours)
            ][:STEAM_INDEX_MAX_SEARCHES_PER_ROUND]

        async def process_searches(queries: list[str]) -> None:
            nonlocal enriched_count
            if not queries:
                return
            searched_queries.extend(queries)
            results = await self._search_queries(queries)
            selected_hits: list[SteamSearchHit] = []
            other_hits: list[SteamSearchHit] = []
            reference_results: list[tuple[str, str, list[SteamSearchHit]]] = []
            for query, hits, succeeded in results:
                if succeeded:
                    coverage[normalize_text(query)] = now
                polarity = reference_polarity_for(query, preference)
                if polarity:
                    selected = best_reference_hit(query, hits)
                    if selected is not None:
                        selected_hits.append(selected)
                    reference_results.append((query, polarity, hits))
                other_hits.extend(hits)

            existing_appids = {
                int(record.candidate.appid)
                for record in records.values()
                if record.candidate.appid is not None
            }
            new_hits: list[SteamSearchHit] = []
            for hit in [*selected_hits, *other_hits]:
                if hit.appid in seen_hits or hit.appid in existing_appids:
                    continue
                seen_hits.add(hit.appid)
                new_hits.append(hit)
            remaining = max(enrichment_limit - enriched_count, 0)
            enriched = await self._enrich_hits(new_hits[:remaining])
            enriched_count += len(enriched)
            for candidate in enriched:
                records[entry_key(candidate)] = SteamIndexEntry(candidate, now)

            confirmed_appids = {
                int(record.candidate.appid)
                for record in records.values()
                if record.candidate.appid is not None
                and is_confirmed_base_game(record.candidate)
            }
            for query, polarity, hits in reference_results:
                confirmed_hits = [hit for hit in hits if hit.appid in confirmed_appids]
                selected = record_reference_resolution(
                    preference,
                    query,
                    confirmed_hits,
                    polarity,
                )
                if selected is not None:
                    markers.setdefault(selected.appid, []).append((query, polarity))

            for key, record in list(records.items()):
                candidate = record.candidate
                if candidate.appid is None or int(candidate.appid) not in markers:
                    continue
                marked = candidate
                for query, polarity in markers[int(candidate.appid)]:
                    marked = mark_reference_query(marked, query, polarity)
                records[key] = SteamIndexEntry(marked, record.refreshed_at)

        await process_searches(initial_queries)

        for key, record in list(records.items()):
            candidate = record.candidate
            if candidate.appid is None or int(candidate.appid) not in markers:
                continue
            marked = candidate
            for query, polarity in markers[int(candidate.appid)]:
                marked = mark_reference_query(marked, query, polarity)
            records[key] = SteamIndexEntry(marked, record.refreshed_at)

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
                and not query_is_covered(query, coverage, now, self.ttl_hours)
            ][:remaining_searches]
            await process_searches(supplemental_queries)

        refreshed_snapshot = prune_snapshot(
            SteamIndexSnapshot(
                entries=list(records.values()),
                search_coverage=coverage,
            )
        )
        await self.cache.set_json(
            STEAM_INDEX_CACHE_KEY,
            snapshot_payload(refreshed_snapshot),
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
        return [record.candidate for record in refreshed_snapshot.entries]

    async def _search_queries(
        self,
        queries: list[str],
    ) -> list[tuple[str, list[SteamSearchHit], bool]]:
        semaphore = asyncio.Semaphore(STEAM_HTTP_CONCURRENCY)

        async def search_one(query: str) -> tuple[str, list[SteamSearchHit], bool]:
            async with semaphore:
                try:
                    return query, await self._search_refs(query), True
                except Exception:
                    return query, [], False

        return list(await asyncio.gather(*(search_one(query) for query in queries)))

    async def _search_refs(self, query: str) -> list[SteamSearchHit]:
        search_refs = getattr(self.steam_client, "search_game_refs", None)
        if search_refs:
            results = await search_refs(
                search=query,
                page_size=self.page_size,
                ordering="-relevance",
            )
            return [validate_search_hit(hit) for hit in results]

        search_games = getattr(self.steam_client, "search_games", None)
        if not search_games:
            return []
        candidates = await search_games(
            search=query,
            page_size=self.page_size,
            ordering="-relevance",
        )
        hits: list[SteamSearchHit] = []
        for candidate in candidates:
            if candidate.appid is None:
                continue
            self._round_prefetched[int(candidate.appid)] = candidate
            hits.append(
                SteamSearchHit(
                    appid=int(candidate.appid),
                    title=candidate.title,
                    store_url=candidate.raw_url,
                )
            )
        return hits

    async def _enrich_hits(self, hits: list[SteamSearchHit]) -> list[GameCandidate]:
        semaphore = asyncio.Semaphore(STEAM_HTTP_CONCURRENCY)

        async def enrich_one(hit: SteamSearchHit) -> GameCandidate | None:
            async with semaphore:
                candidate = self._round_prefetched.get(hit.appid)
                detail_getter = getattr(self.steam_client, "get_game_detail", None)
                if candidate is None and detail_getter:
                    try:
                        candidate = await detail_getter(hit.appid)
                    except Exception:
                        candidate = None
                if candidate is None or not is_confirmed_base_game(candidate):
                    return None
                return await self.enrich_candidate(candidate)

        candidates = await asyncio.gather(*(enrich_one(hit) for hit in hits))
        return [candidate for candidate in candidates if candidate is not None]

    async def enrich_candidate(self, candidate: GameCandidate) -> GameCandidate:
        has_steam_tags = await self.ensure_steam_tag_aliases()
        data = dump_model(candidate)
        markers = list(data.get("internal_source_markers") or [])
        if "steam_index" not in markers:
            markers.append("steam_index")
        if has_steam_tags and "tag_enrichment:steam_popular_tags" not in markers:
            markers.append("tag_enrichment:steam_popular_tags")
        appid = data.get("appid")
        if appid is not None and hasattr(self.steam_client, "get_store_page_tags"):
            try:
                store_tags = await self.steam_client.get_store_page_tags(int(appid))
            except Exception:
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
                summary = await self.steam_client.get_review_summary(int(appid))
            except Exception:
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
    min_review_count: int,
    min_positive_ratio: float,
    profile_tag_weights: dict[str, float] | None = None,
    positive_component_weights: Mapping[str, Any] | None = None,
) -> list[RankedGame]:
    profile = build_profile_from_preference(
        preference,
        reference_candidates=reference_candidates(preference, entries),
        negative_reference_candidates=negative_reference_candidates(preference, entries),
    )
    return rank_steam_candidates(
        entries,
        profile,
        min_review_count,
        min_positive_ratio,
        profile_tag_weights=profile_tag_weights,
        positive_component_weights=positive_component_weights,
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
    resolved_appids = {
        int(item.appid)
        for item in resolved or []
        if item.polarity == polarity
        if item.appid is not None and item.confidence >= REFERENCE_MATCH_THRESHOLD
    }
    references = [title for title in titles if title]
    return [
        entry
        for entry in entries
        if (entry.appid is not None and int(entry.appid) in resolved_appids)
        or (
            not resolved_appids
            and any(
                title_match_confidence(reference, entry.title) >= REFERENCE_MATCH_THRESHOLD
                or game_family_key(reference) == game_family_key(entry.title)
                for reference in references
            )
        )
    ]


def search_terms_for(preference: GamePreference, profile: SteamTagProfile) -> list[str]:
    terms: list[str] = []
    if has_aaa_intent(preference):
        terms.extend(AAA_SEARCH_TERMS)
    terms.extend(preference.reference_games_like[:3])
    terms.extend(preference.reference_search_terms[:3])
    terms.extend(preference.reference_games_dislike[:3])
    include = [tag.replace("_", " ") for tag in profile.include_tags[:6]]
    if include:
        terms.append(" ".join(include[:3]))
        terms.extend(include[:4])
    if preference.players and preference.players >= 2:
        terms.extend(["co-op", "local co-op"])
    if not terms:
        terms.append("popular co-op")
    return dedupe_texts(terms)


def has_aaa_intent(preference: GamePreference) -> bool:
    values = [*preference.genres_like, *preference.extra_tags]
    normalized = {normalize_text(value) for value in values}
    return bool(normalized & AAA_INTENT_MARKERS)


def reference_terms_for(preference: GamePreference) -> list[str]:
    return dedupe_texts(
        [
            *preference.reference_games_like,
            *preference.reference_search_terms,
            *preference.reference_games_dislike,
        ]
    )


def reference_polarity_for(query: str, preference: GamePreference) -> str | None:
    key = normalize_text(query)
    disliked = {normalize_text(title) for title in preference.reference_games_dislike}
    if key in disliked:
        return "dislike"
    liked = {
        normalize_text(title)
        for title in [*preference.reference_games_like, *preference.reference_search_terms]
    }
    return "like" if key in liked else None


def record_reference_resolution(
    preference: GamePreference,
    query: str,
    hits: list[SteamSearchHit],
    polarity: str,
) -> SteamSearchHit | None:
    hit = best_reference_hit(query, hits)
    confidence = title_match_confidence(query, hit.title) if hit else 0.0
    resolved = ResolvedReferenceGame(
        raw_text=query,
        normalized_title=normalize_title_key(query),
        canonical_title=hit.title if hit else "",
        appid=hit.appid if hit else None,
        store_url=hit.store_url if hit else None,
        confidence=confidence,
        source="steam_search",
        polarity=polarity,
        stores=["steam"] if hit else [],
    )
    preference.resolved_reference_games = [
        item
        for item in preference.resolved_reference_games
        if not (
            normalize_text(item.raw_text) == normalize_text(query) and item.polarity == polarity
        )
    ]
    preference.resolved_reference_games.append(resolved)
    if confidence >= REFERENCE_MATCH_THRESHOLD:
        return hit
    warning = f"参考游戏“{query}”未能可靠解析，未扩展其标签。"
    if warning not in preference.parse_warnings:
        preference.parse_warnings.append(warning)
    return None


def best_reference_hit(query: str, hits: list[SteamSearchHit]) -> SteamSearchHit | None:
    if not hits:
        return None
    hit, confidence = max(
        (
            (hit, title_match_confidence(query, hit.title))
            for hit in hits[:STEAM_INDEX_SEARCH_RESULTS_PER_TERM]
        ),
        key=lambda item: item[1],
    )
    return hit if confidence >= REFERENCE_MATCH_THRESHOLD else None


def references_are_resolved(preference: GamePreference) -> bool:
    expected = {
        (normalize_text(query), reference_polarity_for(query, preference) or "like")
        for query in reference_terms_for(preference)
    }
    if not expected:
        return True
    resolved = {
        (normalize_text(item.raw_text), item.polarity)
        for item in preference.resolved_reference_games
        if item.appid is not None and item.confidence >= REFERENCE_MATCH_THRESHOLD
    }
    return expected <= resolved


def title_match_confidence(query: str, candidate_title: str) -> float:
    expected = normalize_title_key(query)
    actual = normalize_title_key(candidate_title)
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0
    if actual.startswith(expected) or expected.startswith(actual):
        coverage = min(len(expected), len(actual)) / max(len(expected), len(actual))
        return min(0.65 + coverage * 0.35, 0.99)
    return SequenceMatcher(None, expected, actual).ratio()


def normalize_title_key(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value or "").lower())


def query_is_covered(
    query: str,
    coverage: dict[str, float],
    now: float,
    ttl_hours: int,
) -> bool:
    covered_at = float(coverage.get(normalize_text(query), 0.0) or 0.0)
    return covered_at > 0 and now - covered_at < max(ttl_hours, 1) * 3600


def parse_snapshot(payload: Any) -> SteamIndexSnapshot:
    if not isinstance(payload, dict) or payload.get("version") != STEAM_INDEX_VERSION:
        return SteamIndexSnapshot()
    entries: list[SteamIndexEntry] = []
    for item in payload.get("entries") or []:
        if not isinstance(item, dict) or not isinstance(item.get("candidate"), dict):
            continue
        try:
            refreshed_at = float(item.get("refreshed_at") or 0.0)
        except (TypeError, ValueError):
            refreshed_at = 0.0
        candidate = validate_candidate(item["candidate"])
        if is_confirmed_base_game(candidate):
            entries.append(
                SteamIndexEntry(
                    candidate=candidate,
                    refreshed_at=refreshed_at,
                )
            )
    raw_coverage = payload.get("search_coverage")
    coverage: dict[str, float] = {}
    if isinstance(raw_coverage, dict):
        for query, covered_at in raw_coverage.items():
            try:
                coverage[normalize_text(query)] = float(covered_at)
            except (TypeError, ValueError):
                continue
    return prune_snapshot(SteamIndexSnapshot(entries=entries, search_coverage=coverage))


def prune_snapshot(snapshot: SteamIndexSnapshot) -> SteamIndexSnapshot:
    newest_by_key: dict[str, SteamIndexEntry] = {}
    for record in snapshot.entries:
        if not is_confirmed_base_game(record.candidate):
            continue
        key = entry_key(record.candidate)
        current = newest_by_key.get(key)
        if current is None or record.refreshed_at >= current.refreshed_at:
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
        "version": STEAM_INDEX_VERSION,
        "entries": [
            {
                "candidate": dump_model(record.candidate),
                "refreshed_at": record.refreshed_at,
            }
            for record in snapshot.entries
        ],
        "search_coverage": snapshot.search_coverage,
    }


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
