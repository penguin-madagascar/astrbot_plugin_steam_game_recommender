from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Iterable, TypeAlias

from ..clients.steam import SteamApiError
from ..storage.models import GameCandidate
from .recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    RecommendationIntent,
)

DEFAULT_SEED_LIMIT = 3
DEFAULT_PER_TAG_LIMIT = 40
DEFAULT_TAG_LIMIT = 120
DEFAULT_TOTAL_LIMIT = 100
RRF_K = 60

LegacyCandidateSource: TypeAlias = tuple[
    str,
    str | None,
    Iterable[GameCandidate],
]


@dataclass(frozen=True)
class RecallSeed:
    tag: str
    role: IntentTagRole
    source: IntentTagSource
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", IntentTagRole(self.role))
        object.__setattr__(self, "source", IntentTagSource(self.source))
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class RecallSource:
    source_id: str
    source_kind: str
    source_tag: str | None
    candidates: tuple[GameCandidate, ...]
    weight: float
    component_tags: tuple[str, ...] = ()
    candidate_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", str(self.source_id).strip())
        object.__setattr__(self, "source_kind", str(self.source_kind).strip())
        object.__setattr__(
            self,
            "source_tag",
            str(self.source_tag).strip() if self.source_tag is not None else None,
        )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "weight", max(float(self.weight), 0.0))
        object.__setattr__(
            self,
            "component_tags",
            tuple(dict.fromkeys(str(tag).strip() for tag in self.component_tags if tag)),
        )
        ranks = tuple(int(rank) for rank in self.candidate_ranks)
        if ranks and (len(ranks) != len(self.candidates) or any(rank <= 0 for rank in ranks)):
            raise ValueError("candidate ranks must align with candidates and be positive")
        object.__setattr__(self, "candidate_ranks", ranks)


CandidateSource: TypeAlias = RecallSource | LegacyCandidateSource


@dataclass(frozen=True)
class CandidateSourceHit:
    source_id: str
    source_kind: str
    source_tag: str | None
    source_rank: int
    source_weight: float
    component_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateHit:
    candidate: GameCandidate
    source_hits: tuple[CandidateSourceHit, ...]
    rrf_score: float
    retrieval_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_hits", tuple(self.source_hits))

    @property
    def source_kind(self) -> str:
        return self.source_hits[0].source_kind if self.source_hits else ""

    @property
    def source_tag(self) -> str | None:
        return self.source_hits[0].source_tag if self.source_hits else None

    @property
    def source_rank(self) -> int:
        return self.source_hits[0].source_rank if self.source_hits else 0


class RecallSourceStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    STALE = "stale"
    TRANSIENT_FAILURE = "transient_failure"
    CONTRACT_FAILURE = "contract_failure"


@dataclass(frozen=True)
class RecallSourceHealth:
    source_id: str
    critical: bool
    status: RecallSourceStatus
    candidate_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RecallSourceStatus(self.status))
        object.__setattr__(self, "candidate_count", max(int(self.candidate_count), 0))


@dataclass(frozen=True)
class RecallHealth:
    sources: tuple[RecallSourceHealth, ...] = ()
    validation_attempts: int = 0
    validation_transient_failures: int = 0
    validation_contract_failures: int = 0
    verified: int = 0
    eligible: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        for field_name in (
            "validation_attempts",
            "validation_transient_failures",
            "validation_contract_failures",
            "verified",
            "eligible",
        ):
            object.__setattr__(self, field_name, max(int(getattr(self, field_name)), 0))

    @property
    def systemic_failure(self) -> bool:
        critical = [source for source in self.sources if source.critical]
        source_failure = bool(critical) and all(
            source.status
            in {
                RecallSourceStatus.TRANSIENT_FAILURE,
                RecallSourceStatus.CONTRACT_FAILURE,
            }
            for source in critical
        )
        validation_failures = (
            self.validation_transient_failures
            + self.validation_contract_failures
        )
        large_sample_failure = (
            self.validation_attempts >= 3
            and validation_failures * 2 > self.validation_attempts
        )
        small_sample_failure = (
            0 < self.validation_attempts < 3
            and validation_failures == self.validation_attempts
            and self.verified == 0
        )
        return source_failure or large_sample_failure or small_sample_failure

    def unavailable(self, *, limit: int) -> bool:
        verified_floor = min(max(int(limit), 0), 3)
        return self.systemic_failure and (
            self.verified < verified_floor or self.eligible == 0
        )


class RecallUnavailableError(SteamApiError):
    def __init__(self, health: RecallHealth) -> None:
        super().__init__("Steam 候选召回暂时不可用，请稍后重试。")
        self.health = health


@dataclass(frozen=True)
class CandidateRecallResult:
    hits: tuple[CandidateHit, ...]
    seeds: tuple[RecallSeed, ...] = ()
    warnings: tuple[str, ...] = ()
    degraded: bool = False
    health: RecallHealth = RecallHealth()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def candidates(self) -> tuple[GameCandidate, ...]:
        return tuple(hit.candidate for hit in self.hits)


def select_recall_seeds(
    intent: RecommendationIntent,
    *,
    limit: int = DEFAULT_SEED_LIMIT,
) -> tuple[RecallSeed, ...]:
    """Select unique hard and anchor tags without relying on global state."""
    if limit <= 0:
        return ()

    selected: list[RecallSeed] = []
    seen: set[str] = set()
    for role in (
        IntentTagRole.REQUIRED,
        IntentTagRole.ANCHOR,
        IntentTagRole.RECALL_ONLY,
    ):
        for intent_tag in intent.tags:
            if intent_tag.role is not role or intent_tag.tag in seen:
                continue
            selected.append(
                RecallSeed(
                    tag=intent_tag.tag,
                    role=intent_tag.role,
                    source=intent_tag.source,
                    weight=intent_tag.weight,
                )
            )
            seen.add(intent_tag.tag)
            if len(selected) >= limit:
                return tuple(selected)
    return tuple(selected)


def merge_candidate_sources(
    sources: Iterable[CandidateSource],
    *,
    seeds: Iterable[RecallSeed] = (),
    warnings: Iterable[str] = (),
    degraded: bool = False,
    health: RecallHealth | None = None,
    per_tag_limit: int = DEFAULT_PER_TAG_LIMIT,
    tag_limit: int = DEFAULT_TAG_LIMIT,
    total_limit: int = DEFAULT_TOTAL_LIMIT,
) -> CandidateRecallResult:
    """Fuse request-local sources with weighted reciprocal-rank fusion."""
    total_limit = max(int(total_limit), 0)
    per_tag_limit = max(int(per_tag_limit), 0)
    tag_limit = max(int(tag_limit), 0)
    prepared = _prepare_sources(sources, per_tag_limit, tag_limit)
    if total_limit == 0:
        return CandidateRecallResult(
            hits=(),
            seeds=tuple(seeds),
            warnings=tuple(warnings),
            degraded=degraded,
            health=health or RecallHealth(),
        )

    candidates: dict[int, GameCandidate] = {}
    source_hits: dict[int, list[CandidateSourceHit]] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for source_index, source in enumerate(prepared):
        seen_in_source: set[int] = set()
        ranks = source.candidate_ranks or tuple(range(1, len(source.candidates) + 1))
        for candidate, source_rank in zip(source.candidates, ranks):
            appid = int(candidate.appid or 0)
            if appid <= 0 or appid in seen_in_source:
                continue
            seen_in_source.add(appid)
            if appid in candidates:
                candidates[appid] = _merge_candidate_evidence(
                    candidates[appid],
                    candidate,
                )
            else:
                candidates[appid] = candidate
            first_seen.setdefault(appid, (source_index, source_rank))
            candidate_hit = CandidateSourceHit(
                source_id=source.source_id,
                source_kind=source.source_kind,
                source_tag=source.source_tag,
                source_rank=source_rank,
                source_weight=source.weight,
                component_tags=source.component_tags,
            )
            hits_for_candidate = source_hits.setdefault(appid, [])
            duplicate_index = next(
                (
                    index
                    for index, existing_hit in enumerate(hits_for_candidate)
                    if existing_hit.source_id == candidate_hit.source_id
                ),
                None,
            )
            if duplicate_index is None:
                hits_for_candidate.append(candidate_hit)
            elif _source_hit_contribution(candidate_hit) > _source_hit_contribution(
                hits_for_candidate[duplicate_index]
            ):
                hits_for_candidate[duplicate_index] = candidate_hit

    intersection_definitions = [
        source for source in prepared if source.source_kind == "intersection"
    ]
    scored = [
        (
            appid,
            _rrf_score(hits, intersection_definitions),
            first_seen[appid],
        )
        for appid, hits in source_hits.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[2], item[0]))

    hits = tuple(
        CandidateHit(
            candidate=candidates[appid],
            source_hits=tuple(source_hits[appid]),
            rrf_score=score,
            retrieval_rank=retrieval_rank,
        )
        for retrieval_rank, (appid, score, _seen) in enumerate(
            scored[:total_limit],
            start=1,
        )
    )
    return CandidateRecallResult(
        hits=hits,
        seeds=tuple(seeds),
        warnings=tuple(warnings),
        degraded=degraded,
        health=health or RecallHealth(),
    )


def _prepare_sources(
    sources: Iterable[CandidateSource],
    per_tag_limit: int,
    tag_limit: int,
) -> list[RecallSource]:
    prepared: list[RecallSource] = []
    remaining_tag_hits = tag_limit
    for source_index, value in enumerate(sources):
        if isinstance(value, RecallSource):
            source = value
        else:
            source_kind, source_tag, candidates = value
            source = RecallSource(
                source_id=f"{source_kind}:{source_tag or source_index}",
                source_kind=source_kind,
                source_tag=source_tag,
                candidates=tuple(candidates),
                weight=_default_source_weight(source_kind),
            )
        source_limit = per_tag_limit if source.source_kind == "tag" else DEFAULT_TOTAL_LIMIT
        if source.source_kind == "tag":
            source_limit = min(source_limit, remaining_tag_hits)
            remaining_tag_hits = max(remaining_tag_hits - source_limit, 0)
        prepared.append(
            RecallSource(
                source_id=source.source_id,
                source_kind=source.source_kind,
                source_tag=source.source_tag,
                candidates=tuple(islice(source.candidates, max(source_limit, 0))),
                weight=source.weight,
                component_tags=source.component_tags,
                candidate_ranks=source.candidate_ranks[: max(source_limit, 0)],
            )
        )
    return prepared


def _default_source_weight(source_kind: str) -> float:
    return {
        "more_like": 1.2,
        "intersection": 1.3,
        "tag": 1.0,
        "tag_text": 1.0,
        "company": 1.0,
        "top_seller": 0.5,
        "index": 0.35,
    }.get(source_kind, 1.0)


def _rrf_score(
    hits: list[CandidateSourceHit],
    intersection_definitions: list[RecallSource],
) -> float:
    contributions = {
        hit.source_id: _source_hit_contribution(hit)
        for hit in hits
    }
    consumed_tag_sources: set[str] = set()
    total = 0.0

    for intersection in intersection_definitions:
        intersection_hit = next(
            (hit for hit in hits if hit.source_id == intersection.source_id),
            None,
        )
        components = [
            hit
            for hit in hits
            if hit.source_kind in {"tag", "tag_text"}
            and hit.source_tag in intersection.component_tags
            and hit.source_id not in consumed_tag_sources
        ]
        consumed_tag_sources.update(hit.source_id for hit in components)
        if intersection_hit is not None:
            total += contributions[intersection_hit.source_id]
            continue
        if components:
            best_rank = min(hit.source_rank for hit in components)
            component_sum = sum(contributions[hit.source_id] for hit in components)
            total += min(
                component_sum,
                intersection.weight / (RRF_K + best_rank),
            )

    for hit in hits:
        if hit.source_kind == "intersection":
            if not any(
                source.source_id == hit.source_id
                for source in intersection_definitions
            ):
                total += contributions[hit.source_id]
            continue
        if (
            hit.source_kind in {"tag", "tag_text"}
            and hit.source_id in consumed_tag_sources
        ):
            continue
        total += contributions[hit.source_id]
    return total


def _source_hit_contribution(hit: CandidateSourceHit) -> float:
    return hit.source_weight / (RRF_K + hit.source_rank)


def _merge_candidate_evidence(
    existing: GameCandidate,
    incoming: GameCandidate,
) -> GameCandidate:
    ordered = list(existing.ordered_tags)
    for tag in incoming.ordered_tags:
        if tag not in ordered:
            ordered.append(tag)
    if ordered == existing.ordered_tags:
        return existing
    copier = getattr(existing, "model_copy", None)
    if copier is not None:
        return copier(update={"ordered_tags": ordered})
    return existing.copy(update={"ordered_tags": ordered})
