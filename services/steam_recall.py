from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterable, TypeAlias

from ..storage.models import GameCandidate
from .recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    RecommendationIntent,
)

DEFAULT_SEED_LIMIT = 3
DEFAULT_PER_TAG_LIMIT = 20
DEFAULT_TAG_LIMIT = 60
DEFAULT_TOTAL_LIMIT = 100

CandidateSource: TypeAlias = tuple[str, str | None, Iterable[GameCandidate]]


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
class CandidateHit:
    candidate: GameCandidate
    source_kind: str
    source_tag: str | None
    source_rank: int
    retrieval_rank: int


@dataclass(frozen=True)
class CandidateRecallResult:
    hits: tuple[CandidateHit, ...]
    seeds: tuple[RecallSeed, ...] = ()
    warnings: tuple[str, ...] = ()
    degraded: bool = False

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
    for role in (IntentTagRole.REQUIRED, IntentTagRole.ANCHOR):
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
    per_tag_limit: int = DEFAULT_PER_TAG_LIMIT,
    tag_limit: int = DEFAULT_TAG_LIMIT,
    total_limit: int = DEFAULT_TOTAL_LIMIT,
) -> CandidateRecallResult:
    """Round-robin request-local candidate sources into one deterministic pool."""
    total_limit = max(0, total_limit)
    per_tag_limit = max(0, per_tag_limit)
    tag_limit = max(0, tag_limit)
    if total_limit == 0:
        return CandidateRecallResult(
            hits=(),
            seeds=tuple(seeds),
            warnings=tuple(warnings),
            degraded=degraded,
        )

    prepared_sources: list[tuple[str, str | None, tuple[GameCandidate, ...]]] = []
    for source_kind, source_tag, candidates in sources:
        source_limit = per_tag_limit if source_tag is not None else total_limit
        prepared_sources.append(
            (
                source_kind,
                source_tag,
                tuple(islice(candidates, source_limit)),
            )
        )

    hits: list[CandidateHit] = []
    seen_appids: set[int] = set()
    positions = [0] * len(prepared_sources)
    tag_hits = 0

    while len(hits) < total_limit:
        advanced = False
        for source_index, (source_kind, source_tag, candidates) in enumerate(
            prepared_sources
        ):
            position = positions[source_index]
            if position >= len(candidates):
                continue
            positions[source_index] += 1
            advanced = True

            candidate = candidates[position]
            appid = candidate.appid
            if appid is None or appid <= 0 or appid in seen_appids:
                continue
            if source_tag is not None and tag_hits >= tag_limit:
                continue

            hits.append(
                CandidateHit(
                    candidate=candidate,
                    source_kind=source_kind,
                    source_tag=source_tag,
                    source_rank=position + 1,
                    retrieval_rank=len(hits) + 1,
                )
            )
            seen_appids.add(appid)
            if source_tag is not None:
                tag_hits += 1
            if len(hits) >= total_limit:
                break
        if not advanced:
            break

    return CandidateRecallResult(
        hits=tuple(hits),
        seeds=tuple(seeds),
        warnings=tuple(warnings),
        degraded=degraded,
    )
