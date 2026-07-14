from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..storage.models import GameCandidate
from .tag_normalizer import canonical_tags_from_terms, extract_description_terms, normalize_tag

REQUIRED_DIRECT_STRENGTH = 0.65
INFERRED_STRENGTH = 0.25
GENRE_TAG_STRENGTH = 0.65

_COOP_TAGS = {"co_op", "local_coop", "online_coop"}
_SPECIFIC_COOP_TAGS = {"local_coop", "online_coop"}


@dataclass(frozen=True)
class CandidateTagEvidence:
    direct: Mapping[str, float]
    supporting: Mapping[str, float]

    def __post_init__(self) -> None:
        direct = dict(self.direct)
        supporting = dict(self.supporting)
        for tag, strength in direct.items():
            supporting[tag] = max(supporting.get(tag, 0.0), strength)
        object.__setattr__(self, "direct", MappingProxyType(direct))
        object.__setattr__(self, "supporting", MappingProxyType(supporting))


def build_candidate_tag_evidence(candidate: GameCandidate) -> CandidateTagEvidence:
    direct: dict[str, float] = {}
    for position, raw_tag in enumerate(candidate.ordered_tags):
        tag = normalize_tag(raw_tag)
        if tag:
            _record_strength(direct, tag, max(0.4, 1.0 - 0.06 * position))

    for tag in canonical_tags_from_terms([*candidate.genres, *candidate.tags]):
        _record_strength(direct, tag, GENRE_TAG_STRENGTH)

    _add_coop_implications(direct)
    supporting = dict(direct)

    inferred_terms = list(candidate.inferred_tags)
    if candidate.description:
        inferred_terms.extend(extract_description_terms(candidate.description))
    for tag in canonical_tags_from_terms(inferred_terms):
        _record_strength(supporting, tag, INFERRED_STRENGTH)

    _add_coop_implications(supporting)
    return CandidateTagEvidence(direct=direct, supporting=supporting)


def required_tag_is_satisfied(
    evidence: CandidateTagEvidence,
    required_tag: str,
) -> bool:
    tag = normalize_tag(required_tag)
    return bool(tag and evidence.direct.get(tag, 0.0) >= REQUIRED_DIRECT_STRENGTH)


def satisfies_required_tags(
    evidence: CandidateTagEvidence,
    required_tags: Iterable[str],
) -> bool:
    return all(required_tag_is_satisfied(evidence, tag) for tag in required_tags)


def excluded_tag_is_hit(
    evidence: CandidateTagEvidence,
    excluded_tag: str,
) -> bool:
    tag = normalize_tag(excluded_tag)
    return bool(tag and evidence.direct.get(tag, 0.0) > 0.0)


def matches_excluded_tags(
    evidence: CandidateTagEvidence,
    excluded_tags: Iterable[str],
) -> bool:
    return any(excluded_tag_is_hit(evidence, tag) for tag in excluded_tags)


def _record_strength(strengths: dict[str, float], tag: str, strength: float) -> None:
    strengths[tag] = max(strengths.get(tag, 0.0), strength)


def _add_coop_implications(strengths: dict[str, float]) -> None:
    for tag, strength in list(strengths.items()):
        if tag in _SPECIFIC_COOP_TAGS:
            _record_strength(strengths, "co_op", strength)
        if tag in _COOP_TAGS:
            _record_strength(strengths, "multiplayer", strength)
