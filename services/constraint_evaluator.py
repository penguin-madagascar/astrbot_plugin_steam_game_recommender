from __future__ import annotations

from dataclasses import dataclass

from ..storage.models import GameCandidate
from .tag_normalizer import normalize_key, normalize_tag

COOPERATIVE_TAGS = {"co_op", "local_coop", "online_coop"}
MULTIPLAYER_TAGS = {*COOPERATIVE_TAGS, "multiplayer"}


@dataclass(frozen=True)
class ConstraintAssessment:
    status: str
    hits: list[str]
    violations: list[str]
    unknowns: list[str]


def evaluate_candidate_constraints(
    candidate: GameCandidate,
    required_tags: list[str],
    exclude_tags: list[str],
) -> ConstraintAssessment:
    direct_tags = set(normalize_terms([*candidate.genres, *candidate.tags]))
    required = normalize_terms(required_tags)
    excluded = normalize_terms(exclude_tags)

    hits: list[str] = []
    violations: list[str] = []
    unknowns: list[str] = []
    for tag in required:
        if requirement_is_satisfied(tag, direct_tags):
            hits.append(tag)
        elif requirement_is_contradicted(tag, direct_tags):
            violations.append(tag)
        else:
            unknowns.append(tag)

    for tag in excluded:
        if requirement_is_satisfied(tag, direct_tags) and tag not in violations:
            violations.append(tag)

    status = "violated" if violations else "unknown" if unknowns else "satisfied"
    return ConstraintAssessment(
        status=status,
        hits=hits,
        violations=violations,
        unknowns=unknowns,
    )


def requirement_is_satisfied(required_tag: str, direct_tags: set[str]) -> bool:
    if required_tag == "co_op":
        return bool(direct_tags & COOPERATIVE_TAGS)
    if required_tag == "multiplayer":
        return bool(direct_tags & MULTIPLAYER_TAGS)
    return required_tag in direct_tags


def requirement_is_contradicted(required_tag: str, direct_tags: set[str]) -> bool:
    singleplayer_only = "singleplayer" in direct_tags and not bool(direct_tags & MULTIPLAYER_TAGS)
    if required_tag in MULTIPLAYER_TAGS and singleplayer_only:
        return True
    if required_tag == "chinese" and "english_only" in direct_tags:
        return True
    return required_tag == "relaxing" and "difficult" in direct_tags


def normalize_terms(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_tag(value) or normalize_key(value).replace(" ", "_")
        if key and key not in seen:
            normalized.append(key)
            seen.add(key)
    return normalized
