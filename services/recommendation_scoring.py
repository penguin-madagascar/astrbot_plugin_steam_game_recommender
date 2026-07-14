from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum

from .candidate_tag_evidence import CandidateTagEvidence
from .recommendation_intent import (
    IntentTagRole,
    QualityIntent,
    RecommendationIntent,
)


class RelevanceTier(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    BROAD = "broad"


def anchor_coverage(
    intent: RecommendationIntent,
    evidence: CandidateTagEvidence,
) -> float:
    anchors = _anchor_weights(intent)
    total_weight = sum(anchors.values())
    if total_weight <= 0.0:
        return 0.0
    matched_weight = sum(
        weight * _unit_interval(evidence.direct.get(tag, 0.0))
        for tag, weight in anchors.items()
    )
    return _unit_interval(matched_weight / total_weight)


def relevance_tier(
    intent: RecommendationIntent,
    evidence: CandidateTagEvidence,
) -> RelevanceTier:
    if not _anchor_weights(intent):
        return RelevanceTier.BROAD
    coverage = anchor_coverage(intent, evidence)
    if coverage >= 0.60:
        return RelevanceTier.A
    if coverage > 0.0:
        return RelevanceTier.B
    return RelevanceTier.C


def evidence_scaled_similarity(
    query_weights: Mapping[str, float],
    candidate_supporting: Mapping[str, float],
) -> float:
    weights = {
        tag: weight
        for tag, raw_weight in query_weights.items()
        if (weight := _unit_interval(raw_weight)) > 0.0
    }
    if not weights:
        return 0.0

    weighted_evidence = {
        tag: weight * _unit_interval(candidate_supporting.get(tag, 0.0))
        for tag, weight in weights.items()
    }
    query_norm = math.sqrt(sum(weight**2 for weight in weights.values()))
    evidence_norm = math.sqrt(sum(value**2 for value in weighted_evidence.values()))
    if query_norm <= 0.0 or evidence_norm <= 0.0:
        return 0.0

    dot_product = sum(
        weight * weighted_evidence[tag] for tag, weight in weights.items()
    )
    cosine = dot_product / (query_norm * evidence_norm)
    coverage = sum(weighted_evidence.values()) / sum(weights.values())
    return _unit_interval(cosine * coverage)


def wilson_lower_bound(
    positive_ratio: float,
    review_count: int | float | None,
    z: float = 1.96,
) -> float:
    count = _normalize_review_count(review_count)
    if count <= 0:
        return 0.0
    ratio = _unit_interval(positive_ratio)
    confidence = abs(float(z))
    if not math.isfinite(confidence):
        confidence = 1.96
    squared = confidence**2
    numerator = ratio + squared / (2 * count) - confidence * math.sqrt(
        (ratio * (1.0 - ratio) + squared / (4 * count)) / count
    )
    return _unit_interval(numerator / (1.0 + squared / count))


def popularity(review_count: int | float | None) -> float:
    count = _normalize_review_count(review_count)
    return _unit_interval(math.log10(count + 1) / 5)


def quality_score(
    positive_ratio: float | None,
    review_count: int | float | None,
) -> float:
    count = _normalize_review_count(review_count)
    if positive_ratio is None or count <= 0:
        return 0.0
    ratio = float(positive_ratio)
    if not math.isfinite(ratio):
        return 0.0
    return _unit_interval(
        0.60 * wilson_lower_bound(ratio, count) + 0.40 * popularity(count)
    )


def semantic_score(
    intent: RecommendationIntent,
    anchor_coverage_value: float,
    supporting_similarity: float,
    negative_similarity: float = 0.0,
) -> float:
    anchor = _unit_interval(anchor_coverage_value)
    supporting = _unit_interval(supporting_similarity)
    negative = _unit_interval(negative_similarity)
    if _anchor_weights(intent):
        positive = 0.70 * anchor + 0.30 * supporting
    else:
        positive = supporting
    return _unit_interval(positive - 0.25 * negative)


def layer_score(
    semantic: float,
    quality: float,
    quality_intent: QualityIntent | str,
) -> float:
    semantic_component = _unit_interval(semantic)
    quality_component = _unit_interval(quality)
    if QualityIntent(quality_intent) == QualityIntent.MAINSTREAM:
        return 0.55 * semantic_component + 0.45 * quality_component
    return 0.70 * semantic_component + 0.30 * quality_component


def _anchor_weights(intent: RecommendationIntent) -> dict[str, float]:
    return {
        tag.tag: weight
        for tag in intent.tags
        if tag.role == IntentTagRole.ANCHOR
        and (weight := _unit_interval(tag.weight)) > 0.0
    }


def _normalize_review_count(value: int | float | None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return 0
    if value < 0:
        return 0
    return int(value)


def _unit_interval(value: float) -> float:
    number = float(value)
    if math.isnan(number):
        return 0.0
    return min(max(number, 0.0), 1.0)
