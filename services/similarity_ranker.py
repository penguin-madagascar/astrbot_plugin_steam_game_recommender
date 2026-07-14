from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..storage.models import (
    GameCandidate,
    GamePreference,
    RankedGame,
    RecommendationEvidence,
    ScoreBreakdown,
    split_language_list,
)
from .candidate_tag_evidence import (
    CandidateTagEvidence,
    build_candidate_tag_evidence,
    matches_excluded_tags,
    satisfies_required_tags,
)
from .constraint_evaluator import ConstraintAssessment
from .recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    WeightedIntentTag,
)
from .recommendation_scoring import (
    RelevanceTier,
    anchor_coverage,
    evidence_scaled_similarity,
    layer_score,
    popularity,
    quality_score,
    relevance_tier,
    semantic_score,
    wilson_lower_bound,
)
from .tag_normalizer import (
    canonical_tags_from_terms,
    extract_description_terms,
    normalize_tag,
)

POSITIVE_COMPONENT_WEIGHTS = {
    "tag_coverage": 35.0,
    "positive_reference": 25.0,
    "library_profile": 5.0,
    "review_reputation": 20.0,
    "popularity": 15.0,
}
TAG_WEIGHTS = {
    "co_op": 1.5,
    "local_coop": 1.6,
    "online_coop": 1.35,
    "multiplayer": 1.15,
    "puzzle": 1.25,
    "casual": 1.15,
    "relaxing": 1.15,
    "farming": 1.35,
    "crafting": 1.25,
    "building": 1.2,
    "management": 1.2,
}
LANGUAGE_LABELS = {
    "schinese": "简体中文",
    "tchinese": "繁体中文",
    "english": "英语",
    "japanese": "日语",
    "koreana": "韩语",
    "french": "法语",
    "german": "德语",
    "spanish": "西班牙语",
    "russian": "俄语",
    "portuguese": "葡萄牙语",
}


@dataclass(frozen=True)
class SteamTagProfile:
    include_tags: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    preferred_languages: list[str] = field(default_factory=list)
    required_languages: list[str] = field(default_factory=list)
    reference_titles: list[str] = field(default_factory=list)
    reference_titles_dislike: list[str] = field(default_factory=list)
    reference_appids: list[int] = field(default_factory=list)
    reference_appids_dislike: list[int] = field(default_factory=list)
    positive_reference_candidates: list[GameCandidate] = field(default_factory=list)
    negative_reference_candidates: list[GameCandidate] = field(default_factory=list)


def build_profile_from_preference(
    preference: GamePreference,
    reference_candidates: list[GameCandidate] | None = None,
    negative_reference_candidates: list[GameCandidate] | None = None,
) -> SteamTagProfile:
    include = canonical_tags_from_terms([*preference.genres_like, *preference.extra_tags])
    required = canonical_tags_from_terms(preference.required_tags)
    exclude = canonical_tags_from_terms(preference.genres_dislike)

    if preference.players and preference.players >= 2:
        include = merge_tags(include, ["co_op", "multiplayer"])
    if preference.difficulty and any(
        word in preference.difficulty for word in ("easy", "简单", "轻松", "休闲")
    ):
        include = merge_tags(include, ["casual", "relaxing"])
    if preference.mood:
        include = merge_tags(include, canonical_tags_from_terms([preference.mood]))

    for candidate in reference_candidates or []:
        include = merge_tags(include, reference_expansion_tags(candidate))

    return SteamTagProfile(
        include_tags=include,
        required_tags=required,
        exclude_tags=exclude,
        preferred_languages=split_language_list(preference.preferred_languages),
        required_languages=split_language_list(preference.required_languages),
        reference_titles=list(preference.reference_games_like),
        reference_titles_dislike=list(preference.reference_games_dislike),
        reference_appids=[
            int(candidate.appid)
            for candidate in reference_candidates or []
            if candidate.appid is not None
        ],
        reference_appids_dislike=[
            int(candidate.appid)
            for candidate in negative_reference_candidates or []
            if candidate.appid is not None
        ],
        positive_reference_candidates=list(reference_candidates or []),
        negative_reference_candidates=list(negative_reference_candidates or []),
    )


def rank_steam_candidates(
    candidates: list[GameCandidate],
    intent_or_profile: RecommendationIntent | SteamTagProfile,
    min_review_count: int = 50,
    min_positive_ratio: float = 0.65,
    profile_tag_weights: dict[str, float] | None = None,
    positive_component_weights: Mapping[str, Any] | None = None,
    *,
    positive_reference_candidates: list[GameCandidate] | None = None,
    negative_reference_candidates: list[GameCandidate] | None = None,
    retrieval_ranks: Mapping[int, int] | None = None,
    language_profile: SteamTagProfile | None = None,
) -> list[RankedGame]:
    del min_review_count, min_positive_ratio, positive_component_weights
    intent, compatibility_profile = resolve_rank_intent(intent_or_profile)
    profile = language_profile or compatibility_profile or SteamTagProfile()
    positive_references = (
        list(positive_reference_candidates)
        if positive_reference_candidates is not None
        else list(profile.positive_reference_candidates)
    )
    negative_references = (
        list(negative_reference_candidates)
        if negative_reference_candidates is not None
        else list(profile.negative_reference_candidates)
    )
    reference_appids = {
        int(reference.appid)
        for reference in [*positive_references, *negative_references]
        if reference.appid is not None
    }
    if compatibility_profile is not None:
        reference_appids.update(compatibility_profile.reference_appids)
        reference_appids.update(compatibility_profile.reference_appids_dislike)

    required_tags = [
        intent_tag.tag
        for intent_tag in intent.tags
        if intent_tag.role is IntentTagRole.REQUIRED
    ]
    excluded_tags = [
        intent_tag.tag
        for intent_tag in intent.tags
        if intent_tag.role is IntentTagRole.EXCLUDE
    ]
    supporting_weights, library_tags = supporting_query_weights(
        intent,
        profile_tag_weights or {},
    )
    negative_evidence = [
        build_candidate_tag_evidence(reference) for reference in negative_references
    ]
    ranked: list[RankedGame] = []
    for input_rank, candidate in enumerate(candidates, start=1):
        if candidate.appid is not None and int(candidate.appid) in reference_appids:
            continue
        if candidate.coming_soon and not intent.allow_unreleased:
            continue
        candidate_evidence = build_candidate_tag_evidence(candidate)
        if not satisfies_required_tags(candidate_evidence, required_tags):
            continue
        if matches_excluded_tags(candidate_evidence, excluded_tags):
            continue

        anchor_value = anchor_coverage(intent, candidate_evidence)
        tier = relevance_tier(intent, candidate_evidence)
        supporting_value = evidence_scaled_similarity(
            supporting_weights,
            candidate_evidence.supporting,
        )
        negative_value = maximum_evidence_similarity(
            candidate_evidence,
            negative_evidence,
        )
        semantic_value = semantic_score(
            intent,
            anchor_value,
            supporting_value,
            negative_value,
        )
        wilson_value = (
            wilson_lower_bound(candidate.review_positive_ratio, candidate.review_total)
            if candidate.review_positive_ratio is not None
            else 0.0
        )
        popularity_value = popularity(candidate.review_total)
        quality_value = quality_score(
            candidate.review_positive_ratio,
            candidate.review_total,
        )
        layer_value = layer_score(
            semantic_value,
            quality_value,
            intent.quality_intent,
        )
        retrieval_rank = resolve_retrieval_rank(
            candidate,
            input_rank,
            retrieval_ranks,
        )
        language_adjustment = language_preference_adjustment(candidate, profile)
        breakdown = ScoreBreakdown(
            relevance_tier=tier.value,
            anchor_coverage=anchor_value,
            supporting_similarity=supporting_value,
            negative_reference_similarity=negative_value,
            semantic_score=semantic_value,
            wilson_lower_bound=wilson_value,
            quality_score=quality_value,
            layer_score=layer_value,
            retrieval_rank=retrieval_rank,
            tag_coverage=anchor_value if tier is not RelevanceTier.BROAD else supporting_value,
            positive_reference=None,
            library_profile=library_match_score(candidate_evidence, library_tags),
            review_reputation=wilson_value,
            popularity=popularity_value,
            positive_score=layer_value * 100,
            negative_reference_penalty=min(negative_value * 25.0, 20.0),
            unknown_constraints_penalty=0.0,
            language_adjustment=language_adjustment,
        )
        explanation = build_anchor_tier_evidence(
            candidate=candidate,
            intent=intent,
            profile=profile,
            candidate_evidence=candidate_evidence,
            tier=tier,
            negative_similarity=negative_value,
            wilson_value=wilson_value,
            library_tags=library_tags,
            has_positive_references=bool(positive_references),
        )
        ranked.append(
            RankedGame.from_candidate(
                mark_index_source(candidate),
                clamp_score(layer_value * 100),
                breakdown,
                explanation,
            )
        )

    return sorted(ranked, key=ranked_game_sort_key)


def resolve_rank_intent(
    value: RecommendationIntent | SteamTagProfile,
) -> tuple[RecommendationIntent, SteamTagProfile | None]:
    if isinstance(value, RecommendationIntent):
        return value, None
    tags: list[WeightedIntentTag] = []
    seen: set[str] = set()
    groups = (
        (value.required_tags, IntentTagRole.REQUIRED, 1.0),
        (value.include_tags, IntentTagRole.ANCHOR, 1.0),
        (value.exclude_tags, IntentTagRole.EXCLUDE, 1.0),
    )
    for terms, role, weight in groups:
        for canonical in canonical_tags_from_terms(terms):
            if canonical in seen:
                continue
            tags.append(
                WeightedIntentTag(
                    canonical,
                    role,
                    IntentTagSource.EXPLICIT,
                    weight,
                )
            )
            seen.add(canonical)
    return (
        RecommendationIntent(
            tags=tuple(tags),
            references=(),
            quality_intent=QualityIntent.NORMAL,
            allow_unreleased=False,
        ),
        value,
    )


def supporting_query_weights(
    intent: RecommendationIntent,
    profile_tag_weights: Mapping[str, float],
) -> tuple[dict[str, float], set[str]]:
    weights = {
        intent_tag.tag: min(max(float(intent_tag.weight), 0.0), 1.0)
        for intent_tag in intent.tags
        if intent_tag.role is IntentTagRole.SUPPORTING and intent_tag.weight > 0.0
    }
    occupied = {intent_tag.tag for intent_tag in intent.tags}
    library_tags: set[str] = set()
    for raw_tag, raw_weight in profile_tag_weights.items():
        canonical = normalize_tag(raw_tag)
        if not canonical or canonical in occupied:
            continue
        try:
            profile_weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(profile_weight) or profile_weight <= 0.0:
            continue
        weights[canonical] = 0.25 * min(profile_weight, 1.0)
        library_tags.add(canonical)
    return weights, library_tags


def library_match_score(
    evidence: CandidateTagEvidence,
    library_tags: set[str],
) -> float | None:
    if not library_tags:
        return None
    return max((evidence.supporting.get(tag, 0.0) for tag in library_tags), default=0.0)


def maximum_evidence_similarity(
    candidate: CandidateTagEvidence,
    references: list[CandidateTagEvidence],
) -> float:
    return max(
        (
            evidence_vector_cosine(candidate.supporting, reference.supporting)
            for reference in references
        ),
        default=0.0,
    )


def evidence_vector_cosine(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    if not left or not right:
        return 0.0
    dot = sum(float(value) * float(right.get(tag, 0.0)) for tag, value in left.items())
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return min(max(dot / (left_norm * right_norm), 0.0), 1.0)


def resolve_retrieval_rank(
    candidate: GameCandidate,
    input_rank: int,
    retrieval_ranks: Mapping[int, int] | None,
) -> int:
    if candidate.appid is None or not retrieval_ranks:
        return input_rank
    try:
        rank = int(retrieval_ranks.get(int(candidate.appid), input_rank))
    except (TypeError, ValueError):
        return input_rank
    return rank if rank > 0 else input_rank


def preference_coverage(
    matched_tags: list[str],
    include_tags: list[str],
) -> float:
    total_weight = sum(tag_weight(tag) for tag in include_tags)
    if total_weight <= 0:
        return 1.0
    matched_weight = sum(tag_weight(tag) for tag in matched_tags)
    return min(max(matched_weight / total_weight, 0.0), 1.0)


def resolve_positive_component_weights(
    weights: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    source = weights or {}
    resolved: dict[str, float] = {}
    for name, default in POSITIVE_COMPONENT_WEIGHTS.items():
        try:
            value = float(source.get(name, default))
        except (TypeError, ValueError):
            value = default
        if not math.isfinite(value):
            value = default
        resolved[name] = min(max(value, 0.0), 100.0)
    if not any(resolved.values()):
        return dict(POSITIVE_COMPONENT_WEIGHTS)
    return resolved


def weighted_positive_score(
    positive_component_weights: Mapping[str, Any] | None = None,
    **components: float | None,
) -> float:
    weights = resolve_positive_component_weights(positive_component_weights)
    available = [(weights[name], value) for name, value in components.items() if value is not None]
    total_weight = sum(weight for weight, _value in available)
    if not total_weight:
        return 0.0
    return (
        sum(weight * min(max(float(value), 0.0), 1.0) for weight, value in available)
        / total_weight
        * 100
    )


def popularity_score(review_total: int | None) -> float:
    total = max(int(review_total or 0), 0)
    return min(math.log10(total + 1) / 5, 1.0)


def language_preference_adjustment(
    candidate: GameCandidate,
    profile: SteamTagProfile,
) -> float:
    requested = merge_tags(profile.preferred_languages, profile.required_languages)
    if not requested:
        return 0.0
    if not candidate.language_data_available:
        return -2.0

    supported = set(candidate.supported_languages)
    if any(language not in supported for language in profile.required_languages):
        return -10.0
    if any(language not in supported for language in profile.preferred_languages):
        return -5.0
    return 0.0


def unknown_constraint_penalty(
    constraints: ConstraintAssessment,
    profile: SteamTagProfile,
) -> float:
    total_required = len(profile.required_tags)
    if not total_required or not constraints.unknowns:
        return 0.0
    return min(len(constraints.unknowns) / total_required, 1.0) * 15.0


def clamp_score(value: float) -> int:
    return min(max(round(float(value)), 0), 100)


def build_anchor_tier_evidence(
    candidate: GameCandidate,
    intent: RecommendationIntent,
    profile: SteamTagProfile,
    candidate_evidence: CandidateTagEvidence,
    tier: RelevanceTier,
    negative_similarity: float,
    wilson_value: float,
    library_tags: set[str],
    has_positive_references: bool,
) -> list[RecommendationEvidence]:
    evidence: list[RecommendationEvidence] = []
    anchors = [
        intent_tag.tag
        for intent_tag in intent.tags
        if intent_tag.role is IntentTagRole.ANCHOR
    ]
    matched_anchors = [
        tag for tag in anchors if candidate_evidence.direct.get(tag, 0.0) > 0.0
    ]
    if matched_anchors:
        evidence.append(
            evidence_item(
                "core_match",
                "core",
                "positive",
                f"命中核心标签：{'、'.join(matched_anchors[:5])}",
            )
        )
    if tier in {RelevanceTier.B, RelevanceTier.C}:
        missing = [
            tag for tag in anchors if candidate_evidence.direct.get(tag, 0.0) < 0.60
        ]
        evidence.append(
            evidence_item(
                "core_missing",
                "core",
                "uncertain",
                f"宽松匹配：缺失或证据不足的核心特征为{'、'.join((missing or anchors)[:5])}",
                important=True,
            )
        )

    supporting = [
        intent_tag.tag
        for intent_tag in intent.tags
        if intent_tag.role is IntentTagRole.SUPPORTING
        and candidate_evidence.supporting.get(intent_tag.tag, 0.0) > 0.0
    ]
    if supporting:
        evidence.append(
            evidence_item(
                "supporting_match",
                "supporting",
                "positive",
                f"命中辅助标签：{'、'.join(supporting[:5])}",
            )
        )
    matched_library = [
        tag for tag in library_tags if candidate_evidence.supporting.get(tag, 0.0) > 0.0
    ]
    if matched_library:
        evidence.append(
            evidence_item(
                "library_profile",
                "library",
                "positive",
                f"命中游戏库辅助偏好：{'、'.join(sorted(matched_library)[:5])}",
            )
        )
    if has_positive_references:
        evidence.append(
            evidence_item(
                "reference_expansion",
                "reference",
                "positive",
                "已从解析成功的参考游戏提取核心与辅助标签",
            )
        )
    if negative_similarity > 0.0:
        evidence.append(
            evidence_item(
                "negative_reference",
                "reference",
                "negative",
                f"与负向参考的玩法标签相似度为 {negative_similarity:.0%}",
                important=negative_similarity >= 0.25,
            )
        )

    if candidate.review_total and candidate.review_positive_ratio is not None:
        evidence.append(
            evidence_item(
                "review_confidence",
                "reviews",
                "positive" if wilson_value >= 0.70 else "uncertain",
                (
                    f"Steam 好评率 {candidate.review_positive_ratio:.0%}，"
                    f"共 {candidate.review_total} 条评测；"
                    f"Wilson 置信下界 {wilson_value:.0%}"
                ),
            )
        )
    else:
        evidence.append(
            evidence_item(
                "review_unknown",
                "reviews",
                "uncertain",
                "Steam 评测缺失或为零，口碑置信度不足",
            )
        )
    if intent.quality_intent is QualityIntent.MAINSTREAM:
        evidence.append(
            evidence_item(
                "mainstream_intent",
                "quality",
                "positive",
                "按高知名度/大作倾向提高成熟口碑在层内的权重",
            )
        )

    append_language_evidence(evidence, candidate, profile)
    return dedupe_evidence(evidence)


def build_recommendation_evidence(
    candidate: GameCandidate,
    profile: SteamTagProfile,
    matched_tags: list[str],
    missing_tags: list[str],
    constraints: ConstraintAssessment,
    positive_reference: float,
    negative_reference: float,
    library_profile: float | None,
    review_reputation: float,
    popularity: float,
) -> list[RecommendationEvidence]:
    evidence: list[RecommendationEvidence] = []
    if matched_tags:
        evidence.append(
            evidence_item(
                "tag_match",
                "preference",
                "positive",
                f"匹配偏好标签：{'、'.join(matched_tags[:5])}",
            )
        )
    elif profile.include_tags:
        evidence.append(
            evidence_item(
                "tag_mismatch",
                "preference",
                "negative",
                f"未命中主要偏好标签：{'、'.join(missing_tags[:5])}",
                important=True,
            )
        )
    if profile.positive_reference_candidates:
        evidence.append(
            evidence_item(
                "positive_reference",
                "reference",
                "positive",
                f"与正向参考的玩法标签相似度为 {positive_reference:.0%}",
            )
        )
    if profile.negative_reference_candidates and negative_reference > 0:
        evidence.append(
            evidence_item(
                "negative_reference",
                "reference",
                "negative",
                f"与不喜欢的参考游戏相似度为 {negative_reference:.0%}",
                important=negative_reference >= 0.25,
            )
        )
    if library_profile is not None and library_profile > 0:
        evidence.append(
            evidence_item(
                "library_profile",
                "library",
                "positive",
                f"命中个人游戏库画像，匹配度为 {library_profile:.0%}",
            )
        )

    if candidate.review_total is not None and candidate.review_positive_ratio is not None:
        evidence.append(
            evidence_item(
                "review_reputation",
                "reviews",
                "positive" if review_reputation >= 0.7 else "negative",
                (
                    f"Steam 好评率 {candidate.review_positive_ratio:.0%}，"
                    f"共 {candidate.review_total} 条评测"
                ),
            )
        )
    else:
        evidence.append(
            evidence_item(
                "review_unknown",
                "reviews",
                "uncertain",
                "Steam 评测数据尚未获取",
            )
        )
    if candidate.review_total:
        evidence.append(
            evidence_item(
                "popularity",
                "popularity",
                "positive",
                f"评测规模对应的知名度指标为 {popularity:.0%}",
            )
        )

    append_language_evidence(evidence, candidate, profile)
    for unknown in constraints.unknowns:
        if unknown.startswith("language:"):
            continue
        evidence.append(
            evidence_item(
                f"constraint_unknown:{unknown}",
                "constraint",
                "uncertain",
                f"硬条件尚未确认：{unknown}",
                important=True,
            )
        )
    return dedupe_evidence(evidence)


def append_language_evidence(
    evidence: list[RecommendationEvidence],
    candidate: GameCandidate,
    profile: SteamTagProfile,
) -> None:
    requested = merge_tags(profile.preferred_languages, profile.required_languages)
    if not requested:
        return
    supported = set(candidate.supported_languages)
    required = set(profile.required_languages)
    for language in requested:
        label = language_label(language)
        if not candidate.language_data_available:
            evidence.append(
                evidence_item(
                    f"language_unknown:{language}",
                    "language",
                    "uncertain",
                    f"Steam 语言数据缺失，无法确认是否支持{label}",
                    important=language in required,
                )
            )
        elif language in supported:
            evidence.append(
                evidence_item(
                    f"language_supported:{language}",
                    "language",
                    "positive",
                    f"Steam 明确标注支持{label}",
                )
            )
        else:
            evidence.append(
                evidence_item(
                    f"language_unsupported:{language}",
                    "language",
                    "negative",
                    f"Steam 语言列表未标注支持{label}",
                    important=language in required,
                )
            )


def evidence_item(
    evidence_id: str,
    category: str,
    sentiment: str,
    text: str,
    important: bool = False,
) -> RecommendationEvidence:
    return RecommendationEvidence(
        evidence_id=evidence_id,
        category=category,
        sentiment=sentiment,
        text=text,
        important=important,
    )


def dedupe_evidence(values: list[RecommendationEvidence]) -> list[RecommendationEvidence]:
    result: list[RecommendationEvidence] = []
    seen: set[str] = set()
    for value in values:
        if value.evidence_id and value.evidence_id not in seen:
            result.append(value)
            seen.add(value.evidence_id)
    return result


def language_label(language: str) -> str:
    return LANGUAGE_LABELS.get(language, language)


def ranked_game_sort_key(game: RankedGame) -> tuple[Any, ...]:
    tier_order = {
        RelevanceTier.A.value: 0,
        RelevanceTier.BROAD.value: 0,
        RelevanceTier.B.value: 1,
        RelevanceTier.C.value: 2,
    }
    breakdown = game.score_breakdown
    raw_layer = float(breakdown.layer_score)
    has_scored_layer = raw_layer != 0.0
    if not has_scored_layer and game.score:
        raw_layer = float(game.score) / 100.0
    effective_layer = (
        raw_layer + float(breakdown.budget_adjustment) / 100.0
        if has_scored_layer
        else raw_layer
    )
    retrieval_rank = int(breakdown.retrieval_rank)
    return (
        tier_order.get(breakdown.relevance_tier, 3),
        -effective_layer,
        -raw_layer,
        retrieval_rank if retrieval_rank > 0 else 1_000_000_000,
        -int(game.review_total or 0),
        -release_year(game.release_date or game.released),
        game.title.casefold(),
    )


def mark_index_source(candidate: GameCandidate) -> GameCandidate:
    if "steam_index" in candidate.internal_source_markers:
        return candidate
    data = dump_model(candidate)
    data["internal_source_markers"] = [
        *candidate.internal_source_markers,
        "steam_index",
    ]
    return validate_candidate(data)


def reference_expansion_tags(candidate: GameCandidate) -> list[str]:
    ignored = {"singleplayer", "chinese"}
    direct_tags = canonical_tags_from_terms(
        [*candidate.ordered_tags, *candidate.tags, *candidate.genres]
    )
    return [tag for tag in direct_tags if tag not in ignored]


def ordered_tag_sequence(candidate: GameCandidate) -> list[str]:
    inferred = [*candidate.inferred_tags]
    if candidate.description:
        inferred.extend(extract_description_terms(candidate.description))
    return merge_tags(
        canonical_tags_from_terms(candidate.ordered_tags),
        merge_tags(
            canonical_tags_from_terms([*candidate.tags, *candidate.genres]),
            canonical_tags_from_terms(inferred),
        ),
    )


def compute_tag_idf(documents: list[list[str]]) -> dict[str, float]:
    document_count = len(documents)
    frequencies: dict[str, int] = {}
    for document in documents:
        for tag in set(document):
            frequencies[tag] = frequencies.get(tag, 0) + 1
    return {
        tag: math.log((document_count + 1) / (frequency + 1)) + 1
        for tag, frequency in frequencies.items()
    }


def position_weight(position: int) -> float:
    return 1 / math.log2(position + 2)


def ordered_sequence_weights(
    tags: list[str],
    idf: dict[str, float],
    scale: float = 1.0,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for position, tag in enumerate(canonical_tags_from_terms(tags)):
        value = idf.get(tag, 1.0) * position_weight(position) * scale
        weights[tag] = max(weights.get(tag, 0.0), value)
    return weights


def candidate_tag_weights(
    candidate: GameCandidate,
    idf: dict[str, float],
) -> dict[str, float]:
    weights = ordered_sequence_weights(candidate.ordered_tags, idf)
    for tag in canonical_tags_from_terms([*candidate.tags, *candidate.genres]):
        weights[tag] = max(weights.get(tag, 0.0), idf.get(tag, 1.0))
    inferred = [*candidate.inferred_tags]
    if candidate.description:
        inferred.extend(extract_description_terms(candidate.description))
    for tag in canonical_tags_from_terms(inferred):
        weights[tag] = max(weights.get(tag, 0.0), idf.get(tag, 1.0) * 0.5)
    return weights


def weighted_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(tag, 0.0) for tag, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return min(max(dot / (left_norm * right_norm), 0.0), 1.0)


def ordered_tfidf_cosine(
    left: list[str],
    right: list[str],
    idf: dict[str, float],
) -> float:
    return weighted_cosine(
        ordered_sequence_weights(left, idf),
        ordered_sequence_weights(right, idf),
    )


def maximum_reference_similarity(
    candidate_weights: dict[str, float],
    reference_candidates: list[GameCandidate],
    idf: dict[str, float],
) -> float:
    return max(
        (
            weighted_cosine(
                candidate_weights,
                candidate_tag_weights(reference, idf),
            )
            for reference in reference_candidates
        ),
        default=0.0,
    )


def candidate_pool_review_prior(candidates: list[GameCandidate]) -> float:
    ratios = [
        min(max(float(candidate.review_positive_ratio), 0.0), 1.0)
        for candidate in candidates
        if candidate.review_positive_ratio is not None
    ]
    return sum(ratios) / len(ratios) if ratios else 0.75


def bayesian_review_score(
    candidate: GameCandidate,
    prior: float,
    prior_strength: int,
) -> float:
    baseline = min(max(float(prior), 0.0), 1.0)
    total = max(int(candidate.review_total or 0), 0)
    ratio = candidate.review_positive_ratio
    if total <= 0 or ratio is None:
        return baseline
    strength = max(int(prior_strength), 1)
    observed = min(max(float(ratio), 0.0), 1.0)
    return (total * observed + strength * baseline) / (total + strength)


def weighted_overlap(matched: list[str], include_tags: list[str]) -> float:
    if not include_tags:
        return 1.0
    matched_weight = sum(tag_weight(tag) for tag in matched)
    total_weight = sum(tag_weight(tag) for tag in include_tags)
    return matched_weight / total_weight if total_weight else 0.0


def tag_weight(tag: str) -> float:
    return TAG_WEIGHTS.get(tag, 1.0)


def profile_weight_bonus(tags: list[str], weights: dict[str, float]) -> float:
    if not weights:
        return 0.0
    matched = [min(max(float(weights[tag]), 0.0), 1.0) for tag in tags if tag in weights]
    if not matched:
        return 0.0
    return min(sum(matched) / 3, 1.0)


def merge_tags(left: list[str], right: list[str]) -> list[str]:
    result = list(left)
    for tag in right:
        if tag and tag not in result:
            result.append(tag)
    return result


def copy_ranked_game(game: RankedGame, update: dict[str, Any]) -> RankedGame:
    copier = getattr(game, "model_copy", None)
    if copier:
        return copier(update=update)
    return game.copy(update=update)


def release_year(value: str | None) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return int(match.group(0)) if match else 0


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_candidate(data: dict[str, Any]) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)
