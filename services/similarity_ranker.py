from __future__ import annotations

import math
import re
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
from .constraint_evaluator import ConstraintAssessment, evaluate_candidate_constraints
from .tag_normalizer import (
    candidate_canonical_tags,
    canonical_tags_from_terms,
    extract_description_terms,
)

MULTIPLAYER_TAGS = {"co_op", "local_coop", "online_coop", "multiplayer"}
POSITIVE_COMPONENT_WEIGHTS = {
    "tag_coverage": 50.0,
    "positive_reference": 15.0,
    "library_profile": 10.0,
    "review_reputation": 10.0,
    "popularity": 10.0,
    "data_completeness": 5.0,
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
    profile: SteamTagProfile,
    min_review_count: int = 50,
    min_positive_ratio: float = 0.65,
    profile_tag_weights: dict[str, float] | None = None,
) -> list[RankedGame]:
    del min_positive_ratio
    ranked: list[RankedGame] = []
    profile_weights = profile_tag_weights or {}
    idf = compute_tag_idf([ordered_tag_sequence(candidate) for candidate in candidates])
    review_prior = candidate_pool_review_prior(candidates)
    prior_strength = max(int(min_review_count), 50)
    desired_languages = merge_tags(profile.preferred_languages, profile.required_languages)

    for candidate in candidates:
        reference_appids = {*profile.reference_appids, *profile.reference_appids_dislike}
        if (
            candidate.appid is not None and int(candidate.appid) in reference_appids
        ) or is_reference_title(
            candidate.title,
            [*profile.reference_titles, *profile.reference_titles_dislike],
        ):
            continue

        constraints = evaluate_candidate_constraints(
            candidate,
            required_tags=profile.required_tags,
            exclude_tags=profile.exclude_tags,
            required_languages=profile.required_languages,
        )
        if constraints.status == "violated":
            continue

        tags = candidate_canonical_tags(candidate)
        matched = [tag for tag in profile.include_tags if tag in tags]
        missing = [tag for tag in profile.include_tags if tag not in matched]
        tag_coverage = preference_coverage(
            matched,
            profile.include_tags,
            candidate,
            desired_languages,
        )
        candidate_weights = candidate_tag_weights(candidate, idf)
        positive_reference = maximum_reference_similarity(
            candidate_weights,
            profile.positive_reference_candidates,
            idf,
        )
        negative_reference = maximum_reference_similarity(
            candidate_weights,
            profile.negative_reference_candidates,
            idf,
        )
        library_profile = profile_weight_bonus(tags, profile_weights) if profile_weights else None
        review_reputation = bayesian_review_score(
            candidate,
            prior=review_prior,
            prior_strength=prior_strength,
        )
        popularity = popularity_score(candidate.review_total)
        completeness = data_completeness_score(
            candidate, language_requested=bool(desired_languages)
        )
        positive_score = weighted_positive_score(
            tag_coverage=tag_coverage,
            positive_reference=(
                positive_reference if profile.positive_reference_candidates else None
            ),
            library_profile=library_profile,
            review_reputation=review_reputation,
            popularity=popularity,
            data_completeness=completeness,
        )
        negative_penalty = min(max(negative_reference, 0.0), 1.0) * 20.0
        unknown_penalty = unknown_constraint_penalty(constraints, profile)
        score = clamp_score(positive_score - negative_penalty - unknown_penalty)
        breakdown = ScoreBreakdown(
            tag_coverage=tag_coverage,
            positive_reference=(
                positive_reference if profile.positive_reference_candidates else None
            ),
            library_profile=library_profile,
            review_reputation=review_reputation,
            popularity=popularity,
            data_completeness=completeness,
            positive_score=positive_score,
            negative_reference_penalty=negative_penalty,
            unknown_constraints_penalty=unknown_penalty,
        )
        evidence = build_recommendation_evidence(
            candidate=candidate,
            profile=profile,
            matched_tags=matched,
            missing_tags=missing,
            constraints=constraints,
            positive_reference=positive_reference,
            negative_reference=negative_reference,
            library_profile=library_profile,
            review_reputation=review_reputation,
            popularity=popularity,
        )
        ranked.append(
            RankedGame.from_candidate(
                mark_index_source(candidate),
                score,
                breakdown,
                evidence,
            )
        )

    return sorted(ranked, key=ranked_game_sort_key)


def preference_coverage(
    matched_tags: list[str],
    include_tags: list[str],
    candidate: GameCandidate,
    desired_languages: list[str],
) -> float:
    total_weight = sum(tag_weight(tag) for tag in include_tags) + len(desired_languages)
    if total_weight <= 0:
        return 1.0
    matched_weight = sum(tag_weight(tag) for tag in matched_tags)
    if candidate.language_data_available:
        supported = set(candidate.supported_languages)
        matched_weight += sum(1.0 for language in desired_languages if language in supported)
    return min(max(matched_weight / total_weight, 0.0), 1.0)


def weighted_positive_score(**components: float | None) -> float:
    available = [
        (POSITIVE_COMPONENT_WEIGHTS[name], value)
        for name, value in components.items()
        if value is not None
    ]
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


def data_completeness_score(
    candidate: GameCandidate,
    language_requested: bool = False,
) -> float:
    checks = [
        candidate.appid is not None,
        bool(candidate.ordered_tags or candidate.tags or candidate.genres),
        candidate.review_total is not None,
        candidate.review_positive_ratio is not None,
        bool(candidate.release_date or candidate.released),
    ]
    if language_requested:
        checks.append(candidate.language_data_available)
    return sum(bool(value) for value in checks) / len(checks)


def unknown_constraint_penalty(
    constraints: ConstraintAssessment,
    profile: SteamTagProfile,
) -> float:
    total_required = len(profile.required_tags) + len(profile.required_languages)
    if not total_required or not constraints.unknowns:
        return 0.0
    return min(len(constraints.unknowns) / total_required, 1.0) * 15.0


def clamp_score(value: float) -> int:
    return min(max(round(float(value)), 0), 100)


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
    return (
        -int(game.score),
        -float(game.score_breakdown.tag_coverage),
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


def is_reference_title(title: str, reference_titles: list[str]) -> bool:
    normalized = normalize_title(title)
    return any(
        normalized == normalize_title(reference) for reference in reference_titles if reference
    )


def normalize_title(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def release_year(value: str | None) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return int(match.group(0)) if match else 0


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_candidate(data: dict[str, Any]) -> GameCandidate:
    validator = getattr(GameCandidate, "model_validate", None)
    return validator(data) if validator else GameCandidate.parse_obj(data)
