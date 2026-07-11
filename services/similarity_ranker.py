from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..storage.models import GameCandidate, GameFacts, GamePreference, RankedGame
from .constraint_evaluator import evaluate_candidate_constraints
from .tag_normalizer import (
    candidate_canonical_tags,
    canonical_tags_from_terms,
    extract_description_terms,
)

TIER_ORDER = {"strong": 0, "recommended": 1, "backup": 2}
MULTIPLAYER_TAGS = {"co_op", "local_coop", "online_coop", "multiplayer"}
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
    "chinese": 1.1,
}


@dataclass(frozen=True)
class SteamTagProfile:
    include_tags: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    reference_titles: list[str] = field(default_factory=list)
    reference_titles_dislike: list[str] = field(default_factory=list)
    positive_reference_tag_sequences: list[list[str]] = field(default_factory=list)
    negative_reference_tag_sequences: list[list[str]] = field(default_factory=list)


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
    if preference.language and ("中文" in preference.language or "chinese" in preference.language):
        include = merge_tags(include, ["chinese"])
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
        reference_titles=list(preference.reference_games_like),
        reference_titles_dislike=list(preference.reference_games_dislike),
        positive_reference_tag_sequences=[
            ordered_tag_sequence(candidate) for candidate in reference_candidates or []
        ],
        negative_reference_tag_sequences=[
            ordered_tag_sequence(candidate) for candidate in negative_reference_candidates or []
        ],
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
    for candidate in candidates:
        if is_reference_title(
            candidate.title,
            [*profile.reference_titles, *profile.reference_titles_dislike],
        ) or is_reference_query(candidate):
            continue
        constraints = evaluate_candidate_constraints(
            candidate,
            required_tags=profile.required_tags,
            exclude_tags=profile.exclude_tags,
        )
        if constraints.status == "violated":
            continue
        tags = candidate_canonical_tags(candidate)

        matched = [tag for tag in profile.include_tags if tag in tags]
        missing = [tag for tag in profile.include_tags if tag not in matched]
        tag_coverage = weighted_overlap(matched, profile.include_tags)
        candidate_weights = candidate_tag_weights(candidate, idf)
        positive_reference = maximum_reference_similarity(
            candidate_weights,
            profile.positive_reference_tag_sequences,
            idf,
        )
        negative_reference = maximum_reference_similarity(
            candidate_weights,
            profile.negative_reference_tag_sequences,
            idf,
        )
        library_profile = profile_weight_bonus(tags, profile_weights) if profile_weights else None
        data_confidence = confidence_for(candidate, tags)
        review_reputation = bayesian_review_score(
            candidate,
            prior=review_prior,
            prior_strength=prior_strength,
        )
        review_confidence = min(review_reputation * 0.8 + data_confidence * 0.2, 1.0)
        base_relevance = blend_relevance_components(
            tag_coverage=tag_coverage if profile.include_tags else None,
            positive_reference=(
                positive_reference if profile.positive_reference_tag_sequences else None
            ),
            library_profile=library_profile,
            review_confidence=review_confidence,
            negative_reference=negative_reference,
        )

        tier = classify_similarity_tier(base_relevance)
        if constraints.status == "unknown" and tier == "strong":
            tier = "recommended"
        facts = GameFacts(
            constraint_status=constraints.status,
            constraint_hits=constraints.hits,
            constraint_violations=constraints.violations,
            constraint_unknowns=constraints.unknowns,
            matched_like_terms=matched,
            missing_like_terms=missing,
            required_hits=constraints.hits,
            required_misses=constraints.violations,
            required_unknowns=constraints.unknowns,
            has_coop=bool(set(tags) & {"co_op", "local_coop", "online_coop"}),
            has_local_coop="local_coop" in tags,
            has_online_coop="online_coop" in tags,
            ordinary_multiplayer=(
                "multiplayer" in tags
                and not bool(set(tags) & {"co_op", "local_coop", "online_coop"})
            ),
            singleplayer_only="singleplayer" in tags and not bool(set(tags) & MULTIPLAYER_TAGS),
            chinese="chinese" in tags,
            reference_similarity=positive_reference,
            match_coverage=(
                len(matched) / len(profile.include_tags) if profile.include_tags else 0.0
            ),
            match_score=tag_coverage,
            base_tag_score=tag_coverage,
            profile_weight_bonus=library_profile or 0.0,
            confidence=data_confidence,
            tag_coverage_score=tag_coverage,
            positive_reference_score=positive_reference,
            negative_reference_score=negative_reference,
            library_profile_score=library_profile or 0.0,
            review_confidence_score=review_confidence,
            base_relevance_score=base_relevance,
        )
        score = similarity_score(facts)
        fit_points = fit_points_for(candidate, matched, tag_coverage)
        if positive_reference > 0:
            fit_points = merge_tags(
                fit_points,
                [f"正向参考相似度 {positive_reference:.0%}"],
            )
        if constraints.hits:
            fit_points = merge_tags(
                fit_points,
                [f"Steam 信息确认硬条件：{'、'.join(constraints.hits)}"],
            )
        if facts.profile_weight_bonus > 0:
            fit_points = merge_tags(
                fit_points,
                [profile_fit_point(tags, profile_weights)],
            )
        risk_points = risk_points_for(
            candidate,
            missing,
            tier,
            min_review_count,
            constraints.unknowns,
            negative_reference,
        )
        ranked.append(
            copy_ranked_game(
                RankedGame.from_candidate(candidate, score, fit_points, risk_points),
                {
                    "tier": tier,
                    "fit_points": fit_points,
                    "risk_points": risk_points,
                    "facts": facts,
                    "index_source": candidate.index_source or "steam_index",
                },
            )
        )

    return sorted(
        ranked,
        key=lambda game: (
            TIER_ORDER.get(game.tier, 9),
            -game.facts.base_relevance_score,
            -game.facts.match_score,
            -float(game.score),
            -release_year(game.release_date or game.released),
            game.title,
        ),
    )


def select_diverse_results(
    games: list[RankedGame],
    limit: int,
    group_by: str = "primary",
    penalty_weight: float = 15,
) -> list[RankedGame]:
    if limit <= 0 or not games:
        return []

    selected: list[RankedGame] = []
    groups = tier_groups(games) if group_by == "tier" else primary_match_groups(games)
    for group in groups:
        group_selected: list[RankedGame] = []
        remaining = list(group)
        while remaining and len(selected) < limit:
            best_index, best_game, best_penalty = max(
                (
                    (
                        index,
                        game,
                        diversity_penalty_for(game, group_selected),
                    )
                    for index, game in enumerate(remaining)
                ),
                key=lambda item: (
                    float(item[1].score) - item[2] * penalty_weight,
                    -item[0],
                ),
            )
            if best_penalty > 0:
                best_game = copy_ranked_game(
                    best_game,
                    {
                        "facts": copy_facts(
                            best_game.facts,
                            {"diversity_penalty": best_penalty},
                        )
                    },
                )
            selected.append(best_game)
            group_selected.append(best_game)
            del remaining[best_index]
        if len(selected) >= limit:
            break
    return selected


def tier_groups(games: list[RankedGame]) -> list[list[RankedGame]]:
    groups: list[list[RankedGame]] = []
    current: list[RankedGame] = []
    current_key: int | None = None
    for game in games:
        key = TIER_ORDER.get(game.tier, 9)
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(game)
        current_key = key
    if current:
        groups.append(current)
    return groups


def primary_match_groups(games: list[RankedGame]) -> list[list[RankedGame]]:
    groups: list[list[RankedGame]] = []
    current: list[RankedGame] = []
    current_key: tuple[int, float] | None = None
    for game in games:
        key = (TIER_ORDER.get(game.tier, 9), round(game.facts.match_score, 6))
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(game)
        current_key = key
    if current:
        groups.append(current)
    return groups


def diversity_penalty_for(game: RankedGame, selected: list[RankedGame]) -> float:
    if not selected:
        return 0.0
    tags = set(diversity_tags_for(game))
    if not tags:
        return 0.0
    used_tags: set[str] = set()
    for selected_game in selected:
        used_tags.update(diversity_tags_for(selected_game))
    if not used_tags:
        return 0.0
    return min(len(tags & used_tags) / len(tags), 1.0)


def diversity_tags_for(game: RankedGame) -> list[str]:
    ignored = {
        *MULTIPLAYER_TAGS,
        "singleplayer",
        "chinese",
        *game.facts.matched_like_terms,
    }
    return [tag for tag in candidate_canonical_tags(game) if tag not in ignored]


def copy_facts(facts: GameFacts, update: dict[str, Any]) -> GameFacts:
    copier = getattr(facts, "model_copy", None)
    if copier:
        return copier(update=update)
    return facts.copy(update=update)


def reference_expansion_tags(candidate: GameCandidate) -> list[str]:
    ignored = {"singleplayer", "chinese"}
    direct_tags = canonical_tags_from_terms([*candidate.tags, *candidate.genres])
    return [tag for tag in direct_tags if tag not in ignored]


def ordered_tag_sequence(candidate: GameCandidate) -> list[str]:
    inferred = [*candidate.inferred_tags]
    if candidate.description:
        inferred.extend(extract_description_terms(candidate.description))
    return merge_tags(
        canonical_tags_from_terms(candidate.tags),
        merge_tags(
            canonical_tags_from_terms(candidate.genres),
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
    weights = ordered_sequence_weights(candidate.tags, idf)
    for tag in canonical_tags_from_terms(candidate.genres):
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
    reference_sequences: list[list[str]],
    idf: dict[str, float],
) -> float:
    return max(
        (
            weighted_cosine(
                candidate_weights,
                ordered_sequence_weights(sequence, idf),
            )
            for sequence in reference_sequences
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


def blend_relevance_components(
    tag_coverage: float | None,
    positive_reference: float | None,
    library_profile: float | None,
    review_confidence: float | None,
    negative_reference: float,
) -> float:
    components = (
        (0.55, tag_coverage),
        (0.20, positive_reference),
        (0.10, library_profile),
        (0.15, review_confidence),
    )
    available = [(weight, value) for weight, value in components if value is not None]
    total_weight = sum(weight for weight, _value in available)
    positive_score = (
        sum(weight * min(max(float(value), 0.0), 1.0) for weight, value in available) / total_weight
        if total_weight
        else 0.0
    )
    score = positive_score - 0.20 * min(max(float(negative_reference), 0.0), 1.0)
    return min(max(score, 0.0), 1.0)


def weighted_overlap(matched: list[str], include_tags: list[str]) -> float:
    if not include_tags:
        return 0.0
    matched_weight = sum(tag_weight(tag) for tag in matched)
    total_weight = sum(tag_weight(tag) for tag in include_tags)
    return matched_weight / total_weight if total_weight else 0.0


def tag_weight(tag: str) -> float:
    return TAG_WEIGHTS.get(tag, 1.0)


def classify_similarity_tier(match_score: float) -> str:
    if match_score >= 0.72:
        return "strong"
    if match_score >= 0.38:
        return "recommended"
    return "backup"


def similarity_score(facts: GameFacts) -> float:
    return round(facts.base_relevance_score * 100, 6)


def confidence_for(candidate: GameCandidate, tags: list[str]) -> float:
    confidence = 0.20
    if tags:
        confidence += 0.25
    if candidate.review_total:
        confidence += 0.20
    if candidate.review_positive_ratio is not None:
        confidence += 0.15
    if candidate.appid is not None:
        confidence += 0.10
    return min(confidence, 1.0)


def profile_weight_bonus(tags: list[str], weights: dict[str, float]) -> float:
    if not weights:
        return 0.0
    matched = [min(max(float(weights[tag]), 0.0), 1.0) for tag in tags if tag in weights]
    if not matched:
        return 0.0
    return min(sum(matched) / 3, 1.0)


def profile_fit_point(tags: list[str], weights: dict[str, float]) -> str:
    matched = [tag for tag in tags if tag in weights]
    ranked = sorted(matched, key=lambda tag: (-weights.get(tag, 0.0), tag))
    return f"个人游戏库偏好命中：{'、'.join(ranked[:4])}"


def fit_points_for(candidate: GameCandidate, matched: list[str], match_score: float) -> list[str]:
    points = [f"相似标签：{'、'.join(matched[:6])}"] if matched else []
    points.append(f"Steam 索引匹配度 {match_score:.0%}")
    if candidate.review_total is not None and candidate.review_positive_ratio is not None:
        review = f"{candidate.review_positive_ratio:.0%} 好评，{candidate.review_total} 条"
        points.append(f"Steam 评测：{review}")
    if "chinese" in matched:
        points.append("Steam 信息确认支持中文")
    return dedupe(points)


def risk_points_for(
    candidate: GameCandidate,
    missing: list[str],
    tier: str,
    min_review_count: int,
    constraint_unknowns: list[str] | None = None,
    negative_reference_similarity: float = 0.0,
) -> list[str]:
    risks: list[str] = []
    if constraint_unknowns:
        risks.append(f"硬条件尚未确认：{'、'.join(constraint_unknowns[:5])}")
    if negative_reference_similarity > 0:
        risks.append(f"与负向参考相似度 {negative_reference_similarity:.0%}")
    if tier == "backup":
        risks.append("标签相似度较弱，仅作为备选")
    if missing and tier != "strong":
        risks.append(f"部分偏好标签未命中：{'、'.join(missing[:5])}")
    if candidate.review_total is None:
        risks.append("Steam 评测量未获取到")
    elif candidate.review_total < min_review_count * 2:
        risks.append("Steam 评测量偏少，口碑稳定性较弱")
    if not risks:
        risks.append("价格和具体版本仍需以商店页面确认")
    return dedupe(risks)


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


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def is_reference_title(title: str, reference_titles: list[str]) -> bool:
    normalized = normalize_title(title)
    return any(
        normalized == normalize_title(reference) for reference in reference_titles if reference
    )


def is_reference_query(candidate: GameCandidate) -> bool:
    return any(reason.startswith("reference_query:") for reason in candidate.source_reasons)


def normalize_title(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def release_year(value: str | None) -> int:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return int(match.group(0)) if match else 0
