from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..storage.models import GameCandidate, GameFacts, GamePreference, RankedGame
from .tag_normalizer import candidate_canonical_tags, canonical_tags_from_terms

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
    exclude_tags: list[str] = field(default_factory=list)
    reference_titles: list[str] = field(default_factory=list)


def build_profile_from_preference(
    preference: GamePreference,
    reference_candidates: list[GameCandidate] | None = None,
) -> SteamTagProfile:
    include = canonical_tags_from_terms(preference.genres_like)
    exclude = canonical_tags_from_terms(preference.genres_dislike)

    if preference.players and preference.players >= 2:
        include = merge_tags(include, ["co_op", "multiplayer"])
    if preference.language and (
        "中文" in preference.language or "chinese" in preference.language
    ):
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
        exclude_tags=exclude,
        reference_titles=list(preference.reference_games_like),
    )


def rank_steam_candidates(
    candidates: list[GameCandidate],
    profile: SteamTagProfile,
    min_review_count: int = 50,
    min_positive_ratio: float = 0.65,
) -> list[RankedGame]:
    ranked: list[RankedGame] = []
    for candidate in candidates:
        tags = candidate_canonical_tags(candidate)
        if excluded_by_tags(tags, profile):
            continue
        if singleplayer_only(tags, profile):
            continue
        if below_review_floor(candidate, min_review_count, min_positive_ratio):
            continue

        matched = [tag for tag in profile.include_tags if tag in tags]
        missing = [tag for tag in profile.include_tags if tag not in matched]
        match_score = weighted_overlap(matched, profile.include_tags)
        if profile.include_tags and match_score <= 0:
            continue

        tier = classify_similarity_tier(match_score)
        facts = GameFacts(
            matched_like_terms=matched,
            missing_like_terms=missing,
            required_hits=matched,
            required_misses=[],
            has_coop=bool(set(tags) & {"co_op", "local_coop", "online_coop"}),
            has_local_coop="local_coop" in tags,
            has_online_coop="online_coop" in tags,
            ordinary_multiplayer=(
                "multiplayer" in tags
                and not bool(set(tags) & {"co_op", "local_coop", "online_coop"})
            ),
            singleplayer_only="singleplayer" in tags and not bool(set(tags) & MULTIPLAYER_TAGS),
            chinese="chinese" in tags,
            reference_similarity=match_score,
            match_coverage=(
                len(matched) / len(profile.include_tags) if profile.include_tags else 0.0
            ),
            match_score=match_score,
            confidence=confidence_for(candidate, tags),
        )
        score = similarity_score(candidate, facts, tier)
        fit_points = fit_points_for(candidate, matched, match_score)
        risk_points = risk_points_for(candidate, missing, tier, min_review_count)
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
            -game.facts.match_score,
            -float(game.score),
            game.title,
        ),
    )


def reference_expansion_tags(candidate: GameCandidate) -> list[str]:
    ignored = {"singleplayer", "chinese"}
    return [tag for tag in candidate_canonical_tags(candidate) if tag not in ignored]


def excluded_by_tags(tags: list[str], profile: SteamTagProfile) -> bool:
    return bool(set(tags) & set(profile.exclude_tags))


def singleplayer_only(tags: list[str], profile: SteamTagProfile) -> bool:
    wants_multiplayer = bool(set(profile.include_tags) & MULTIPLAYER_TAGS)
    return wants_multiplayer and "singleplayer" in tags and not bool(set(tags) & MULTIPLAYER_TAGS)


def below_review_floor(
    candidate: GameCandidate,
    min_review_count: int,
    min_positive_ratio: float,
) -> bool:
    if candidate.review_total is not None and candidate.review_total < min_review_count:
        return True
    ratio = candidate.review_positive_ratio
    return ratio is not None and ratio < min_positive_ratio


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


def similarity_score(candidate: GameCandidate, facts: GameFacts, tier: str) -> float:
    score = {"strong": 300.0, "recommended": 200.0, "backup": 100.0}[tier]
    score += facts.match_score * 120
    score += facts.confidence * 10
    if candidate.review_positive_ratio is not None:
        score += candidate.review_positive_ratio * 8
    if candidate.review_total:
        score += min(math.log10(max(candidate.review_total, 1)), 6) * 0.8
    return score


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
) -> list[str]:
    risks: list[str] = []
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
