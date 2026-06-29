from __future__ import annotations

from ..storage.models import GameCandidate, GameFacts, GamePreference, RankedGame

TIER_LABELS = {
    "strong": "强烈推荐",
    "recommended": "推荐",
    "backup": "备选",
}
TIER_ORDER = {"strong": 0, "recommended": 1, "backup": 2}


def build_ranked_game(
    candidate: GameCandidate,
    preference: GamePreference,
    facts: GameFacts,
) -> RankedGame | None:
    if is_hard_blocked(candidate, facts, preference):
        return None

    tier = classify_tier(facts, preference)
    score = score_candidate(candidate, facts, preference, tier)
    fit_points = build_fit_points(candidate, facts, preference)
    risk_points = build_risk_points(facts, preference, tier)
    if not risk_points:
        risk_points.append("价格、中文或具体平台版本仍需以商店页面确认")

    ranked = RankedGame.from_candidate(
        candidate,
        score=score,
        reasons=fit_points,
        warnings=risk_points,
    )
    return copy_ranked_game(
        ranked,
        {
            "tier": tier,
            "fit_points": fit_points,
            "risk_points": risk_points,
            "facts": facts,
        },
    )


def is_hard_blocked(
    candidate: GameCandidate,
    facts: GameFacts,
    preference: GamePreference,
) -> bool:
    title = candidate.title.lower()
    if facts.horror and any(term in {"horror", "恐怖"} for term in preference.genres_dislike):
        return True
    if facts.singleplayer_only:
        return True
    if "未匹配指定平台" in facts.hard_blocks:
        return True
    if preference.players and preference.players >= 2 and not (facts.has_coop or facts.ordinary_multiplayer):
        return True
    if any(reference.lower() in title for reference in preference.reference_games_dislike):
        return True
    return False


def classify_tier(facts: GameFacts, preference: GamePreference) -> str:
    all_platforms = not preference.platforms or not facts.missing_platforms
    needs_multiplayer = bool(preference.players and preference.players >= 2)
    play_mode_ok = not needs_multiplayer or facts.has_coop
    strong_match = (
        facts.match_coverage >= 0.70
        or facts.match_score >= 0.82
        or facts.reference_similarity >= 0.85
    )
    medium_match = (
        facts.match_coverage >= 0.40
        or facts.match_score >= 0.45
        or facts.reference_similarity >= 0.45
    )
    if play_mode_ok and not facts.required_misses and strong_match and all_platforms:
        return "strong"
    if play_mode_ok and not facts.required_misses and medium_match:
        return "recommended"
    return "backup"


def score_candidate(
    candidate: GameCandidate,
    facts: GameFacts,
    preference: GamePreference,
    tier: str,
) -> float:
    score = {"strong": 300.0, "recommended": 200.0, "backup": 100.0}[tier]
    score += facts.match_score * 80
    score += facts.match_coverage * 25
    score += facts.reference_similarity * 20
    score += len(facts.matched_platforms) * 12
    if preference.platforms and not facts.missing_platforms:
        score += 18
    if facts.has_local_coop or facts.has_split_screen:
        score += 18
    if facts.has_online_coop:
        score += 10
    if facts.has_remote_play:
        score += 6
    if facts.chinese:
        score += 10
    if preference.difficulty and ("easy" in preference.difficulty or "轻松" in preference.difficulty):
        if any(mode in facts.coop_modes for mode in ("本地合作", "分屏/同屏")):
            score += 4
    if candidate.rating is not None:
        score += min(max(candidate.rating, 0), 5) * 0.75
    if candidate.metacritic is not None:
        score += min(max(candidate.metacritic, 0), 100) / 80
    score += facts.confidence * 10
    return score


def build_fit_points(
    candidate: GameCandidate,
    facts: GameFacts,
    preference: GamePreference,
) -> list[str]:
    points: list[str] = []
    points.extend(candidate.source_reasons)
    if facts.matched_like_terms:
        points.append(f"需求命中：{'、'.join(facts.matched_like_terms[:6])}")
    if facts.has_coop:
        source = "Steam 分类" if "Steam" in facts.data_sources else "标签"
        modes = "、".join(facts.coop_modes) if facts.coop_modes else "合作"
        points.append(f"{source}确认支持{modes}")
    elif facts.ordinary_multiplayer:
        points.append("标签显示支持多人游玩")
    if facts.matched_platforms:
        points.append(f"匹配平台：{'、'.join(facts.matched_platforms)}")
    if preference.platforms and not facts.missing_platforms:
        points.append("覆盖你指定的平台组合")
    if facts.chinese:
        points.append("Steam/标签信息确认支持中文")
    if candidate.rating is not None:
        points.append(f"口碑参考：RAWG 评分 {candidate.rating:.1f}/5")
    return dedupe(points)


def build_risk_points(
    facts: GameFacts,
    preference: GamePreference,
    tier: str,
) -> list[str]:
    risks: list[str] = []
    if tier == "backup":
        risks.append("与参考游戏的相似度较弱，仅作为备选")
    if facts.required_misses:
        risks.append(f"核心标签未确认：{'、'.join(facts.required_misses[:4])}")
    elif facts.missing_like_terms and tier != "strong":
        risks.append(f"部分偏好标签未确认：{'、'.join(facts.missing_like_terms[:4])}")
    if facts.missing_platforms:
        risks.append(f"未确认支持平台：{'、'.join(facts.missing_platforms)}")
    if facts.switch2_only:
        risks.append("Nintendo 侧为 Switch 2，不是原版 Switch")
    if facts.ordinary_multiplayer and not facts.has_coop:
        risks.append("只确认普通多人，未确认双人合作/同屏协作")
    if preference.language and not facts.chinese:
        risks.append("中文支持未确认，需以商店页面为准")
    if preference.budget is not None:
        risks.append("实时价格未确认，预算匹配需以商店页面或价格插件为准")
    return dedupe(risks)


def sort_ranked_games(games: list[RankedGame]) -> list[RankedGame]:
    return sorted(
        games,
        key=lambda game: (
            TIER_ORDER.get(game.tier, 9),
            -game.facts.match_score,
            -game.facts.match_coverage,
            -game.score,
            game.title,
        ),
    )


def copy_ranked_game(game: RankedGame, update: dict) -> RankedGame:
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
