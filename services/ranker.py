from __future__ import annotations

from ..storage.models import GameCandidate, GamePreference
from .platforms import (
    candidate_matches_any_platform as game_matches_any_platform,
    candidate_matches_platform as game_matches_platform,
    is_switch2_only,
    matched_requested_platforms,
)

DISLIKE_ALIASES = {
    "horror": ("horror", "恐怖", "心理恐怖"),
    "恐怖": ("horror", "恐怖", "心理恐怖"),
    "soulslike": ("souls-like", "soulslike", "魂", "dark souls"),
    "魂": ("souls-like", "soulslike", "魂", "dark souls"),
    "roguelike": ("roguelike", "rogue-like", "roguelite", "肉鸽"),
    "肉鸽": ("roguelike", "rogue-like", "roguelite", "肉鸽"),
    "violent": ("violent", "gore", "blood", "血腥"),
    "血腥": ("violent", "gore", "blood", "血腥"),
}

MULTIPLAYER_TERMS = (
    "co-op",
    "coop",
    "cooperative",
    "multiplayer",
    "local co-op",
    "online co-op",
    "split screen",
    "shared/split screen",
    "shared/split screen co-op",
    "多人",
    "合作",
)
SINGLEPLAYER_TERMS = ("singleplayer", "single player", "单人")
CHINESE_TERMS = ("chinese", "simplified chinese", "traditional chinese", "中文", "简体中文")
DIFFICULT_TERMS = ("souls-like", "soulslike", "difficult", "hard", "permadeath", "roguelike")
EASY_TERMS = ("casual", "relaxing", "family friendly", "cute", "party", "cozy")


def score_game(game: GameCandidate, preference: GamePreference) -> tuple[float, list[str], list[str]]:
    score = 0.0
    reasons: list[str] = list(game.source_reasons)
    warnings: list[str] = list(game.source_warnings)

    matched_platforms = matched_requested_platforms(game, preference.platforms)
    if preference.platforms:
        if len(matched_platforms) == len(preference.platforms):
            score += 38
            reasons.append(f"覆盖你指定的平台：{', '.join(matched_platforms)}")
        elif matched_platforms:
            score += 8
            reasons.append(f"至少匹配平台：{', '.join(matched_platforms)}")
            missing = [item for item in preference.platforms if item not in matched_platforms]
            warnings.append(f"未确认支持这些平台：{', '.join(missing)}")
        else:
            score -= 80
            warnings.append("未匹配到指定平台")
    else:
        score += 5
    warnings.extend(platform_family_warnings(game, preference.platforms))

    like_hits = match_terms(game, preference.genres_like)
    if like_hits:
        score += min(len(like_hits) * 8, 24)
        reasons.append(f"类型/标签匹配：{', '.join(like_hits[:4])}")

    dislike_hits = match_disliked_terms(game, preference.genres_dislike)
    if dislike_hits:
        score -= 35 * len(dislike_hits)
        warnings.append(f"命中你不想要的元素：{', '.join(dislike_hits)}")

    reference_warnings = reference_game_warnings(game, preference)
    warnings.extend(reference_warnings)

    if game.rating is not None:
        score += min(max(game.rating, 0), 5) * 3
        reasons.append(f"RAWG 评分 {game.rating:.1f}/5")
    if game.metacritic is not None:
        score += min(max(game.metacritic, 0), 100) / 15
        reasons.append(f"Metacritic {game.metacritic}")

    if preference.players and preference.players >= 2:
        if has_multiplayer_signal(game):
            score += 25
            coop_detail = cooperative_play_detail(game)
            reasons.append(coop_detail or "标签显示支持多人/合作")
        else:
            score -= 45
            warnings.append("RAWG 数据中没有明确多人/合作标签")
        if has_singleplayer_only_signal(game):
            score -= 35
            warnings.append("标签显示主要是单人体验")

    if preference.language and ("中文" in preference.language or "chinese" in preference.language):
        if match_any(game_haystack(game), CHINESE_TERMS):
            score += 10
            reasons.append("RAWG 标签中出现中文相关信息")
        else:
            warnings.append("RAWG 未明确给出中文支持，需以商店页面为准")

    if preference.difficulty:
        difficulty = preference.difficulty.lower()
        haystack = game_haystack(game)
        if any(word in difficulty for word in ("easy", "简单", "轻松", "别太难", "casual")):
            if match_any(haystack, DIFFICULT_TERMS):
                score -= 12
                warnings.append("标签里有高难或重复挑战倾向，可能不符合低难度偏好")
            if match_any(haystack, EASY_TERMS):
                score += 8
                reasons.append("标签偏休闲/轻松")
        elif any(word in difficulty for word in ("hard", "困难", "高难")):
            if match_any(haystack, DIFFICULT_TERMS):
                score += 8
                reasons.append("标签符合高难度偏好")

    if preference.mood:
        mood_hits = match_terms(game, [preference.mood])
        if mood_hits:
            score += 6
            reasons.append(f"氛围匹配：{', '.join(mood_hits)}")

    if preference.budget is not None:
        warnings.append("RAWG 不提供实时地区价格，预算匹配无法确认")

    if game.stores:
        score += 4
        reasons.append(f"RAWG 记录了购买渠道：{', '.join(game.stores[:3])}")
    else:
        warnings.append("RAWG 未返回购买渠道")

    if game.playtime is not None and preference.difficulty:
        if game.playtime > 60 and any(word in preference.difficulty for word in ("轻松", "简单")):
            score -= 4
            warnings.append("平均游玩时长偏长，可能不适合短平快需求")

    return score, dedupe(reasons), dedupe(warnings)


def platform_family_warnings(game: GameCandidate, requested: list[str]) -> list[str]:
    warnings: list[str] = []
    if "nintendo switch" not in requested:
        return warnings
    if is_switch2_only(game.platforms):
        warnings.append("Nintendo 侧为 Switch 2，不是原版 Switch；请按设备确认版本")
    return warnings


def game_has_disliked_term(game: GameCandidate, disliked: list[str]) -> bool:
    return bool(match_disliked_terms(game, disliked))


def match_disliked_terms(game: GameCandidate, disliked: list[str]) -> list[str]:
    haystack = game_haystack(game)
    hits: list[str] = []
    for term in disliked:
        aliases = DISLIKE_ALIASES.get(term.lower(), (term.lower(),))
        if match_any(haystack, aliases):
            hits.append(term)
    return dedupe(hits)


def match_terms(game: GameCandidate, terms: list[str]) -> list[str]:
    haystack = game_haystack(game)
    hits = []
    for term in terms:
        if term and match_any(haystack, (term.lower(),)):
            hits.append(term)
    return dedupe(hits)


def reference_game_warnings(game: GameCandidate, preference: GamePreference) -> list[str]:
    title = game.title.lower()
    warnings = []
    for reference in preference.reference_games_dislike:
        if reference and reference.lower() in title:
            warnings.append(f"可能接近你明确不喜欢的参考游戏：{reference}")
    for reference in preference.reference_games_like:
        if reference and reference.lower() == title:
            warnings.append(f"这可能是参考游戏本身：{reference}")
    return warnings


def has_multiplayer_signal(game: GameCandidate) -> bool:
    return match_any(game_haystack(game), MULTIPLAYER_TERMS)


def has_singleplayer_only_signal(game: GameCandidate) -> bool:
    haystack = game_haystack(game)
    return match_any(haystack, SINGLEPLAYER_TERMS) and not has_multiplayer_signal(game)


def cooperative_play_detail(game: GameCandidate) -> str:
    haystack = game_haystack(game)
    if match_any(haystack, ("local co-op", "split screen", "shared/split screen")):
        return "支持本地/同屏合作，适合双人一起玩"
    if match_any(haystack, ("online co-op", "co-op", "coop", "cooperative")):
        return "标签显示支持双人/多人合作"
    if match_any(haystack, ("multiplayer", "多人")):
        return "标签显示支持多人游玩"
    return ""


def game_haystack(game: GameCandidate, include_stores: bool = False) -> str:
    values = [game.title, *game.platforms, *game.genres, *game.tags]
    if include_stores:
        values.extend([*game.stores, game.raw_url or ""])
    return " | ".join(str(item).lower() for item in values if item)


def match_any(haystack: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in haystack for term in terms if term)


def dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result
