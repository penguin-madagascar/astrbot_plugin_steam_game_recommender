from __future__ import annotations

from ..storage.models import GameCandidate, GameFacts, GamePreference
from .platforms import is_switch2_only, platform_families_for, platform_matches
from .reference_data import REFERENCE_PROFILES, ReferenceProfile
from .reference_resolver import alias_for_title, reference_profile_for

HORROR_TERMS = ("horror", "恐怖", "psychological horror")
SINGLEPLAYER_TERMS = ("single-player", "singleplayer", "single player", "单人")
MULTIPLAYER_TERMS = ("multiplayer", "多人")
COOP_TERMS = ("co-op", "coop", "cooperative", "合作")
LOCAL_COOP_TERMS = ("local co-op", "shared/split screen co-op", "shared/split screen", "本地合作")
ONLINE_COOP_TERMS = ("online co-op", "online coop", "在线合作")
SPLIT_SCREEN_TERMS = ("split screen", "shared/split screen")
REMOTE_PLAY_TERMS = ("remote play together", "远程同乐")
CHINESE_TERMS = ("simplified chinese", "traditional chinese", "chinese", "中文", "简体中文")
TERM_ALIASES = {
    "co-op": (
        *COOP_TERMS,
        *LOCAL_COOP_TERMS,
        *ONLINE_COOP_TERMS,
        *SPLIT_SCREEN_TERMS,
        *REMOTE_PLAY_TERMS,
    ),
    "coop": (
        *COOP_TERMS,
        *LOCAL_COOP_TERMS,
        *ONLINE_COOP_TERMS,
        *SPLIT_SCREEN_TERMS,
        *REMOTE_PLAY_TERMS,
    ),
    "local co-op": (*LOCAL_COOP_TERMS, *SPLIT_SCREEN_TERMS, "remote play together"),
    "online co-op": (*ONLINE_COOP_TERMS, "co-op", "coop"),
    "multiplayer": (*MULTIPLAYER_TERMS, *COOP_TERMS, *LOCAL_COOP_TERMS, *ONLINE_COOP_TERMS),
    "relaxing": ("relaxing", "cozy", "chill", "轻松", "休闲"),
    "casual": ("casual", "family friendly", "relaxing", "休闲", "轻松"),
}
BROAD_REFERENCE_TERMS = {
    "action",
    "adventure",
    "casual",
    "co-op",
    "coop",
    "local co-op",
    "multiplayer",
    "online co-op",
    "rpg",
    "simulation",
    "strategy",
}


def build_game_facts(
    candidate: GameCandidate,
    preference: GamePreference,
    steam_candidate: GameCandidate | None = None,
) -> GameFacts:
    rawg_haystack = haystack(candidate)
    steam_haystack = haystack(steam_candidate) if steam_candidate else ""
    combined_haystack = " | ".join(value for value in (rawg_haystack, steam_haystack) if value)

    platform_families = platform_families_for(candidate, steam_candidate)
    matched = [platform for platform in preference.platforms if platform_matches(platform, platform_families)]
    missing = [platform for platform in preference.platforms if platform not in matched]

    has_local = contains_any(combined_haystack, LOCAL_COOP_TERMS)
    has_online = contains_any(combined_haystack, ONLINE_COOP_TERMS)
    has_split = contains_any(combined_haystack, SPLIT_SCREEN_TERMS)
    has_remote = contains_any(combined_haystack, REMOTE_PLAY_TERMS)
    has_generic_coop = contains_any(combined_haystack, COOP_TERMS)
    has_coop = has_local or has_online or has_split or has_remote or has_generic_coop
    ordinary_multiplayer = contains_any(combined_haystack, MULTIPLAYER_TERMS) and not has_coop
    singleplayer = contains_any(combined_haystack, SINGLEPLAYER_TERMS) and not has_coop

    coop_modes: list[str] = []
    add_if(coop_modes, has_local, "本地合作")
    add_if(coop_modes, has_online, "在线合作")
    add_if(coop_modes, has_split, "分屏/同屏")
    add_if(coop_modes, has_remote, "Remote Play Together")
    if has_generic_coop and not coop_modes:
        coop_modes.append("合作")
    if ordinary_multiplayer:
        coop_modes.append("普通多人")

    like_terms = dedupe(preference.genres_like)
    matched_like_terms = [
        term for term in like_terms if term_matches(combined_haystack, term)
    ]
    missing_like_terms = [
        term for term in like_terms if term not in matched_like_terms
    ]
    required_terms = reference_required_terms(preference)
    required_hits = [
        term for term in required_terms if term_matches(combined_haystack, term)
    ]
    required_misses = [
        term for term in required_terms if term not in required_hits
    ]
    match_coverage = len(matched_like_terms) / len(like_terms) if like_terms else 0.0
    required_coverage = (
        len(required_hits) / len(required_terms) if required_terms else 1.0
    )

    source_reasons = " | ".join(candidate.source_reasons).lower()
    match_score = (
        match_coverage if not required_terms else match_coverage * 0.75 + required_coverage * 0.25
    )
    if "参考画像种子" in source_reasons:
        match_score = max(match_score, 0.95)

    reference_similarity = 0.0
    if "参考画像种子" in source_reasons:
        reference_similarity = 1.0
    else:
        reference_similarity = min(match_score, 0.85)
        reference_similarity = apply_reference_profile_specificity(
            reference_similarity,
            combined_haystack,
            preference,
        )
        if has_coop and any(term in combined_haystack for term in ("puzzle", "adventure", "platformer", "casual")):
            reference_similarity = max(reference_similarity, 0.55)
        elif has_coop:
            reference_similarity = max(reference_similarity, 0.35)
        elif ordinary_multiplayer:
            reference_similarity = max(reference_similarity, 0.15)

    data_sources = ["RAWG"]
    if steam_candidate:
        data_sources.append("Steam")
    if candidate.source_reasons:
        data_sources.append("参考画像")

    hard_blocks: list[str] = []
    if contains_any(combined_haystack, HORROR_TERMS):
        hard_blocks.append("命中恐怖元素")
    if singleplayer:
        hard_blocks.append("主要是单人体验")
    if preference.players and preference.players >= 2 and not (has_coop or ordinary_multiplayer):
        hard_blocks.append("未确认支持双人或多人游玩")
    if preference.platforms and not matched:
        hard_blocks.append("未匹配指定平台")

    confidence = 0.15
    if candidate.rating is not None:
        confidence += 0.10
    if candidate.platforms:
        confidence += 0.10
    if candidate.tags or candidate.genres:
        confidence += 0.10
    if steam_candidate:
        confidence += 0.25
    if has_coop:
        confidence += 0.20
    if candidate.source_reasons:
        confidence += 0.20

    return GameFacts(
        platform_families=platform_families,
        matched_platforms=matched,
        missing_platforms=missing,
        coop_modes=coop_modes,
        data_sources=data_sources,
        hard_blocks=hard_blocks,
        matched_like_terms=matched_like_terms,
        missing_like_terms=missing_like_terms,
        required_hits=required_hits,
        required_misses=required_misses,
        has_coop=has_coop,
        has_local_coop=has_local,
        has_online_coop=has_online,
        has_split_screen=has_split,
        has_remote_play=has_remote,
        ordinary_multiplayer=ordinary_multiplayer,
        singleplayer_only=singleplayer,
        horror=contains_any(combined_haystack, HORROR_TERMS),
        chinese=contains_any(combined_haystack, CHINESE_TERMS),
        switch2_only=is_switch2_only(candidate.platforms),
        reference_similarity=reference_similarity,
        match_coverage=match_coverage,
        match_score=match_score,
        confidence=confidence,
    )


def apply_reference_profile_specificity(
    similarity: float,
    text: str,
    preference: GamePreference,
) -> float:
    required_terms = reference_required_specific_terms(preference)
    if required_terms and not any(term_matches(text, term) for term in required_terms):
        return min(similarity, 0.55)

    specific_terms = reference_specific_terms(preference)
    if not specific_terms:
        return similarity

    hits = [term for term in specific_terms if term_matches(text, term)]
    if not hits:
        return min(similarity, 0.55)
    return max(similarity, min(0.70 + len(hits) * 0.05, 0.85))


def reference_required_terms(preference: GamePreference) -> list[str]:
    terms: list[str] = []
    for profile in active_reference_profiles(preference):
        terms.extend(term.lower() for term in profile.required_tags)
    return dedupe(terms)


def reference_required_specific_terms(preference: GamePreference) -> list[str]:
    terms: list[str] = []
    for profile in active_reference_profiles(preference):
        for term in profile.required_tags:
            key = term.lower()
            if key not in BROAD_REFERENCE_TERMS:
                terms.append(key)
    return dedupe(terms)


def reference_specific_terms(preference: GamePreference) -> list[str]:
    terms: list[str] = []
    for profile in active_reference_profiles(preference):
        for term in (*profile.genres_like, *profile.required_tags):
            key = term.lower()
            if key not in BROAD_REFERENCE_TERMS:
                terms.append(key)
    return dedupe(terms)


def active_reference_profiles(preference: GamePreference) -> list[ReferenceProfile]:
    profiles: list[ReferenceProfile] = []
    for entity in preference.resolved_reference_games:
        profile = reference_profile_for(entity)
        if profile:
            profiles.append(profile)
    for title in preference.reference_games_like:
        alias = alias_for_title(title)
        if alias:
            profile = REFERENCE_PROFILES.get(alias.rawg_slug)
            if profile:
                profiles.append(profile)
    return dedupe_profiles(profiles)


def haystack(game: GameCandidate | None, include_tags: bool = True) -> str:
    if not game:
        return ""
    values = [game.title, *game.platforms, *game.stores]
    if include_tags:
        values.extend([*game.genres, *game.tags, game.description or ""])
    return " | ".join(str(value).lower() for value in values if value)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def term_matches(text: str, term: str) -> bool:
    key = term.lower().strip()
    if not key:
        return False
    aliases = TERM_ALIASES.get(key, (key,))
    return contains_any(text, aliases)


def add_if(values: list[str], condition: bool, text: str) -> None:
    if condition and text not in values:
        values.append(text)


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def dedupe_profiles(values: list[ReferenceProfile]) -> list[ReferenceProfile]:
    result: list[ReferenceProfile] = []
    seen: set[str] = set()
    for value in values:
        if value.rawg_slug not in seen:
            result.append(value)
            seen.add(value.rawg_slug)
    return result
