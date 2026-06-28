from __future__ import annotations

from ..storage.models import GameCandidate, GameFacts, GamePreference

STEAM_ALIASES = ("steam", "pc", "windows")
SWITCH_ALIASES = ("nintendo switch", "switch", "nintendo store")
SWITCH2_ALIASES = ("nintendo switch 2", "switch 2")
HORROR_TERMS = ("horror", "恐怖", "psychological horror")
SINGLEPLAYER_TERMS = ("single-player", "singleplayer", "single player", "单人")
MULTIPLAYER_TERMS = ("multiplayer", "多人")
COOP_TERMS = ("co-op", "coop", "cooperative", "合作")
LOCAL_COOP_TERMS = ("local co-op", "shared/split screen co-op", "shared/split screen", "本地合作")
ONLINE_COOP_TERMS = ("online co-op", "online coop", "在线合作")
SPLIT_SCREEN_TERMS = ("split screen", "shared/split screen")
REMOTE_PLAY_TERMS = ("remote play together", "远程同乐")
CHINESE_TERMS = ("simplified chinese", "traditional chinese", "chinese", "中文", "简体中文")


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

    source_reasons = " | ".join(candidate.source_reasons).lower()
    reference_similarity = 0.0
    if "参考画像种子" in source_reasons:
        reference_similarity = 1.0
    else:
        like_hits = sum(
            1 for term in preference.genres_like if term and term.lower() in combined_haystack
        )
        reference_similarity = min(like_hits / 5, 0.8)
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
        has_coop=has_coop,
        has_local_coop=has_local,
        has_online_coop=has_online,
        has_split_screen=has_split,
        has_remote_play=has_remote,
        ordinary_multiplayer=ordinary_multiplayer,
        singleplayer_only=singleplayer,
        horror=contains_any(combined_haystack, HORROR_TERMS),
        chinese=contains_any(combined_haystack, CHINESE_TERMS),
        switch2_only=switch2_only(candidate),
        reference_similarity=reference_similarity,
        confidence=confidence,
    )


def platform_families_for(
    candidate: GameCandidate,
    steam_candidate: GameCandidate | None = None,
) -> list[str]:
    text = haystack(candidate, include_tags=False)
    if steam_candidate:
        text = f"{text} | {haystack(steam_candidate, include_tags=False)}"
    families: list[str] = []
    if contains_any(text, STEAM_ALIASES):
        families.append("steam")
    if contains_any(text, SWITCH_ALIASES) or contains_any(text, SWITCH2_ALIASES):
        families.append("nintendo switch")
    if contains_any(text, ("playstation", "ps4", "ps5")):
        families.append("playstation")
    if contains_any(text, ("xbox", "xbox one", "xbox series")):
        families.append("xbox")
    return dedupe(families)


def platform_matches(requested: str, families: list[str]) -> bool:
    key = requested.lower()
    if key in {"pc", "steam"}:
        return "steam" in families
    if key == "nintendo switch":
        return "nintendo switch" in families
    return key in families


def switch2_only(candidate: GameCandidate) -> bool:
    text = " | ".join(candidate.platforms).lower()
    return "switch 2" in text and "nintendo switch" not in {
        platform.lower() for platform in candidate.platforms
    }


def haystack(game: GameCandidate | None, include_tags: bool = True) -> str:
    if not game:
        return ""
    values = [game.title, *game.platforms, *game.stores]
    if include_tags:
        values.extend([*game.genres, *game.tags, game.description or ""])
    return " | ".join(str(value).lower() for value in values if value)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


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
