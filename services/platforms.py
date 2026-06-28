from __future__ import annotations

from typing import Protocol


class PlatformEvidence(Protocol):
    platforms: list[str]
    stores: list[str]
    raw_url: str | None


STEAM_ALIASES = ("steam", "store.steampowered.com")
PC_PLATFORM_ALIASES = ("pc", "windows", "macos", "mac", "linux")
SWITCH_PLATFORM_ALIASES = ("nintendo switch", "switch", "nintendo switch 2", "switch 2")
SWITCH_STORE_ALIASES = ("nintendo store",)
PLAYSTATION_PLATFORM_ALIASES = ("playstation", "ps4", "ps5")
PLAYSTATION_STORE_ALIASES = ("playstation store",)
XBOX_PLATFORM_ALIASES = ("xbox", "xbox one", "xbox series")
XBOX_STORE_ALIASES = ("xbox store", "microsoft store")


def platform_families_for(
    candidate: PlatformEvidence,
    attached_candidate: PlatformEvidence | None = None,
) -> list[str]:
    platform_text = platform_haystack(candidate)
    store_text = store_haystack(candidate)
    if attached_candidate:
        platform_text = join_text(platform_text, platform_haystack(attached_candidate))
        store_text = join_text(store_text, store_haystack(attached_candidate))

    families: list[str] = []
    if contains_any(store_text, STEAM_ALIASES):
        families.append("steam")
    if contains_any(platform_text, PC_PLATFORM_ALIASES) or "steam" in families:
        families.append("pc")
    if contains_any(platform_text, SWITCH_PLATFORM_ALIASES) or contains_any(
        store_text,
        SWITCH_STORE_ALIASES,
    ):
        families.append("nintendo switch")
    if contains_any(platform_text, PLAYSTATION_PLATFORM_ALIASES) or contains_any(
        store_text,
        PLAYSTATION_STORE_ALIASES,
    ):
        families.append("playstation")
    if contains_any(platform_text, XBOX_PLATFORM_ALIASES) or contains_any(
        store_text,
        XBOX_STORE_ALIASES,
    ):
        families.append("xbox")
    return dedupe(families)


def matched_requested_platforms(
    candidate: PlatformEvidence,
    requested: list[str],
    attached_candidate: PlatformEvidence | None = None,
) -> list[str]:
    families = platform_families_for(candidate, attached_candidate)
    return [platform for platform in requested if platform_matches(platform, families)]


def candidate_matches_any_platform(candidate: PlatformEvidence, requested: list[str]) -> bool:
    return not requested or bool(matched_requested_platforms(candidate, requested))


def candidate_matches_platform(candidate: PlatformEvidence, requested: str) -> bool:
    return platform_matches(requested, platform_families_for(candidate))


def platform_matches(requested: str, families: list[str]) -> bool:
    key = requested.lower()
    if key == "steam":
        return "steam" in families
    if key == "pc":
        return "pc" in families
    if key == "nintendo switch":
        return "nintendo switch" in families
    return key in families


def is_switch2_only(platforms: list[str]) -> bool:
    normalized = [platform.lower() for platform in platforms]
    text = " | ".join(normalized)
    has_switch_2 = any("switch 2" in platform for platform in normalized)
    has_switch_1 = any(
        platform in {"nintendo switch", "switch"} or platform.endswith(" nintendo switch")
        for platform in normalized
    )
    return has_switch_2 and not has_switch_1


def platform_haystack(candidate: PlatformEvidence | None) -> str:
    if candidate is None:
        return ""
    return " | ".join(str(value).lower() for value in candidate.platforms if value)


def store_haystack(candidate: PlatformEvidence | None) -> str:
    if candidate is None:
        return ""
    values = [*candidate.stores, candidate.raw_url or ""]
    return " | ".join(str(value).lower() for value in values if value)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text for term in terms if term)


def join_text(*values: str) -> str:
    return " | ".join(value for value in values if value)


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result
