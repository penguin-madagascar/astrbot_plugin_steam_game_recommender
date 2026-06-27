from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Protocol

from ..storage.models import GameCandidate, GamePreference, ResolvedReferenceGame
from .reference_data import REFERENCE_ALIASES, REFERENCE_PROFILES, ReferenceAlias, ReferenceProfile

SOFT_CONFIDENCE = 0.70


class ReferenceGameSource(Protocol):
    async def search_games(self, **kwargs) -> list[GameCandidate]:
        ...


class ReferenceGameResolver:
    def __init__(self, game_source: ReferenceGameSource | None = None) -> None:
        self.game_source = game_source

    async def resolve_reference_games(
        self,
        text: str,
        preference: GamePreference,
    ) -> list[ResolvedReferenceGame]:
        titles = merge_texts(preference.reference_games_like, extract_reference_titles(text))
        resolved: list[ResolvedReferenceGame] = []
        seen: set[str] = set()
        for title in titles:
            entity = await self.resolve_one(title)
            if not entity:
                continue
            key = entity.rawg_slug or entity.normalized_title
            if key and key not in seen:
                resolved.append(entity)
                seen.add(key)
        return resolved

    async def resolve_one(self, raw_title: str) -> ResolvedReferenceGame | None:
        raw_title = cleanup_reference_text(raw_title)
        if not raw_title:
            return None
        alias = alias_for_title(raw_title)
        if alias:
            return resolved_from_alias(raw_title, alias)

        if self.game_source is not None:
            rawg_match = await self.resolve_from_rawg(raw_title)
            if rawg_match:
                return rawg_match

        normalized = normalize_reference_title(raw_title)
        return ResolvedReferenceGame(
            raw_text=raw_title,
            normalized_title=normalized,
            canonical_title=raw_title,
            confidence=0.5,
            source="text",
        )

    async def resolve_from_rawg(self, raw_title: str) -> ResolvedReferenceGame | None:
        candidates = await self.game_source.search_games(
            search=raw_title,
            page_size=5,
            ordering="-relevance",
        )
        best: tuple[float, GameCandidate] | None = None
        for candidate in candidates:
            score = title_similarity(raw_title, candidate.title)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None or best[0] < SOFT_CONFIDENCE:
            return None
        score, candidate = best
        return ResolvedReferenceGame(
            raw_text=raw_title,
            normalized_title=normalize_reference_title(raw_title),
            canonical_title=candidate.title,
            rawg_id=candidate.rawg_id,
            rawg_slug=slug_from_candidate(candidate),
            confidence=min(max(score, SOFT_CONFIDENCE), 0.95),
            source="rawg",
            genres=candidate.genres,
            tags=candidate.tags,
            platforms=candidate.platforms,
            stores=candidate.stores,
        )


def extract_reference_titles(text: str) -> list[str]:
    if not text:
        return []
    patterns = (
        r"(?:类似于|类似|像|接近|参考|玩法像|同类于)\s*(?:《([^》]+)》|([^，。,.；;\n]+))",
        r"(?:similar to|like)\s+(?:\"([^\"]+)\"|'([^']+)'|([^，。,.；;\n]+))",
    )
    titles: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            raw = next((group for group in match.groups() if group), "")
            cleaned = cleanup_reference_text(raw)
            if cleaned:
                titles.append(cleaned)
    return merge_texts([], titles)


def cleanup_reference_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip(" \t\r\n《》<>「」『』“”\"'")
    text = re.split(r"\s*(?:但|不过|别|不要|且|并且|，|。|；|;)\s*", text, maxsplit=1)[0]
    text = re.split(r"\s+(?:but|except|without|and)\s+", text, maxsplit=1, flags=re.I)[0]
    text = re.split(r"的(?:游戏|玩法|轻松|合作|同类|类型)?", text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text).strip(" -:：")


def normalize_reference_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.strip(" \t\r\n《》<>「」『』“”\"'")
    text = re.sub(
        r"\b(?:friend'?s pass|friends pass|game of the year|goty|complete edition|"
        r"definitive edition|special edition|ultimate edition|deluxe edition|"
        r"remastered|remaster|remake|edition)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def alias_for_title(value: str) -> ReferenceAlias | None:
    normalized = normalize_reference_title(value)
    for alias in REFERENCE_ALIASES.values():
        if normalized == normalize_reference_title(alias.canonical_title):
            return alias
        if any(normalized == normalize_reference_title(item) for item in alias.aliases):
            return alias
    return None


def resolved_from_alias(raw_title: str, alias: ReferenceAlias) -> ResolvedReferenceGame:
    return ResolvedReferenceGame(
        raw_text=raw_title,
        normalized_title=normalize_reference_title(raw_title),
        canonical_title=alias.canonical_title,
        rawg_id=alias.rawg_id,
        rawg_slug=alias.rawg_slug,
        confidence=0.98,
        source="alias",
    )


def reference_profile_for(entity: ResolvedReferenceGame) -> ReferenceProfile | None:
    if entity.rawg_slug and entity.confidence >= SOFT_CONFIDENCE:
        return REFERENCE_PROFILES.get(entity.rawg_slug)
    return None


def title_similarity(left: str, right: str) -> float:
    left_key = normalize_reference_title(left)
    right_key = normalize_reference_title(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 0.98
    if left_key in right_key or right_key in left_key:
        shorter = min(len(left_key), len(right_key))
        longer = max(len(left_key), len(right_key))
        return max(0.75, shorter / longer)
    return SequenceMatcher(None, left_key, right_key).ratio()


def slug_from_candidate(candidate: GameCandidate) -> str | None:
    if candidate.raw_url:
        slug = candidate.raw_url.rstrip("/").split("/")[-1]
        if slug:
            return slug
    normalized = normalize_reference_title(candidate.title)
    return normalized.replace(" ", "-") if normalized else None


def merge_texts(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in [*left, *right]:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        key = normalize_reference_title(text)
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result
