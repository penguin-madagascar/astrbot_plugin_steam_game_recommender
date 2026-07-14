from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..storage.models import SteamSearchHit
from .recommendation_intent import ReferenceQuery

FUZZY_MATCH_THRESHOLD = 0.90
FUZZY_MATCH_MARGIN = 0.08
EDITION_SUFFIXES = (
    "game of the year edition",
    "definitive edition",
    "complete edition",
    "remastered",
    "goty",
)


@dataclass(frozen=True)
class ReferenceMatch:
    hit: SteamSearchHit
    confidence: float
    matched_alias: str
    match_kind: str


@dataclass(frozen=True)
class _ScoredHit:
    hit: SteamSearchHit
    priority: int
    confidence: float
    matched_alias: str
    match_kind: str


def match_reference_query(
    reference: ReferenceQuery,
    hits: list[SteamSearchHit] | tuple[SteamSearchHit, ...],
) -> ReferenceMatch | None:
    grouped: dict[int, list[SteamSearchHit]] = {}
    for hit in hits:
        grouped.setdefault(int(hit.appid), []).append(hit)

    scored = [
        best_appid_match(reference, observed_hits)
        for observed_hits in grouped.values()
    ]
    candidates = sorted(
        (item for item in scored if item is not None),
        key=lambda item: (item.priority, item.confidence),
        reverse=True,
    )
    if not candidates:
        return None

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    if runner_up and (best.priority, best.confidence) == (
        runner_up.priority,
        runner_up.confidence,
    ):
        return None
    if best.match_kind == "fuzzy":
        second_score = runner_up.confidence if runner_up else 0.0
        if best.confidence < FUZZY_MATCH_THRESHOLD:
            return None
        if runner_up and best.confidence - second_score < FUZZY_MATCH_MARGIN:
            return None
    return ReferenceMatch(
        hit=best.hit,
        confidence=best.confidence,
        matched_alias=best.matched_alias,
        match_kind=best.match_kind,
    )


def best_appid_match(
    reference: ReferenceQuery,
    hits: list[SteamSearchHit],
) -> _ScoredHit | None:
    matches = [
        score_title_pair(alias, hit)
        for alias in reference.aliases
        for hit in hits
        if alias and hit.title
    ]
    return max(matches, key=lambda item: (item.priority, item.confidence), default=None)


def score_title_pair(alias: str, hit: SteamSearchHit) -> _ScoredHit:
    expected = title_key(alias)
    actual = title_key(hit.title)
    if expected and expected == actual:
        return _ScoredHit(hit, 3, 1.0, alias, "exact")
    expected_base = title_base_key(alias)
    actual_base = title_base_key(hit.title)
    if expected_base and expected_base == actual_base:
        return _ScoredHit(hit, 2, 0.96, alias, "base")
    confidence = SequenceMatcher(None, expected, actual).ratio() if expected and actual else 0.0
    return _ScoredHit(hit, 1, confidence, alias, "fuzzy")


def title_key(value: str) -> str:
    normalized = normalized_title_words(value)
    return "".join(character for character in normalized if character.isalnum())


def title_base_key(value: str) -> str:
    normalized = normalized_title_words(value)
    suffix_pattern = "|".join(re.escape(suffix) for suffix in EDITION_SUFFIXES)
    normalized = re.sub(rf"(?:\s+|:\s*)({suffix_pattern})$", "", normalized).strip()
    return "".join(character for character in normalized if character.isalnum())


def normalized_title_words(value: str) -> str:
    text = str(value or "").replace("™", "").replace("®", "").replace("©", "")
    text = unicodedata.normalize("NFKC", text).casefold()
    words = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(words.split())
