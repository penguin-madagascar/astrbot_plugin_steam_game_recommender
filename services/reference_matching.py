from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..storage.models import SteamSearchHit
from .game_identity import game_family_key
from .recommendation_intent import ReferenceQuery

FUZZY_MATCH_THRESHOLD = 0.90
FUZZY_MATCH_MARGIN = 0.08


def roman_numeral(value: int) -> str:
    remainder = value
    parts: list[str] = []
    for number, token in (
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ):
        count, remainder = divmod(remainder, number)
        parts.extend(token for _ in range(count))
    return "".join(parts)


ROMAN_NUMBERS = {roman_numeral(value): str(value) for value in range(1, 51)}


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
    if title_number_signature(alias) != title_number_signature(hit.title):
        return _ScoredHit(hit, 1, 0.0, alias, "fuzzy")
    confidence = SequenceMatcher(None, expected, actual).ratio() if expected and actual else 0.0
    return _ScoredHit(hit, 1, confidence, alias, "fuzzy")


def title_key(value: str) -> str:
    normalized = normalize_title_number_words(normalized_title_words(value))
    return "".join(character for character in normalized if character.isalnum())


def title_base_key(value: str) -> str:
    normalized = normalize_title_number_words(game_family_key(value))
    return "".join(
        character for character in normalized if character.isalnum()
    )


def title_number_signature(value: str) -> tuple[str, ...]:
    signature: list[str] = []
    for word in normalized_title_words(value).split():
        digits = "".join(character for character in word if character.isdecimal())
        if digits:
            signature.append(str(int(digits)))
        elif word in ROMAN_NUMBERS:
            signature.append(ROMAN_NUMBERS[word])
    return tuple(signature)


def normalize_title_number_words(value: str) -> str:
    return " ".join(
        ROMAN_NUMBERS.get(word, word)
        for word in normalized_title_words(value).split()
    )


def normalized_title_words(value: str) -> str:
    text = str(value or "").replace("™", "").replace("®", "").replace("©", "")
    text = unicodedata.normalize("NFKC", text).casefold()
    words = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(words.split())
