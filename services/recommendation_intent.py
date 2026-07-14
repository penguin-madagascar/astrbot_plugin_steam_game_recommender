from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..storage.models import GamePreference
from .tag_normalizer import canonical_tags_from_terms


class IntentTagRole(str, Enum):
    REQUIRED = "required"
    ANCHOR = "anchor"
    SUPPORTING = "supporting"
    EXCLUDE = "exclude"


class IntentTagSource(str, Enum):
    EXPLICIT = "explicit"
    REFERENCE = "reference"
    DERIVED = "derived"
    LIBRARY = "library"


class ReferencePolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class QualityIntent(str, Enum):
    NORMAL = "normal"
    MAINSTREAM = "mainstream"


@dataclass(frozen=True)
class ReferenceQuery:
    display_title: str
    aliases: tuple[str, ...]
    polarity: ReferencePolarity

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "polarity", ReferencePolarity(self.polarity))


@dataclass(frozen=True)
class WeightedIntentTag:
    tag: str
    role: IntentTagRole
    source: IntentTagSource
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", IntentTagRole(self.role))
        object.__setattr__(self, "source", IntentTagSource(self.source))
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class RecommendationIntent:
    tags: tuple[WeightedIntentTag, ...]
    references: tuple[ReferenceQuery, ...]
    quality_intent: QualityIntent
    allow_unreleased: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "quality_intent", QualityIntent(self.quality_intent))


_ROLE_PRIORITY = {
    IntentTagRole.SUPPORTING: 1,
    IntentTagRole.ANCHOR: 2,
    IntentTagRole.REQUIRED: 3,
    IntentTagRole.EXCLUDE: 3,
}
_SOURCE_PRIORITY = {
    IntentTagSource.LIBRARY: 1,
    IntentTagSource.DERIVED: 2,
    IntentTagSource.REFERENCE: 3,
    IntentTagSource.EXPLICIT: 4,
}


def build_recommendation_intent(preference: GamePreference) -> RecommendationIntent:
    tags_by_canonical: dict[str, WeightedIntentTag] = {}
    groups = (
        (preference.required_tags, IntentTagRole.REQUIRED, IntentTagSource.EXPLICIT, 1.0),
        (preference.genres_like, IntentTagRole.ANCHOR, IntentTagSource.EXPLICIT, 1.0),
        (preference.extra_tags, IntentTagRole.SUPPORTING, IntentTagSource.DERIVED, 0.35),
        (preference.genres_dislike, IntentTagRole.EXCLUDE, IntentTagSource.EXPLICIT, 1.0),
    )
    for values, role, source, weight in groups:
        for tag in canonical_tags_from_terms(values):
            candidate = WeightedIntentTag(tag, role, source, weight)
            current = tags_by_canonical.get(tag)
            if current is None or _tag_priority(candidate) > _tag_priority(current):
                tags_by_canonical[tag] = candidate

    positive_references = _group_positive_references(
        preference.reference_games_like,
        preference.reference_search_terms,
    )
    negative_references = tuple(
        ReferenceQuery(title, (title,), ReferencePolarity.NEGATIVE)
        for title in preference.reference_games_dislike
    )
    return RecommendationIntent(
        tags=tuple(tags_by_canonical.values()),
        references=(*positive_references, *negative_references),
        quality_intent=QualityIntent(preference.quality_intent),
        allow_unreleased=preference.allow_unreleased,
    )


def _tag_priority(tag: WeightedIntentTag) -> tuple[int, float, int]:
    return (_ROLE_PRIORITY[tag.role], tag.weight, _SOURCE_PRIORITY[tag.source])


def _group_positive_references(
    titles: list[str],
    search_terms: list[str],
) -> tuple[ReferenceQuery, ...]:
    if len(titles) == 1:
        aliases = _unique_aliases([titles[0], *search_terms])
        return (ReferenceQuery(titles[0], aliases, ReferencePolarity.POSITIVE),)
    if titles and len(titles) == len(search_terms):
        return tuple(
            ReferenceQuery(
                title,
                _unique_aliases([title, search_term]),
                ReferencePolarity.POSITIVE,
            )
            for title, search_term in zip(titles, search_terms)
        )
    return tuple(
        ReferenceQuery(title, (title,), ReferencePolarity.POSITIVE) for title in titles
    )


def _unique_aliases(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return tuple(result)
