from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..storage.models import GameCandidate, GamePreference
from .tag_normalizer import (
    canonical_tags_from_terms,
    steam_tag_result_count_for,
)


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

    derived_groups: list[list[str]] = []
    if preference.players is not None and preference.players >= 2:
        derived_groups.append(["co_op", "multiplayer"])
    if preference.difficulty and any(
        word in preference.difficulty for word in ("easy", "简单", "轻松", "休闲")
    ):
        derived_groups.append(["casual", "relaxing"])
    if preference.mood:
        derived_groups.append([preference.mood])
    for values in derived_groups:
        for tag in canonical_tags_from_terms(values):
            candidate = WeightedIntentTag(
                tag,
                IntentTagRole.SUPPORTING,
                IntentTagSource.DERIVED,
                0.35,
            )
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


def expand_intent_with_reference_tags(
    intent: RecommendationIntent,
    positive_reference_candidates: list[GameCandidate] | tuple[GameCandidate, ...],
) -> RecommendationIntent:
    tags_by_canonical = {tag.tag: tag for tag in intent.tags}
    for reference in positive_reference_candidates:
        source_tags = (
            reference.ordered_tags
            if reference.ordered_tags
            else [*reference.tags, *reference.genres]
        )
        reference_tags = canonical_tags_from_terms(source_tags)[:10]
        anchor_pool = canonical_tags_from_terms(source_tags[:5])
        counted = [
            (tag, count)
            for tag in anchor_pool
            if (count := steam_tag_result_count_for(tag)) is not None
        ]
        if len(counted) >= 2:
            selected = {tag for tag, _count in sorted(counted, key=lambda item: item[1])[:2]}
            anchors = [tag for tag in anchor_pool if tag in selected]
        else:
            anchors = anchor_pool[:2]
        anchor_weights = {
            tag: weight for tag, weight in zip(anchors, (1.0, 0.8))
        }

        for position, tag in enumerate(reference_tags):
            if tag in anchor_weights:
                candidate = WeightedIntentTag(
                    tag,
                    IntentTagRole.ANCHOR,
                    IntentTagSource.REFERENCE,
                    anchor_weights[tag],
                )
            else:
                candidate = WeightedIntentTag(
                    tag,
                    IntentTagRole.SUPPORTING,
                    IntentTagSource.REFERENCE,
                    0.5 * 0.85**position,
                )
            current = tags_by_canonical.get(tag)
            if current is None or _tag_priority(candidate) > _tag_priority(current):
                tags_by_canonical[tag] = candidate

    return RecommendationIntent(
        tags=tuple(tags_by_canonical.values()),
        references=intent.references,
        quality_intent=intent.quality_intent,
        allow_unreleased=intent.allow_unreleased,
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
