from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import islice

from ..storage.models import (
    MAX_REFERENCE_ALIASES_PER_ENTITY,
    MAX_REFERENCE_ENTITIES,
    GameCandidate,
    GamePreference,
)
from .tag_normalizer import (
    ASCII_CANONICAL_TAG_PATTERN,
    canonical_tag_from_vocabulary,
    canonical_steam_tag_name,
    canonical_tags_from_terms,
    static_canonical_tags,
)


class IntentTagRole(str, Enum):
    REQUIRED = "required"
    ANCHOR = "anchor"
    SUPPORTING = "supporting"
    RECALL_ONLY = "recall_only"
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
        object.__setattr__(
            self,
            "aliases",
            _unique_aliases([self.display_title, *self.aliases]),
        )
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
        object.__setattr__(
            self,
            "references",
            tuple(self.references)[:MAX_REFERENCE_ENTITIES],
        )
        object.__setattr__(self, "quality_intent", QualityIntent(self.quality_intent))


_ROLE_PRIORITY = {
    IntentTagRole.RECALL_ONLY: 0,
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
_FEATURE_ROLE_PRIORITY = {"required": 0, "core": 1, "optional": 2}
MAX_PROXY_TAGS_PER_FEATURE = 2


def build_recommendation_intent(
    preference: GamePreference,
    *,
    known_tags: set[str] | frozenset[str] | None = None,
    known_tag_aliases: Mapping[str, str] | None = None,
) -> RecommendationIntent:
    vocabulary = frozenset(known_tags) if known_tags is not None else static_canonical_tags()
    tags_by_canonical: dict[str, WeightedIntentTag] = {}
    groups = (
        (preference.required_tags, IntentTagRole.REQUIRED, IntentTagSource.EXPLICIT, 1.0),
        (preference.genres_like, IntentTagRole.ANCHOR, IntentTagSource.EXPLICIT, 1.0),
        (preference.genres_dislike, IntentTagRole.EXCLUDE, IntentTagSource.EXPLICIT, 1.0),
    )
    for values, role, source, weight in groups:
        for tag in canonical_explicit_terms(
            values,
            vocabulary,
            allow_unknown=known_tags is None,
            aliases=known_tag_aliases,
        ):
            candidate = WeightedIntentTag(tag, role, source, weight)
            current = tags_by_canonical.get(tag)
            if current is None or _tag_priority(candidate) > _tag_priority(current):
                tags_by_canonical[tag] = candidate

    for derived in preference.derived_intent_tags:
        if derived.tag not in vocabulary:
            continue
        candidate = WeightedIntentTag(
            derived.tag,
            IntentTagRole.SUPPORTING,
            IntentTagSource.DERIVED,
            0.25,
        )
        current = tags_by_canonical.get(derived.tag)
        if current is None or _tag_priority(candidate) > _tag_priority(current):
            tags_by_canonical[derived.tag] = candidate

    ordered_features = sorted(
        enumerate(preference.soft_features),
        key=lambda item: (_FEATURE_ROLE_PRIORITY[item[1].role], item[0]),
    )
    for _position, feature in ordered_features:
        valid_proxy_tags = (tag for tag in feature.proxy_tags if tag in vocabulary)
        for tag in islice(valid_proxy_tags, MAX_PROXY_TAGS_PER_FEATURE):
            candidate = WeightedIntentTag(
                tag,
                IntentTagRole.RECALL_ONLY,
                IntentTagSource.DERIVED,
                0.25,
            )
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
        for tag in canonical_terms_from_vocabulary(values, vocabulary):
            candidate = WeightedIntentTag(
                tag,
                IntentTagRole.SUPPORTING,
                IntentTagSource.DERIVED,
                0.35,
            )
            current = tags_by_canonical.get(tag)
            if current is None or _tag_priority(candidate) > _tag_priority(current):
                tags_by_canonical[tag] = candidate

    if preference.reference_entities:
        references = tuple(
            ReferenceQuery(
                entity.display_title,
                tuple(entity.aliases),
                (
                    ReferencePolarity.NEGATIVE
                    if entity.polarity == "negative"
                    else ReferencePolarity.POSITIVE
                ),
            )
            for entity in preference.reference_entities
        )
    else:
        positive_references = _group_positive_references(
            preference.reference_games_like,
            preference.reference_search_terms,
        )
        negative_references = tuple(
            ReferenceQuery(title, (title,), ReferencePolarity.NEGATIVE)
            for title in preference.reference_games_dislike
        )
        references = (*positive_references, *negative_references)
    return RecommendationIntent(
        tags=tuple(tags_by_canonical.values()),
        references=references,
        quality_intent=QualityIntent(preference.quality_intent),
        allow_unreleased=preference.allow_unreleased,
    )


def expand_intent_with_reference_tags(
    intent: RecommendationIntent,
    positive_reference_candidates: list[GameCandidate] | tuple[GameCandidate, ...],
    tag_result_counts: Mapping[str, int] | None = None,
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
            if (count := (tag_result_counts or {}).get(tag))
            is not None
            and type(count) is int
            and count >= 0
        ]
        if counted and len(counted) == len(anchor_pool):
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


def _tag_priority(tag: WeightedIntentTag) -> tuple[int, int, float]:
    return (_ROLE_PRIORITY[tag.role], _SOURCE_PRIORITY[tag.source], tag.weight)


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
        if len(result) >= MAX_REFERENCE_ALIASES_PER_ENTITY:
            break
    return tuple(result)


def canonical_terms_from_vocabulary(
    values: list[str] | tuple[str, ...],
    vocabulary: frozenset[str],
) -> list[str]:
    result: list[str] = []
    for value in values:
        canonical = canonical_tag_from_vocabulary(value, vocabulary)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def canonical_explicit_terms(
    values: list[str] | tuple[str, ...],
    vocabulary: frozenset[str],
    *,
    allow_unknown: bool,
    aliases: Mapping[str, str] | None = None,
) -> list[str]:
    result: list[str] = []
    for value in values:
        canonical = canonical_tag_from_vocabulary(value, vocabulary, aliases)
        if canonical is None and allow_unknown:
            canonical = canonical_steam_tag_name(value)
            if not ASCII_CANONICAL_TAG_PATTERN.fullmatch(canonical):
                continue
        if canonical is None:
            continue
        if canonical not in result:
            result.append(canonical)
    return result
