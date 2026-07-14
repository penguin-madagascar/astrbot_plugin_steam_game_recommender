from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..storage.models import GamePreference
from .company_preferences import normalize_company_name
from .played_filter import detect_library_filter_mode
from .tag_normalizer import (
    canonical_tag_occurrences,
    canonical_tags_from_terms,
    normalize_tag,
    normalized_alias_occurrences,
    steam_tag_canonical_key,
)

SOULSLIKE_TERMS = (
    "魂like",
    "魂系",
    "魂类",
    "类魂",
    "soulslike",
    "souls-like",
)

TAG_INTENT_TERMS: dict[str, tuple[str, ...]] = {
    "co_op": ("双人", "两人", "合作", "coop", "co-op"),
    "local_coop": (
        "本地双人合作",
        "本地双人",
        "本地合作",
        "同屏",
        "分屏",
        "local coop",
        "local co-op",
    ),
    "online_coop": ("线上合作", "在线合作", "联机合作", "online coop", "online co-op"),
    "multiplayer": ("多人联机", "多人", "联机", "multiplayer"),
    "puzzle": ("解谜", "谜题", "puzzle"),
    "adventure": ("冒险", "剧情", "adventure"),
    "casual": ("休闲", "casual"),
    "relaxing": ("轻松", "治愈", "cozy", "relaxing"),
    "action": ("动作", "action"),
    "rpg": ("角色扮演", "rpg"),
    "party": ("聚会", "派对", "party"),
    "simulation": ("模拟", "simulation"),
    "farming": ("种田", "农场", "farming", "farm"),
    "management": ("经营", "management"),
    "crafting": ("制作", "crafting"),
    "building": ("建造", "building"),
    "racing": ("赛车", "竞速", "racing"),
    "horror": ("恐怖", "horror"),
    "soulslike": SOULSLIKE_TERMS,
    "roguelike": ("肉鸽", "roguelike", "rogue-like", "roguelite"),
    "violent": ("血腥", "violent", "gore"),
    "singleplayer": ("纯单人", "单机", "单人", "singleplayer", "single-player"),
    "pvp": ("pvp",),
}

REQUIRED_TAG_TERMS: dict[str, tuple[str, ...]] = dict(TAG_INTENT_TERMS)

POLARITY_ONLY_TAGS = {"horror", "soulslike", "roguelike", "violent", "singleplayer", "pvp"}
REFERENCE_DESCRIPTION_SUFFIXES = (
    "游戏",
    "作品",
    "玩法",
    "题材",
    "风格",
    "类型",
    "战斗",
    "氛围",
    "合作",
    "解谜",
)
NEGATIVE_MARKERS = (
    "不要",
    "不想",
    "不喜欢",
    "不需要",
    "别",
    "排除",
    "避免",
    "拒绝",
    "讨厌",
    "no ",
    "not ",
    "without ",
    "exclude ",
    "avoid ",
    "dislike ",
)
HARD_REQUIREMENT_MARKERS = (
    "必须",
    "必需",
    "一定要",
    "务必",
    "只接受",
    "只要",
    "需要支持",
    "must ",
    "required ",
    "need ",
)

PLATFORM_SPAN_PATTERN = re.compile(
    r"(?<![0-9A-Za-z])(?:Steam|PC|Windows|Switch|Nintendo|任天堂|"
    r"PlayStation|PS[45]?|Xbox|电脑)(?![0-9A-Za-z])",
    flags=re.I,
)
COMPANY_ENTITY_AFTER_SPAN_PATTERN = re.compile(
    r"^\s*(?:的\s*)?(?:开发商|发行商|开发|发行|公司|工作室|"
    r"studios?\b|interactive\b|entertainment\b|developer\b|"
    r"publisher\b|company\b|corporation\b)",
    flags=re.I,
)
COMPANY_ENTITY_BEFORE_SPAN_PATTERN = re.compile(
    r"(?:(?:开发商|发行商|开发公司|发行公司|开发方|发行方|"
    r"developer\b|publisher\b|company\b)\s*(?:是|为|:|：|=)?|"
    r"(?:developed|published)\s+by)\s*$",
    flags=re.I,
)
RESULT_QUANTITY_PATTERN = re.compile(
    r"(?P<count>\d+|十|[一二两三四五六七八九])\s*"
    r"(?:个|款|部|(?:games?|results?|titles?)\b)",
    flags=re.I,
)
BUDGET_CURRENCY_PATTERN = (
    r"美元|美金|usd|日元|日币|jpy|円|欧元|eur|英镑|gbp|港币|hkd|"
    r"台币|新台币|twd|韩元|krw|人民币|rmb|cny|元|块"
)
BUDGET_PATTERNS = tuple(
    re.compile(pattern, flags=re.I)
    for pattern in (
        rf"(?:预算|价格|价位|budget|price)\s*"
        rf"(?:(?:改为|改成|调整到|设为|到|为|是|约|最多|不超过|不得超过|"
        rf"不能超过|低于|小于|必须|一定要|只接受|务必|must|be|required|to|"
        rf"under|below|at\s+most|less\s+than|only|accept)\s*)*"
        rf"(?P<symbol>[$€£¥￥]?)\s*(?P<amount>\d+(?:\.\d+)?)\s*"
        rf"(?P<currency>{BUDGET_CURRENCY_PATTERN})?",
        r"(?P<symbol>[$€£¥￥])\s*(?P<amount>\d+(?:\.\d+)?)",
        rf"(?P<amount>\d+(?:\.\d+)?)\s*"
        rf"(?P<currency>{BUDGET_CURRENCY_PATTERN})"
        rf"\s*(?:以内|以下|左右|上下)?",
    )
)
QUALITY_COMPANY_IDENTITIES = {
    "3a",
    "aaa",
    "blockbuster",
    "mainstream",
    "triple a",
    "大作",
}


@dataclass(frozen=True)
class TagPolarityEvent:
    start: int
    end: int
    tag: str
    polarity: str


def infer_preference_from_text(
    text: str,
    *,
    reference_titles: list[str] | None = None,
) -> GamePreference:
    reference_like = extract_reference_games(text)
    reference_dislike = extract_disliked_reference_games(text)
    reference_like = remove_reference_titles(reference_like, reference_dislike)
    intent_text = mask_reference_titles(
        text,
        [*reference_like, *reference_dislike, *(reference_titles or [])],
    )
    lower = intent_text.lower()
    tag_polarities = detect_tag_polarities(intent_text)
    positive_only_tags = [
        tag
        for tag, polarity in tag_polarities.items()
        if polarity == "positive" and tag in POLARITY_ONLY_TAGS
    ]
    negative_tags = [tag for tag, polarity in tag_polarities.items() if polarity == "negative"]
    platforms: list[str] = []
    if "steam" in lower:
        platforms.append("steam")
    if any(word in lower for word in ("pc", "电脑", "windows")):
        platforms.append("pc")
    if any(word in lower for word in ("switch", "任天堂", "ns")):
        platforms.append("nintendo switch")
    if any(word in lower for word in ("playstation", "ps5", "ps4", "psn")):
        platforms.append("playstation")
    if "xbox" in lower:
        platforms.append("xbox")

    genres_like = keyword_hits(
        lower,
        {
            "co-op": ("双人", "两人", "合作", "coop", "co-op"),
            "local co-op": ("本地双人", "本地合作", "同屏", "分屏"),
            "multiplayer": ("多人", "联机"),
            "puzzle": ("解谜", "谜题", "puzzle"),
            "adventure": ("冒险", "剧情", "adventure"),
            "casual": ("休闲", "轻松", "casual", "别太难", "不要太难"),
            "action": ("动作", "action"),
            "rpg": ("rpg", "角色扮演"),
            "party": ("聚会", "派对", "party"),
            "simulation": ("模拟", "simulation"),
            "farming": ("种田", "农场", "farming", "farm"),
            "management": ("经营", "management"),
            "crafting": ("制作", "crafting"),
            "building": ("建造", "building"),
            "racing": ("赛车", "竞速", "racing"),
        },
    )
    genres_like = remove_terms_matching_tags(genres_like, set(negative_tags))
    genres_like = merge_lists(genres_like, positive_only_tags)
    genres_dislike = negative_tags

    wants_multiple_players = any(
        tag_polarities.get(tag) == "positive"
        for tag in ("co_op", "local_coop", "online_coop", "multiplayer")
    )
    players = 2 if wants_multiple_players else None

    budget, budget_currency, budget_is_required = extract_budget(lower)

    result_count = extract_result_count(lower) or 5

    difficulty = None
    if any(
        word in lower
        for word in (
            "别太难",
            "不要太难",
            "简单",
            "轻松",
            "休闲",
            "不要高难",
            "别高难",
            "不高难",
        )
    ):
        difficulty = "easy"
    elif any(word in lower for word in ("高难", "困难", "挑战")):
        difficulty = "hard"

    extra_tags = keyword_hits(
        lower,
        {
            "relaxing": ("轻松", "休闲", "治愈", "别太难", "不要太难", "cozy", "relaxing"),
            "local co-op": ("本地合作", "同屏", "分屏", "远程同乐", "remote play"),
            "online co-op": ("线上合作", "在线合作", "联机合作", "online co-op"),
            "family": ("亲子", "家庭", "family"),
            "party": ("聚会", "派对", "party"),
        },
    )
    extra_tags = remove_terms_matching_tags(extra_tags, set(negative_tags))
    extra_tags = expand_related_extra_tags(extra_tags)
    genres_like = remove_terms_matching_tags(genres_like, set(negative_tags))
    extra_tags = remove_terms_matching_tags(extra_tags, set(negative_tags))
    library_filter_mode = detect_library_filter_mode(intent_text)

    preferred_languages, required_languages = extract_language_preferences(intent_text)
    return GamePreference(
        platforms=platforms,
        required_tags=extract_required_tags(intent_text, tag_polarities),
        genres_like=genres_like,
        extra_tags=extra_tags,
        genres_dislike=genres_dislike,
        reference_games_like=reference_like,
        reference_search_terms=search_terms_from_reference_titles(reference_like),
        reference_games_dislike=reference_dislike,
        players=players,
        budget=budget,
        budget_is_required=budget_is_required,
        budget_currency=budget_currency,
        preferred_languages=preferred_languages,
        required_languages=required_languages,
        difficulty=difficulty,
        mood="轻松" if tag_polarities.get("relaxing") == "positive" else None,
        quality_intent="mainstream" if has_aaa_intent(lower) else "normal",
        allow_unreleased=has_unreleased_intent(lower),
        result_count=result_count,
        library_filter_mode=library_filter_mode,
    )


def merge_text_preference(preference: GamePreference, text: str) -> GamePreference:
    preference_reference_like = clean_reference_titles(
        preference.reference_games_like
    )
    preference_reference_dislike = clean_reference_titles(
        preference.reference_games_dislike
    )
    inferred = infer_preference_from_text(
        text,
        reference_titles=[
            *preference_reference_like,
            *preference.reference_search_terms,
            *preference_reference_dislike,
        ],
    )
    reference_titles = [
        *preference_reference_like,
        *preference.reference_search_terms,
        *preference_reference_dislike,
        *inferred.reference_games_like,
        *inferred.reference_search_terms,
        *inferred.reference_games_dislike,
    ]
    intent_text = mask_reference_titles(text, reference_titles)
    evidence_tags, evidence_events = validated_explicit_tag_evidence(
        preference,
        text,
        reference_titles=reference_titles,
    )
    lexical_tags, lexical_events = validated_same_language_tag_evidence(
        preference,
        intent_text,
    )
    tag_polarities = final_tag_polarities(
        [
            *tag_polarity_events(intent_text),
            *evidence_events,
            *lexical_events,
        ]
    )
    evidenced_tags = {
        event.tag for event in [*evidence_events, *lexical_events]
    }
    corrected_negative_tags = {
        tag
        for tag in evidenced_tags
        if tag_polarities.get(tag) == "negative"
    }
    positive_tags = {tag for tag, polarity in tag_polarities.items() if polarity == "positive"}
    negative_tags = {tag for tag, polarity in tag_polarities.items() if polarity == "negative"}
    data = dump_preference(preference)
    data["required_tags"] = merge_lists(
        remove_terms_matching_tags(preference.required_tags, negative_tags),
        inferred.required_tags,
    )
    data["genres_like"] = merge_lists(
        remove_terms_matching_tags(preference.genres_like, negative_tags),
        inferred.genres_like,
    )
    data["extra_tags"] = merge_lists(
        remove_terms_matching_tags(preference.extra_tags, negative_tags),
        inferred.extra_tags,
    )
    data["genres_dislike"] = merge_lists(
        remove_terms_matching_tags(preference.genres_dislike, positive_tags),
        inferred.genres_dislike,
    )
    data["reference_games_like"] = merge_lists(
        remove_reference_titles(
            preference_reference_like,
            inferred.reference_games_dislike,
        ),
        inferred.reference_games_like,
    )
    data["reference_games_dislike"] = merge_lists(
        remove_reference_titles(
            preference_reference_dislike,
            inferred.reference_games_like,
        ),
        inferred.reference_games_dislike,
    )
    data["reference_search_terms"] = merge_lists(
        merge_lists(
            preference.reference_search_terms,
            search_terms_from_reference_titles(preference_reference_like),
        ),
        inferred.reference_search_terms,
    )
    data["parse_warnings"] = merge_lists(
        preference.parse_warnings,
        inferred.parse_warnings,
    )
    data["platforms"] = merge_platforms(preference.platforms, inferred.platforms)
    data["preferred_languages"] = list(inferred.preferred_languages)
    data["required_languages"] = list(inferred.required_languages)
    data["budget_is_required"] = inferred.budget_is_required
    data["quality_intent"] = (
        "mainstream"
        if "mainstream" in {preference.quality_intent, inferred.quality_intent}
        else "normal"
    )
    data["allow_unreleased"] = preference.allow_unreleased or inferred.allow_unreleased
    companies = validated_company_preferences(preference, text)
    company_reference_titles = [
        value
        for company in companies
        for value in (company.source_span, company.display_name)
    ]
    for field_name in (
        "reference_games_like",
        "reference_search_terms",
        "reference_games_dislike",
    ):
        data[field_name] = remove_reference_titles(
            data[field_name],
            company_reference_titles,
        )
    merged_reference_spans = [
        *data["reference_games_like"],
        *data["reference_search_terms"],
        *data["reference_games_dislike"],
    ]
    company_spans = merge_lists(
        [company.source_span for company in companies],
        contextual_raw_company_spans(preference, text),
    )
    blocked_semantic_spans = semantic_blocked_ranges(
        preference,
        text,
        company_spans=company_spans,
        reference_spans=merged_reference_spans,
    )
    data["company_preferences"] = companies
    data["derived_intent_tags"] = [
        item
        for item in preference.derived_intent_tags
        if exact_span_outside_ranges(item.source_span, text, blocked_semantic_spans)
        and not company_span_has_entity_context(item.source_span, text)
    ][:3]
    data["soft_features"] = [
        item
        for item in preference.soft_features
        if exact_span_outside_ranges(item.source_span, text, blocked_semantic_spans)
        and not company_span_has_entity_context(item.source_span, text)
    ][:3]
    explicit_tags = {
        tag for tag, polarity in tag_polarities.items() if polarity == "positive"
    }
    inferred_required = set(canonical_tags_from_terms(inferred.required_tags))
    inferred_genres = set(canonical_tags_from_terms(inferred.genres_like))
    inferred_extra = set(canonical_tags_from_terms(inferred.extra_tags))
    required, unverified_required = partition_verified_tags(
        data["required_tags"],
        inferred_required
        | evidence_tags["required_tags"]
        | lexical_tags["required_tags"],
        blocked_tags=corrected_negative_tags,
    )
    genres, unverified_genres = partition_verified_tags(
        data["genres_like"],
        explicit_tags
        | inferred_genres
        | evidence_tags["genres_like"]
        | lexical_tags["genres_like"],
        blocked_tags=corrected_negative_tags,
    )
    verified_extra, _unverified_extra = partition_verified_tags(
        data["extra_tags"],
        explicit_tags
        | inferred_extra
        | evidence_tags["extra_tags"]
        | lexical_tags["extra_tags"],
        blocked_tags=corrected_negative_tags,
    )
    inferred_dislikes = set(canonical_tags_from_terms(inferred.genres_dislike))
    verified_dislikes, _unverified_dislikes = partition_verified_tags(
        data["genres_dislike"],
        (
            inferred_dislikes
            | evidence_tags["genres_dislike"]
            | lexical_tags["genres_dislike"]
        )
        & negative_tags,
    )
    data["required_tags"] = required
    data["genres_like"] = genres
    data["genres_dislike"] = merge_lists(
        verified_dislikes,
        sorted(corrected_negative_tags),
    )
    if inferred.quality_intent == "mainstream":
        data["extra_tags"] = verified_extra
    else:
        data["extra_tags"] = merge_lists(
            remove_quality_tags(data["extra_tags"]),
            [*unverified_required, *unverified_genres],
        )
    for field in ("players", "budget", "difficulty", "mood"):
        if getattr(preference, field) in (None, "", []):
            data[field] = getattr(inferred, field)
    if not preference.library_filter_mode:
        data["library_filter_mode"] = inferred.library_filter_mode
    if explicit_count := extract_result_count(text):
        data["result_count"] = explicit_count
    elif not preference.result_count:
        data["result_count"] = inferred.result_count
    validator = getattr(GamePreference, "model_validate", None)
    return validator(data) if validator else GamePreference.parse_obj(data)


def extract_reference_games(text: str) -> list[str]:
    titles: list[str] = []
    patterns = [
        r"(?:类似|像是|接近|参考|像\s*)(?:《([^》]+)》|([^，。,.；;!?！？\n]{1,60}))",
        r"(?<![0-9a-z_])(?:similar to|like)\b\s+([^，。,.；;!?！？\n]{2,60})",
        r"(?:喜欢|偏爱|钟爱)\s*(?:《([^》]+)》|([^，。,.；;!?！？\n]{2,60}))",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            if has_negative_reference_prefix(text, match.start()):
                continue
            for group in match.groups():
                title = clean_reference_title(group)
                if title and is_probable_reference_title(title):
                    titles.append(title)
                    break
    return merge_lists([], titles)


def extract_disliked_reference_games(text: str) -> list[str]:
    titles: list[str] = []
    patterns = [
        (
            r"(?:不要|别|不想要|不喜欢|排除|避免)\s*(?:再)?\s*"
            r"(?:类似|像|接近|参考)\s*(?:《([^》]+)》|([^，。,.；;!?！？\n]{1,60}))"
        ),
        r"(?:不要|别|不想要|不喜欢|排除|避免)\s*《([^》]+)》",
        r"(?:不喜欢|讨厌)\s*([^，。,.；;!?！？\n]{1,60}?)(?=\s*(?:这类|这种|这一类|这款))",
        r"(?<![0-9a-z_])(?:not like|unlike|avoid|dislike)\b\s+([^，。,.；;!?！？\n]{2,60})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            for group in match.groups():
                title = clean_reference_title(group)
                if title and is_probable_reference_title(title):
                    titles.append(title)
                    break
    return merge_lists([], titles)


def clean_reference_title(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" 《》\"'“”‘’")
    if not text:
        return ""
    text = re.split(
        r"(?:这类|这种|这一类|但|不过|不要|别|最好|可以|能|，|。|,|\.|；|;|!|！|\?)",
        text,
        maxsplit=1,
    )[0].strip()
    if "的" in text:
        title, suffix = text.rsplit("的", maxsplit=1)
        descriptive_prefixes = (
            "游戏",
            "作品",
            "玩法",
            "多人",
            "双人",
            "单人",
            "合作",
            "同类",
            "风格",
        )
        if (
            suffix.startswith(descriptive_prefixes)
            or suffix.endswith(REFERENCE_DESCRIPTION_SUFFIXES)
            or has_aaa_intent(suffix)
        ):
            text = title.strip()
    return text[:80]


def clean_reference_titles(values: list[str]) -> list[str]:
    return merge_lists(
        [],
        [title for value in values if (title := clean_reference_title(value))],
    )


def mask_reference_titles(text: str, titles: list[str]) -> str:
    masked = str(text or "")
    unique_titles = sorted(
        {title.strip() for title in titles if title and title.strip()},
        key=len,
        reverse=True,
    )
    for title in unique_titles:
        pattern = r"\s+".join(re.escape(part) for part in title.split())
        if not pattern:
            continue
        masked = re.sub(
            pattern,
            lambda match: " " * len(match.group(0)),
            masked,
            flags=re.I,
        )
    return masked


def is_probable_reference_title(value: str) -> bool:
    title = normalize_reference_title(value)
    if not title or canonical_tags_from_terms([title]):
        return False
    if title in {"高难", "太难", "简单", "困难"}:
        return False
    return not title.endswith(REFERENCE_DESCRIPTION_SUFFIXES)


def has_negative_reference_prefix(text: str, start: int) -> bool:
    left = clause_left(text.lower(), start)[-12:]
    return any(marker.strip() in left for marker in NEGATIVE_MARKERS) or bool(
        re.search(r"(?:不|没)\s*$", left)
    )


def detect_tag_polarities(text: str) -> dict[str, str]:
    return final_tag_polarities(tag_polarity_events(text))


def tag_polarity_events(text: str) -> list[TagPolarityEvent]:
    lower = unicodedata.normalize("NFKC", str(text or "")).casefold()
    lower = re.sub(
        r"排除\s*已有|exclude[-_ ]owned",
        lambda match: " " * len(match.group(0)),
        lower,
    )
    events: list[TagPolarityEvent] = []
    for tag, terms in TAG_INTENT_TERMS.items():
        for term in terms:
            for match in re.finditer(re.escape(term.lower()), lower):
                polarity = (
                    "negative"
                    if is_negative_context(lower, match.start(), match.end())
                    else "positive"
                )
                events.append(
                    TagPolarityEvent(match.start(), match.end(), tag, polarity)
                )
    for tag, start, end in canonical_tag_occurrences(lower):
        polarity = "negative" if is_negative_context(lower, start, end) else "positive"
        events.append(TagPolarityEvent(start, end, tag, polarity))
    return events


def final_tag_polarities(events: list[TagPolarityEvent]) -> dict[str, str]:
    polarities: dict[str, str] = {}
    for event in sorted(events, key=lambda item: (item.start, item.end)):
        polarities[event.tag] = event.polarity
    return polarities


def extract_required_tags(text: str, polarities: dict[str, str] | None = None) -> list[str]:
    lower = text.lower()
    final_polarities = polarities or detect_tag_polarities(text)
    matches: list[tuple[int, int, str]] = []
    for tag, terms in REQUIRED_TAG_TERMS.items():
        for term in terms:
            for match in re.finditer(re.escape(term.lower()), lower):
                matches.append((match.start(), match.end(), tag))

    required: list[str] = []
    covered_spans: list[tuple[int, int]] = []
    for start, end, tag in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(
            start >= covered_start and end <= covered_end
            for covered_start, covered_end in covered_spans
        ):
            continue
        covered_spans.append((start, end))
        if final_polarities.get(tag) == "negative":
            continue
        requirement_left = re.split(r"的", clause_left(lower, start))[-1]
        if any(marker in requirement_left[-24:] for marker in HARD_REQUIREMENT_MARKERS):
            required = merge_lists(required, [tag])
    return required


def extract_language_preferences(text: str) -> tuple[list[str], list[str]]:
    lower = text.lower()
    matches: list[tuple[int, int, str]] = []
    language_terms = {
        "schinese": ("simplified chinese", "schinese", "简体中文", "简中"),
        "tchinese": ("traditional chinese", "tchinese", "繁体中文", "繁中"),
        "english": ("english", "英语", "英文"),
        "japanese": ("japanese", "日语", "日文"),
        "koreana": ("korean", "韩语", "韩文"),
    }
    explicit_spans: list[tuple[int, int]] = []
    for language, terms in language_terms.items():
        for term in terms:
            for match in re.finditer(re.escape(term), lower):
                matches.append((match.start(), match.end(), language))
                explicit_spans.append((match.start(), match.end()))
    for term in ("中文", "chinese", "汉化"):
        for match in re.finditer(re.escape(term), lower):
            if any(match.start() >= start and match.end() <= end for start, end in explicit_spans):
                continue
            matches.append((match.start(), match.end(), "schinese"))

    preferred: list[str] = []
    required: list[str] = []
    for start, end, language in sorted(matches, key=lambda item: (item[0], item[1])):
        if is_negative_context(lower, start, end):
            preferred = [item for item in preferred if item != language]
            required = [item for item in required if item != language]
            continue
        requirement_left = re.split(r"的", clause_left(lower, start))[-1]
        if any(marker in requirement_left[-24:] for marker in HARD_REQUIREMENT_MARKERS):
            required = merge_lists(required, [language])
            preferred = [item for item in preferred if item != language]
        elif language not in required:
            preferred = merge_lists(preferred, [language])
    return preferred, required


def is_negative_context(text: str, start: int, end: int) -> bool:
    left = clause_left(text, start)[-18:]
    right = clause_right(text, end)[:10].lstrip()
    if any(marker in left for marker in NEGATIVE_MARKERS):
        return True
    if re.search(r"(?:不|无)\s*(?:太\s*)?(?:想|喜欢|要|是)?\s*$", left):
        return True
    return right.startswith(("不要", "别", "排除", "算了", "free"))


def clause_left(text: str, position: int) -> str:
    return re.split(
        r"[，。,.；;!?！？\n]|(?:但|不过|然而|其实|改成|改为|现在|后来|还是)",
        text[:position],
    )[-1]


def clause_right(text: str, position: int) -> str:
    return re.split(
        r"[，。,.；;!?！？\n]|(?:但|不过|然而|其实|改成|改为|现在|后来|还是)",
        text[position:],
        maxsplit=1,
    )[0]


def search_terms_from_reference_titles(titles: list[str]) -> list[str]:
    terms = [title for title in titles if re.search(r"[A-Za-z]", title)]
    return merge_lists([], terms)


def has_aaa_intent(text: str) -> bool:
    lower = text.lower()
    return bool(re.search(r"(?<![0-9a-z])(?:3a|aaa)(?![0-9a-z])", lower)) or any(
        term in lower for term in ("triple-a", "triple a", "大作")
    )


def has_unreleased_intent(text: str) -> bool:
    lower = text.lower()
    terms = (
        "尚未发售",
        "未发售",
        "即将发售",
        "尚未发行",
        "未发行",
        "即将推出",
        "unreleased",
        "upcoming",
        "coming-soon",
        "coming soon",
    )
    for term in terms:
        for match in re.finditer(re.escape(term), lower):
            if not is_negative_context(lower, match.start(), match.end()):
                return True
    return False


QUALITY_TAG_KEYS = {
    "3a",
    "aaa",
    "blockbuster",
    "high_quality",
    "mainstream",
    "quality",
    "triple_a",
}


def is_quality_only_tag(value: str) -> bool:
    return has_aaa_intent(value) or steam_tag_canonical_key(value) in QUALITY_TAG_KEYS


def partition_verified_tags(
    values: list[str],
    allowed_tags: set[str],
    *,
    blocked_tags: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    verified: list[str] = []
    unverified: list[str] = []
    blocked = blocked_tags or set()
    for value in values:
        if is_quality_only_tag(value):
            continue
        canonical = canonical_tag_key(value)
        if canonical in blocked:
            continue
        if canonical in allowed_tags:
            verified.append(value)
        else:
            unverified.append(value)
    return verified, unverified


def remove_quality_tags(values: list[str]) -> list[str]:
    return [value for value in values if not is_quality_only_tag(value)]


CANONICAL_TAG_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
EVIDENCE_TARGETS = (
    "required_tags",
    "genres_like",
    "extra_tags",
    "genres_dislike",
)


def canonical_tag_key(value: str) -> str | None:
    if canonical := normalize_tag(value):
        return canonical
    fallback = steam_tag_canonical_key(value)
    return fallback if CANONICAL_TAG_PATTERN.fullmatch(fallback) else None


def validated_explicit_tag_evidence(
    preference: GamePreference,
    text: str,
    *,
    reference_titles: list[str] | None = None,
) -> tuple[dict[str, set[str]], list[TagPolarityEvent]]:
    allowed = {target: set() for target in EVIDENCE_TARGETS}
    events: list[TagPolarityEvent] = []
    source = unicodedata.normalize("NFKC", str(text or "")).casefold()
    all_reference_titles = [
        *preference.reference_games_like,
        *preference.reference_search_terms,
        *preference.reference_games_dislike,
        *(reference_titles or []),
    ]
    normalized_references = [
        unicodedata.normalize("NFKC", title).casefold()
        for title in all_reference_titles
        if title
    ]
    field_tags = {
        target: {
            tag
            for value in getattr(preference, target)
            if (tag := canonical_tag_key(value))
        }
        for target in EVIDENCE_TARGETS
    }

    for evidence in preference.explicit_tag_evidence:
        target = evidence.target
        tag = canonical_tag_key(evidence.tag)
        raw_tag = steam_tag_canonical_key(evidence.tag)
        span = unicodedata.normalize("NFKC", evidence.span).casefold()
        if (
            target not in allowed
            or not tag
            or not CANONICAL_TAG_PATTERN.fullmatch(raw_tag)
            or tag not in field_tags[target]
            or not span
            or is_quality_only_tag(evidence.tag)
            or is_quality_only_tag(span)
        ):
            continue
        start = source.rfind(span)
        if start < 0 or any(span in title for title in normalized_references):
            continue
        if (span_tag := normalize_tag(span)) and span_tag != tag:
            continue
        is_negative = is_negative_context(source, start, start + len(span))
        events.append(
            TagPolarityEvent(
                start=start,
                end=start + len(span),
                tag=tag,
                polarity="negative" if is_negative else "positive",
            )
        )
        if is_negative:
            if target == "genres_dislike":
                allowed[target].add(tag)
            continue
        if target == "genres_dislike":
            continue
        if target == "required_tags" and not span_has_hard_requirement(source, start):
            continue
        allowed[target].add(tag)
    return allowed, events


def validated_same_language_tag_evidence(
    preference: GamePreference,
    text: str,
) -> tuple[dict[str, set[str]], list[TagPolarityEvent]]:
    allowed = {target: set() for target in EVIDENCE_TARGETS}
    events: list[TagPolarityEvent] = []
    source = unicodedata.normalize("NFKC", str(text or "")).casefold()
    for target in EVIDENCE_TARGETS:
        for value in getattr(preference, target):
            tag = canonical_tag_key(value)
            if not tag or is_quality_only_tag(value):
                continue
            occurrences = normalized_alias_occurrences(value, source)
            if not occurrences:
                continue
            start, end = max(occurrences, key=lambda occurrence: occurrence[0])
            is_negative = is_negative_context(source, start, end)
            events.append(
                TagPolarityEvent(
                    start=start,
                    end=end,
                    tag=tag,
                    polarity="negative" if is_negative else "positive",
                )
            )
            if is_negative:
                if target == "genres_dislike":
                    allowed[target].add(tag)
            elif target == "genres_dislike":
                continue
            elif target != "required_tags" or span_has_hard_requirement(source, start):
                allowed[target].add(tag)
    return allowed, events


def span_has_hard_requirement(text: str, start: int) -> bool:
    requirement_scope = re.split(
        r"的|(?:(?:并且?|同时|另外|而且)\s*)?"
        r"(?:想玩|想要|偏好|喜欢|推荐|寻找|找|"
        r"want|prefer|recommend|look(?:ing)?\s+for)",
        clause_left(text, start),
        flags=re.I,
    )[-1]
    return any(
        marker.strip() in requirement_scope[-24:]
        for marker in HARD_REQUIREMENT_MARKERS
    )


def expand_related_extra_tags(tags: list[str]) -> list[str]:
    expanded = list(tags)
    if "soulslike" in expanded:
        expanded = merge_lists(expanded, ["action", "rpg"])
    return expanded


def extract_result_count(text: str) -> int | None:
    count_match = RESULT_QUANTITY_PATTERN.search(text)
    if not count_match:
        return None
    raw_count = count_match.group("count")
    chinese_counts = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    count = int(raw_count) if raw_count.isdigit() else chinese_counts[raw_count]
    return min(max(count, 1), 10)


def extract_budget(text: str) -> tuple[float | None, str | None, bool]:
    match = next(
        (result for pattern in BUDGET_PATTERNS if (result := pattern.search(text))),
        None,
    )
    if match is None:
        return None, None, False
    amount = float(match.group("amount"))
    symbol = str(match.groupdict().get("symbol") or "")
    currency_text = str(match.groupdict().get("currency") or "").lower()
    currency = currency_from_budget_token(symbol or currency_text)
    clause_start = max(
        (text.rfind(mark, 0, match.start()) for mark in ",，。；;!?！？\n"),
        default=-1,
    )
    clause_ends = [
        position
        for mark in ",，。；;!?！？\n"
        if (position := text.find(mark, match.end())) >= 0
    ]
    clause_end = min(clause_ends) if clause_ends else len(text)
    clause = text[clause_start + 1 : clause_end]
    is_required = bool(
        re.search(
            r"必须|一定要|只接受|务必|不得超过|不能超过|"
            r"\bmust\b|\brequired\b|\bonly\s+accept\b",
            clause,
        )
    )
    return amount, currency, is_required


def currency_from_budget_token(value: str) -> str | None:
    token = str(value or "").strip().lower()
    if token == "$" or token in {"美元", "美金", "usd"}:
        return "USD"
    if token == "€" or token in {"欧元", "eur"}:
        return "EUR"
    if token == "£" or token in {"英镑", "gbp"}:
        return "GBP"
    if token in {"日元", "日币", "jpy", "円"}:
        return "JPY"
    if token in {"港币", "hkd"}:
        return "HKD"
    if token in {"台币", "新台币", "twd"}:
        return "TWD"
    if token in {"韩元", "krw"}:
        return "KRW"
    if token in {"人民币", "rmb", "cny", "元", "块"}:
        return "CNY"
    return None


def keyword_hits(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, words in mapping.items() if any(word in text for word in words)]


def merge_lists(left: list[str], right: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in [*left, *right]:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def remove_terms_matching_tags(values: list[str], blocked_tags: set[str]) -> list[str]:
    if not blocked_tags:
        return list(values)
    return [
        value for value in values if not (set(canonical_tags_from_terms([value])) & blocked_tags)
    ]


def remove_reference_titles(values: list[str], blocked_titles: list[str]) -> list[str]:
    blocked = {normalize_reference_title(value) for value in blocked_titles}
    return [value for value in values if normalize_reference_title(value) not in blocked]


def normalize_reference_title(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def merge_platforms(llm_platforms: list[str], text_platforms: list[str]) -> list[str]:
    if text_platforms:
        return merge_lists([], text_platforms)
    return merge_lists([], llm_platforms)


def dump_preference(preference: GamePreference) -> dict:
    dumper = getattr(preference, "model_dump", None)
    return dumper() if dumper else preference.dict()


def validated_company_preferences(
    preference: GamePreference,
    text: str,
    *,
    reference_spans: list[str] | None = None,
) -> list:
    blocked = semantic_blocked_ranges(
        preference,
        text,
        company_spans=[],
        block_platforms=False,
        reference_spans=reference_spans,
    )
    result = []
    for company in preference.company_preferences:
        source_span = company.source_span
        source_identity = normalize_company_name(source_span)
        if (
            not source_identity
            or is_quality_company_identity(source_span)
            or is_quality_company_identity(company.display_name)
            or not platform_company_span_is_grounded(source_span, text)
            or normalize_company_name(company.display_name) != source_identity
            or not exact_span_outside_ranges(source_span, text, blocked)
        ):
            continue
        result.append(company)
        if len(result) >= 3:
            break
    return result


def is_quality_company_identity(value: str) -> bool:
    return (
        has_aaa_intent(value)
        or normalize_company_name(value) in QUALITY_COMPANY_IDENTITIES
    )


def contextual_raw_company_spans(
    preference: GamePreference,
    text: str,
) -> list[str]:
    return merge_lists(
        [],
        [
            company.source_span
            for company in preference.company_preferences
            if company_span_has_entity_context(company.source_span, text)
        ],
    )


def company_span_has_entity_context(source_span: str, text: str) -> bool:
    for start, end in exact_span_occurrences(source_span, text):
        if COMPANY_ENTITY_AFTER_SPAN_PATTERN.search(text[end:]):
            return True
        if COMPANY_ENTITY_BEFORE_SPAN_PATTERN.search(text[:start]):
            return True
    return False


def platform_company_span_is_grounded(source_span: str, text: str) -> bool:
    if not PLATFORM_SPAN_PATTERN.search(source_span):
        return True
    return company_span_has_entity_context(source_span, text)


def semantic_blocked_ranges(
    preference: GamePreference,
    text: str,
    *,
    company_spans: list[str],
    block_platforms: bool = True,
    reference_spans: list[str] | None = None,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    references = reference_spans or [
        *preference.reference_games_like,
        *preference.reference_search_terms,
        *preference.reference_games_dislike,
    ]
    for value in references:
        ranges.extend(reference_span_occurrences(value, text))
    for value in company_spans:
        ranges.extend(exact_span_occurrences(value, text))

    ranges.extend(budget_span_occurrences(text))
    ranges.extend(result_quantity_span_occurrences(text))
    if block_platforms:
        ranges.extend(
            (match.start(), match.end()) for match in PLATFORM_SPAN_PATTERN.finditer(text)
        )
    return sorted(set(ranges))


def budget_span_occurrences(text: str) -> list[tuple[int, int]]:
    return sorted(
        {
            (match.start(), match.end())
            for pattern in BUDGET_PATTERNS
            for match in pattern.finditer(text)
        }
    )


def result_quantity_span_occurrences(text: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in RESULT_QUANTITY_PATTERN.finditer(text)
    ]


def exact_span_outside_ranges(
    span: str,
    text: str,
    blocked: list[tuple[int, int]],
) -> bool:
    return any(
        not any(start < blocked_end and end > blocked_start for blocked_start, blocked_end in blocked)
        for start, end in exact_span_occurrences(span, text)
    )


def exact_span_occurrences(span: str, text: str) -> list[tuple[int, int]]:
    needle = str(span or "")
    if not needle:
        return []
    occurrences: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = text.find(needle, offset)
        if start < 0:
            return occurrences
        occurrences.append((start, start + len(needle)))
        offset = start + 1


def reference_span_occurrences(span: str, text: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    markers = (
        "类似",
        "像是",
        "像",
        "接近",
        "参考",
        "喜欢",
        "偏爱",
        "钟爱",
        "不要",
        "别",
        "不想要",
        "不喜欢",
        "排除",
        "避免",
        "讨厌",
        "like",
        "similar to",
        "not like",
        "unlike",
        "avoid",
        "dislike",
    )
    for start, end in exact_span_occurrences(span, text):
        bracketed = start > 0 and end < len(text) and text[start - 1] == "《" and text[end] == "》"
        left = text[max(start - 16, 0) : start].casefold().rstrip()
        introduced = any(left.endswith(marker) for marker in markers)
        if bracketed or introduced:
            result.append((start, end))
    return result
