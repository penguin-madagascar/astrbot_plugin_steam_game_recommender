from __future__ import annotations

import re

from ..storage.models import GamePreference
from .played_filter import detect_library_filter_mode
from .tag_normalizer import canonical_tags_from_terms

SOULSLIKE_TERMS = (
    "魂like",
    "魂系",
    "魂类",
    "类魂",
    "soulslike",
    "souls-like",
    "dark souls",
)

AAA_GENRE_TAGS = ["action", "adventure", "rpg"]
AAA_EXTRA_TAGS = ["aaa", "story rich", "open world"]

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
    "soulslike": SOULSLIKE_TERMS[:-1],
    "roguelike": ("肉鸽", "roguelike", "rogue-like", "roguelite"),
    "violent": ("血腥", "violent", "gore"),
    "singleplayer": ("纯单人", "singleplayer", "single-player"),
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


def infer_preference_from_text(text: str) -> GamePreference:
    lower = text.lower()
    tag_polarities = detect_tag_polarities(text)
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

    reference_like = extract_reference_games(text)
    reference_dislike = extract_disliked_reference_games(text)
    reference_like = remove_reference_titles(reference_like, reference_dislike)
    if references_imply_soulslike(reference_like):
        genres_like = merge_lists(genres_like, ["soulslike"])
    if has_aaa_intent(lower):
        genres_like = merge_lists(genres_like, AAA_GENRE_TAGS)
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
    if references_imply_soulslike(reference_like):
        extra_tags = merge_lists(extra_tags, ["soulslike"])
    if has_aaa_intent(lower):
        extra_tags = merge_lists(extra_tags, AAA_EXTRA_TAGS)
    extra_tags = expand_related_extra_tags(extra_tags)
    genres_like = remove_terms_matching_tags(genres_like, set(negative_tags))
    extra_tags = remove_terms_matching_tags(extra_tags, set(negative_tags))
    library_filter_mode = detect_library_filter_mode(text)

    preferred_languages, required_languages = extract_language_preferences(text)
    return GamePreference(
        platforms=platforms,
        required_tags=extract_required_tags(text, tag_polarities),
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
        result_count=result_count,
        library_filter_mode=library_filter_mode,
    )


def merge_text_preference(preference: GamePreference, text: str) -> GamePreference:
    inferred = infer_preference_from_text(text)
    tag_polarities = detect_tag_polarities(text)
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
            preference.reference_games_like,
            inferred.reference_games_dislike,
        ),
        inferred.reference_games_like,
    )
    data["reference_games_dislike"] = merge_lists(
        remove_reference_titles(
            preference.reference_games_dislike,
            inferred.reference_games_like,
        ),
        inferred.reference_games_dislike,
    )
    data["reference_search_terms"] = merge_lists(
        preference.reference_search_terms,
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
        if suffix.startswith(descriptive_prefixes) or suffix.endswith(
            REFERENCE_DESCRIPTION_SUFFIXES
        ):
            text = title.strip()
    return text[:80]


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
    lower = text.lower()
    lower = re.sub(
        r"排除\s*已有|exclude[-_ ]owned",
        lambda match: " " * len(match.group(0)),
        lower,
    )
    events: list[tuple[int, int, str, str]] = []
    for tag, terms in TAG_INTENT_TERMS.items():
        for term in terms:
            for match in re.finditer(re.escape(term.lower()), lower):
                polarity = (
                    "negative"
                    if is_negative_context(lower, match.start(), match.end())
                    else "positive"
                )
                events.append((match.start(), match.end(), tag, polarity))

    polarities: dict[str, str] = {}
    for _start, _end, tag, polarity in sorted(events, key=lambda item: (item[0], item[1])):
        polarities[tag] = polarity
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


def references_imply_soulslike(titles: list[str]) -> bool:
    return any("魂" in title or "souls" in title.lower() for title in titles)


def has_aaa_intent(text: str) -> bool:
    lower = text.lower()
    return bool(re.search(r"(?<![0-9a-z])(?:3a|aaa)(?![0-9a-z])", lower)) or any(
        term in lower for term in ("triple-a", "triple a", "大作")
    )


def expand_related_extra_tags(tags: list[str]) -> list[str]:
    expanded = list(tags)
    if "soulslike" in expanded:
        expanded = merge_lists(expanded, ["action", "rpg"])
    return expanded


def extract_result_count(text: str) -> int | None:
    count_match = re.search(r"(\d+)\s*(?:个|款|部)", text.lower())
    if not count_match:
        return None
    return min(max(int(count_match.group(1)), 1), 10)


def extract_budget(text: str) -> tuple[float | None, str | None, bool]:
    currency_pattern = (
        r"美元|美金|usd|日元|日币|jpy|円|欧元|eur|英镑|gbp|港币|hkd|"
        r"台币|新台币|twd|韩元|krw|人民币|rmb|cny|元|块"
    )
    patterns = (
        rf"(?:预算|价格|价位|budget|price)\s*"
        rf"(?:(?:改为|改成|调整到|设为|到|为|是|约|最多|不超过|不得超过|"
        rf"不能超过|低于|小于|必须|一定要|只接受|务必|must|be|required|to|"
        rf"under|below|at\s+most|less\s+than|only|accept)\s*)*"
        rf"(?P<symbol>[$€£¥￥]?)\s*(?P<amount>\d+(?:\.\d+)?)\s*"
        rf"(?P<currency>{currency_pattern})?",
        r"(?P<symbol>[$€£])\s*(?P<amount>\d+(?:\.\d+)?)",
        rf"(?P<amount>\d+(?:\.\d+)?)\s*(?P<currency>{currency_pattern})"
        rf"\s*(?:以内|以下|左右|上下)?",
    )
    match = next((result for pattern in patterns if (result := re.search(pattern, text))), None)
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
