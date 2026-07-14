from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import Any, Iterable

from ..storage.models import COMPANY_ALIAS_LIMIT, CompanyPreference, GameCandidate

LEGAL_SUFFIXES = {
    "ab",
    "ag",
    "bv",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "kg",
    "limited",
    "llc",
    "llp",
    "lp",
    "ltd",
    "nv",
    "oy",
    "plc",
    "pte",
    "sa",
    "sarl",
}
CJK_LEGAL_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "株式会社",
)


class CompanyMatchStatus(str, Enum):
    MATCH = "match"
    UNKNOWN = "unknown"
    MISMATCH = "mismatch"


def normalize_company_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = strip_terminal_punctuation(text)
    text = collapse_terminal_dotted_initialism(text)
    text = strip_terminal_punctuation(text)
    for suffix in CJK_LEGAL_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = strip_terminal_punctuation(text[: -len(suffix)])
            break
    text = text.replace("&", " and ")
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def strip_terminal_punctuation(value: str) -> str:
    return re.sub(r"[\W_]+$", "", str(value or ""), flags=re.UNICODE).rstrip()


def collapse_terminal_dotted_initialism(value: str) -> str:
    pattern = re.compile(r"(?:\b[a-z]\s*\.\s*){1,3}[a-z]\.?(?:\s*)$", flags=re.I)
    return pattern.sub(
        lambda match: "".join(
            character for character in match.group(0) if character.isalnum()
        ),
        str(value or ""),
    )


def company_alias_keys(preference: CompanyPreference) -> set[str]:
    return {
        normalized
        for value in [preference.display_name, *preference.aliases]
        if (normalized := normalize_company_name(value))
    }


def merge_company_preferences(
    values: Iterable[Any],
    *,
    limit: int = 3,
) -> list[CompanyPreference]:
    resolved_limit = max(int(limit), 0)
    groups: list[tuple[CompanyPreference, set[str]]] = []
    for value in values:
        item = validated_company_preference(value)
        if item is None:
            continue
        keys = company_alias_keys(item)
        matches = [
            index
            for index, (existing, existing_keys) in enumerate(groups)
            if existing.role == item.role
            and company_source_supports_display(existing)
            and company_source_supports_display(item)
            and existing_keys & keys
        ]
        if not matches:
            if len(groups) < resolved_limit:
                groups.append((item, keys))
            continue
        first = matches[0]
        additions = [groups[index][0] for index in matches[1:]]
        additions.append(item)
        merged = merge_company_preference(groups[first][0], additions)
        merged_keys = keys | set().union(*(groups[index][1] for index in matches))
        groups[first] = (merged, merged_keys)
        for index in reversed(matches[1:]):
            groups.pop(index)
    return [preference for preference, _keys in groups]


def merge_retry_company_preferences(
    left: Iterable[Any],
    right: Iterable[Any],
    *,
    limit: int = 3,
) -> list[CompanyPreference]:
    resolved_limit = max(int(limit), 0)
    entries = [
        [preference, False]
        for preference in merge_company_preferences(left, limit=resolved_limit)
    ]
    for item in merge_company_preferences(right, limit=resolved_limit):
        keys = company_alias_keys(item)
        matches = [
            index
            for index, (existing, _protected) in enumerate(entries)
            if existing.role == item.role and company_alias_keys(existing) & keys
        ]
        if matches:
            first = matches[0]
            additions = [entries[index][0] for index in matches]
            entries[first] = [merge_company_preference(item, additions), True]
            for index in reversed(matches[1:]):
                entries.pop(index)
        else:
            entries.append([item, True])

    while len(entries) > resolved_limit:
        removable = next(
            (
                index
                for index in range(len(entries) - 1, -1, -1)
                if not entries[index][1]
            ),
            None,
        )
        if removable is None:
            entries = entries[-resolved_limit:] if resolved_limit else []
            break
        entries.pop(removable)
    return [preference for preference, _protected in entries]


def merge_company_preference(
    primary: CompanyPreference,
    additions: Iterable[CompanyPreference],
) -> CompanyPreference:
    merged_items = [primary, *additions]
    aliases: list[str] = []
    seen = {normalize_company_name(primary.display_name)}
    for item in merged_items:
        for value in [item.display_name, *item.aliases]:
            key = normalize_company_name(value)
            if key and key not in seen:
                aliases.append(value)
                seen.add(key)
            if len(aliases) >= COMPANY_ALIAS_LIMIT:
                break
        if len(aliases) >= COMPANY_ALIAS_LIMIT:
            break
    return CompanyPreference(
        display_name=primary.display_name,
        aliases=aliases,
        role=primary.role,
        strength=(
            "strong"
            if any(item.strength == "strong" for item in merged_items)
            else "preferred"
        ),
        source_span=primary.source_span,
    )


def validated_company_preference(value: Any) -> CompanyPreference | None:
    if isinstance(value, CompanyPreference):
        return value
    try:
        validator = getattr(CompanyPreference, "model_validate", None)
        return validator(value) if validator else CompanyPreference.parse_obj(value)
    except (TypeError, ValueError):
        return None


def company_source_supports_display(preference: CompanyPreference) -> bool:
    display = normalize_company_name(preference.display_name)
    source = normalize_company_name(preference.source_span)
    return bool(display and source and display in source)


def matches_company_preference(
    candidate: GameCandidate,
    preference: CompanyPreference,
) -> bool:
    expected = company_alias_keys(preference)
    if not expected:
        return False
    actual = {
        normalize_company_name(value)
        for value in candidate_companies(candidate, preference.role)
    }
    actual.discard("")
    return bool(expected & actual)


def company_match_status(
    candidate: GameCandidate,
    preference: CompanyPreference,
) -> CompanyMatchStatus:
    if matches_company_preference(candidate, preference):
        return CompanyMatchStatus.MATCH
    if preference.role == "developer":
        available = candidate.developer_data_available
    elif preference.role == "publisher":
        available = candidate.publisher_data_available
    else:
        available = (
            candidate.developer_data_available
            and candidate.publisher_data_available
        )
    return CompanyMatchStatus.MISMATCH if available else CompanyMatchStatus.UNKNOWN


def company_preference_adjustment(
    candidate: GameCandidate,
    preferences: Iterable[CompanyPreference],
) -> float:
    adjustments = [0.0]
    for preference in preferences:
        status = company_match_status(candidate, preference)
        if status is CompanyMatchStatus.UNKNOWN:
            adjustments.append(-2.0)
        elif status is CompanyMatchStatus.MISMATCH:
            adjustments.append(-10.0 if preference.strength == "strong" else -5.0)
    return min(adjustments)


def candidate_companies(candidate: GameCandidate, role: str) -> list[str]:
    if role == "developer":
        return list(candidate.developers)
    if role == "publisher":
        return list(candidate.publishers)
    return [*candidate.developers, *candidate.publishers]
