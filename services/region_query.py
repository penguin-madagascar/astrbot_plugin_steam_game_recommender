from __future__ import annotations

import re
from dataclasses import dataclass

REGION_ALIASES = {
    "中国区": "CN",
    "国区": "CN",
    "美国区": "US",
    "美区": "US",
    "日本区": "JP",
    "日区": "JP",
    "香港区": "HK",
    "港区": "HK",
    "台湾区": "TW",
    "台区": "TW",
    "韩国区": "KR",
    "韩区": "KR",
    "英国区": "GB",
    "英区": "GB",
    "德国区": "DE",
    "德区": "DE",
    "乌克兰区": "UA",
    "乌区": "UA",
    "土耳其区": "TR",
    "土区": "TR",
    "阿根廷区": "AR",
    "阿区": "AR",
    "巴西区": "BR",
    "巴区": "BR",
    "俄罗斯区": "RU",
    "俄区": "RU",
}
REGION_CURRENCIES = {
    "CN": "CNY",
    "US": "USD",
    "JP": "JPY",
    "HK": "HKD",
    "TW": "TWD",
    "KR": "KRW",
    "GB": "GBP",
    "DE": "EUR",
    "UA": "UAH",
    "TR": "TRY",
    "AR": "ARS",
    "BR": "BRL",
    "RU": "RUB",
}
REGION_ALIAS_PATTERN = re.compile(
    "|".join(re.escape(alias) for alias in sorted(REGION_ALIASES, key=len, reverse=True))
)
REGION_CODE_PATTERN = re.compile(r"(?<!\S)-([A-Za-z]{2})(?=\s|$)")


@dataclass(frozen=True)
class ParsedRegionQuery:
    query: str
    region: str
    explicit: bool = False


def parse_region_query(text: str, default_region: str = "CN") -> ParsedRegionQuery:
    source = str(text or "").strip()
    matches: list[tuple[int, int, str]] = []
    for match in REGION_CODE_PATTERN.finditer(source):
        matches.append((match.start(), match.end(), match.group(1).upper()))
    for match in REGION_ALIAS_PATTERN.finditer(source):
        matches.append((match.start(), match.end(), REGION_ALIASES[match.group(0)]))

    region = normalize_region(default_region)
    if matches:
        region = max(matches, key=lambda item: item[0])[2]
    query = source
    for start, end, _region in sorted(matches, reverse=True):
        query = f"{query[:start]} {query[end:]}"
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"\s+([，,。；;])", r"\1", query)
    query = query.strip(" \t,，。；;")
    return ParsedRegionQuery(query=query, region=region, explicit=bool(matches))


def normalize_region(value: str) -> str:
    text = str(value or "").strip()
    if len(text) == 2 and text.isalpha():
        return text.upper()
    return REGION_ALIASES.get(text, "CN")


def region_currency(region: str) -> str | None:
    return REGION_CURRENCIES.get(normalize_region(region))
