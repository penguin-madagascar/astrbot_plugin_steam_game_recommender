from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .tag_normalizer import canonical_steam_tag_name, normalize_tag

STATIC_TAG_TRANSLATIONS: dict[str, str] = {
    "action": "动作",
    "adventure": "冒险",
    "automation": "自动化",
    "building": "建造",
    "card_battler": "卡牌对战",
    "casual": "休闲",
    "chinese": "中文",
    "choices_matter": "选择影响剧情",
    "co_op": "合作",
    "crafting": "制作",
    "deckbuilding": "牌组构筑",
    "family": "家庭友好",
    "farming": "农场经营",
    "farming_sim": "农场模拟",
    "horror": "恐怖",
    "life_sim": "生活模拟",
    "local_coop": "本地合作",
    "management": "经营",
    "metroidvania": "银河城",
    "multiplayer": "多人",
    "online_coop": "在线合作",
    "open_world": "开放世界",
    "open_world_survival_craft": "开放世界生存制作",
    "party": "聚会",
    "pixel_graphics": "像素画面",
    "platformer": "平台跳跃",
    "puzzle": "解谜",
    "pve": "玩家对抗环境",
    "pvp": "玩家对战",
    "racing": "竞速",
    "relaxing": "轻松",
    "roguelike": "类 Rogue",
    "rpg": "角色扮演",
    "sandbox": "沙盒",
    "shooter": "射击",
    "simulation": "模拟",
    "singleplayer": "单人",
    "soulslike": "类魂",
    "story_rich": "剧情丰富",
    "strategy": "策略",
    "survival": "生存",
    "turn_based": "回合制",
    "violent": "暴力",
}
INTERNAL_TAG_PATTERN = re.compile(r"(?<![0-9A-Za-z_])[a-z0-9]+(?:_[a-z0-9]+)+(?![0-9A-Za-z_])")
STRUCTURED_TAG_LIST_PATTERN = re.compile(
    r"(?P<prefix>(?:(?:核心|辅助)?玩法特征|(?:核心|辅助|偏好)?标签|类型)\s*[：:])"
    r"(?P<values>[^。；\n]+)",
    flags=re.I,
)
STRUCTURED_CORE_TAG_PATTERN = re.compile(
    r"(?P<prefix>核心(?:玩法)?特征(?:为|是)?\s*)"
    r"(?P<values>(?:[A-Za-z0-9_-]+\s*(?:、|,|/)?\s*)+?)"
    r"(?P<suffix>(?=缺失|证据|未命中|不足|未知|[。；，,\n]|$))",
    flags=re.I,
)


def build_tag_presentations(
    english_tags: Iterable[Mapping[str, Any]],
    schinese_tags: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    english_by_id = vocabulary_by_id(english_tags)
    chinese_by_id = vocabulary_by_id(schinese_tags)
    result: dict[str, str] = {}
    for tag_id, english_name in english_by_id.items():
        chinese_name = chinese_by_id.get(tag_id, "").strip()
        if not chinese_name or not contains_cjk(chinese_name):
            continue
        canonical = canonical_steam_tag_name(english_name)
        if canonical:
            result[canonical] = chinese_name
    return result


def presentation_tag(
    value: str,
    localized: Mapping[str, str] | None = None,
) -> str | None:
    canonical = normalize_tag(value) or canonical_steam_tag_name(value)
    translated = str((localized or {}).get(canonical) or "").strip()
    if translated and contains_cjk(translated):
        return translated
    return STATIC_TAG_TRANSLATIONS.get(canonical)


def presentation_tags(
    values: Iterable[str],
    localized: Mapping[str, str] | None = None,
    *,
    limit: int = 5,
) -> list[str]:
    result: list[str] = []
    for value in values:
        label = presentation_tag(value, localized)
        if label and label not in result:
            result.append(label)
        if len(result) >= max(int(limit), 1):
            break
    return result


def sanitize_user_facing_tag_text(value: str) -> str:
    text = str(value or "")

    def translate_tag_tokens(values: str) -> str:
        for canonical in sorted(STATIC_TAG_TRANSLATIONS, key=len, reverse=True):
            label = STATIC_TAG_TRANSLATIONS[canonical]
            values = re.sub(
                rf"(?<![0-9A-Za-z_]){re.escape(canonical)}(?![0-9A-Za-z_])",
                label,
                values,
                flags=re.I,
            )
        return values

    def translate_structured_list(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{translate_tag_tokens(match.group('values'))}"

    def translate_structured_core(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{translate_tag_tokens(match.group('values'))}"

    text = STRUCTURED_TAG_LIST_PATTERN.sub(translate_structured_list, text)
    text = STRUCTURED_CORE_TAG_PATTERN.sub(translate_structured_core, text)
    text = INTERNAL_TAG_PATTERN.sub("相关玩法特征", text)
    text = re.sub(r"(?:、\s*相关玩法特征){2,}", "、相关玩法特征", text)
    return text


def vocabulary_by_id(values: Iterable[Mapping[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in values:
        tag_id = item.get("tagid", item.get("id"))
        name = str(item.get("name") or "").strip()
        try:
            resolved_id = int(tag_id)
        except (TypeError, ValueError):
            continue
        if resolved_id > 0 and name:
            result[resolved_id] = name
    return result


def contains_cjk(value: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in value)
