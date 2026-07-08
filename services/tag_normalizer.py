from __future__ import annotations

import re
import unicodedata

from ..storage.models import GameCandidate

TAG_ALIASES = {
    "co op": "co_op",
    "coop": "co_op",
    "cooperative": "co_op",
    "双人": "co_op",
    "两人": "co_op",
    "合作": "co_op",
    "local co op": "local_coop",
    "local coop": "local_coop",
    "shared split screen co op": "local_coop",
    "split screen co op": "local_coop",
    "split screen": "local_coop",
    "remote play together": "local_coop",
    "本地合作": "local_coop",
    "同屏": "local_coop",
    "分屏": "local_coop",
    "online co op": "online_coop",
    "online coop": "online_coop",
    "在线合作": "online_coop",
    "multiplayer": "multiplayer",
    "multi player": "multiplayer",
    "多人": "multiplayer",
    "singleplayer": "singleplayer",
    "single player": "singleplayer",
    "单人": "singleplayer",
    "puzzle": "puzzle",
    "解谜": "puzzle",
    "casual": "casual",
    "休闲": "casual",
    "relaxing": "relaxing",
    "cozy": "relaxing",
    "轻松": "relaxing",
    "治愈": "relaxing",
    "adventure": "adventure",
    "冒险": "adventure",
    "platformer": "platformer",
    "platform": "platformer",
    "平台跳跃": "platformer",
    "simulation": "simulation",
    "模拟": "simulation",
    "rpg": "rpg",
    "role playing": "rpg",
    "角色扮演": "rpg",
    "strategy": "strategy",
    "策略": "strategy",
    "action": "action",
    "动作": "action",
    "racing": "racing",
    "竞速": "racing",
    "赛车": "racing",
    "party": "party",
    "派对": "party",
    "聚会": "party",
    "family friendly": "family",
    "family": "family",
    "家庭": "family",
    "farming": "farming",
    "farm": "farming",
    "种田": "farming",
    "农场": "farming",
    "crafting": "crafting",
    "制作": "crafting",
    "building": "building",
    "建造": "building",
    "management": "management",
    "经营": "management",
    "survival": "survival",
    "生存": "survival",
    "automation": "automation",
    "factory automation": "automation",
    "自动化": "automation",
    "deckbuilding": "deckbuilding",
    "deck builder": "deckbuilding",
    "deck building": "deckbuilding",
    "牌组构建": "deckbuilding",
    "card battler": "card_battler",
    "card game": "card_battler",
    "卡牌": "card_battler",
    "open world survival craft": "open_world_survival_craft",
    "open world survival crafting": "open_world_survival_craft",
    "开放世界生存制作": "open_world_survival_craft",
    "choices matter": "choices_matter",
    "choice matters": "choices_matter",
    "选择影响剧情": "choices_matter",
    "open world": "open_world",
    "开放世界": "open_world",
    "story rich": "story_rich",
    "剧情丰富": "story_rich",
    "剧情向": "story_rich",
    "sandbox": "sandbox",
    "沙盒": "sandbox",
    "turn based": "turn_based",
    "turn based combat": "turn_based",
    "回合制": "turn_based",
    "metroidvania": "metroidvania",
    "类银河战士恶魔城": "metroidvania",
    "shooter": "shooter",
    "射击": "shooter",
    "pvp": "pvp",
    "pve": "pve",
    "horror": "horror",
    "psychological horror": "horror",
    "恐怖": "horror",
    "soulslike": "soulslike",
    "souls like": "soulslike",
    "魂like": "soulslike",
    "魂系": "soulslike",
    "roguelike": "roguelike",
    "rogue like": "roguelike",
    "roguelite": "roguelike",
    "肉鸽": "roguelike",
    "violent": "violent",
    "violence": "violent",
    "gore": "violent",
    "blood": "violent",
    "血腥": "violent",
    "simplified chinese": "chinese",
    "traditional chinese": "chinese",
    "chinese": "chinese",
    "中文": "chinese",
    "简体中文": "chinese",
    "繁体中文": "chinese",
}

CANONICAL_TAGS = set(TAG_ALIASES.values())


def normalize_tag(value: str) -> str | None:
    key = normalize_key(value)
    if not key:
        return None
    if key in TAG_ALIASES:
        return TAG_ALIASES[key]
    compact = key.replace(" ", "_")
    return compact if compact in CANONICAL_TAGS else None


def canonical_tags_from_terms(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_tag(value)
        if tag and tag not in seen:
            result.append(tag)
            seen.add(tag)
    return result


def candidate_canonical_tags(candidate: GameCandidate) -> list[str]:
    values = [*candidate.genres, *candidate.tags]
    if candidate.description:
        values.extend(extract_description_terms(candidate.description))
    return canonical_tags_from_terms(values)


def extract_description_terms(text: str) -> list[str]:
    normalized = normalize_key(text)
    return [
        raw
        for raw, canonical in TAG_ALIASES.items()
        if raw in normalized and canonical not in {"violent"}
    ]


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/_\\-]+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
