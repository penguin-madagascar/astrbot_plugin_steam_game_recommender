from __future__ import annotations

import json
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from ..storage.models import GamePreference

SYSTEM_PROMPT = """你是游戏推荐插件的偏好解析器。
只把用户自然语言解析成 JSON，不要推荐游戏，不要补充解释，不要使用 Markdown。
未知字段使用空数组或 null，不要编造价格、平台、语言支持等事实。"""

PREFERENCE_SCHEMA_HINT = """
返回 JSON 字段必须包括：
{
  "platforms": ["steam", "playstation", "nintendo switch"],
  "genres_like": [],
  "genres_dislike": [],
  "reference_games_like": [],
  "reference_games_dislike": [],
  "players": null,
  "budget": null,
  "language": null,
  "difficulty": null,
  "mood": null,
  "result_count": 5
}
"""


class PreferenceParser:
    def __init__(self, context: Context, provider_id: str = "") -> None:
        self.context = context
        self.provider_id = provider_id.strip()

    async def parse_preference(self, event: AstrMessageEvent, text: str) -> GamePreference:
        text = text.strip()
        if not text:
            return GamePreference(parse_warnings=["需求为空，已使用默认偏好。"])

        try:
            raw = await self._llm_parse(event, text)
            return parse_preference_json(raw)
        except Exception as exc:
            logger.warning(f"游戏推荐偏好解析失败，尝试修复 JSON：{exc}")

        try:
            fixed = await self._llm_repair(event, text)
            return parse_preference_json(fixed)
        except Exception as exc:
            logger.warning(f"游戏推荐偏好 JSON 修复失败，使用关键词 fallback：{exc}")

        preference = keyword_fallback(text)
        preference.parse_warnings.append("LLM 偏好解析失败，已使用关键词 fallback，结果可能不完整。")
        return preference

    async def _llm_parse(self, event: AstrMessageEvent, text: str) -> str:
        prompt = f"{PREFERENCE_SCHEMA_HINT}\n用户需求：{text}\n只返回 JSON："
        return await self._llm_generate_text(event, prompt)

    async def _llm_repair(self, event: AstrMessageEvent, text: str) -> str:
        prompt = f"{PREFERENCE_SCHEMA_HINT}\n请重新解析这段用户需求，只返回合法 JSON：{text}"
        return await self._llm_generate_text(event, prompt)

    async def _llm_generate_text(self, event: AstrMessageEvent, prompt: str) -> str:
        kwargs: dict[str, Any] = {"prompt": prompt, "system_prompt": SYSTEM_PROMPT}
        provider_id = await self._resolve_provider_id(event)
        if provider_id:
            kwargs["chat_provider_id"] = provider_id
        response = await self.context.llm_generate(**kwargs)
        return str(getattr(response, "completion_text", "") or "").strip()

    async def _resolve_provider_id(self, event: AstrMessageEvent) -> str:
        if self.provider_id:
            return self.provider_id
        getter = getattr(self.context, "get_current_chat_provider_id", None)
        if not getter:
            return ""
        try:
            return str(await getter(umo=event.unified_msg_origin) or "")
        except Exception as exc:
            logger.debug(f"获取当前 LLM provider 失败：{exc}")
            return ""


def parse_preference_json(text: str) -> GamePreference:
    payload = extract_json_object(text)
    data = json.loads(payload)
    return GamePreference.parse_obj(data)


def extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM did not return a JSON object")
    return cleaned[start : end + 1]


def keyword_fallback(text: str) -> GamePreference:
    lower = text.lower()
    platforms = []
    if any(word in lower for word in ("steam", "pc", "电脑")):
        platforms.append("steam")
    if any(word in lower for word in ("switch", "任天堂", "ns")):
        platforms.append("nintendo switch")
    if any(word in lower for word in ("playstation", "ps5", "ps4", "psn")):
        platforms.append("playstation")
    if "xbox" in lower:
        platforms.append("xbox")

    genres_like = keyword_hits(
        lower,
        {
            "co-op": ("双人", "合作", "coop", "co-op"),
            "multiplayer": ("多人", "联机"),
            "puzzle": ("解谜", "谜题", "puzzle"),
            "adventure": ("冒险", "剧情", "adventure"),
            "casual": ("休闲", "轻松", "casual"),
            "action": ("动作", "action"),
            "rpg": ("rpg", "角色扮演"),
            "party": ("聚会", "派对", "party"),
            "simulation": ("模拟", "simulation"),
            "racing": ("赛车", "竞速", "racing"),
        },
    )
    genres_dislike = keyword_hits(
        lower,
        {
            "horror": ("不要恐怖", "不恐怖", "恐怖", "horror"),
            "soulslike": ("魂like", "魂系", "soulslike", "souls-like"),
            "roguelike": ("肉鸽", "roguelike", "rogue-like"),
            "violent": ("血腥", "violent", "gore"),
        },
    )

    players = 2 if any(word in lower for word in ("双人", "两人", "合作", "co-op")) else None
    if players is None and "多人" in lower:
        players = 2

    budget = None
    budget_match = re.search(r"(?:预算|价格|价位)?\s*(\d+(?:\.\d+)?)\s*(?:以内|以下|元|块|rmb)", lower)
    if budget_match:
        budget = float(budget_match.group(1))

    result_count = 5
    count_match = re.search(r"(\d+)\s*(?:个|款|部)", lower)
    if count_match:
        result_count = int(count_match.group(1))

    difficulty = None
    if any(word in lower for word in ("别太难", "不要太难", "简单", "轻松", "休闲")):
        difficulty = "easy"
    elif any(word in lower for word in ("高难", "困难", "挑战")):
        difficulty = "hard"

    reference_like = []
    like_match = re.search(r"类似([^，。,.；;]+)", text)
    if like_match:
        reference = re.split(r"但|不过|别|不要|且|并且", like_match.group(1).strip(), maxsplit=1)[
            0
        ].strip()
        if reference:
            reference_like.append(reference)

    return GamePreference(
        platforms=platforms,
        genres_like=genres_like,
        genres_dislike=genres_dislike,
        reference_games_like=reference_like,
        players=players,
        budget=budget,
        language="中文" if "中文" in text or "汉化" in text else None,
        difficulty=difficulty,
        mood="轻松" if any(word in lower for word in ("轻松", "休闲", "治愈")) else None,
        result_count=result_count,
    )


def keyword_hits(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, words in mapping.items() if any(word in text for word in words)]
