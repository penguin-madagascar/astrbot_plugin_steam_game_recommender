from __future__ import annotations

import json
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from ..storage.models import GamePreference
from .preference_rules import infer_preference_from_text, merge_text_preference

SYSTEM_PROMPT = """你是游戏推荐插件的偏好解析器。
只把用户自然语言解析成 JSON，不要推荐游戏，不要补充解释，不要使用 Markdown。
插件只覆盖 Steam/PC；你只负责抽取用户明确或隐含的标签、排除项、相似游戏名和多样性要求。
未知字段使用空数组或 null，不要编造价格、平台、语言支持、评测或商店事实。"""

PREFERENCE_SCHEMA_HINT = """
返回 JSON 字段必须包括：
{
  "platforms": ["steam", "pc"],
  "genres_like": [],
  "extra_tags": [],
  "genres_dislike": [],
  "reference_games_like": [],
  "reference_search_terms": [],
  "reference_games_dislike": [],
  "library_filter_mode": null,
  "players": null,
  "budget": null,
  "language": null,
  "difficulty": null,
  "mood": null,
  "diversity_mode": "strict",
  "result_count": 5
}
说明：
- genres_like 放用户明确说出的类型/玩法标签。
- extra_tags 放你从自然语言总结出的补充标签，例如“轻松”“本地合作”“剧情合作”“短流程”。
- reference_games_like 只放用户提到的相似游戏名，不要把相似游戏扩写成推荐结果。
- reference_search_terms 放参考游戏的 Steam 搜索友好标题候选，例如“黑暗之魂”对应 “Dark Souls”。
- genres_dislike 放排除标签，例如恐怖、魂类、肉鸽、纯单人、pvp。
- library_filter_mode 只在用户明确要求时填写：排除已有/exclude-owned 为 "exclude_owned"；仅查看已有/only-owned 为 "only_owned"；否则为 null。
- diversity_mode 只允许 "strict"、"balanced"、"high"：
  默认使用 "strict"；用户要求更像、同类优先、严格匹配时也使用 "strict"。
  用户要求适度变化但仍以相似为主时使用 "balanced"。
  只有用户明确希望更多样、不同题材/玩法、避免同质化时使用 "high"。
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
            return merge_text_preference(parse_preference_json(raw), text)
        except Exception as exc:
            logger.warning(f"游戏推荐偏好解析失败，尝试修复 JSON：{exc}")

        try:
            fixed = await self._llm_repair(event, text)
            return merge_text_preference(parse_preference_json(fixed), text)
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
    validator = getattr(GamePreference, "model_validate", None)
    return validator(data) if validator else GamePreference.parse_obj(data)


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
    return infer_preference_from_text(text)
