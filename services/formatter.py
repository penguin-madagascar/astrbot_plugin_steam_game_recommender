from __future__ import annotations

import json
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from ..storage.models import GameCandidate, GamePreference, RankedGame

DISCLAIMER = "以下推荐基于当前可查询到的数据，价格和平台信息可能因地区变化。"


def format_recommendations(
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
) -> str:
    count = min(limit or preference.result_count or 5, len(ranked_games))
    if not ranked_games:
        return (
            "一句话结论：暂时没有找到满足这些硬条件的游戏。\n"
            f"{DISCLAIMER}\n"
            "可以尝试放宽平台、排除标签或多人条件后再查一次。"
        )

    lines = [
        f"一句话结论：优先看前 {count} 款，它们和你的平台、类型与游玩人数偏好最接近。",
        DISCLAIMER,
    ]
    if preference.parse_warnings:
        lines.append("偏好解析提示：" + "；".join(preference.parse_warnings))

    lines.append("推荐列表：")
    for index, game in enumerate(ranked_games[:count], start=1):
        lines.extend(format_game_block(index, game))
    return "\n".join(lines)


async def format_recommendations_with_llm(
    context: Context,
    event: AstrMessageEvent,
    provider_id: str,
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
) -> str:
    fallback = format_recommendations(preference, ranked_games, limit=limit)
    if not ranked_games:
        return fallback

    resolved_provider = await resolve_provider_id(context, event, provider_id)
    if not resolved_provider:
        return fallback

    payload = {
        "preference": preference.dict(),
        "games": [game.dict() for game in ranked_games[: limit or preference.result_count or 5]],
        "rules": [
            "只能基于 games 中已有字段写推荐说明。",
            "不要编造当前价格、史低、平台支持、中文支持。",
            "字段为空时写不确定或省略。",
            "必须包含免责声明。",
        ],
    }
    prompt = (
        "请用中文生成固定格式游戏推荐结果：一句话结论、免责声明、推荐列表；"
        "每款游戏包含名称、平台、推荐理由、可能不适合的点、购买/平台建议、数据不确定说明。\n"
        f"数据 JSON：{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        response = await context.llm_generate(
            chat_provider_id=resolved_provider,
            prompt=prompt,
            system_prompt="你只能改写给定 JSON 中的事实，不得补充外部知识或猜测。",
        )
        text = str(getattr(response, "completion_text", "") or "").strip()
    except Exception as exc:
        logger.warning(f"游戏推荐结果 LLM 格式化失败，使用规则 formatter：{exc}")
        return fallback

    if not text or "价格和平台信息可能因地区变化" not in text:
        return fallback
    return text


def format_game_block(index: int, game: RankedGame) -> list[str]:
    platforms = "、".join(game.platforms) if game.platforms else "不确定"
    reasons = "；".join(game.reasons[:4]) if game.reasons else "RAWG 数据与偏好有一定匹配"
    warnings = "；".join(game.warnings[:4]) if game.warnings else "暂未发现明显不适合点"
    stores = "、".join(game.stores[:4]) if game.stores else "不确定"
    uncertain = uncertain_fields(game)
    lines = [
        f"{index}. 《{game.title}》",
        f"   平台：{platforms}",
        f"   推荐理由：{reasons}",
        f"   可能不适合的点：{warnings}",
        f"   购买 / 平台建议：RAWG 记录的商店为 {stores}；具体价格请以对应商店页面为准。",
    ]
    if game.raw_url:
        lines.append(f"   数据来源：{game.raw_url}")
    if uncertain:
        lines.append(f"   数据不确定：{uncertain}")
    return lines


def format_game_detail(game: GameCandidate) -> str:
    lines = [
        f"《{game.title}》",
        f"平台：{'、'.join(game.platforms) if game.platforms else '不确定'}",
        f"类型：{'、'.join(game.genres) if game.genres else '不确定'}",
        f"标签：{'、'.join(game.tags[:10]) if game.tags else '不确定'}",
        f"RAWG 评分：{game.rating if game.rating is not None else '不确定'}",
        f"Metacritic：{game.metacritic if game.metacritic is not None else '不确定'}",
        f"发售日：{game.released or '不确定'}",
        f"平均游玩时长：{str(game.playtime) + ' 小时' if game.playtime is not None else '不确定'}",
        f"商店：{'、'.join(game.stores) if game.stores else '不确定'}",
        "价格 / 中文支持：RAWG 不提供可靠实时地区价格，中文支持也可能缺失，请以商店页面为准。",
    ]
    if game.raw_url:
        lines.append(f"数据来源：{game.raw_url}")
    return "\n".join(lines)


def uncertain_fields(game: RankedGame) -> str:
    fields = []
    if not game.stores:
        fields.append("购买渠道")
    fields.append("实时价格")
    if not any("中文" in reason or "chinese" in reason.lower() for reason in game.reasons):
        fields.append("中文支持")
    return "、".join(fields)


async def resolve_provider_id(context: Context, event: AstrMessageEvent, provider_id: str) -> str:
    if provider_id:
        return provider_id
    getter = getattr(context, "get_current_chat_provider_id", None)
    if not getter:
        return ""
    try:
        return str(await getter(umo=event.unified_msg_origin) or "")
    except Exception as exc:
        logger.debug(f"获取当前 LLM provider 失败：{exc}")
        return ""

