from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

try:
    from astrbot.api import logger
except ModuleNotFoundError:  # Allows formatter-only unit tests outside AstrBot.
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.star import Context

from ..storage.models import GameCandidate, GamePreference, GamePriceSummary, RankedGame

TIER_LABELS = {
    "strong": "强烈推荐",
    "recommended": "推荐",
    "backup": "备选",
}

DISCLAIMER = "以下推荐基于当前可查询到的 Steam 公开数据，价格和商店信息可能因地区变化。"
EMPTY_LLM_FALLBACK_TITLE = "LLM 兜底建议（未经过 Steam 索引验证）"


def format_recommendations(
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
) -> str:
    return "\n".join(format_recommendation_messages(preference, ranked_games, limit=limit))


def format_recommendation_messages(
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
) -> list[str]:
    count = min(limit or preference.result_count or 5, len(ranked_games))
    if not ranked_games:
        return [
            (
                "暂时没有找到满足当前条件的游戏。\n"
                f"{DISCLAIMER}\n"
                "可以尝试改用 Steam/PC 请求，或放宽排除标签、人数和类型条件后再查一次。"
            )
        ]

    lines = [
        (f"优先看前 {count} 款，它们和你的 Steam 标签、游玩人数与参考游戏偏好最接近。"),
        tier_summary(ranked_games[:count]),
        DISCLAIMER,
    ]
    if preference.parse_warnings:
        lines.append("偏好解析提示：" + "；".join(preference.parse_warnings))

    lines.append("推荐列表将分条发送。")
    messages = ["\n".join(lines)]
    for index, game in enumerate(ranked_games[:count], start=1):
        messages.append("\n".join(format_game_block(index, game)))
    return messages


async def format_recommendations_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
    enable_empty_fallback: bool = False,
    raw_query: str = "",
) -> str:
    return "\n".join(
        await format_recommendation_messages_with_llm(
            context,
            event,
            provider_id,
            preference,
            ranked_games,
            limit=limit,
            enable_empty_fallback=enable_empty_fallback,
            raw_query=raw_query,
        )
    )


async def format_recommendation_messages_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
    enable_empty_fallback: bool = False,
    raw_query: str = "",
) -> list[str]:
    fallback = format_recommendation_messages(preference, ranked_games, limit=limit)
    if not ranked_games:
        if enable_empty_fallback:
            empty_fallback = await format_empty_recommendations_with_llm(
                context,
                event,
                provider_id,
                preference,
                limit=limit,
                raw_query=raw_query,
            )
            if empty_fallback:
                return [empty_fallback]
        return fallback

    return fallback


async def format_empty_recommendations_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
    preference: GamePreference,
    limit: int | None = None,
    raw_query: str = "",
) -> str:
    resolved_provider = await resolve_provider_id(context, event, provider_id)
    if not resolved_provider:
        return ""

    count = min(limit or preference.result_count or 5, 10)
    payload = {
        "raw_query": raw_query,
        "preference": dump_model(preference),
        "result_limit": count,
        "rules": [
            f"回复必须以“{EMPTY_LLM_FALLBACK_TITLE}”开头。",
            "只给游戏名和简短理由，不要输出价格、评测数、中文支持、商店链接或数据来源。",
            "必须明确这些建议未经过 Steam 索引验证，需要用户自行确认平台和商店信息。",
            "不要使用 Markdown 表格。",
        ],
    }
    prompt = (
        "Steam 索引没有找到可验证的游戏结果。"
        "请基于用户需求生成已标注的 LLM 兜底建议。\n"
        f"数据 JSON：{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        response = await context.llm_generate(
            chat_provider_id=resolved_provider,
            prompt=prompt,
            system_prompt=(
                "你是游戏推荐兜底助手。你不能声称建议经过数据库、Steam 索引、价格、"
                "评测或商店信息验证，只能给未验证候选和简短匹配理由。"
            ),
        )
    except Exception as exc:
        logger.warning(f"游戏推荐空结果 LLM 兜底失败，使用规则 formatter：{exc}")
        return ""

    text = str(getattr(response, "completion_text", "") or "").strip()
    if not text:
        return ""
    if EMPTY_LLM_FALLBACK_TITLE not in text:
        text = f"{EMPTY_LLM_FALLBACK_TITLE}\n{text}"
    return text


def format_game_block(index: int, game: RankedGame) -> list[str]:
    platforms = "、".join(game.platforms) if game.platforms else "不确定"
    tier = TIER_LABELS.get(getattr(game, "tier", ""), "")
    reasons = "；".join(display_points(game.fit_points, game.reasons)[:5]) or (
        "当前数据与偏好有一定匹配，但具体玩法仍需以商店页面确认"
    )
    warnings = "；".join(display_points(game.risk_points, game.warnings)[:5]) or (
        "仍需以商店页面确认平台版本、中文支持和实时价格"
    )
    stores = "、".join(game.stores[:4]) if game.stores else "不确定"
    uncertain = uncertain_fields(game)
    lines = [
        f"{index}. 《{game.title}》",
        *([f"   层级：{tier}"] if tier else []),
        f"   平台：{platforms}",
        f"   推荐理由：{reasons}",
        f"   可能不适合的点：{warnings}",
    ]
    price_summary = getattr(game, "price_summary", None)
    if price_summary:
        lines.append(f"   价格：{format_price_summary(price_summary)}")
        links = format_price_links(price_summary)
        if links:
            lines.append(f"   购买链接：{links}")
    else:
        lines.append(f"   购买 / 平台建议：Steam 商店记录为 {stores}；实时价格请以商店页面为准。")
    if game.raw_url:
        lines.append(f"   数据来源：{game.raw_url}")
    if uncertain:
        lines.append(f"   数据不确定：{uncertain}")
    return lines


def valid_game_message(text: str, index: int, title: str) -> bool:
    if not text:
        return False
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return first_line.startswith(f"{index}.") and title.lower() in text.lower()


def format_game_detail(game: GameCandidate, price_summary: GamePriceSummary | None = None) -> str:
    lines = [
        f"《{game.title}》",
        f"平台：{'、'.join(game.platforms) if game.platforms else '不确定'}",
        f"类型：{'、'.join(game.genres) if game.genres else '不确定'}",
        f"标签：{'、'.join(game.tags[:10]) if game.tags else '不确定'}",
        f"Steam 好评率：{format_review_ratio(game.review_positive_ratio)}",
        f"Steam 评测数：{game.review_total if game.review_total is not None else '不确定'}",
        f"Metacritic：{game.metacritic if game.metacritic is not None else '不确定'}",
        f"发售日：{game.released or '不确定'}",
        (
            "平均游玩时长："
            f"{str(game.playtime) + ' 小时' if game.playtime is not None else '不确定'}"
        ),
        f"商店：{'、'.join(game.stores) if game.stores else '不确定'}",
    ]
    if price_summary:
        lines.append(f"Steam 价格：{format_price_summary(price_summary)}")
        links = format_price_links(price_summary)
        if links:
            lines.append(f"购买链接：{links}")
        lines.append("中文支持：Steam 数据可能缺失，请以商店页面为准。")
    else:
        lines.append("价格 / 中文支持：实时地区价格和语言信息请以 Steam 商店页面为准。")
    if game.raw_url:
        lines.append(f"数据来源：{game.raw_url}")
    return "\n".join(lines)


def uncertain_fields(game: RankedGame | GameCandidate) -> str:
    fields = []
    if not game.stores:
        fields.append("购买渠道")
    if not getattr(game, "price_summary", None):
        fields.append("实时价格")
    points = display_points(getattr(game, "fit_points", []), game.reasons)
    if not any("中文" in reason or "chinese" in reason.lower() for reason in points):
        fields.append("中文支持")
    return "、".join(fields)


def format_price_summary(summary: GamePriceSummary) -> str:
    parts: list[str] = []
    if summary.current_price:
        parts.append(f"Steam 当前价 {summary.current_price}")
    if summary.lowest_price:
        lowest = f"史低 {summary.lowest_price}"
        annotations = []
        if summary.lowest_date:
            annotations.append(summary.lowest_date)
        if summary.lowest_discount:
            annotations.append(f"-{summary.lowest_discount}%")
        if annotations:
            lowest += f"（{'，'.join(annotations)}）"
        parts.append(lowest)
    if summary.sale_status:
        parts.append(summary.sale_status)
    if summary.region_summary:
        parts.append(summary.region_summary)
    return "；".join(parts) if parts else "暂时不可用"


def format_price_links(summary: GamePriceSummary) -> str:
    links = []
    if summary.store_url:
        links.append(f"Steam：{summary.store_url}")
    if summary.heybox_url:
        links.append(f"小黑盒：{summary.heybox_url}")
    return "；".join(links)


def format_review_ratio(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "不确定"


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def tier_summary(games: list[RankedGame]) -> str:
    counts = {key: 0 for key in TIER_LABELS}
    for game in games:
        if game.tier in counts:
            counts[game.tier] += 1
    parts = [f"{label} {counts[key]} 款" for key, label in TIER_LABELS.items() if counts[key]]
    return "分层统计：" + ("；".join(parts) if parts else "未分层")


def display_points(primary: list[str], secondary: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in [*primary, *secondary]:
        key = value.lower()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


async def resolve_provider_id(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
) -> str:
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
