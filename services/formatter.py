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

from ..storage.models import GamePreference, GamePriceSummary, RankedGame
from .explanation_builder import fallback_reason

EMPTY_LLM_FALLBACK_TITLE = "⚠️ LLM 兜底建议（未经过 Steam 索引验证）"


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
                "可以尝试放宽排除标签、人数、语言或类型条件后再查一次。"
            )
        ]

    displayed_games = ranked_games[:count]
    has_anchor_tiers = any(
        game.score_breakdown.relevance_tier in {"A", "B", "C"}
        for game in displayed_games
    )
    order_text = (
        "按核心匹配层级及层内推荐分排列"
        if has_anchor_tiers
        else "按推荐分从高到低排列"
    )
    lines = [f"找到 {count} 款 Steam 游戏，{order_text}。"]
    if preference.parse_warnings:
        lines.append("偏好解析提示：" + "；".join(preference.parse_warnings))

    messages = ["\n".join(lines)]
    for index, game in enumerate(ranked_games[:count], start=1):
        messages.append("\n".join(format_game_block(index, game, region=preference.region or "CN")))
    return messages


async def format_recommendations_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
    fallback_provider_id: str = "",
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
            fallback_provider_id=fallback_provider_id,
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
    fallback_provider_id: str = "",
    raw_query: str = "",
) -> list[str]:
    fallback = format_recommendation_messages(preference, ranked_games, limit=limit)
    if not ranked_games and fallback_provider_id:
        empty_fallback = await format_empty_recommendations_with_llm(
            context,
            event,
            fallback_provider_id,
            preference,
            limit=limit,
            raw_query=raw_query,
        )
        if empty_fallback:
            return [empty_fallback]
    if not ranked_games:
        return fallback

    return fallback


async def format_empty_recommendations_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    fallback_provider_id: str,
    preference: GamePreference,
    limit: int | None = None,
    raw_query: str = "",
) -> str:
    del event
    selected_provider = str(fallback_provider_id or "").strip()
    if not selected_provider:
        return ""

    count = min(limit or preference.result_count or 5, 10)
    payload = {
        "raw_query": raw_query,
        "preference": dump_model(preference),
        "result_limit": count,
        "rules": [
            f"回复必须以“{EMPTY_LLM_FALLBACK_TITLE}”开头。",
            "只给游戏名和简短理由，不要输出价格、评测数、中文支持、商店链接或数据来源。",
            "必须明确这些建议未经过 Steam 索引验证，也未经过 Steam 应用类型、同作版本和套餐校验。",
            "尽量避开 DLC、原声、工具、套餐和同一游戏的不同版本，但不得声称能够硬性保证。",
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
            chat_provider_id=selected_provider,
            prompt=prompt,
            system_prompt=(
                "你是游戏推荐兜底助手。你不能声称建议经过数据库、Steam 索引、价格、"
                "评测、应用类型、同作版本或套餐验证，只能给未验证候选和简短匹配理由。"
            ),
        )
    except Exception as exc:
        logger.warning(f"游戏推荐空结果 LLM 兜底失败，使用规则 formatter：{exc}")
        return ""

    text = str(getattr(response, "completion_text", "") or "").strip()
    if not text:
        return ""
    if not text.startswith(EMPTY_LLM_FALLBACK_TITLE):
        text = f"{EMPTY_LLM_FALLBACK_TITLE}\n{text}"
    return text


def format_game_block(index: int, game: RankedGame, region: str | None = None) -> list[str]:
    reason = game.recommendation_reason or fallback_recommendation_reason(game)
    reason = disclose_relaxed_match(game, reason)
    price_summary = game.price_summary
    selected_region = price_summary.region if price_summary else (region or "CN")
    lines = [
        f"{index}. 《{game.title}》｜推荐分：{game.score}/100",
        f"推荐理由：{reason}",
        (
            f"价格（{selected_region}）："
            f"{format_price_summary(price_summary) if price_summary else unavailable_price_text()}"
        ),
    ]
    if link := steam_store_url(game):
        lines.append(f"购买链接：{link}")
    return lines


def fallback_recommendation_reason(game: RankedGame) -> str:
    return fallback_reason(game.recommendation_evidence)


def disclose_relaxed_match(game: RankedGame, reason: str) -> str:
    if game.score_breakdown.relevance_tier not in {"B", "C"}:
        return reason
    has_relaxed_label = "宽松匹配" in reason
    has_core_gap = "核心" in reason and any(
        marker in reason for marker in ("缺失", "不足", "未命中", "证据")
    )
    if has_relaxed_label and has_core_gap:
        return reason
    core_missing = next(
        (
            item.text
            for item in game.recommendation_evidence
            if item.evidence_id == "core_missing" and item.text
        ),
        "宽松匹配：部分核心特征缺失或证据不足",
    )
    if "宽松匹配" not in core_missing:
        core_missing = f"宽松匹配：{core_missing}"
    return f"{reason.rstrip()}{sentence(core_missing)}"


def sentence(value: str) -> str:
    text = str(value or "").strip().rstrip("。！？.!?")
    return f"{text}。" if text else ""


def valid_game_message(text: str, index: int, title: str) -> bool:
    if not text:
        return False
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return first_line.startswith(f"{index}.") and title.lower() in text.lower()


def format_price_summary(summary: GamePriceSummary) -> str:
    recent = summary.recent_sale_price or "暂无数据"
    if summary.sale_time_status:
        recent += f"（{summary.sale_time_status}）"
    return "；".join(
        [
            f"当前价 {summary.current_price or '暂无数据'}",
            f"历史最低 {summary.historic_low or '暂无数据'}",
            f"最近促销 {recent}",
        ]
    )


def unavailable_price_text() -> str:
    return "当前价 暂无数据；历史最低 暂无数据；最近促销 暂无数据"


def steam_store_url(game: RankedGame) -> str:
    if game.appid is not None:
        return f"https://store.steampowered.com/app/{int(game.appid)}/"
    raw_url = str(game.raw_url or "").strip()
    return raw_url if raw_url.startswith("https://store.steampowered.com/app/") else ""


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()
