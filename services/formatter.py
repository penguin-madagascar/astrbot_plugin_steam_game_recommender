from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.star import Context

from ..storage.models import GamePreference, GamePriceSummary, RankedGame
from .explanation_builder import (
    fallback_caution_reason,
    fallback_reason,
    user_facing_evidence_text,
)
from .llm_fallback import UnverifiedGameSuggestion, safe_unverified_title
from .recommendation_limits import DEFAULT_RECOMMENDATION_COUNT
from .run_notices import RunNotice

UNVERIFIED_FALLBACK_DISCLAIMER = (
    "⚠️ LLM 兜底建议（名称经 Steam 目录确认，需求匹配未验证）"
)
UNVERIFIED_FALLBACK_REASON = (
    "Steam 仅确认了该名称对应游戏；模型认为它可能符合需求，"
    "需求匹配未经过 Steam 数据验证。"
)


def format_recommendations(
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
) -> str:
    return "\n\n".join(
        format_recommendation_messages(preference, ranked_games, limit=limit)
    )


def format_recommendation_messages(
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
    run_notices: list[RunNotice] | tuple[RunNotice, ...] = (),
    unverified_suggestions: tuple[UnverifiedGameSuggestion, ...] = (),
) -> list[str]:
    notice_messages = [notice.text for notice in run_notices]
    count = min(
        limit or preference.result_count or DEFAULT_RECOMMENDATION_COUNT,
        len(ranked_games),
    )
    if not ranked_games:
        lines = [
            "暂时没有找到满足当前条件的游戏。",
            "可以尝试放宽排除标签、人数、语言或类型条件后再查一次。",
        ]
        if preference.parse_warnings:
            lines.append(format_parse_warnings(preference.parse_warnings))
        messages = [*notice_messages, "\n".join(lines)]
        if unverified_suggestions:
            messages.append(UNVERIFIED_FALLBACK_DISCLAIMER)
            messages.extend(
                format_unverified_suggestion(index, suggestion)
                for index, suggestion in enumerate(unverified_suggestions, start=1)
            )
        return messages

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
        lines.append(format_parse_warnings(preference.parse_warnings))

    messages = [*notice_messages, "\n".join(lines)]
    for index, game in enumerate(ranked_games[:count], start=1):
        messages.append("\n".join(format_game_block(index, game, region=preference.region or "CN")))
    return messages


def format_unverified_suggestion(
    index: int,
    suggestion: UnverifiedGameSuggestion,
) -> str:
    # LLM prose is untrusted.  The model may select a title, but user-visible
    # claims are rendered from a fixed, explicitly unverified statement.
    if not suggestion.title_verified:
        return (
            f"{index}. 模型候选名称未通过 Steam 目录确认，已省略\n"
            f"系统说明：{UNVERIFIED_FALLBACK_REASON}"
        )
    title = safe_unverified_title(
        suggestion.title,
        title_verified=suggestion.title_verified,
    )
    return (
        f"{index}. 模型候选（名称经 Steam 目录确认）：“{title}”\n"
        f"系统说明：{UNVERIFIED_FALLBACK_REASON}"
    )


async def format_recommendations_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
    run_notices: list[RunNotice] | tuple[RunNotice, ...] = (),
) -> str:
    return "\n\n".join(
        await format_recommendation_messages_with_llm(
            context,
            event,
            provider_id,
            preference,
            ranked_games,
            limit=limit,
            run_notices=run_notices,
        )
    )


async def format_recommendation_messages_with_llm(
    context: "Context",
    event: "AstrMessageEvent",
    provider_id: str,
    preference: GamePreference,
    ranked_games: list[RankedGame],
    limit: int | None = None,
    run_notices: list[RunNotice] | tuple[RunNotice, ...] = (),
) -> list[str]:
    del context, event, provider_id
    return format_recommendation_messages(
        preference,
        ranked_games,
        limit=limit,
        run_notices=run_notices,
    )


def format_game_block(index: int, game: RankedGame, region: str | None = None) -> list[str]:
    reason = user_facing_evidence_text(
        game.recommendation_reason or fallback_recommendation_reason(game)
    )
    caution = user_facing_evidence_text(
        game.caution_reason or fallback_caution_reason(game.recommendation_evidence) or ""
    )
    price_summary = game.price_summary
    selected_region = price_summary.region if price_summary else (region or "CN")
    lines = [
        f"{index}. 《{game.title}》｜推荐分：{game.score}/100",
        "",
        f"推荐理由：{reason}",
    ]
    if caution:
        lines.extend(["", f"不推荐理由：{caution}"])
    lines.extend(
        [
            "",
            (
            f"价格（{selected_region}）："
            f"{format_price_summary(price_summary) if price_summary else unavailable_price_text()}"
            ),
        ]
    )
    if link := steam_store_url(game):
        lines.extend(["", f"购买链接：{link}"])
    return lines


def fallback_recommendation_reason(game: RankedGame) -> str:
    return fallback_reason(game.recommendation_evidence)


def disclose_relaxed_match(game: RankedGame, reason: str) -> str:
    del game
    return reason


def format_parse_warnings(values: list[str]) -> str:
    return "偏好解析提示：\n" + "\n".join(f"- {value}" for value in values if value)


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
