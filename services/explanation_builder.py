from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ..storage.models import GameCandidate, RankedGame, RecommendationEvidence
from .similarity_ranker import popularity_score

logger = logging.getLogger(__name__)

MAX_REASON_CONCURRENCY = 5
MAX_REASON_LENGTH = 180
MAX_REASON_EVIDENCE = 8
SYSTEM_PROMPT = (
    "你是 Steam 游戏推荐理由编辑器。只能使用输入的可信证据，不得补充未提供的玩法、"
    "语言、价格、口碑或平台事实。输出 2 至 3 句简短中文理由，并只返回指定 JSON。"
)


@dataclass(frozen=True)
class ValidatedReason:
    reason: str
    evidence_ids: list[str]


async def generate_recommendation_reasons(
    context: Any,
    event: Any,
    provider_id: str,
    games: list[RankedGame],
) -> list[RankedGame]:
    if not games:
        return []
    resolved_provider = await resolve_provider_id(context, event, provider_id)
    if not resolved_provider:
        return [with_reason(game, fallback_reason(game.recommendation_evidence)) for game in games]

    semaphore = asyncio.Semaphore(MAX_REASON_CONCURRENCY)

    async def generate_one(game: RankedGame) -> RankedGame:
        evidence = select_reason_evidence(game.recommendation_evidence)
        try:
            async with semaphore:
                raw = await generate_reason_text(
                    context,
                    resolved_provider,
                    appid=game.appid,
                    title=game.title,
                    evidence=evidence,
                    unplayed=False,
                )
            result = validate_reason_response(raw, game.appid, evidence)
        except Exception as exc:
            logger.warning(
                "Steam recommendation reason generation failed for %s: %s", game.appid, exc
            )
            result = None
        reason = result.reason if result else fallback_reason(evidence)
        return with_reason(game, reason)

    return list(await asyncio.gather(*(generate_one(game) for game in games)))


async def generate_unplayed_reason(
    context: Any,
    event: Any,
    provider_id: str,
    game: GameCandidate,
) -> str:
    evidence = build_unplayed_evidence(game)
    resolved_provider = await resolve_provider_id(context, event, provider_id)
    if not resolved_provider:
        return fallback_unplayed_reason(evidence)
    try:
        raw = await generate_reason_text(
            context,
            resolved_provider,
            appid=game.appid,
            title=game.title,
            evidence=evidence,
            unplayed=True,
        )
        result = validate_reason_response(raw, game.appid, evidence)
    except Exception as exc:
        logger.warning("Steam unplayed reason generation failed for %s: %s", game.appid, exc)
        result = None
    return result.reason if result else fallback_unplayed_reason(evidence)


async def generate_reason_text(
    context: Any,
    provider_id: str,
    appid: int | None,
    title: str,
    evidence: list[RecommendationEvidence],
    unplayed: bool,
) -> str:
    prompt = reason_prompt(appid, title, evidence, unplayed=unplayed)
    response = await context.llm_generate(
        chat_provider_id=provider_id,
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
    )
    return str(getattr(response, "completion_text", "") or "").strip()


def reason_prompt(
    appid: int | None,
    title: str,
    evidence: list[RecommendationEvidence],
    unplayed: bool = False,
) -> str:
    numbered = "\n".join(
        (f"{index}. ID={item.evidence_id} | {item.sentiment} | {item.category} | {item.text}")
        for index, item in enumerate(evidence, start=1)
    )
    important_ids = [
        item.evidence_id
        for item in evidence
        if item.important and item.sentiment in {"negative", "uncertain"}
    ]
    focus = (
        "这是未玩游戏库的通用理由，重点概括玩法、口碑和知名度；不要输出分数、价格、链接或用户偏好。"
        if unplayed
        else "优先保留最关键的匹配点；次要优点和非重要缺点可以省略。"
    )
    return (
        f"APPID={appid if appid is not None else 'null'}\n"
        f"TITLE={title}\n"
        f"{focus}\n"
        "从下面编号证据中选择通常 2 至 4 条来写 2 至 3 句理由，总长度不超过 180 字。\n"
        f"重要风险 ID（必须全部保留）：{json.dumps(important_ids, ensure_ascii=False)}\n"
        f"可信证据：\n{numbered}\n"
        "只返回 JSON："
        '{"appid":123,"reason":"……。……。","evidence_ids":["证据ID"]}'
    )


def validate_reason_response(
    raw_text: str,
    appid: int | None,
    evidence: list[RecommendationEvidence],
) -> ValidatedReason | None:
    try:
        payload = json.loads(extract_json_object(raw_text))
    except Exception:
        return None
    if not isinstance(payload, dict) or normalize_appid(payload.get("appid")) != appid:
        return None

    reason = re.sub(r"\s+", " ", str(payload.get("reason") or "")).strip()
    if len(reason) > MAX_REASON_LENGTH or sentence_count(reason) not in {2, 3}:
        return None
    if not reason.endswith(("。", "！", "？", ".", "!", "?")):
        return None

    evidence_ids = normalize_evidence_ids(payload.get("evidence_ids"))
    available_ids = {item.evidence_id for item in evidence}
    if evidence and not evidence_ids:
        return None
    if any(evidence_id not in available_ids for evidence_id in evidence_ids):
        return None
    important_ids = {
        item.evidence_id
        for item in evidence
        if item.important and item.sentiment in {"negative", "uncertain"}
    }
    if not important_ids.issubset(evidence_ids):
        return None
    important_evidence = [item for item in evidence if item.evidence_id in important_ids]
    if any(not important_risk_is_mentioned(item, reason) for item in important_evidence):
        return None
    if len(evidence_ids) > max(4, len(important_ids) + 2):
        return None
    return ValidatedReason(reason=reason, evidence_ids=evidence_ids)


def select_reason_evidence(
    evidence: list[RecommendationEvidence],
    limit: int = MAX_REASON_EVIDENCE,
) -> list[RecommendationEvidence]:
    maximum = max(int(limit), 1)
    important = [
        item for item in evidence if item.important and item.sentiment in {"negative", "uncertain"}
    ]
    priority = {
        "preference": 0,
        "reference": 1,
        "language": 2,
        "budget": 3,
        "reviews": 4,
        "popularity": 5,
        "library": 6,
    }
    remaining = [
        item for item in evidence if item.evidence_id not in {x.evidence_id for x in important}
    ]
    remaining.sort(
        key=lambda item: (
            0 if item.sentiment == "positive" else 1,
            priority.get(item.category, 9),
        )
    )
    selected = [*important, *remaining[: max(maximum - len(important), 0)]]
    return selected if len(important) > maximum else selected[:maximum]


def fallback_reason(evidence: list[RecommendationEvidence]) -> str:
    selected = select_reason_evidence(evidence)
    positives = [item.text for item in selected if item.sentiment == "positive"]
    important_risks = [
        item.text
        for item in selected
        if item.important and item.sentiment in {"negative", "uncertain"}
    ]
    sentences: list[str] = []
    if positives:
        sentences.append(short_sentence(positives[0], 55))
    else:
        sentences.append("现有 Steam 数据只能提供有限的匹配信息。")
    if len(positives) > 1:
        sentences.append(short_sentence(positives[1], 55))
    elif not important_risks:
        sentences.append("建议结合实际玩法偏好再做最终选择。")
    if important_risks:
        risk_text = "；".join(important_risks[:2])
        sentences.append(short_sentence(risk_text, 60))
    return "".join(sentences[:3])


def fallback_unplayed_reason(evidence: list[RecommendationEvidence]) -> str:
    by_id = {item.evidence_id: item.text for item in evidence}
    sentences = [
        short_sentence(by_id[evidence_id], 55)
        for evidence_id in ("gameplay", "reviews", "popularity")
        if evidence_id in by_id
    ]
    while len(sentences) < 2:
        sentences.append("建议结合实际玩法偏好再决定是否现在开玩。")
    return "".join(sentences[:3])


def important_risk_is_mentioned(item: RecommendationEvidence, reason: str) -> bool:
    text = reason.casefold()
    if item.category == "language":
        return any(word in text for word in ("语言", "中文", "英文", "日语", "韩语")) and any(
            word in text for word in ("未确认", "不支持", "缺失", "未知")
        )
    if item.category == "budget":
        return "预算" in text or "价格" in text
    if item.category == "reference":
        return "参考" in text or "相似" in text
    if item.category == "constraint":
        return any(word in text for word in ("硬条件", "未确认", "不支持", "缺失", "未知"))
    return any(
        word in text
        for word in ("未确认", "不支持", "高于", "不一致", "未命中", "相似", "缺失", "未知")
    )


def build_unplayed_evidence(game: GameCandidate) -> list[RecommendationEvidence]:
    evidence: list[RecommendationEvidence] = []
    gameplay_parts: list[str] = []
    if game.genres:
        gameplay_parts.append(f"类型：{'、'.join(game.genres[:4])}")
    if game.tags:
        gameplay_parts.append(f"标签：{'、'.join(game.tags[:6])}")
    if gameplay_parts:
        evidence.append(
            RecommendationEvidence(
                evidence_id="gameplay",
                category="gameplay",
                sentiment="positive",
                text="；".join(gameplay_parts),
            )
        )
    if game.review_total is not None and game.review_positive_ratio is not None:
        evidence.append(
            RecommendationEvidence(
                evidence_id="reviews",
                category="reviews",
                sentiment="positive",
                text=(
                    f"Steam 好评率 {game.review_positive_ratio:.0%}，共 {game.review_total} 条评测"
                ),
            )
        )
        evidence.append(
            RecommendationEvidence(
                evidence_id="popularity",
                category="popularity",
                sentiment="positive",
                text=f"评测规模对应的知名度指标为 {popularity_score(game.review_total):.0%}",
            )
        )
    return evidence


def sentence_count(value: str) -> int:
    parts = re.split(r"[。！？!?]+|(?<!\d)\.(?!\d)", value)
    return len([part for part in parts if part.strip()])


def normalize_evidence_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def normalize_appid(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def short_sentence(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().rstrip("。！？.!?")
    if len(text) > limit:
        text = text[: max(limit - 1, 1)].rstrip("，；、 ") + "…"
    return f"{text}。"


def extract_json_object(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return cleaned[start : end + 1]


def with_reason(game: RankedGame, reason: str) -> RankedGame:
    copier = getattr(game, "model_copy", None)
    if copier:
        return copier(update={"recommendation_reason": reason})
    return game.copy(update={"recommendation_reason": reason})


async def resolve_provider_id(context: Any, event: Any, provider_id: str) -> str:
    if provider_id:
        return provider_id
    getter = getattr(context, "get_current_chat_provider_id", None)
    if not getter:
        return ""
    try:
        return str(await getter(umo=event.unified_msg_origin) or "")
    except Exception as exc:
        logger.debug("Failed to resolve current LLM provider: %s", exc)
        return ""
