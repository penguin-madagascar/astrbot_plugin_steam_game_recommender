from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolishedPoints:
    fit_points: list[str]
    risk_points: list[str]


def validate_polished_points(
    raw_text: str,
    fallback_fit_points: list[str],
    fallback_risk_points: list[str],
) -> PolishedPoints:
    """Accept LLM-polished points only when it preserves trusted evidence.

    The LLM cannot add unsupported claims or remove rule-generated evidence.
    If validation is not conservative, fallback.
    """

    try:
        payload = json.loads(extract_json_object(raw_text))
    except Exception:
        return PolishedPoints(fallback_fit_points, fallback_risk_points)

    fit_points = normalize_points(payload.get("fit_points"))
    risk_points = normalize_points(payload.get("risk_points"))
    if not fit_points or not risk_points:
        return PolishedPoints(fallback_fit_points, fallback_risk_points)

    trusted_fit = {point.lower() for point in fallback_fit_points}
    trusted_risk = {point.lower() for point in fallback_risk_points}
    if any(point.lower() not in trusted_fit for point in fit_points):
        return PolishedPoints(fallback_fit_points, fallback_risk_points)
    if any(point.lower() not in trusted_risk for point in risk_points):
        return PolishedPoints(fallback_fit_points, fallback_risk_points)
    if not trusted_fit.issubset({point.lower() for point in fit_points}):
        return PolishedPoints(fallback_fit_points, fallback_risk_points)
    if not trusted_risk.issubset({point.lower() for point in risk_points}):
        return PolishedPoints(fallback_fit_points, fallback_risk_points)

    return PolishedPoints(fit_points, risk_points)


def normalize_points(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    points: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = " ".join(str(item or "").split()).strip()
        key = text.lower()
        if text and key not in seen:
            points.append(text)
            seen.add(key)
    return points


def extract_json_object(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return cleaned[start : end + 1]
