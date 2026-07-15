from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from ..storage.models import GameCandidate, SoftFeature

FEATURE_PROMPT_VERSION = "semantic-feature-v3"
FEATURE_SCHEMA_VERSION = "feature-verdict-v2"
MAX_INPUT_CHARS = 48_000
MAX_EVIDENCE_TITLE_CHARS = 256
MAX_EVIDENCE_LIST_ITEMS = 24
MAX_EVIDENCE_LIST_ITEM_CHARS = 128
MAX_SHORT_DESCRIPTION_CHARS = 1_000
MAX_DETAILED_DESCRIPTION_CHARS = 4_000
MAX_EVIDENCE_QUOTE_CHARS = 512
VALID_STATUSES = frozenset({"satisfied", "unknown", "violated"})
VERDICT_FIELDS = frozenset(
    {
        "appid",
        "constraint_id",
        "polarity",
        "status",
        "evidence_quote",
    }
)

VerdictPair = tuple[int, str, str]


class FeatureVerificationContractError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureVerdict:
    appid: int
    constraint_id: str
    polarity: str
    status: str
    evidence_quote: str = ""


def build_verification_payload(
    features: Iterable[SoftFeature],
    candidates: Iterable[GameCandidate],
    *,
    requested_pairs: Iterable[VerdictPair] | None = None,
    prompt_version: str = FEATURE_PROMPT_VERSION,
    schema_version: str = FEATURE_SCHEMA_VERSION,
    short_description_chars: int = MAX_SHORT_DESCRIPTION_CHARS,
    detailed_description_chars: int = MAX_DETAILED_DESCRIPTION_CHARS,
) -> dict[str, Any]:
    selected_features = tuple(features)
    selected_candidates = tuple(candidates)
    allowed = expected_verdict_pairs(selected_features, selected_candidates)
    requested = tuple(requested_pairs) if requested_pairs is not None else allowed
    if len(set(requested)) != len(requested) or not set(requested).issubset(allowed):
        raise FeatureVerificationContractError("invalid verifier request pairs")

    requested_appids = {pair[0] for pair in requested}
    requested_constraints = {(pair[1], pair[2]) for pair in requested}
    return {
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "constraints": [
            constraint_payload(feature)
            for feature in selected_features
            if (feature.constraint_id, feature.polarity) in requested_constraints
        ],
        "candidates": [
            candidate_evidence_payload(
                candidate,
                short_description_chars=short_description_chars,
                detailed_description_chars=detailed_description_chars,
            )
            for candidate in selected_candidates
            if int(candidate.appid or 0) in requested_appids
        ],
        "requests": [
            {
                "appid": appid,
                "constraint_id": constraint_id,
                "polarity": polarity,
            }
            for appid, constraint_id, polarity in requested
        ],
        "allowed_statuses": ["satisfied", "unknown", "violated"],
    }


def validate_verdict_response(
    raw_text: str,
    features: Iterable[SoftFeature],
    candidates: Iterable[GameCandidate],
    *,
    expected_pairs: Iterable[VerdictPair] | None = None,
    candidate_payloads: Iterable[dict[str, Any]] | None = None,
) -> tuple[FeatureVerdict, ...]:
    selected_features = tuple(features)
    selected_candidates = tuple(candidates)
    allowed = expected_verdict_pairs(selected_features, selected_candidates)
    expected = tuple(expected_pairs) if expected_pairs is not None else allowed
    if len(set(expected)) != len(expected) or not set(expected).issubset(allowed):
        raise FeatureVerificationContractError("invalid expected verdict pairs")

    payload = extract_json_object(raw_text)
    if set(payload) != {"verdicts"}:
        raise FeatureVerificationContractError("verifier response has unexpected fields")
    items = payload.get("verdicts")
    if not isinstance(items, list):
        raise FeatureVerificationContractError("verifier verdicts must be an array")

    expected_set = set(expected)
    occurrences: dict[VerdictPair, list[dict[str, Any]]] = {}
    for item in items:
        pair = mapping_pair(item)
        if pair is None or pair not in expected_set:
            continue
        occurrences.setdefault(pair, []).append(item)

    if candidate_payloads is None:
        evidence_by_appid = {
            int(candidate.appid or 0): candidate_evidence_payload(candidate)
            for candidate in selected_candidates
        }
    else:
        evidence_by_appid = {
            int(item.get("appid") or 0): item
            for item in candidate_payloads
            if isinstance(item, dict)
        }

    valid: dict[VerdictPair, FeatureVerdict] = {}
    for pair in expected:
        pair_items = occurrences.get(pair, [])
        if not pair_items or any(item != pair_items[0] for item in pair_items[1:]):
            continue
        try:
            verdict = verdict_from_mapping(pair_items[0])
        except FeatureVerificationContractError:
            continue
        evidence = evidence_by_appid.get(verdict.appid)
        if evidence is None or not verdict_evidence_is_valid(verdict, evidence):
            continue
        valid[pair] = verdict
    return tuple(valid[pair] for pair in expected if pair in valid)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    raise FeatureVerificationContractError("verifier did not return a JSON object")


def mapping_pair(value: Any) -> VerdictPair | None:
    if not isinstance(value, dict):
        return None
    appid = value.get("appid")
    constraint_id = value.get("constraint_id")
    polarity = value.get("polarity")
    if (
        type(appid) is not int
        or not isinstance(constraint_id, str)
        or not isinstance(polarity, str)
    ):
        return None
    return appid, constraint_id, polarity


def expected_verdict_pairs(
    features: Iterable[SoftFeature],
    candidates: Iterable[GameCandidate],
) -> tuple[VerdictPair, ...]:
    selected_features = tuple(features)
    selected_candidates = tuple(candidates)
    appids = [item.appid for item in selected_candidates]
    if any(type(appid) is not int or appid <= 0 for appid in appids):
        raise FeatureVerificationContractError("candidate AppIDs must be positive integers")
    if len(set(appids)) != len(appids):
        raise FeatureVerificationContractError("candidate AppIDs must be unique")
    constraints = [(item.constraint_id, item.polarity) for item in selected_features]
    if len(set(constraints)) != len(constraints):
        raise FeatureVerificationContractError("feature constraints must be unique")
    return tuple(
        (int(candidate.appid), feature.constraint_id, feature.polarity)
        for candidate in selected_candidates
        for feature in selected_features
    )


def verdict_from_mapping(value: Any) -> FeatureVerdict:
    if not isinstance(value, dict) or set(value) != VERDICT_FIELDS:
        raise FeatureVerificationContractError("verdict has unexpected fields")
    appid = value.get("appid")
    if type(appid) is not int or appid <= 0:
        raise FeatureVerificationContractError("verdict AppID is invalid")
    strings = {
        name: value.get(name)
        for name in ("constraint_id", "polarity", "status", "evidence_quote")
    }
    if any(not isinstance(item, str) for item in strings.values()):
        raise FeatureVerificationContractError("verdict text fields are invalid")
    if len(strings["evidence_quote"]) > MAX_EVIDENCE_QUOTE_CHARS:
        raise FeatureVerificationContractError("evidence quote is too long")
    if strings["polarity"] not in {"positive", "negative"}:
        raise FeatureVerificationContractError("verdict polarity is invalid")
    if strings["status"] not in VALID_STATUSES:
        raise FeatureVerificationContractError("verdict status is invalid")
    if not strings["constraint_id"]:
        raise FeatureVerificationContractError("verdict constraint ID is empty")
    if strings["status"] != "unknown" and not strings["evidence_quote"].strip():
        raise FeatureVerificationContractError("decisive verdict requires evidence quote")
    return FeatureVerdict(appid=appid, **strings)


def verdict_evidence_is_valid(
    verdict: FeatureVerdict,
    evidence: dict[str, Any],
) -> bool:
    if verdict.status != "unknown" and not verdict.evidence_quote.strip():
        return False
    if not verdict.evidence_quote:
        return True
    values = [
        str(evidence.get("short_description") or ""),
        str(evidence.get("detailed_description") or ""),
        *(str(item) for item in evidence.get("ordered_tags", [])),
        *(str(item) for item in evidence.get("genres", [])),
        *(str(item) for item in evidence.get("categories", [])),
    ]
    return any(verdict.evidence_quote in value for value in values)


def candidate_evidence_payload(
    candidate: GameCandidate,
    *,
    short_description_chars: int = MAX_SHORT_DESCRIPTION_CHARS,
    detailed_description_chars: int = MAX_DETAILED_DESCRIPTION_CHARS,
) -> dict[str, Any]:
    return {
        "appid": int(candidate.appid or 0),
        "title": bounded_text(candidate.title, MAX_EVIDENCE_TITLE_CHARS),
        "ordered_tags": bounded_text_list(candidate.ordered_tags),
        "genres": bounded_text_list(candidate.genres),
        "categories": bounded_text_list(getattr(candidate, "categories", [])),
        "short_description": bounded_text(
            getattr(candidate, "short_description", "") or "",
            min(max(int(short_description_chars), 0), MAX_SHORT_DESCRIPTION_CHARS),
        ),
        "detailed_description": bounded_text(
            getattr(candidate, "detailed_description", "") or candidate.description or "",
            min(max(int(detailed_description_chars), 0), MAX_DETAILED_DESCRIPTION_CHARS),
        ),
    }


def bounded_text(value: Any, limit: int) -> str:
    return str(value or "")[: max(int(limit), 0)]


def bounded_text_list(values: Iterable[Any]) -> list[str]:
    return [
        bounded_text(value, MAX_EVIDENCE_LIST_ITEM_CHARS)
        for value in tuple(values)[:MAX_EVIDENCE_LIST_ITEMS]
    ]


def constraint_payload(feature: SoftFeature) -> dict[str, str]:
    return {
        "constraint_id": feature.constraint_id,
        "source_span": feature.source_span,
        "normalized_text": feature.normalized_text,
        "role": feature.role,
        "polarity": feature.polarity,
    }


def verdict_cache_key(
    feature: SoftFeature,
    candidate: GameCandidate,
    *,
    provider_id: str,
    locale: str,
    prompt_version: str = FEATURE_PROMPT_VERSION,
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> str:
    evidence_hash = sha256_json(candidate_evidence_payload(candidate))
    identity = {
        "appid": int(candidate.appid or 0),
        "constraint": constraint_payload(feature),
        "evidence_hash": evidence_hash,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "provider_id": str(provider_id or ""),
        "locale": str(locale or ""),
    }
    return f"semantic-feature-verdict:{sha256_json(identity)}"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
