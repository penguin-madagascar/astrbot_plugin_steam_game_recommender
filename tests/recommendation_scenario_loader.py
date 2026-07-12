from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recommendation_quality_scenarios.json"
LEGACY_OUTPUT_PATH = Path(__file__).parent / "fixtures" / "legacy_recommendation_outputs.json"
EXPECTED_LEGACY_OUTPUT_SHA256 = "de68b5e1e2b6812045935336b5027f382221fc69d67aafeee0ece55871d2cc09"


def load_recommendation_quality_fixture() -> dict[str, Any]:
    actual_hash = hashlib.sha256(LEGACY_OUTPUT_PATH.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_LEGACY_OUTPUT_SHA256:
        raise ValueError("legacy recommendation output fixture hash changed")

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    legacy = json.loads(LEGACY_OUTPUT_PATH.read_text(encoding="utf-8"))
    outputs_by_id = {output["id"]: output for output in legacy["outputs"]}
    scenario_ids = {scenario["id"] for scenario in fixture["scenarios"]}
    if scenario_ids != outputs_by_id.keys():
        raise ValueError("scenario and frozen legacy output ids differ")

    return {
        **fixture,
        "legacy_source": legacy["legacy_source"],
        "legacy_baseline": legacy["legacy_baseline"],
        "scenarios": [
            {**scenario, **outputs_by_id[scenario["id"]]} for scenario in fixture["scenarios"]
        ],
    }
