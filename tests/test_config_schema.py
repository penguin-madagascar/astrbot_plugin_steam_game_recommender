from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfigSchemaTest(unittest.TestCase):
    def test_steam_index_settings_are_exposed_in_dashboard_schema(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(schema["steam_index_ttl_hours"]["default"], 168)
        self.assertEqual(schema["steam_min_review_count"]["default"], 50)
        self.assertEqual(schema["steam_min_positive_ratio"]["default"], 0.65)

    def test_steam_web_api_key_is_exposed_for_owned_game_filter(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(schema["steam_api_key"]["type"], "string")
        self.assertEqual(schema["steam_api_key"]["default"], "")
        self.assertIn("GetOwnedGames", schema["steam_api_key"]["hint"])

    def test_llm_fallback_toggle_is_exposed_after_provider(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        keys = list(schema)

        self.assertEqual(keys.index("enable_llm_fallback"), keys.index("llm_provider_id") + 1)
        self.assertEqual(schema["enable_llm_fallback"]["type"], "bool")
        self.assertIs(schema["enable_llm_fallback"]["default"], False)
        self.assertTrue(schema["enable_llm_fallback"]["hint"].strip())


if __name__ == "__main__":
    unittest.main()
