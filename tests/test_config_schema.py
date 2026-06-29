from __future__ import annotations

import json
from pathlib import Path
import unittest


class ConfigSchemaTest(unittest.TestCase):
    def test_steam_index_settings_are_exposed_in_dashboard_schema(self) -> None:
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(schema["steam_index_ttl_hours"]["default"], 168)
        self.assertEqual(schema["steam_min_review_count"]["default"], 50)
        self.assertEqual(schema["steam_min_positive_ratio"]["default"], 0.65)


if __name__ == "__main__":
    unittest.main()
