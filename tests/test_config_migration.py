from __future__ import annotations

import codecs
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_migration_module():
    try:
        return importlib.import_module("astrbot_plugin_steam_game_recommender.config_migration")
    except ModuleNotFoundError as exc:
        raise AssertionError("config migration module is missing") from exc


def legacy_config() -> dict:
    return {
        "llm_provider_id": "provider/custom-model",
        "enable_llm_fallback": True,
        "default_region": "US",
        "steam_api_key": "test-steam-key",
        "steam_price_heybox_notice": "legacy notice",
        "max_results": 7,
        "tag_coverage_weight": 90,
        "positive_reference_weight": 1,
        "library_profile_weight": 2,
        "review_reputation_weight": 3,
        "popularity_weight": 4,
        "steam_index_ttl_hours": 72,
        "steam_min_review_count": 120,
        "steam_min_positive_ratio": 0.75,
        "cache_ttl_hours": 36,
        "timeout_seconds": 30,
    }


class ConfigMigrationTest(unittest.TestCase):
    def test_bom_legacy_file_is_atomically_migrated_with_custom_values(self) -> None:
        migration = load_migration_module()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "plugin_config.json"
            config_path.write_bytes(
                codecs.BOM_UTF8 + json.dumps(legacy_config(), ensure_ascii=False).encode("utf-8")
            )

            changed = migration.migrate_config_file(
                config_path,
                ROOT / "_conf_schema.json",
            )

            self.assertIs(changed, True)
            raw = config_path.read_bytes()
            self.assertFalse(raw.startswith(codecs.BOM_UTF8))
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))
            migrated = json.loads(raw.decode("utf-8"))
            self.assertEqual(
                list(migrated),
                [
                    "model_and_access",
                    "price_and_region",
                    "recommendation_and_scoring",
                    "cache_and_network",
                ],
            )
            self.assertEqual(
                migrated["model_and_access"],
                {
                    "llm_provider_id": "provider/custom-model",
                    "steam_api_key": "test-steam-key",
                },
            )
            self.assertEqual(migrated["price_and_region"]["default_region"], "US")
            self.assertEqual(
                migrated["recommendation_and_scoring"]["max_results"],
                7,
            )
            self.assertTrue(
                migration.OBSOLETE_SCORING_WEIGHT_KEYS.isdisjoint(
                    migrated["recommendation_and_scoring"]
                )
            )
            self.assertEqual(migrated["cache_and_network"]["timeout_seconds"], 30)

            first_write = raw
            self.assertIs(
                migration.migrate_config_file(
                    config_path,
                    ROOT / "_conf_schema.json",
                ),
                False,
            )
            self.assertEqual(config_path.read_bytes(), first_write)

    def test_existing_nested_values_override_legacy_values_and_defaults_fill_gaps(
        self,
    ) -> None:
        migration = load_migration_module()
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        config = {
            "model_and_access": {
                "llm_provider_id": "provider/new-model",
            },
            "llm_provider_id": "provider/legacy-model",
            "steam_api_key": "legacy-key",
            "default_region": "JP",
        }

        migrated, changed = migration.migrate_config_data(config, schema)

        self.assertIs(changed, True)
        self.assertEqual(
            migrated["model_and_access"]["llm_provider_id"],
            "provider/new-model",
        )
        self.assertEqual(migrated["model_and_access"]["steam_api_key"], "legacy-key")
        self.assertNotIn(
            "llm_fallback_provider_id",
            migrated["model_and_access"],
        )
        self.assertEqual(migrated["price_and_region"]["default_region"], "JP")
        self.assertTrue(
            migration.OBSOLETE_SCORING_WEIGHT_KEYS.isdisjoint(
                migrated["recommendation_and_scoring"]
            )
        )
        for key in legacy_config():
            self.assertNotIn(key, migrated)

    def test_obsolete_weights_are_removed_from_grouped_config_and_trigger_rewrite(
        self,
    ) -> None:
        migration = load_migration_module()
        config = {
            "model_and_access": {"llm_provider_id": "provider/current"},
            "recommendation_and_scoring": {
                "max_results": 9,
                "steam_index_ttl_hours": 48,
                "steam_min_review_count": 75,
                "steam_min_positive_ratio": 0.7,
                "tag_coverage_weight": 100,
                "popularity_weight": 0,
                "custom_future_value": "keep-me",
            },
            "positive_reference_weight": 42,
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "plugin_config.json"
            original = json.dumps(config, ensure_ascii=False, indent=4).encode("utf-8")
            config_path.write_bytes(original)

            changed = migration.migrate_config_file(
                config_path,
                ROOT / "_conf_schema.json",
            )

            self.assertIs(changed, True)
            raw = config_path.read_bytes()
            self.assertNotEqual(raw, original)
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))
            migrated = json.loads(raw.decode("utf-8"))
            self.assertEqual(
                list(migrated),
                ["model_and_access", "recommendation_and_scoring"],
            )
            self.assertEqual(
                migrated["model_and_access"],
                {"llm_provider_id": "provider/current"},
            )
            scoring = migrated["recommendation_and_scoring"]
            self.assertTrue(migration.OBSOLETE_SCORING_WEIGHT_KEYS.isdisjoint(migrated))
            self.assertTrue(migration.OBSOLETE_SCORING_WEIGHT_KEYS.isdisjoint(scoring))
            self.assertEqual(scoring["max_results"], 9)
            self.assertEqual(scoring["steam_min_review_count"], 75)
            self.assertEqual(scoring["custom_future_value"], "keep-me")

            self.assertIs(
                migration.migrate_config_file(
                    config_path,
                    ROOT / "_conf_schema.json",
                ),
                False,
            )
            self.assertEqual(config_path.read_bytes(), raw)

    def test_flat_and_grouped_legacy_fallback_fields_are_removed(self) -> None:
        migration = load_migration_module()
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        cases = [
            {
                "enable_llm_fallback": False,
                "llm_fallback_provider_id": "provider/flat-selection",
            },
            {
                "model_and_access": {
                    "enable_llm_fallback": True,
                    "llm_fallback_provider_id": "provider/grouped-selection",
                }
            },
        ]

        for config in cases:
            with self.subTest(config=config):
                migrated, changed = migration.migrate_config_data(config, schema)
                self.assertIs(changed, True)
                self.assertNotIn("enable_llm_fallback", migrated)
                self.assertNotIn("llm_fallback_provider_id", migrated)
                model_config = migrated.get("model_and_access", {})
                self.assertNotIn("enable_llm_fallback", model_config)
                self.assertNotIn("llm_fallback_provider_id", model_config)

    def test_grouped_file_without_legacy_keys_is_not_rewritten(self) -> None:
        migration = load_migration_module()
        grouped = {
            "model_and_access": {
                "llm_provider_id": "provider/current",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "plugin_config.json"
            original = json.dumps(grouped, ensure_ascii=False, indent=4).encode("utf-8")
            config_path.write_bytes(original)

            changed = migration.migrate_config_file(
                config_path,
                ROOT / "_conf_schema.json",
            )

            self.assertIs(changed, False)
            self.assertEqual(config_path.read_bytes(), original)

    def test_malformed_config_is_unchanged_when_migration_aborts(self) -> None:
        migration = load_migration_module()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "plugin_config.json"
            original = codecs.BOM_UTF8 + b'{"llm_provider_id":'
            config_path.write_bytes(original)

            with self.assertRaises(json.JSONDecodeError):
                migration.migrate_config_file(
                    config_path,
                    ROOT / "_conf_schema.json",
                )

            self.assertEqual(config_path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [config_path])

    def test_repository_import_does_not_run_installed_config_migration(self) -> None:
        migration = load_migration_module()

        self.assertIs(migration.migrate_installed_config(ROOT), False)

    def test_package_entry_invokes_migration_before_plugin_loading(self) -> None:
        migration = load_migration_module()
        package = importlib.import_module("astrbot_plugin_steam_game_recommender")

        with patch.object(
            migration,
            "migrate_installed_config",
            return_value=False,
        ) as migrate:
            importlib.reload(package)

        migrate.assert_called_once_with(ROOT)


if __name__ == "__main__":
    unittest.main()
