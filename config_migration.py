from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLUGIN_NAME = "astrbot_plugin_steam_game_recommender"
OBSOLETE_LLM_FALLBACK_KEYS = frozenset({"enable_llm_fallback"})
OBSOLETE_SCORING_WEIGHT_KEYS = frozenset(
    {
        "tag_coverage_weight",
        "positive_reference_weight",
        "library_profile_weight",
        "review_reputation_weight",
        "popularity_weight",
    }
)
GROUP_LEGACY_KEYS = {
    "model_and_access": (
        "llm_provider_id",
        "llm_fallback_provider_id",
        "steam_api_key",
    ),
    "price_and_region": (
        "steam_price_heybox_notice",
        "default_region",
    ),
    "recommendation_and_scoring": (
        "max_results",
        "steam_index_ttl_hours",
        "steam_min_review_count",
        "steam_min_positive_ratio",
    ),
    "cache_and_network": (
        "cache_ttl_hours",
        "timeout_seconds",
        "reuse_identical_query_cache",
    ),
}
LEGACY_KEYS = frozenset(
    {
        *OBSOLETE_LLM_FALLBACK_KEYS,
        *(key for group_keys in GROUP_LEGACY_KEYS.values() for key in group_keys),
    }
)


def migrate_config_data(
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    existing_model_config = config.get("model_and_access")
    has_obsolete_llm_fallback = any(
        key in config for key in OBSOLETE_LLM_FALLBACK_KEYS
    ) or (
        isinstance(existing_model_config, Mapping)
        and any(key in existing_model_config for key in OBSOLETE_LLM_FALLBACK_KEYS)
    )
    existing_scoring_config = config.get("recommendation_and_scoring")
    has_obsolete_scoring_weights = any(
        key in config for key in OBSOLETE_SCORING_WEIGHT_KEYS
    ) or (
        isinstance(existing_scoring_config, Mapping)
        and any(key in existing_scoring_config for key in OBSOLETE_SCORING_WEIGHT_KEYS)
    )
    has_flat_legacy_values = any(
        key in config
        for key in LEGACY_KEYS
        if key not in OBSOLETE_LLM_FALLBACK_KEYS
    )
    if not has_flat_legacy_values:
        if has_obsolete_scoring_weights or has_obsolete_llm_fallback:
            cleaned = dict(config)
            for key in OBSOLETE_SCORING_WEIGHT_KEYS:
                cleaned.pop(key, None)
            for key in OBSOLETE_LLM_FALLBACK_KEYS:
                cleaned.pop(key, None)
            if isinstance(existing_model_config, Mapping):
                model_config = dict(existing_model_config)
                for key in OBSOLETE_LLM_FALLBACK_KEYS:
                    model_config.pop(key, None)
                cleaned["model_and_access"] = model_config
            if isinstance(existing_scoring_config, Mapping):
                scoring_config = dict(existing_scoring_config)
                for key in OBSOLETE_SCORING_WEIGHT_KEYS:
                    scoring_config.pop(key, None)
                cleaned["recommendation_and_scoring"] = scoring_config
            return cleaned, True
        return dict(config), False

    migrated = {
        key: value
        for key, value in config.items()
        if key not in LEGACY_KEYS
        and key not in GROUP_LEGACY_KEYS
        and key not in OBSOLETE_SCORING_WEIGHT_KEYS
    }
    for group_name, group_schema in schema.items():
        if group_name not in GROUP_LEGACY_KEYS:
            continue
        existing = config.get(group_name)
        group_values = dict(existing) if isinstance(existing, Mapping) else {}
        for key in OBSOLETE_LLM_FALLBACK_KEYS:
            group_values.pop(key, None)
        if group_name == "recommendation_and_scoring":
            for key in OBSOLETE_SCORING_WEIGHT_KEYS:
                group_values.pop(key, None)
        for item_name, item_schema in group_schema["items"].items():
            if item_name in group_values:
                continue
            if item_name in config:
                group_values[item_name] = config[item_name]
            else:
                group_values[item_name] = item_schema.get("default")
        migrated[group_name] = group_values
    return migrated, True


def migrate_config_file(config_path: Path, schema_path: Path) -> bool:
    config_path = Path(config_path)
    schema_path = Path(schema_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    migrated, changed = migrate_config_data(config, schema)
    if not changed:
        return False

    _atomic_write_json(config_path, migrated)
    return True


def migrate_installed_config(plugin_dir: Path) -> bool:
    plugin_dir = Path(plugin_dir).absolute()
    if plugin_dir.name != PLUGIN_NAME or plugin_dir.parent.name != "plugins":
        return False

    from astrbot.core.utils.astrbot_path import get_astrbot_data_path

    data_dir = Path(get_astrbot_data_path()).absolute()
    installed_dir = (data_dir / "plugins" / PLUGIN_NAME).absolute()
    if plugin_dir != installed_dir:
        return False

    config_path = data_dir / "config" / f"{PLUGIN_NAME}_config.json"
    if not config_path.exists():
        return False
    return migrate_config_file(config_path, plugin_dir / "_conf_schema.json")


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(data, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
