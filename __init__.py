"""AstrBot Steam game recommender plugin."""

from pathlib import Path

from .config_migration import migrate_installed_config

migrate_installed_config(Path(__file__).resolve().parent)
