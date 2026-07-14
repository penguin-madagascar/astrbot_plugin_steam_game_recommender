from __future__ import annotations

import importlib
import importlib.util
import unittest

from astrbot_plugin_steam_game_recommender.clients.steam import parse_steam_game
from astrbot_plugin_steam_game_recommender.services.ranking_precedence import effective_score
from astrbot_plugin_steam_game_recommender.services.steam_index import rank_entries
from astrbot_plugin_steam_game_recommender.storage.models import (
    CompanyPreference,
    GameCandidate,
    GamePreference,
    ScoreBreakdown,
)


MODULE = "astrbot_plugin_steam_game_recommender.services.company_preferences"


def company(
    name: str,
    *,
    role: str = "either",
    strength: str = "preferred",
) -> CompanyPreference:
    return CompanyPreference(
        display_name=name,
        aliases=[name],
        role=role,
        strength=strength,
        source_span=name,
    )


class CompanyMetadataTest(unittest.TestCase):
    def test_appdetails_preserves_company_category_and_both_descriptions(self) -> None:
        candidate = parse_steam_game(
            10,
            {
                "name": "Example",
                "type": "game",
                "developers": ["Acme Studio, Inc."],
                "publishers": ["Publisher House LLC"],
                "genres": [{"id": "3", "description": "RPG"}],
                "categories": [
                    {"id": 2, "description": "Single-player"},
                    {"id": 9, "description": "Co-op"},
                ],
                "short_description": "A <b>short</b> summary.",
                "detailed_description": "A <i>complete store</i> description.",
                "about_the_game": "A shorter about-the-game description.",
                "release_date": {"coming_soon": False, "date": "1 Jan, 2025"},
            },
        )

        self.assertEqual(candidate.developers, ["Acme Studio, Inc."])
        self.assertEqual(candidate.publishers, ["Publisher House LLC"])
        self.assertTrue(candidate.developer_data_available)
        self.assertTrue(candidate.publisher_data_available)
        self.assertTrue(candidate.company_data_available)
        self.assertEqual(candidate.categories, ["single-player", "co-op"])
        self.assertEqual(candidate.category_ids, [2, 9])
        self.assertEqual(candidate.short_description, "A short summary.")
        self.assertEqual(candidate.detailed_description, "A complete store description.")

    def test_missing_company_fields_are_distinct_from_present_empty_lists(self) -> None:
        missing = parse_steam_game(1, {"name": "Missing", "type": "game"})
        empty = parse_steam_game(
            2,
            {"name": "Empty", "type": "game", "developers": [], "publishers": []},
        )

        self.assertFalse(missing.company_data_available)
        self.assertTrue(empty.company_data_available)


class CompanyPreferenceScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.find_spec(MODULE)
        self.assertIsNotNone(spec)
        self.module = importlib.import_module(MODULE)

    def test_normalization_is_exact_and_removes_only_terminal_legal_suffixes(self) -> None:
        normalize = self.module.normalize_company_name

        self.assertEqual(normalize("ＡＣＭＥ， Inc."), "acme")
        self.assertEqual(normalize("Acme Studios"), "acme studios")
        self.assertEqual(normalize("Limited Run Games"), "limited run games")
        self.assertEqual(normalize("某某有限公司。"), normalize("某某"))
        self.assertEqual(normalize("某某株式会社！"), normalize("某某"))
        self.assertEqual(normalize("Acme Ltd.，"), "acme")
        self.assertEqual(normalize("CD PROJEKT S.A."), normalize("CD PROJEKT SA"))
        self.assertEqual(normalize("CD PROJEKT S.A."), "cd projekt")
        self.assertEqual(normalize("CD PROJEKT S.A.，"), "cd projekt")
        self.assertTrue(normalize("ソニー・インタラクティブエンタテインメント"))
        self.assertTrue(normalize("Сабер Интерактив"))
        self.assertNotEqual(normalize("Acme"), normalize("Acme North"))

    def test_non_latin_company_names_match_exactly(self) -> None:
        candidate = GameCandidate(
            appid=11,
            title="International Studio Game",
            developers=["ソニー・インタラクティブエンタテインメント"],
            developer_data_available=True,
        )

        self.assertTrue(
            self.module.matches_company_preference(
                candidate,
                company("ソニー・インタラクティブエンタテインメント", role="developer"),
            )
        )
        self.assertFalse(
            self.module.matches_company_preference(
                candidate,
                company("ソニー", role="developer"),
            )
        )

    def test_developer_publisher_and_either_match_only_company_fields(self) -> None:
        candidate = GameCandidate(
            appid=10,
            title="Acme Adventure",
            developers=["Acme Studio, Inc."],
            publishers=["Publisher House LLC"],
            developer_data_available=True,
            publisher_data_available=True,
            company_data_available=True,
        )

        self.assertTrue(
            self.module.matches_company_preference(
                candidate,
                company("Acme Studio", role="developer"),
            )
        )
        self.assertTrue(
            self.module.matches_company_preference(
                candidate,
                company("Publisher House", role="publisher"),
            )
        )
        self.assertTrue(
            self.module.matches_company_preference(
                candidate,
                company("Publisher House", role="either"),
            )
        )
        self.assertFalse(
            self.module.matches_company_preference(
                candidate,
                company("Acme", role="either"),
            )
        )

    def test_unknown_and_mismatch_penalties_take_only_the_most_severe(self) -> None:
        unknown = GameCandidate(appid=1, title="Unknown")
        empty = GameCandidate(
            appid=2,
            title="Empty",
            developers=[],
            publishers=[],
            developer_data_available=True,
            publisher_data_available=True,
            company_data_available=True,
        )
        partial = GameCandidate(
            appid=3,
            title="Partial",
            developers=["Acme LLC"],
            publishers=["Elsewhere Ltd."],
            developer_data_available=True,
            publisher_data_available=True,
            company_data_available=True,
        )
        preferred = company("Acme", role="developer", strength="preferred")
        strong = company("Wanted Publisher", role="publisher", strength="strong")

        self.assertEqual(self.module.company_preference_adjustment(unknown, [strong]), -2.0)
        self.assertEqual(self.module.company_preference_adjustment(empty, [preferred]), -5.0)
        self.assertEqual(self.module.company_preference_adjustment(empty, [strong]), -10.0)
        self.assertEqual(
            self.module.company_preference_adjustment(partial, [preferred, strong]),
            -10.0,
        )

    def test_title_word_never_counts_as_company_match(self) -> None:
        candidate = GameCandidate(
            appid=10,
            title="Valve Puzzle Collection",
            developers=["Different Studio"],
            publishers=["Different Publisher"],
            developer_data_available=True,
            publisher_data_available=True,
            company_data_available=True,
        )

        self.assertEqual(
            self.module.company_preference_adjustment(
                candidate,
                [company("Valve", strength="strong")],
            ),
            -10.0,
        )

    def test_effective_score_includes_company_adjustment_exactly_once(self) -> None:
        breakdown = ScoreBreakdown(
            relevance_tier="A",
            layer_score=0.50,
            language_adjustment=-2,
            budget_adjustment=5,
            company_adjustment=-10,
        )

        self.assertEqual(effective_score(breakdown), 43.0)

    def test_ranker_applies_company_adjustment_within_the_existing_tier(self) -> None:
        preference = GamePreference(
            company_preferences=[
                company("Acme", role="developer", strength="preferred")
            ]
        )
        shared = {
            "app_type": "game",
            "review_total": 1_000,
            "review_positive_ratio": 0.90,
        }
        ranked = rank_entries(
            [
                GameCandidate(
                    appid=1,
                    title="Mismatch",
                    developers=["Elsewhere LLC"],
                    developer_data_available=True,
                    publisher_data_available=True,
                    company_data_available=True,
                    **shared,
                ),
                GameCandidate(appid=2, title="Unknown", **shared),
                GameCandidate(
                    appid=3,
                    title="Match",
                    developers=["Acme, Inc."],
                    developer_data_available=True,
                    publisher_data_available=True,
                    company_data_available=True,
                    **shared,
                ),
            ],
            preference,
        )

        self.assertEqual([game.title for game in ranked], ["Match", "Unknown", "Mismatch"])
        self.assertEqual(
            [game.score_breakdown.company_adjustment for game in ranked],
            [0.0, -2.0, -5.0],
        )
        self.assertEqual({game.score_breakdown.relevance_tier for game in ranked}, {"broad"})
        evidence_by_title = {
            game.title: [
                item
                for item in game.recommendation_evidence
                if item.evidence_id.startswith("company_preference:")
            ][0]
            for game in ranked
        }
        self.assertEqual(evidence_by_title["Unknown"].sentiment, "uncertain")
        self.assertTrue(evidence_by_title["Unknown"].important)
        self.assertEqual(evidence_by_title["Mismatch"].sentiment, "negative")


if __name__ == "__main__":
    unittest.main()
