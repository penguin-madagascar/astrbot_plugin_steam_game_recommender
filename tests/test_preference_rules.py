from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.preference_rules import (
    extract_reference_games,
    infer_preference_from_text,
    merge_text_preference,
)
from astrbot_plugin_steam_game_recommender.storage.models import GamePreference


class PreferenceRulesTest(unittest.TestCase):
    def test_library_exclusion_prefix_does_not_negate_requested_gameplay(self) -> None:
        preference = infer_preference_from_text("排除已有的合作游戏")

        self.assertIn("co-op", preference.genres_like)
        self.assertNotIn("co_op", preference.genres_dislike)
        self.assertEqual(preference.library_filter_mode, "exclude_owned")

    def test_positive_horror_and_soulslike_intent_is_not_treated_as_exclusion(self) -> None:
        cases = (
            ("想玩恐怖合作游戏", "horror"),
            ("想玩魂类动作游戏", "soulslike"),
        )

        for text, tag in cases:
            with self.subTest(text=text):
                preference = infer_preference_from_text(text)
                self.assertIn(tag, [*preference.genres_like, *preference.extra_tags])
                self.assertNotIn(tag, preference.genres_dislike)

    def test_explicit_negative_horror_and_soulslike_intent_is_excluded(self) -> None:
        cases = (
            ("推荐合作游戏，不要恐怖", "horror"),
            ("别推荐魂类游戏", "soulslike"),
        )

        for text, tag in cases:
            with self.subTest(text=text):
                preference = infer_preference_from_text(text)
                self.assertIn(tag, preference.genres_dislike)
                self.assertNotIn(tag, [*preference.genres_like, *preference.extra_tags])

    def test_latest_explicit_text_polarity_overrides_llm_conflicts(self) -> None:
        negative = merge_text_preference(
            GamePreference(genres_like=["horror"], extra_tags=["horror"]),
            "这次不要恐怖游戏",
        )
        positive = merge_text_preference(
            GamePreference(genres_dislike=["horror"]),
            "之前不要恐怖，但现在想玩恐怖游戏",
        )
        chinese_alias = merge_text_preference(
            GamePreference(genres_like=["魂类"], extra_tags=["类魂"]),
            "不要魂类",
        )

        self.assertIn("horror", negative.genres_dislike)
        self.assertNotIn("horror", [*negative.genres_like, *negative.extra_tags])
        self.assertIn("horror", [*positive.genres_like, *positive.extra_tags])
        self.assertNotIn("horror", positive.genres_dislike)
        self.assertNotIn("魂类", chinese_alias.genres_like)
        self.assertNotIn("类魂", chinese_alias.extra_tags)
        self.assertIn("soulslike", chinese_alias.genres_dislike)

    def test_latest_polarity_wins_without_punctuation(self) -> None:
        preference = infer_preference_from_text("不要恐怖但现在想玩恐怖，也不要魂类但改成想玩魂类")

        self.assertIn("horror", [*preference.genres_like, *preference.extra_tags])
        self.assertNotIn("horror", preference.genres_dislike)
        self.assertIn("soulslike", [*preference.genres_like, *preference.extra_tags])
        self.assertNotIn("soulslike", preference.genres_dislike)

    def test_extracts_explicit_hard_requirements_as_required_tags(self) -> None:
        preference = infer_preference_from_text(
            "必须支持中文和本地合作，而且一定要多人联机，最好轻松一点"
        )

        self.assertEqual(
            preference.required_tags,
            ["local_coop", "multiplayer"],
        )
        self.assertEqual(preference.required_languages, ["schinese"])
        self.assertNotIn("relaxing", preference.required_tags)
        self.assertIn("relaxing", preference.extra_tags)

    def test_all_recognized_tags_can_be_explicit_hard_requirements(self) -> None:
        preference = infer_preference_from_text("必须恐怖，而且一定要魂类动作")

        self.assertEqual(
            preference.required_tags,
            ["horror", "soulslike", "action"],
        )

    def test_hard_requirement_scope_stops_before_soft_gameplay_description(self) -> None:
        preference = infer_preference_from_text("必须支持中文的合作解谜游戏")

        self.assertEqual(preference.required_tags, [])
        self.assertEqual(preference.required_languages, ["schinese"])
        self.assertIn("co-op", preference.genres_like)
        self.assertIn("puzzle", preference.genres_like)

        contrasted = infer_preference_from_text("必须中文但合作解谜只要轻松就好")

        self.assertEqual(contrasted.required_tags, ["relaxing"])
        self.assertEqual(contrasted.required_languages, ["schinese"])

    def test_local_coop_requires_explicit_local_semantics(self) -> None:
        required = infer_preference_from_text("必须本地双人合作")
        soft = infer_preference_from_text("想找双人合作游戏")

        self.assertEqual(required.required_tags, ["local_coop"])
        self.assertNotIn("local co-op", soft.genres_like)

    def test_separates_positive_and_negative_reference_games(self) -> None:
        preference = infer_preference_from_text(
            "想找类似星露谷物语的游戏，不要像黑暗之魂，也不喜欢杀戮尖塔这类游戏"
        )

        self.assertEqual(preference.reference_games_like, ["星露谷物语"])
        self.assertEqual(
            preference.reference_games_dislike,
            ["黑暗之魂", "杀戮尖塔"],
        )

    def test_extracts_explicit_positive_reference_wording(self) -> None:
        preference = infer_preference_from_text("喜欢双人成行，不喜欢胡闹厨房这类游戏")

        self.assertEqual(preference.reference_games_like, ["双人成行"])
        self.assertEqual(preference.reference_games_dislike, ["胡闹厨房"])

    def test_english_negative_reference_does_not_leak_into_positive_references(self) -> None:
        preference = infer_preference_from_text("I dislike Dark Souls")

        self.assertEqual(preference.reference_games_like, [])
        self.assertEqual(preference.reference_games_dislike, ["Dark Souls"])
        self.assertNotIn("soulslike", preference.genres_like)

    def test_preserves_chinese_titles_containing_de_and_matches_xiangshi(self) -> None:
        plain = infer_preference_from_text("类似我的世界的游戏")
        bracketed = infer_preference_from_text("类似《我的世界》的游戏")
        xiangshi = infer_preference_from_text("像是双人成行")

        self.assertEqual(plain.reference_games_like, ["我的世界"])
        self.assertEqual(bracketed.reference_games_like, ["我的世界"])
        self.assertEqual(xiangshi.reference_games_like, ["双人成行"])

    def test_plain_tag_exclusions_are_not_misclassified_as_reference_games(self) -> None:
        preference = infer_preference_from_text("不要恐怖，不要魂类，不要高难，也不要双人合作")

        self.assertEqual(preference.reference_games_dislike, [])

    def test_explicit_negative_tag_wins_over_positive_reference_expansion(self) -> None:
        preference = infer_preference_from_text("类似黑暗之魂的氛围，但不要魂类战斗")

        self.assertEqual(preference.reference_games_like, ["黑暗之魂"])
        self.assertIn("soulslike", preference.genres_dislike)
        self.assertNotIn("soulslike", [*preference.genres_like, *preference.extra_tags])

    def test_explicit_negative_reference_overrides_llm_positive_reference(self) -> None:
        merged = merge_text_preference(
            GamePreference(reference_games_like=["黑暗之魂"]),
            "不要像黑暗之魂",
        )

        self.assertNotIn("黑暗之魂", merged.reference_games_like)
        self.assertIn("黑暗之魂", merged.reference_games_dislike)

    def test_infers_steampeek_profile_terms_from_user_text(self) -> None:
        preference = infer_preference_from_text(
            "推荐几个适合 Switch 和 Steam 的双人游戏，不要恐怖，"
            "最好支持中文，预算 100 以内，类似双人成行但别太难。"
        )

        self.assertIn("nintendo switch", preference.platforms)
        self.assertIn("steam", preference.platforms)
        self.assertEqual(preference.players, 2)
        self.assertEqual(preference.budget, 100)
        self.assertEqual(preference.preferred_languages, ["schinese"])
        self.assertEqual(preference.difficulty, "easy")
        self.assertIn("horror", preference.genres_dislike)
        self.assertIn("双人成行", preference.reference_games_like)
        self.assertIn("co-op", preference.genres_like)
        self.assertIn("relaxing", preference.extra_tags)
        self.assertNotIn("platformer", preference.genres_like)

    def test_merges_llm_extra_tags_with_keyword_rules(self) -> None:
        llm_preference = GamePreference(
            platforms=[],
            genres_like=["puzzle"],
            extra_tags=["剧情合作"],
            genres_dislike=[],
            reference_games_like=[],
            players=None,
            budget=None,
            preferred_languages=[],
            difficulty=None,
            result_count=5,
        )

        merged = merge_text_preference(
            llm_preference,
            "推荐几个适合 Steam 的双人游戏，不要恐怖，"
            "最好支持中文，预算 100 以内，类似双人成行但别太难。",
        )

        self.assertEqual(merged.platforms, ["steam"])
        self.assertEqual(merged.players, 2)
        self.assertEqual(merged.budget, 100)
        self.assertEqual(merged.preferred_languages, ["schinese"])
        self.assertEqual(merged.difficulty, "easy")
        self.assertIn("horror", merged.genres_dislike)
        self.assertIn("双人成行", merged.reference_games_like)
        self.assertIn("剧情合作", merged.extra_tags)
        self.assertIn("relaxing", merged.extra_tags)

    def test_text_platforms_override_llm_platform_hallucinations(self) -> None:
        llm_preference = GamePreference(
            platforms=["steam", "playstation", "nintendo switch"],
            result_count=5,
        )

        merged = merge_text_preference(
            llm_preference,
            "推荐几个适合 Steam 的双人游戏，类似双人成行。",
        )

        self.assertEqual(merged.platforms, ["steam"])

    def test_pc_and_steam_are_distinct_platform_preferences(self) -> None:
        pc_preference = infer_preference_from_text("我想找 PC 上玩的合作射击游戏")
        steam_preference = infer_preference_from_text("Steam 上有没有双人合作游戏")

        self.assertIn("pc", pc_preference.platforms)
        self.assertNotIn("steam", pc_preference.platforms)
        self.assertIn("steam", steam_preference.platforms)

    def test_simplified_and_traditional_chinese_preferences_are_distinct(self) -> None:
        simplified = infer_preference_from_text("最好支持简体中文")
        traditional = infer_preference_from_text("最好支持繁体中文")

        self.assertEqual(simplified.preferred_languages, ["schinese"])
        self.assertEqual(traditional.preferred_languages, ["tchinese"])

    def test_explicit_language_requirement_is_not_a_tag_requirement(self) -> None:
        preference = infer_preference_from_text("必须支持繁体中文，想玩解谜")

        self.assertEqual(preference.required_languages, ["tchinese"])
        self.assertNotIn("chinese", preference.required_tags)

    def test_extracts_explicit_budget_currency_without_guessing_implicit_currency(self) -> None:
        usd = infer_preference_from_text("美区合作游戏，预算 $30")
        jpy = infer_preference_from_text("日区解谜，3000 日元以内")
        implicit = infer_preference_from_text("预算 100 以内")

        self.assertEqual((usd.budget, usd.budget_currency), (30, "USD"))
        self.assertEqual((jpy.budget, jpy.budget_currency), (3000, "JPY"))
        self.assertEqual((implicit.budget, implicit.budget_currency), (100, None))

    def test_explicit_text_count_overrides_llm_default_count(self) -> None:
        llm_preference = GamePreference(result_count=5)

        merged = merge_text_preference(
            llm_preference,
            "想找 3 款 PC/Steam 上可以和朋友线上合作的轻松解谜游戏",
        )

        self.assertEqual(merged.result_count, 3)

    def test_reference_title_extraction_is_generic(self) -> None:
        self.assertEqual(
            extract_reference_games("想找类似星露谷物语的多人种田经营游戏"),
            ["星露谷物语"],
        )
        self.assertEqual(
            extract_reference_games("Steam Deck 上找短局卡牌策略，similar to Slay the Spire"),
            ["Slay the Spire"],
        )

    def test_dark_souls_like_request_extracts_searchable_soulslike_profile(self) -> None:
        preference = infer_preference_from_text("类似黑暗之魂的游戏")

        self.assertEqual(preference.reference_games_like, ["黑暗之魂"])
        self.assertIn("soulslike", preference.extra_tags)
        self.assertIn("action", preference.extra_tags)
        self.assertIn("rpg", preference.extra_tags)

    def test_aaa_request_extracts_broad_blockbuster_profile(self) -> None:
        preference = infer_preference_from_text("推荐一下3a游戏")

        self.assertIn("action", preference.genres_like)
        self.assertIn("adventure", preference.genres_like)
        self.assertIn("rpg", preference.genres_like)
        self.assertIn("aaa", preference.extra_tags)
        self.assertIn("story rich", preference.extra_tags)
        self.assertIn("open world", preference.extra_tags)
        self.assertEqual(preference.result_count, 5)

    def test_infers_library_filter_mode_from_text(self) -> None:
        self.assertEqual(
            infer_preference_from_text("推荐几个合作游戏，排除已有").library_filter_mode,
            "exclude_owned",
        )
        self.assertEqual(
            infer_preference_from_text("recommend co-op games only-owned").library_filter_mode,
            "only_owned",
        )

    def test_merges_llm_library_filter_mode(self) -> None:
        merged = merge_text_preference(
            GamePreference(library_filter_mode="exclude_owned"),
            "推荐几个 Steam 合作游戏",
        )

        self.assertEqual(merged.library_filter_mode, "exclude_owned")


if __name__ == "__main__":
    unittest.main()
