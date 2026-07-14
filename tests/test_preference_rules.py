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

    def test_text_languages_override_llm_language_hallucinations(self) -> None:
        absent = merge_text_preference(
            GamePreference(
                preferred_languages=["japanese"],
                required_languages=["english"],
            ),
            "推荐合作解谜游戏",
        )
        explicit = merge_text_preference(
            GamePreference(preferred_languages=["japanese"]),
            "最好支持简体中文",
        )

        self.assertEqual(absent.preferred_languages, [])
        self.assertEqual(absent.required_languages, [])
        self.assertEqual(explicit.preferred_languages, ["schinese"])
        self.assertEqual(explicit.required_languages, [])

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

    def test_marks_only_budget_clause_with_hard_marker_as_required(self) -> None:
        unrelated = infer_preference_from_text("必须支持本地合作，预算 100 元")
        required = infer_preference_from_text("合作游戏，预算必须低于 100 元")
        required_english = infer_preference_from_text(
            "Co-op games where the price must be under $30"
        )

        self.assertFalse(unrelated.budget_is_required)
        self.assertTrue(required.budget_is_required)
        self.assertTrue(required_english.budget_is_required)

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

    def test_reference_title_does_not_hardcode_gameplay_tags(self) -> None:
        preference = infer_preference_from_text("类似黑暗之魂的游戏")

        self.assertEqual(preference.reference_games_like, ["黑暗之魂"])
        self.assertNotIn(
            "soulslike",
            [*preference.required_tags, *preference.genres_like, *preference.extra_tags],
        )

        unrelated = infer_preference_from_text("类似Souls Harbor的游戏")

        self.assertEqual(unrelated.reference_games_like, ["Souls Harbor"])
        self.assertNotIn(
            "soulslike",
            [*unrelated.required_tags, *unrelated.genres_like, *unrelated.extra_tags],
        )

    def test_singleplayer_aaa_request_sets_intents_without_fabricated_genres(self) -> None:
        preference = infer_preference_from_text("推荐的单机3a大作")

        self.assertEqual(preference.genres_like, ["singleplayer"])
        self.assertEqual(preference.extra_tags, [])
        self.assertEqual(preference.quality_intent, "mainstream")
        self.assertFalse(preference.allow_unreleased)
        self.assertEqual(preference.result_count, 5)

    def test_normal_query_keeps_default_quality_and_release_policy(self) -> None:
        preference = infer_preference_from_text("推荐轻松解谜游戏")

        self.assertEqual(preference.quality_intent, "normal")
        self.assertFalse(preference.allow_unreleased)

    def test_explicit_upcoming_query_allows_unreleased_games(self) -> None:
        cases = (
            "推荐尚未发售的单人游戏",
            "show me upcoming single-player games",
            "推荐 coming-soon games",
        )

        for text in cases:
            with self.subTest(text=text):
                preference = infer_preference_from_text(text)
                self.assertTrue(preference.allow_unreleased)

    def test_singleplayer_terms_share_one_canonical_positive_tag(self) -> None:
        terms = ("单机游戏", "单人游戏", "singleplayer game", "single-player game", "纯单人")
        for text in terms:
            with self.subTest(text=text):
                preference = infer_preference_from_text(text)
                self.assertIn("singleplayer", preference.genres_like)
                self.assertNotIn("singleplayer", preference.genres_dislike)

    def test_text_rules_remove_llm_fabricated_aaa_expansion(self) -> None:
        merged = merge_text_preference(
            GamePreference(
                genres_like=["action", "adventure", "rpg"],
                extra_tags=["aaa", "story rich", "open world"],
            ),
            "推荐的单机3a大作",
        )

        self.assertEqual(merged.genres_like, ["singleplayer"])
        self.assertEqual(merged.extra_tags, [])
        self.assertEqual(merged.quality_intent, "mainstream")

    def test_mainstream_removes_any_gameplay_tag_without_text_evidence(self) -> None:
        merged = merge_text_preference(
            GamePreference(
                required_tags=["strategy"],
                genres_like=["simulation", "shooter", "singleplayer"],
                extra_tags=["deckbuilding", "colony sim"],
            ),
            "推荐的单机3a大作",
        )

        self.assertEqual(merged.required_tags, [])
        self.assertEqual(merged.genres_like, ["singleplayer"])
        self.assertEqual(merged.extra_tags, [])

    def test_mainstream_keeps_only_gameplay_tags_explicit_in_the_query(self) -> None:
        merged = merge_text_preference(
            GamePreference(
                genres_like=["strategy", "simulation", "shooter"],
                extra_tags=["deckbuilding"],
            ),
            "推荐单机策略模拟 AAA 大作",
        )

        self.assertEqual(
            set(merged.genres_like),
            {"singleplayer", "strategy", "simulation"},
        )
        self.assertEqual(merged.extra_tags, [])

    def test_mainstream_accepts_cross_language_tags_with_verified_source_spans(self) -> None:
        cases = (
            ("推荐撤离射击 AAA 大作", "extraction_shooter", "撤离射击"),
            ("银河城 AAA", "metroidvania", "银河城"),
            ("牌组构筑 AAA", "deckbuilding", "牌组构筑"),
            ("殖民模拟 3A", "colony_sim", "殖民模拟"),
        )
        for text, tag, span in cases:
            with self.subTest(text=text, tag=tag):
                merged = merge_text_preference(
                    GamePreference(
                        genres_like=[tag, "action", "rpg", "open_world"],
                        explicit_tag_evidence=[
                            {"target": "genres_like", "tag": tag, "span": span}
                        ],
                    ),
                    text,
                )

                self.assertIn(tag, merged.genres_like)
                self.assertFalse(
                    {"action", "rpg", "open_world"} & set(merged.genres_like)
                )

    def test_mainstream_rejects_unverifiable_or_misassigned_tag_evidence(self) -> None:
        cases = (
            {"target": "genres_like", "tag": "action", "span": "AAA"},
            {"target": "genres_like", "tag": "action", "span": "不存在"},
            {"target": "extra_tags", "tag": "action", "span": "未知玩法"},
            {"target": "genres_like", "tag": "action", "span": "单机"},
        )
        for evidence in cases:
            with self.subTest(evidence=evidence):
                merged = merge_text_preference(
                    GamePreference(
                        genres_like=["action"],
                        explicit_tag_evidence=[evidence],
                    ),
                    "推荐单机未知玩法 AAA 大作",
                )

                self.assertNotIn("action", merged.genres_like)

    def test_mainstream_evidence_respects_required_negative_and_reference_context(self) -> None:
        required = merge_text_preference(
            GamePreference(
                required_tags=["precision_duel"],
                explicit_tag_evidence=[
                    {
                        "target": "required_tags",
                        "tag": "precision_duel",
                        "span": "精确对决",
                    }
                ],
            ),
            "必须精确对决的 AAA 大作",
        )
        optional = merge_text_preference(
            GamePreference(
                required_tags=["precision_duel"],
                explicit_tag_evidence=[
                    {
                        "target": "required_tags",
                        "tag": "precision_duel",
                        "span": "精确对决",
                    }
                ],
            ),
            "想玩精确对决的 AAA 大作",
        )
        negative = merge_text_preference(
            GamePreference(
                genres_like=["precision_duel"],
                explicit_tag_evidence=[
                    {
                        "target": "genres_like",
                        "tag": "precision_duel",
                        "span": "精确对决",
                    }
                ],
            ),
            "想玩精确对决，但不要精确对决的 AAA 大作",
        )
        reference = merge_text_preference(
            GamePreference(
                genres_like=["translated_gameplay"],
                reference_games_like=["未知玩法"],
                explicit_tag_evidence=[
                    {
                        "target": "genres_like",
                        "tag": "translated_gameplay",
                        "span": "未知玩法",
                    }
                ],
            ),
            "想玩类似未知玩法的 AAA 大作",
        )

        self.assertEqual(required.required_tags, ["precision_duel"])
        self.assertEqual(optional.required_tags, [])
        self.assertNotIn("precision_duel", negative.genres_like)
        self.assertNotIn("translated_gameplay", reference.genres_like)

    def test_mainstream_masks_negative_tags_and_reference_title_words(self) -> None:
        negative = merge_text_preference(
            GamePreference(genres_like=["strategy"]),
            "不要策略的 AAA 大作",
        )
        action_title = merge_text_preference(
            GamePreference(
                genres_like=["action"],
                reference_games_like=["Action Henk"],
                reference_search_terms=["Action Henk"],
            ),
            "/gamerec 类似 Action Henk 的 AAA 大作",
        )
        rpg_title = merge_text_preference(
            GamePreference(
                genres_like=["rpg", "action"],
                reference_games_like=["RPG Maker"],
                reference_search_terms=["RPG Maker"],
            ),
            "/gamerec 类似 RPG Maker 的 AAA 大作",
        )
        unclean_llm_title = merge_text_preference(
            GamePreference(
                reference_games_like=["RPG Maker 的 AAA 大作"],
                reference_search_terms=["RPG Maker"],
            ),
            "/gamerec 类似 RPG Maker 的 AAA 大作",
        )

        self.assertNotIn("strategy", negative.genres_like)
        self.assertIn("strategy", negative.genres_dislike)
        self.assertNotIn("action", action_title.genres_like)
        self.assertEqual(action_title.reference_games_like, ["Action Henk"])
        self.assertNotIn("rpg", rpg_title.genres_like)
        self.assertEqual(rpg_title.reference_games_like, ["RPG Maker"])
        self.assertEqual(unclean_llm_title.reference_games_like, ["RPG Maker"])

    def test_deterministic_reference_extraction_blocks_fabricated_evidence(self) -> None:
        merged = merge_text_preference(
            GamePreference(
                genres_like=["fabricated_anchor"],
                explicit_tag_evidence=[
                    {
                        "target": "genres_like",
                        "tag": "fabricated_anchor",
                        "span": "灰区战争",
                    }
                ],
            ),
            "类似灰区战争，推荐 AAA 大作",
        )

        self.assertEqual(merged.reference_games_like, ["灰区战争"])
        self.assertNotIn("fabricated_anchor", merged.genres_like)

    def test_required_evidence_uses_the_nearest_requirement_scope(self) -> None:
        preference = GamePreference(
            required_tags=["precision_duel"],
            explicit_tag_evidence=[
                {
                    "target": "required_tags",
                    "tag": "precision_duel",
                    "span": "精确对决",
                }
            ],
        )

        unrelated = merge_text_preference(
            preference,
            "必须高质量的精确对决 AAA 大作",
        )
        direct = merge_text_preference(
            preference,
            "必须精确对决的 AAA 大作",
        )

        self.assertEqual(unrelated.required_tags, [])
        self.assertEqual(direct.required_tags, ["precision_duel"])

    def test_mainstream_keeps_same_language_explicit_tag_before_vocab_load(self) -> None:
        merged = merge_text_preference(
            GamePreference(genres_like=["precision_platformer", "open_world"]),
            "推荐 precision platformer AAA 大作",
        )

        self.assertIn("precision_platformer", merged.genres_like)
        self.assertNotIn("open_world", merged.genres_like)

    def test_quality_words_never_become_gameplay_tags(self) -> None:
        cases = (
            ("推荐 AAA 大作", GamePreference(extra_tags=["aaa"]), "aaa"),
            ("推荐 3A 大作", GamePreference(genres_like=["3a"]), "3a"),
            (
                "recommend high quality AAA games",
                GamePreference(genres_like=["quality"]),
                "quality",
            ),
        )
        for text, preference, forbidden in cases:
            with self.subTest(text=text):
                merged = merge_text_preference(preference, text)
                self.assertNotIn(
                    forbidden,
                    [
                        *merged.required_tags,
                        *merged.genres_like,
                        *merged.extra_tags,
                    ],
                )

    def test_required_evidence_does_not_cross_parallel_intent_boundaries(self) -> None:
        preference = GamePreference(
            required_tags=["precision_duel"],
            explicit_tag_evidence=[
                {
                    "target": "required_tags",
                    "tag": "precision_duel",
                    "span": "精确对决",
                }
            ],
        )
        cases = (
            "必须支持中文并想玩精确对决的 AAA 大作",
            "必须高画质同时想玩精确对决 AAA 大作",
        )

        for text in cases:
            with self.subTest(text=text):
                merged = merge_text_preference(preference, text)
                self.assertEqual(merged.required_tags, [])

    def test_normal_reference_title_tags_are_demoted_from_explicit_anchors(self) -> None:
        merged = merge_text_preference(
            GamePreference(
                reference_games_like=["Action Henk"],
                reference_search_terms=["Action Henk"],
                genres_like=["action"],
            ),
            "/gamerec 类似 Action Henk 的游戏",
        )
        alias_only = merge_text_preference(
            GamePreference(
                reference_games_like=["动作亨克"],
                reference_search_terms=["Action Henk"],
                genres_like=["action"],
            ),
            "/gamerec 围绕 Action Henk 找游戏",
        )

        self.assertNotIn("action", merged.genres_like)
        self.assertIn("action", merged.extra_tags)
        self.assertNotIn("action", alias_only.genres_like)

    def test_negative_evidence_never_demotes_into_positive_supporting_tags(self) -> None:
        cases = (
            ("不要精确对决游戏", "precision_duel", "精确对决"),
            ("不要撤离射击游戏", "extraction_shooter", "撤离射击"),
        )
        for text, tag, span in cases:
            with self.subTest(text=text):
                merged = merge_text_preference(
                    GamePreference(
                        genres_like=[tag],
                        explicit_tag_evidence=[
                            {"target": "genres_like", "tag": tag, "span": span}
                        ],
                    ),
                    text,
                )

                self.assertNotIn(tag, merged.genres_like)
                self.assertNotIn(tag, merged.extra_tags)
                self.assertIn(tag, merged.genres_dislike)

    def test_exclusion_tags_require_negative_source_evidence(self) -> None:
        hallucinated = merge_text_preference(
            GamePreference(
                genres_like=["action"],
                genres_dislike=["soulslike"],
            ),
            "想玩动作游戏",
        )
        same_language = merge_text_preference(
            GamePreference(genres_dislike=["soulslike"]),
            "不要 soulslike 游戏",
        )
        translated = merge_text_preference(
            GamePreference(
                genres_dislike=["extraction_shooter"],
                explicit_tag_evidence=[
                    {
                        "target": "genres_dislike",
                        "tag": "extraction_shooter",
                        "span": "撤离射击",
                    }
                ],
            ),
            "不要撤离射击游戏",
        )

        self.assertNotIn("soulslike", hallucinated.genres_dislike)
        self.assertIn("soulslike", same_language.genres_dislike)
        self.assertIn("extraction_shooter", translated.genres_dislike)

    def test_last_evidence_polarity_wins_across_aliases_and_spans(self) -> None:
        translated = merge_text_preference(
            GamePreference(
                genres_like=["precision_duel"],
                explicit_tag_evidence=[
                    {
                        "target": "genres_like",
                        "tag": "precision_duel",
                        "span": "精确对决",
                    },
                    {
                        "target": "genres_like",
                        "tag": "precision_duel",
                        "span": "精准交锋",
                    },
                ],
            ),
            "不要精确对决，但想要精准交锋游戏",
        )
        same_language = merge_text_preference(
            GamePreference(genres_like=["动作", "action"]),
            "不要动作，但想要 action 游戏",
        )

        self.assertIn("precision_duel", translated.genres_like)
        self.assertNotIn("precision_duel", translated.genres_dislike)
        self.assertNotIn("precision_duel", translated.extra_tags)
        self.assertIn("action", same_language.genres_like)
        self.assertNotIn("action", same_language.genres_dislike)

    def test_last_negative_evidence_wins_across_translated_spans(self) -> None:
        merged = merge_text_preference(
            GamePreference(
                genres_like=["precision_duel"],
                explicit_tag_evidence=[
                    {
                        "target": "genres_like",
                        "tag": "precision_duel",
                        "span": "精准交锋",
                    },
                    {
                        "target": "genres_like",
                        "tag": "precision_duel",
                        "span": "精确对决",
                    },
                ],
            ),
            "想要精准交锋，但不要精确对决游戏",
        )

        self.assertNotIn("precision_duel", merged.genres_like)
        self.assertNotIn("precision_duel", merged.extra_tags)
        self.assertIn("precision_duel", merged.genres_dislike)

    def test_merge_preserves_validated_llm_intents_unknown_to_phrase_rules(self) -> None:
        merged = merge_text_preference(
            GamePreference(quality_intent="mainstream", allow_unreleased=True),
            "推荐几款待发售的精品单人游戏",
        )

        self.assertEqual(merged.quality_intent, "mainstream")
        self.assertTrue(merged.allow_unreleased)

    def test_deterministic_positive_intents_override_llm_defaults(self) -> None:
        merged = merge_text_preference(
            GamePreference(quality_intent="normal", allow_unreleased=False),
            "推荐即将发售的 AAA 游戏",
        )
        normal = merge_text_preference(GamePreference(), "推荐轻松解谜游戏")

        self.assertEqual(merged.quality_intent, "mainstream")
        self.assertTrue(merged.allow_unreleased)
        self.assertEqual(normal.quality_intent, "normal")
        self.assertFalse(normal.allow_unreleased)

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
