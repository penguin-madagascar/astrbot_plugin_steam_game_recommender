from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services import recommendation_intent
from astrbot_plugin_steam_game_recommender.services.preference_rules import (
    infer_preference_from_text,
    merge_text_preference,
)
from astrbot_plugin_steam_game_recommender.services.retry_command import (
    merge_retry_preferences,
)
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    supporting_query_weights,
)
from astrbot_plugin_steam_game_recommender.services.steam_recall import (
    select_recall_seeds,
)
from astrbot_plugin_steam_game_recommender.services.steam_index import SteamGameIndexService
from astrbot_plugin_steam_game_recommender.services.tag_normalizer import (
    register_canonical_tag_keys,
)
from astrbot_plugin_steam_game_recommender.storage import models


class SemanticPreferenceValidationTest(unittest.TestCase):
    def test_semantic_structures_cap_entries_and_reject_unknown_tags(self) -> None:
        derived_type = getattr(models, "DerivedIntentTag", None)
        feature_type = getattr(models, "SoftFeature", None)
        company_type = getattr(models, "CompanyPreference", None)
        self.assertIsNotNone(derived_type)
        self.assertIsNotNone(feature_type)
        self.assertIsNotNone(company_type)

        preference = models.GamePreference(
            derived_intent_tags=[
                {"tag": "choices_matter", "source_span": "选择会改变剧情"},
                {"tag": "story_rich", "source_span": "重剧情"},
                {"tag": "puzzle", "source_span": "解谜"},
                {"tag": "action", "source_span": "动作"},
                {"tag": "invented_free_form", "source_span": "随便编的"},
            ],
            soft_features=[
                {
                    "constraint_id": f"feature-{index}",
                    "source_span": f"特征{index}",
                    "normalized_text": f"feature {index}",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["puzzle"],
                }
                for index in range(4)
            ],
            company_preferences=[
                {
                    "display_name": f"Company {index}",
                    "aliases": [f"Company {index}"],
                    "role": "either",
                    "strength": "preferred",
                    "source_span": f"Company {index}",
                }
                for index in range(4)
            ],
        )

        self.assertEqual(
            [item.tag for item in preference.derived_intent_tags],
            ["choices_matter", "story_rich", "puzzle"],
        )
        self.assertTrue(all(item.weight == 0.25 for item in preference.derived_intent_tags))
        self.assertEqual(len(preference.soft_features), 3)
        self.assertEqual(len(preference.company_preferences), 3)

    def test_merge_keeps_only_verbatim_unblocked_spans(self) -> None:
        query = (
            "想玩选择会改变剧情的游戏，类似《Reference Quest》，Steam，"
            "预算100元，推荐3款，偏好Acme Games开发的作品"
        )
        preference = models.GamePreference(
            reference_games_like=["Reference Quest"],
            platforms=["steam"],
            budget=100,
            result_count=3,
            derived_intent_tags=[
                {"tag": "choices_matter", "source_span": "选择会改变剧情"},
                {"tag": "rpg", "source_span": "Reference Quest"},
            ],
            soft_features=[
                {
                    "constraint_id": "valid-feature",
                    "source_span": "选择会改变剧情",
                    "normalized_text": "choices change the story",
                    "role": "core",
                    "polarity": "positive",
                    "proxy_tags": ["choices_matter"],
                },
                {
                    "constraint_id": "reference-feature",
                    "source_span": "Reference Quest",
                    "normalized_text": "reference title",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["rpg"],
                },
                {
                    "constraint_id": "platform-feature",
                    "source_span": "Steam",
                    "normalized_text": "platform",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["multiplayer"],
                },
                {
                    "constraint_id": "price-feature",
                    "source_span": "100元",
                    "normalized_text": "price",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["casual"],
                },
                {
                    "constraint_id": "quantity-feature",
                    "source_span": "3款",
                    "normalized_text": "quantity",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["action"],
                },
                {
                    "constraint_id": "company-feature",
                    "source_span": "Acme Games",
                    "normalized_text": "company",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["strategy"],
                },
            ],
            company_preferences=[
                {
                    "display_name": "Acme Games",
                    "aliases": ["Acme Games"],
                    "role": "developer",
                    "strength": "strong",
                    "source_span": "Acme Games",
                }
            ],
        )

        merged = merge_text_preference(preference, query)

        self.assertEqual(
            [item.tag for item in getattr(merged, "derived_intent_tags", [])],
            ["choices_matter"],
        )
        self.assertEqual(
            [item.constraint_id for item in getattr(merged, "soft_features", [])],
            ["valid-feature"],
        )
        self.assertEqual(
            [item.display_name for item in getattr(merged, "company_preferences", [])],
            ["Acme Games"],
        )

    def test_inferred_reference_titles_block_semantic_spans(self) -> None:
        preference = models.GamePreference(
            soft_features=[
                {
                    "constraint_id": "reference-as-feature",
                    "source_span": "Inferred Quest",
                    "normalized_text": "inferred reference title",
                    "role": "optional",
                    "polarity": "positive",
                },
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                },
            ]
        )

        merged = merge_text_preference(
            preference,
            "喜欢《Inferred Quest》，还想要分支剧情",
        )

        self.assertEqual(merged.reference_games_like, ["Inferred Quest"])
        self.assertEqual(
            [item.constraint_id for item in merged.soft_features],
            ["branching"],
        )

    def test_explicit_company_role_span_blocks_feature_when_payload_identity_is_invalid(self) -> None:
        preference = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Valve",
                    "role": "developer",
                    "source_span": "Acme",
                }
            ],
            soft_features=[
                {
                    "constraint_id": "company-as-feature",
                    "source_span": "Acme",
                    "normalized_text": "company name",
                    "role": "optional",
                    "polarity": "positive",
                }
            ],
        )

        merged = merge_text_preference(preference, "想玩 Acme 开发的游戏")

        self.assertEqual(merged.company_preferences, [])
        self.assertEqual(merged.soft_features, [])

    def test_company_role_context_blocks_semantic_spans_without_company_payload(self) -> None:
        preference = models.GamePreference(
            derived_intent_tags=[
                {"tag": "puzzle", "source_span": "Valve"},
            ],
            soft_features=[
                {
                    "constraint_id": "valve-mechanic",
                    "source_span": "Valve",
                    "normalized_text": "a mechanic named Valve",
                    "role": "optional",
                    "polarity": "positive",
                }
            ],
        )

        company_context = merge_text_preference(
            preference,
            "偏好 Valve 开发的游戏",
        )
        ordinary_context = merge_text_preference(
            preference,
            "关卡机制叫 Valve，想要解谜",
        )
        neighboring_company_context = merge_text_preference(
            models.GamePreference(
                soft_features=[
                    {
                        "constraint_id": "branching",
                        "source_span": "分支剧情",
                        "normalized_text": "branching story",
                        "role": "optional",
                        "polarity": "positive",
                    }
                ],
            ),
            "想要分支剧情，Acme 开发的游戏",
        )

        self.assertEqual(company_context.company_preferences, [])
        self.assertEqual(company_context.derived_intent_tags, [])
        self.assertEqual(company_context.soft_features, [])
        self.assertEqual(
            [item.tag for item in ordinary_context.derived_intent_tags],
            ["puzzle"],
        )
        self.assertEqual(
            [item.constraint_id for item in ordinary_context.soft_features],
            ["valve-mechanic"],
        )
        self.assertEqual(
            [item.constraint_id for item in neighboring_company_context.soft_features],
            ["branching"],
        )

    def test_ungrounded_raw_company_span_does_not_block_a_feature(self) -> None:
        preference = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Valve",
                    "role": "developer",
                    "source_span": "分支剧情",
                }
            ],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                }
            ],
        )

        merged = merge_text_preference(preference, "想要分支剧情")

        self.assertEqual(merged.company_preferences, [])
        self.assertEqual(
            [item.constraint_id for item in merged.soft_features],
            ["branching"],
        )

    def test_extra_tags_do_not_score_and_proxy_tags_are_recall_only(self) -> None:
        preference = models.GamePreference(
            extra_tags=["story_rich"],
            derived_intent_tags=[
                {"tag": "puzzle", "source_span": "解谜"},
            ],
            soft_features=[
                {
                    "constraint_id": "branching-story",
                    "source_span": "选择影响剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["choices_matter"],
                }
            ],
        )

        intent = recommendation_intent.build_recommendation_intent(preference)
        by_tag = {item.tag: item for item in intent.tags}
        recall_only = getattr(recommendation_intent.IntentTagRole, "RECALL_ONLY", None)

        self.assertNotIn("story_rich", by_tag)
        self.assertEqual(by_tag["puzzle"].weight, 0.25)
        self.assertEqual(by_tag["choices_matter"].role, recall_only)
        weights, _library = supporting_query_weights(intent, {})
        self.assertEqual(weights, {"puzzle": 0.25})
        self.assertIn("choices_matter", [seed.tag for seed in select_recall_seeds(intent)])

    def test_historical_global_registration_cannot_make_unknown_request_tags_score(self) -> None:
        raw = {
            "tag": "request_polluted_tag",
            "source_span": "unknown feature",
        }
        before = recommendation_intent.build_recommendation_intent(
            models.GamePreference(derived_intent_tags=[raw])
        )
        register_canonical_tag_keys(["request_polluted_tag"])
        after = recommendation_intent.build_recommendation_intent(
            models.GamePreference(derived_intent_tags=[raw])
        )

        self.assertNotIn("request_polluted_tag", {item.tag for item in before.tags})
        self.assertNotIn("request_polluted_tag", {item.tag for item in after.tags})

    def test_service_local_official_vocabulary_accepts_dynamic_derived_and_proxy_tags(self) -> None:
        service = SteamGameIndexService(object(), object(), clock=lambda: 100.0)
        service._steam_tag_ids = {"precision_platformer": 123}
        service._canonical_tag_by_id = {123: "precision_platformer"}
        service._tag_vocabulary_payloads["english"] = ((), 100.0)
        preference = models.GamePreference(
            derived_intent_tags=[
                {"tag": "precision_platformer", "source_span": "精准跳跃"}
            ],
            soft_features=[
                {
                    "constraint_id": "movement-tech",
                    "source_span": "高级移动技巧",
                    "normalized_text": "advanced movement techniques",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["precision_platformer"],
                }
            ],
        )

        intent = service._build_request_intent(preference)

        tag = next(item for item in intent.tags if item.tag == "precision_platformer")
        self.assertEqual(tag.role, recommendation_intent.IntentTagRole.SUPPORTING)
        self.assertEqual(tag.weight, 0.25)

    def test_loaded_localized_vocabulary_is_service_local_for_explicit_tags(self) -> None:
        def loaded_service(
            canonical_name: str,
            localized_name: str,
            tag_id: int,
        ) -> SteamGameIndexService:
            service = SteamGameIndexService(object(), object(), clock=lambda: 100.0)
            service._tag_vocabulary_payloads = {
                "english": (({"tagid": tag_id, "name": canonical_name},), 100.0),
                "schinese": (({"tagid": tag_id, "name": localized_name},), 100.0),
            }
            service._rebuild_tag_vocabulary()
            return service

        alpha = loaded_service("Alpha Dynamic", "阿尔法动态", 101)
        beta = loaded_service("Beta Dynamic", "贝塔动态", 202)

        alpha_intent = alpha._build_request_intent(
            models.GamePreference(genres_like=["阿尔法动态", "贝塔动态"])
        )
        beta_intent = beta._build_request_intent(
            models.GamePreference(genres_like=["贝塔动态"])
        )

        self.assertEqual([item.tag for item in alpha_intent.tags], ["alpha_dynamic"])
        self.assertEqual([item.tag for item in beta_intent.tags], ["beta_dynamic"])

    def test_core_feature_proxy_is_not_starved_by_optional_proxy_tags(self) -> None:
        preference = models.GamePreference(
            soft_features=[
                {
                    "constraint_id": "optional-style",
                    "source_span": "最好有额外风格",
                    "normalized_text": "optional style",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["action", "strategy", "rpg"],
                },
                {
                    "constraint_id": "core-puzzles",
                    "source_span": "核心必须是解谜",
                    "normalized_text": "puzzles are core",
                    "role": "core",
                    "polarity": "positive",
                    "proxy_tags": ["puzzle"],
                },
            ]
        )

        seeds = select_recall_seeds(
            recommendation_intent.build_recommendation_intent(preference)
        )

        self.assertEqual(seeds[0].tag, "puzzle")
        self.assertEqual([seed.tag for seed in seeds], ["puzzle", "action", "strategy"])

    def test_unknown_proxies_are_filtered_before_per_feature_cap(self) -> None:
        service = SteamGameIndexService(object(), object(), clock=lambda: 100.0)
        service._steam_tag_ids = {"puzzle": 1664}
        service._canonical_tag_by_id = {1664: "puzzle"}
        service._tag_vocabulary_payloads["english"] = ((), 100.0)
        preference = models.GamePreference(
            soft_features=[
                {
                    "constraint_id": "puzzle-mechanic",
                    "source_span": "解谜机制",
                    "normalized_text": "puzzle mechanic",
                    "role": "core",
                    "polarity": "positive",
                    "proxy_tags": [
                        "unknown_proxy_one",
                        "unknown_proxy_two",
                        "puzzle",
                    ],
                }
            ]
        )

        intent = service._build_request_intent(preference)

        self.assertEqual([item.tag for item in intent.tags], ["puzzle"])
        self.assertEqual(
            intent.tags[0].role,
            recommendation_intent.IntentTagRole.RECALL_ONLY,
        )

    def test_loaded_official_vocabulary_rejects_fabricated_explicit_anchor(self) -> None:
        service = SteamGameIndexService(object(), object(), clock=lambda: 100.0)
        service._steam_tag_ids = {"puzzle": 1664}
        service._canonical_tag_by_id = {1664: "puzzle"}
        service._tag_vocabulary_payloads["english"] = ((), 100.0)
        preference = models.GamePreference(
            genres_like=["puzzle", "fabricated precision duel"]
        )

        runtime_intent = service._build_request_intent(preference)
        cold_intent = recommendation_intent.build_recommendation_intent(preference)

        self.assertEqual(
            {item.tag for item in runtime_intent.tags},
            {"puzzle"},
        )
        self.assertIn(
            "fabricated_precision_duel",
            {item.tag for item in cold_intent.tags},
        )

    def test_verbatim_span_preserves_spaces_and_accepts_a_later_unblocked_occurrence(self) -> None:
        query = "类似《Puzzle》，但还想要choices  matter机制和Puzzle式关卡"
        preference = models.GamePreference(
            reference_games_like=["Puzzle"],
            derived_intent_tags=[
                {"tag": "choices_matter", "source_span": "choices  matter"}
            ],
            soft_features=[
                {
                    "constraint_id": "puzzle-mechanic",
                    "source_span": "Puzzle",
                    "normalized_text": "puzzle-like stages",
                    "role": "optional",
                    "polarity": "positive",
                    "proxy_tags": ["puzzle"],
                }
            ],
        )

        merged = merge_text_preference(preference, query)

        self.assertEqual(merged.derived_intent_tags[0].source_span, "choices  matter")
        self.assertEqual(merged.soft_features[0].source_span, "Puzzle")

    def test_unbracketed_reference_introducers_block_only_the_reference_occurrence(self) -> None:
        for query in (
            "我喜欢 Dark Souls",
            "我偏爱 Dark Souls",
            "我钟爱 Dark Souls",
            "想找像是 Dark Souls 的游戏",
            "想找接近 Dark Souls 的游戏",
        ):
            with self.subTest(query=query):
                preference = models.GamePreference(
                    reference_games_like=["Dark Souls"],
                    derived_intent_tags=[
                        {"tag": "soulslike", "source_span": "Dark Souls"}
                    ],
                )

                merged = merge_text_preference(preference, query)

                self.assertEqual(merged.derived_intent_tags, [])

        repeated = merge_text_preference(
            models.GamePreference(
                reference_games_like=["Dark Souls"],
                derived_intent_tags=[
                    {"tag": "soulslike", "source_span": "Dark Souls"}
                ],
            ),
            "我喜欢 Dark Souls，但后文把 Dark Souls 当作玩法描述",
        )
        self.assertEqual([item.tag for item in repeated.derived_intent_tags], ["soulslike"])

    def test_mainstream_never_infers_company(self) -> None:
        self.assertEqual(
            infer_preference_from_text("只想要一款热门3A大作").company_preferences,
            [],
        )

    def test_company_constraints_dedupe_by_normalized_name_and_role(self) -> None:
        preference = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Acme Ltd.",
                    "aliases": ["Acme"],
                    "role": "developer",
                    "strength": "preferred",
                    "source_span": "Acme Ltd.",
                },
                {
                    "display_name": "ACME",
                    "aliases": ["ACME"],
                    "role": "publisher",
                    "strength": "strong",
                    "source_span": "ACME",
                },
                {
                    "display_name": "Acme LLC",
                    "aliases": ["Acme"],
                    "role": "developer",
                    "strength": "strong",
                    "source_span": "Acme LLC",
                },
            ]
        )

        self.assertEqual(
            [(item.display_name, item.role) for item in preference.company_preferences],
            [("Acme Ltd.", "developer"), ("ACME", "publisher")],
        )
        self.assertEqual(preference.company_preferences[0].strength, "strong")

    def test_same_role_company_alias_overlap_merges_names_and_preserves_first_source(self) -> None:
        preference = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Ubisoft",
                    "aliases": ["育碧", "Ubisoft Entertainment"],
                    "role": "developer",
                    "strength": "preferred",
                    "source_span": "Ubisoft",
                },
                {
                    "display_name": "育碧",
                    "aliases": ["Ubisoft", "Ubisoft Montreal", "Ubi"],
                    "role": "developer",
                    "strength": "strong",
                    "source_span": "育碧",
                },
                {
                    "display_name": "育碧",
                    "aliases": ["Ubisoft"],
                    "role": "publisher",
                    "strength": "preferred",
                    "source_span": "育碧发行",
                },
            ]
        )

        self.assertEqual(
            [(item.display_name, item.role) for item in preference.company_preferences],
            [("Ubisoft", "developer"), ("育碧", "publisher")],
        )
        developer = preference.company_preferences[0]
        self.assertEqual(developer.source_span, "Ubisoft")
        self.assertEqual(developer.strength, "strong")
        self.assertEqual(
            developer.aliases,
            ["育碧", "Ubisoft Entertainment", "Ubisoft Montreal", "Ubi"],
        )

    def test_company_identity_and_aliases_must_be_grounded_in_verbatim_spans(self) -> None:
        query = "偏好 Acme，也喜欢 Xbox Game Studios 开发的作品，但只在 Xbox 玩"
        preference = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Valve",
                    "aliases": ["Valve"],
                    "role": "developer",
                    "source_span": "Acme",
                },
                {
                    "display_name": "Acme",
                    "aliases": ["Acme", "Valve"],
                    "role": "developer",
                    "source_span": "Acme",
                },
                {
                    "display_name": "Xbox Game Studios",
                    "aliases": ["Xbox Game Studios"],
                    "role": "developer",
                    "source_span": "Xbox Game Studios",
                },
            ]
        )

        merged = merge_text_preference(preference, query)

        self.assertEqual(
            [item.display_name for item in merged.company_preferences],
            ["Acme", "Xbox Game Studios"],
        )
        self.assertEqual(merged.company_preferences[0].aliases, ["Acme", "Valve"])

    def test_plain_platform_token_cannot_be_recast_as_a_company(self) -> None:
        preference = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Xbox 平台",
                    "role": "either",
                    "source_span": "Xbox 平台",
                }
            ]
        )

        merged = merge_text_preference(preference, "只在 Xbox 平台玩合作游戏")

        self.assertEqual(merged.company_preferences, [])

    def test_platform_brand_is_allowed_in_explicit_company_role_context(self) -> None:
        for display_name, role_word in (
            ("任天堂", "开发"),
            ("PlayStation", "发行"),
        ):
            with self.subTest(display_name=display_name):
                preference = models.GamePreference(
                    company_preferences=[
                        {
                            "display_name": display_name,
                            "role": "either",
                            "source_span": display_name,
                        }
                    ]
                )

                merged = merge_text_preference(
                    preference,
                    f"只要 {display_name} {role_word}的游戏",
                )

                self.assertEqual(
                    [item.display_name for item in merged.company_preferences],
                    [display_name],
                )

    def test_chinese_result_quantity_cannot_become_a_soft_feature(self) -> None:
        for numeral in "一二三四五六七八九十":
            for unit in ("款", "个", "部"):
                span = f"{numeral}{unit}"
                with self.subTest(span=span):
                    merged = merge_text_preference(
                        models.GamePreference(
                            soft_features=[
                                {
                                    "constraint_id": "result-quantity",
                                    "source_span": span,
                                    "normalized_text": "result quantity",
                                    "role": "optional",
                                    "polarity": "positive",
                                }
                            ]
                        ),
                        f"请推荐{span}分支剧情游戏",
                    )

                    self.assertEqual(merged.soft_features, [])

    def test_currency_prefixes_and_english_result_quantities_cannot_be_features(self) -> None:
        for span in ("$100", "€20", "¥100", "3 games", "4 results", "5 titles"):
            with self.subTest(span=span):
                merged = merge_text_preference(
                    models.GamePreference(
                        soft_features=[
                            {
                                "constraint_id": "numeric-metadata",
                                "source_span": span,
                                "normalized_text": "numeric metadata",
                                "role": "optional",
                                "polarity": "positive",
                            }
                        ]
                    ),
                    f"预算或数量是 {span}，还想要分支剧情",
                )

                self.assertEqual(merged.soft_features, [])

    def test_quality_markers_cannot_be_company_preferences(self) -> None:
        for marker in ("AAA", "3A", "triple-A", "大作"):
            with self.subTest(marker=marker):
                merged = merge_text_preference(
                    models.GamePreference(
                        company_preferences=[
                            {
                                "display_name": marker,
                                "aliases": [marker],
                                "role": "either",
                                "strength": "strong",
                                "source_span": marker,
                            }
                        ]
                    ),
                    f"只想找 {marker}",
                )

                self.assertEqual(merged.quality_intent, "mainstream")
                self.assertEqual(merged.company_preferences, [])

    def test_quality_phrases_cannot_be_company_identities(self) -> None:
        for phrase in ("3A大作", "热门AAA大作", "AAA games"):
            with self.subTest(phrase=phrase):
                merged = merge_text_preference(
                    models.GamePreference(
                        company_preferences=[
                            {
                                "display_name": phrase,
                                "role": "either",
                                "strength": "strong",
                                "source_span": phrase,
                            }
                        ]
                    ),
                    f"只想找 {phrase}",
                )

                self.assertEqual(merged.quality_intent, "mainstream")
                self.assertEqual(merged.company_preferences, [])

    def test_company_name_containing_aaa_letters_is_not_a_quality_phrase(self) -> None:
        name = "Baaad Robot Games"
        merged = merge_text_preference(
            models.GamePreference(
                company_preferences=[
                    {
                        "display_name": name,
                        "role": "developer",
                        "source_span": name,
                    }
                ]
            ),
            f"偏好 {name} 开发的作品",
        )

        self.assertEqual(merged.quality_intent, "normal")
        self.assertEqual(
            [item.display_name for item in merged.company_preferences],
            [name],
        )

    def test_retry_merge_preserves_and_merges_new_preference_structures(self) -> None:
        base = models.GamePreference(
            derived_intent_tags=[{"tag": "puzzle", "source_span": "解谜"}],
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "分支剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                }
            ],
            company_preferences=[
                {
                    "display_name": "Acme",
                    "aliases": ["Acme"],
                    "role": "developer",
                    "strength": "preferred",
                    "source_span": "Acme",
                }
            ],
        )
        supplement = models.GamePreference(
            derived_intent_tags=[{"tag": "action", "source_span": "动作"}],
            soft_features=[
                {
                    "constraint_id": "short-session",
                    "source_span": "短局",
                    "normalized_text": "short sessions",
                    "role": "core",
                    "polarity": "positive",
                }
            ],
            company_preferences=[
                {
                    "display_name": "Publisher House",
                    "aliases": ["Publisher House"],
                    "role": "publisher",
                    "strength": "strong",
                    "source_span": "Publisher House",
                }
            ],
        )

        merged = merge_retry_preferences(base, supplement)

        self.assertEqual([item.tag for item in merged.derived_intent_tags], ["puzzle", "action"])
        self.assertEqual(
            [item.constraint_id for item in merged.soft_features],
            ["branching", "short-session"],
        )
        self.assertEqual(
            [item.display_name for item in merged.company_preferences],
            ["Acme", "Publisher House"],
        )

    def test_retry_keeps_same_company_in_distinct_developer_and_publisher_roles(self) -> None:
        base = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Acme Ltd.",
                    "role": "developer",
                    "source_span": "Acme Ltd.",
                }
            ]
        )
        supplement = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Acme Ltd.",
                    "role": "publisher",
                    "source_span": "Acme Ltd.",
                },
                {
                    "display_name": "ACME",
                    "role": "developer",
                    "source_span": "ACME",
                }
            ]
        )

        merged = merge_retry_preferences(base, supplement)

        self.assertEqual(
            [(item.display_name, item.role) for item in merged.company_preferences],
            [("ACME", "developer"), ("Acme Ltd.", "publisher")],
        )

    def test_retry_upgrades_existing_soft_feature_at_its_original_position(self) -> None:
        base = models.GamePreference(
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "最好有分支剧情",
                    "normalized_text": "branching story",
                    "role": "optional",
                    "polarity": "positive",
                },
                {
                    "constraint_id": "short-session",
                    "source_span": "短局",
                    "normalized_text": "short sessions",
                    "role": "optional",
                    "polarity": "positive",
                },
            ]
        )
        supplement = models.GamePreference(
            soft_features=[
                {
                    "constraint_id": "branching",
                    "source_span": "这次必须有分支剧情",
                    "normalized_text": "branching story",
                    "role": "core",
                    "polarity": "positive",
                }
            ]
        )

        merged = merge_retry_preferences(base, supplement)

        self.assertEqual(
            [item.constraint_id for item in merged.soft_features],
            ["branching", "short-session"],
        )
        self.assertEqual(merged.soft_features[0].role, "core")
        self.assertEqual(merged.soft_features[0].source_span, "这次必须有分支剧情")

    def test_retry_upgrades_existing_company_strength_at_its_original_position(self) -> None:
        base = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Acme Ltd.",
                    "role": "developer",
                    "strength": "preferred",
                    "source_span": "偏好 Acme Ltd.",
                },
                {
                    "display_name": "Other Publisher",
                    "role": "publisher",
                    "strength": "preferred",
                    "source_span": "Other Publisher",
                },
            ]
        )
        supplement = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "ACME",
                    "role": "developer",
                    "strength": "strong",
                    "source_span": "这次强烈偏好 ACME",
                }
            ]
        )

        merged = merge_retry_preferences(base, supplement)

        self.assertEqual(
            [item.display_name for item in merged.company_preferences],
            ["ACME", "Other Publisher"],
        )
        self.assertEqual(merged.company_preferences[0].strength, "strong")
        self.assertEqual(merged.company_preferences[0].source_span, "这次强烈偏好 ACME")

    def test_retry_uses_alias_overlap_with_right_precedence_but_keeps_roles_distinct(self) -> None:
        base = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "Ubisoft",
                    "aliases": ["育碧"],
                    "role": "developer",
                    "strength": "preferred",
                    "source_span": "偏好 Ubisoft",
                },
                {
                    "display_name": "Ubisoft",
                    "aliases": ["育碧"],
                    "role": "publisher",
                    "strength": "preferred",
                    "source_span": "Ubisoft publisher",
                },
            ]
        )
        supplement = models.GamePreference(
            company_preferences=[
                {
                    "display_name": "育碧",
                    "aliases": ["Ubisoft Entertainment"],
                    "role": "developer",
                    "strength": "strong",
                    "source_span": "这次强烈偏好育碧",
                }
            ]
        )

        merged = merge_retry_preferences(base, supplement)

        self.assertEqual(
            [(item.display_name, item.role) for item in merged.company_preferences],
            [("育碧", "developer"), ("Ubisoft", "publisher")],
        )
        developer = merged.company_preferences[0]
        self.assertEqual(developer.source_span, "这次强烈偏好育碧")
        self.assertEqual(developer.strength, "strong")
        self.assertEqual(
            developer.aliases,
            ["Ubisoft Entertainment", "Ubisoft"],
        )

    def test_retry_reserves_three_item_capacity_for_latest_structured_supplement(self) -> None:
        def feature(constraint_id: str, *, role: str = "optional") -> dict[str, str]:
            return {
                "constraint_id": constraint_id,
                "source_span": constraint_id,
                "normalized_text": constraint_id,
                "role": role,
                "polarity": "positive",
            }

        cases = (
            ([feature("D")], ["A", "B", "D"]),
            ([feature("B", role="core"), feature("D")], ["A", "B", "D"]),
            ([feature("D"), feature("E"), feature("F")], ["D", "E", "F"]),
        )
        for supplement_features, expected_ids in cases:
            with self.subTest(expected_ids=expected_ids):
                merged = merge_retry_preferences(
                    models.GamePreference(
                        soft_features=[feature("A"), feature("B"), feature("C")]
                    ),
                    models.GamePreference(soft_features=supplement_features),
                )

                self.assertEqual(
                    [item.constraint_id for item in merged.soft_features],
                    expected_ids,
                )
                if "B" in expected_ids and any(
                    item["constraint_id"] == "B" and item["role"] == "core"
                    for item in supplement_features
                ):
                    self.assertEqual(merged.soft_features[1].role, "core")


if __name__ == "__main__":
    unittest.main()
