from __future__ import annotations

import math
import unittest

from astrbot_plugin_steam_game_recommender.services.explanation_builder import (
    generate_recommendation_reasons,
    user_facing_evidence_text,
    validate_recommendation_reason_response,
)
from astrbot_plugin_steam_game_recommender.services.formatter import (
    format_recommendation_messages,
)
from astrbot_plugin_steam_game_recommender.services.recommendation_intent import (
    IntentTagRole,
    IntentTagSource,
    QualityIntent,
    RecommendationIntent,
    WeightedIntentTag,
)
from astrbot_plugin_steam_game_recommender.services.run_notices import RunNotice
from astrbot_plugin_steam_game_recommender.services.similarity_ranker import (
    rank_steam_candidates,
)
from astrbot_plugin_steam_game_recommender.services.semantic_feature_verifier import (
    copy_score_breakdown as copy_semantic_score_breakdown,
)
from astrbot_plugin_steam_game_recommender.services.steam_price_bridge import (
    attach_missing_price_warning,
)
from astrbot_plugin_steam_game_recommender.services.tag_presentation import (
    build_tag_presentations,
    presentation_tag,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    GamePreference,
    RankedGame,
    RecommendationEvidence,
    ScoreBreakdown,
)

UNRELEASED_CAUTION = (
    "使用 60/100 未发售质量先验，仅用于排序，不代表玩家实评或实际知名度"
)


class UnreleasedQualityTest(unittest.TestCase):
    def test_ordinary_and_preorder_unreleased_games_share_the_same_prior(self) -> None:
        games = rank_steam_candidates(
            [
                candidate(1, "Ordinary Upcoming", coming_soon=True),
                candidate(
                    2,
                    "Preorder Upcoming",
                    coming_soon=True,
                    markers=["preorder"],
                ),
            ],
            intent(allow_unreleased=True),
        )

        self.assertEqual(len(games), 2)
        for game in games:
            with self.subTest(game=game.title):
                breakdown = game.score_breakdown
                self.assertEqual(breakdown.review_reputation, 0.60)
                self.assertEqual(breakdown.popularity, 0.60)
                self.assertEqual(breakdown.quality_score, 0.60)
                self.assertEqual(breakdown.wilson_lower_bound, 0.0)
                self.assertEqual(breakdown.quality_source, "unreleased_prior")
                self.assertIsNone(game.review_total)
                self.assertIsNone(game.review_positive_ratio)
                self.assertTrue(
                    any(item.text == UNRELEASED_CAUTION for item in game.recommendation_evidence)
                )

    def test_actual_reviews_override_unreleased_prior(self) -> None:
        game = rank_steam_candidates(
            [
                candidate(
                    1,
                    "Reviewed Upcoming",
                    coming_soon=True,
                    review_total=1000,
                    review_positive_ratio=0.90,
                )
            ],
            intent(allow_unreleased=True),
        )[0]

        self.assertEqual(game.score_breakdown.quality_source, "actual_reviews")
        self.assertNotEqual(game.score_breakdown.quality_score, 0.60)
        self.assertNotEqual(game.score_breakdown.review_reputation, 0.60)
        self.assertFalse(
            any(item.text == UNRELEASED_CAUTION for item in game.recommendation_evidence)
        )

    def test_low_actual_review_quality_still_overrides_prior(self) -> None:
        game = rank_steam_candidates(
            [
                candidate(
                    1,
                    "Poorly Reviewed Upcoming",
                    coming_soon=True,
                    review_total=1,
                    review_positive_ratio=0.0,
                )
            ],
            intent(allow_unreleased=True),
        )[0]

        self.assertEqual(game.score_breakdown.quality_source, "actual_reviews")
        self.assertLess(game.score_breakdown.quality_score, 0.60)

    def test_invalid_review_shapes_cannot_be_coerced_into_actual_reviews(self) -> None:
        invalid_values = (
            (True, 0.9),
            (1.2, 0.9),
            (math.nan, 0.9),
            (-1, 0.9),
            (100, True),
            (100, math.nan),
            (100, -0.1),
            (100, 1.1),
        )

        for review_total, ratio in invalid_values:
            with self.subTest(review_total=review_total, ratio=ratio):
                game = rank_steam_candidates(
                    [
                        candidate(
                            1,
                            "Invalid Reviews",
                            coming_soon=True,
                            review_total=review_total,  # type: ignore[arg-type]
                            review_positive_ratio=ratio,  # type: ignore[arg-type]
                        )
                    ],
                    intent(allow_unreleased=True),
                )[0]
                self.assertEqual(
                    game.score_breakdown.quality_source,
                    "unreleased_prior",
                )

    def test_released_missing_reviews_remain_zero_quality(self) -> None:
        game = rank_steam_candidates([candidate(1, "Released")], intent())[0]

        breakdown = game.score_breakdown
        self.assertEqual(breakdown.review_reputation, 0.0)
        self.assertEqual(breakdown.popularity, 0.0)
        self.assertEqual(breakdown.quality_score, 0.0)
        self.assertEqual(breakdown.quality_source, "none")

    def test_default_query_filters_unreleased_games(self) -> None:
        games = rank_steam_candidates(
            [candidate(1, "Upcoming", coming_soon=True)],
            intent(allow_unreleased=False),
        )

        self.assertEqual(games, [])

    def test_quality_source_rejects_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            ScoreBreakdown(quality_source="fabricated")

    def test_quality_source_survives_price_and_semantic_score_copies(self) -> None:
        game = rank_steam_candidates(
            [candidate(1, "Upcoming", coming_soon=True)],
            intent(allow_unreleased=True),
        )[0]

        price_updated = attach_missing_price_warning(game)
        semantic_updated = copy_semantic_score_breakdown(
            price_updated.score_breakdown,
            supporting_similarity=0.5,
        )

        self.assertEqual(price_updated.score_breakdown.quality_source, "unreleased_prior")
        self.assertEqual(semantic_updated.quality_source, "unreleased_prior")
        self.assertIsNone(price_updated.review_total)
        self.assertIsNone(price_updated.review_positive_ratio)


class RecommendationOutputTest(unittest.IsolatedAsyncioTestCase):
    def test_llm_caution_must_cover_only_structured_risk_evidence(self) -> None:
        evidence = [
            RecommendationEvidence(
                evidence_id="core_match",
                category="core",
                sentiment="positive",
                text="命中核心玩法特征：类魂",
            ),
            RecommendationEvidence(
                evidence_id="core_missing",
                category="core",
                sentiment="uncertain",
                text="部分核心特征证据不足",
                important=True,
            ),
        ]
        valid = validate_recommendation_reason_response(
            '{"appid":1,"recommendation_reason":"核心玩法方向匹配。现有标签证据较明确。",'
            '"recommendation_evidence_ids":["core_match"],'
            '"caution_reason":"部分核心特征证据仍不足。",'
            '"caution_evidence_ids":["core_missing"]}',
            1,
            evidence,
        )
        invented = validate_recommendation_reason_response(
            '{"appid":1,"recommendation_reason":"核心玩法方向匹配。现有标签证据较明确。",'
            '"recommendation_evidence_ids":["core_match"],'
            '"caution_reason":"可能存在性能问题。",'
            '"caution_evidence_ids":["invented"]}',
            1,
            evidence,
        )
        fabricated_text = validate_recommendation_reason_response(
            '{"appid":1,"recommendation_reason":"核心玩法方向匹配。现有标签证据较明确。",'
            '"recommendation_evidence_ids":["core_match"],'
            '"caution_reason":"游戏可能存在严重性能问题。",'
            '"caution_evidence_ids":["core_missing"]}',
            1,
            evidence,
        )

        self.assertIsNotNone(valid)
        self.assertIsNone(invented)
        self.assertIsNone(fabricated_text)

    def test_unreleased_llm_caution_cannot_weaken_exact_disclosure(self) -> None:
        evidence = [
            RecommendationEvidence(
                evidence_id="unreleased_quality_prior",
                category="reviews",
                sentiment="uncertain",
                text=UNRELEASED_CAUTION,
                important=True,
            )
        ]
        result = validate_recommendation_reason_response(
            '{"appid":1,"recommendation_reason":"现有信息有限。建议结合玩法偏好判断。",'
            '"recommendation_evidence_ids":[],'
            '"caution_reason":"这是未发售作品，评分仅供参考。",'
            '"caution_evidence_ids":["unreleased_quality_prior"]}',
            1,
            evidence,
        )

        self.assertIsNone(result)

    async def test_structured_risk_becomes_optional_caution_line(self) -> None:
        game = RankedGame(
            appid=1,
            title="Upcoming",
            score=65,
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_match",
                    category="core",
                    sentiment="positive",
                    text="命中核心玩法特征：类魂",
                ),
                RecommendationEvidence(
                    evidence_id="unreleased_quality_prior",
                    category="reviews",
                    sentiment="uncertain",
                    text=UNRELEASED_CAUTION,
                    important=True,
                ),
            ],
        )

        generated = await generate_recommendation_reasons(
            NoProviderContext(),
            object(),
            "",
            [game],
        )

        self.assertIn("核心", generated[0].recommendation_reason)
        self.assertEqual(generated[0].caution_reason, f"{UNRELEASED_CAUTION}。")
        block = format_recommendation_messages(GamePreference(), generated)[1]
        self.assertIn(f"不推荐理由：{UNRELEASED_CAUTION}。", block)

    async def test_llm_caution_cannot_mix_fabricated_risk_into_output(self) -> None:
        game = RankedGame(
            appid=1,
            title="Grounded Caution",
            score=65,
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_match",
                    category="core",
                    sentiment="positive",
                    text="命中核心玩法特征：类魂",
                ),
                RecommendationEvidence(
                    evidence_id="core_missing",
                    category="core",
                    sentiment="uncertain",
                    text="部分核心特征证据不足",
                    important=True,
                ),
            ],
        )
        context = StaticReasonContext(
            '{"appid":1,'
            '"recommendation_reason":"核心玩法方向匹配。现有标签证据较明确。",'
            '"recommendation_evidence_ids":["core_match"],'
            '"caution_reason":"部分核心特征缺失，而且存在严重性能问题并频繁崩溃。",'
            '"caution_evidence_ids":["core_missing"]}'
        )

        generated = await generate_recommendation_reasons(
            context,
            object(),
            "provider/test",
            [game],
        )

        self.assertIn("命中核心玩法特征：类魂", generated[0].recommendation_reason)
        self.assertEqual(generated[0].caution_reason, "部分核心特征证据不足。")
        block = format_recommendation_messages(GamePreference(), generated)[1]
        self.assertNotIn("性能", block)
        self.assertNotIn("崩溃", block)

    async def test_unreleased_prior_does_not_authorize_review_or_popularity_claims(
        self,
    ) -> None:
        game = RankedGame(
            appid=1,
            title="Unreviewed Upcoming",
            score=65,
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_match",
                    category="core",
                    sentiment="positive",
                    text="命中核心玩法特征：类魂",
                ),
                RecommendationEvidence(
                    evidence_id="unreleased_quality_prior",
                    category="reviews",
                    sentiment="uncertain",
                    text=UNRELEASED_CAUTION,
                    important=True,
                ),
            ],
        )
        context = StaticReasonContext(
            '{"appid":1,'
            '"recommendation_reason":"核心玩法方向匹配。Steam玩家口碑极佳且知名度很高。",'
            '"recommendation_evidence_ids":["core_match"],'
            f'"caution_reason":"{UNRELEASED_CAUTION}。",'
            '"caution_evidence_ids":["unreleased_quality_prior"]}'
        )

        generated = await generate_recommendation_reasons(
            context,
            object(),
            "provider/test",
            [game],
        )

        self.assertNotIn("口碑", generated[0].recommendation_reason)
        self.assertNotIn("知名度", generated[0].recommendation_reason)
        self.assertEqual(generated[0].caution_reason, f"{UNRELEASED_CAUTION}。")

    async def test_unreleased_prior_always_uses_deterministic_positive_reason(self) -> None:
        game = RankedGame(
            appid=1,
            title="Unreviewed Upcoming",
            score=65,
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_match",
                    category="core",
                    sentiment="positive",
                    text="命中核心玩法特征：类魂",
                ),
                RecommendationEvidence(
                    evidence_id="unreleased_quality_prior",
                    category="reviews",
                    sentiment="uncertain",
                    text=UNRELEASED_CAUTION,
                    important=True,
                ),
            ],
        )
        context = StaticReasonContext(
            '{"appid":1,'
            '"recommendation_reason":"核心玩法方向匹配。广受好评且拥有庞大玩家群体。",'
            '"recommendation_evidence_ids":["core_match"],'
            f'"caution_reason":"{UNRELEASED_CAUTION}。",'
            '"caution_evidence_ids":["unreleased_quality_prior"]}'
        )

        generated = await generate_recommendation_reasons(
            context,
            object(),
            "provider/test",
            [game],
        )
        block = format_recommendation_messages(GamePreference(), generated)[1]

        self.assertIn("核心", generated[0].recommendation_reason)
        self.assertNotIn("广受好评", block)
        self.assertNotIn("庞大玩家群体", block)
        self.assertIn(f"不推荐理由：{UNRELEASED_CAUTION}。", block)

    def test_notices_then_summary_then_game_and_blank_line_fields(self) -> None:
        game = RankedGame(
            appid=1,
            title="Test Game",
            score=69,
            recommendation_reason="玩法契合。",
            caution_reason="语言支持未知。",
        )
        messages = format_recommendation_messages(
            GamePreference(parse_warnings=["第一条", "第二条"]),
            [game],
            run_notices=[
                RunNotice("parser", "error", "偏好解析模型暂不可用，已降级。")
            ],
        )

        self.assertEqual(messages[0], "偏好解析模型暂不可用，已降级。")
        self.assertTrue(messages[1].startswith("找到 1 款 Steam 游戏"))
        self.assertIn("偏好解析提示：\n- 第一条\n- 第二条", messages[1])
        self.assertTrue(messages[2].startswith("1. 《Test Game》｜推荐分：69/100"))
        for field in ("推荐理由：", "不推荐理由：", "价格（CN）：", "购买链接："):
            self.assertIn(f"\n\n{field}", messages[2])

    def test_caution_line_is_omitted_without_structured_risk(self) -> None:
        messages = format_recommendation_messages(
            GamePreference(),
            [RankedGame(title="Safe", score=80, recommendation_reason="玩法契合。")],
        )

        self.assertNotIn("不推荐理由：", messages[1])

    def test_default_output_contains_summary_and_ten_game_nodes(self) -> None:
        games = [
            RankedGame(title=f"Game {index}", score=80 - index)
            for index in range(1, 13)
        ]

        messages = format_recommendation_messages(GamePreference(), games)

        self.assertEqual(len(messages), 11)
        self.assertIn("找到 10 款", messages[0])
        self.assertTrue(messages[-1].startswith("10. 《Game 10》"))

    async def test_nonimportant_uncertainty_does_not_create_caution(self) -> None:
        game = RankedGame(
            title="Unknown Reviews",
            score=50,
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="review_unknown",
                    category="reviews",
                    sentiment="uncertain",
                    text="Steam 评测缺失或为零",
                    important=False,
                )
            ],
        )

        generated = await generate_recommendation_reasons(
            NoProviderContext(), object(), "", [game]
        )

        self.assertIsNone(generated[0].caution_reason)

    def test_tag_presentations_join_vocabulary_by_tag_id_and_hide_unknown_snake_case(
        self,
    ) -> None:
        presentations = build_tag_presentations(
            [{"tagid": 29482, "name": "Souls-like"}],
            [{"tagid": 29482, "name": "类魂"}],
        )
        self.assertEqual(presentation_tag("soulslike", presentations), "类魂")

        game = RankedGame(
            title="Localized",
            score=70,
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_match",
                    category="core",
                    sentiment="positive",
                    text="命中核心标签：soulslike、unknown_feature",
                )
            ],
        )
        messages = format_recommendation_messages(GamePreference(), [game])

        self.assertIn("类魂", messages[1])
        self.assertNotIn("soulslike", messages[1])
        self.assertNotIn("unknown_feature", messages[1])

    def test_free_text_company_and_product_names_are_not_translated_as_tags(self) -> None:
        text = "由 Action Games 开发，发行商 Adventure Works，使用 RPG Maker 制作"

        self.assertEqual(user_facing_evidence_text(text), text)

    def test_dynamic_tag_presentations_are_joined_by_id_and_request_local(self) -> None:
        first = build_tag_presentations(
            [{"tagid": 101, "name": "Precision Platformer"}],
            [{"tagid": 101, "name": "精准平台跳跃"}],
        )
        second = build_tag_presentations(
            [{"tagid": 101, "name": "Precision Platformer"}],
            [{"tagid": 101, "name": "高精度平台动作"}],
        )
        mismatched = build_tag_presentations(
            [{"tagid": 101, "name": "Precision Platformer"}],
            [{"tagid": 202, "name": "精准平台跳跃"}],
        )

        self.assertEqual(
            presentation_tag("precision_platformer", first),
            "精准平台跳跃",
        )
        self.assertEqual(
            presentation_tag("precision_platformer", second),
            "高精度平台动作",
        )
        self.assertIsNone(presentation_tag("precision_platformer", mismatched))


class NoProviderContext:
    async def get_current_chat_provider_id(self, **_kwargs):
        return ""


class StaticReasonContext:
    def __init__(self, response: str) -> None:
        self.response = response

    async def llm_generate(self, **_kwargs):
        return type("Response", (), {"completion_text": self.response})()


def candidate(
    appid: int,
    title: str,
    *,
    coming_soon: bool = False,
    markers: list[str] | None = None,
    review_total: int | None = None,
    review_positive_ratio: float | None = None,
) -> GameCandidate:
    return GameCandidate(
        appid=appid,
        app_type="game",
        title=title,
        tags=["Action"],
        ordered_tags=["Action"],
        coming_soon=coming_soon,
        internal_source_markers=markers or [],
        review_total=review_total,
        review_positive_ratio=review_positive_ratio,
    )


def intent(*, allow_unreleased: bool = False) -> RecommendationIntent:
    return RecommendationIntent(
        tags=(
            WeightedIntentTag(
                "action",
                IntentTagRole.ANCHOR,
                IntentTagSource.EXPLICIT,
                1.0,
            ),
        ),
        references=(),
        quality_intent=QualityIntent.NORMAL,
        allow_unreleased=allow_unreleased,
    )


if __name__ == "__main__":
    unittest.main()
