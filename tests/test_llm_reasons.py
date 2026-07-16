from __future__ import annotations

import asyncio
import json
import re
import unittest

from astrbot_plugin_steam_game_recommender.services.explanation_builder import (
    build_unplayed_evidence,
    fallback_reason,
    generate_recommendation_reasons,
    generate_unplayed_reason,
    reason_prompt,
    select_reason_evidence,
    validate_reason_response,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    RankedGame,
    RecommendationEvidence,
)


class ReasonValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = [
            evidence("tag_match", "preference", "positive", "匹配合作与解谜偏好"),
            evidence("reviews", "reviews", "positive", "Steam 好评率 90%"),
            evidence(
                "language_unknown",
                "language",
                "uncertain",
                "简体中文支持尚未确认",
                important=True,
            ),
        ]

    def test_accepts_two_or_three_grounded_sentences(self) -> None:
        result = validate_reason_response(
            json.dumps(
                {
                    "appid": 123,
                    "reason": "合作解谜玩法与需求很契合。Steam 口碑表现稳定，但中文支持尚未确认。",
                    "evidence_ids": ["tag_match", "reviews", "language_unknown"],
                },
                ensure_ascii=False,
            ),
            appid=123,
            evidence=self.evidence,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.evidence_ids[-1], "language_unknown")

    def test_rejects_wrong_appid_unknown_evidence_and_missing_important_risk(self) -> None:
        cases = (
            {
                "appid": 999,
                "reason": "合作解谜玩法契合。Steam 口碑稳定，中文支持尚未确认。",
                "evidence_ids": ["tag_match", "reviews", "language_unknown"],
            },
            {
                "appid": 123,
                "reason": "合作解谜玩法契合。Steam 口碑稳定，中文支持尚未确认。",
                "evidence_ids": ["tag_match", "invented", "language_unknown"],
            },
            {
                "appid": 123,
                "reason": "合作解谜玩法契合。Steam 口碑表现稳定。",
                "evidence_ids": ["tag_match", "reviews"],
            },
            {
                "appid": 123,
                "reason": "合作解谜玩法契合。Steam 口碑表现稳定。",
                "evidence_ids": ["tag_match", "reviews", "language_unknown"],
            },
        )

        for payload in cases:
            with self.subTest(payload=payload):
                self.assertIsNone(
                    validate_reason_response(
                        json.dumps(payload, ensure_ascii=False),
                        appid=123,
                        evidence=self.evidence,
                    )
                )

    def test_rejects_wrong_sentence_count_and_overlong_reason(self) -> None:
        one_sentence = {
            "appid": 123,
            "reason": "合作解谜玩法与需求契合。",
            "evidence_ids": ["tag_match", "language_unknown"],
        }
        overlong = {
            "appid": 123,
            "reason": f"{'很' * 90}。{'好' * 90}。",
            "evidence_ids": ["tag_match", "language_unknown"],
        }

        self.assertIsNone(
            validate_reason_response(
                json.dumps(one_sentence, ensure_ascii=False),
                appid=123,
                evidence=self.evidence,
            )
        )
        self.assertIsNone(
            validate_reason_response(
                json.dumps(overlong, ensure_ascii=False),
                appid=123,
                evidence=self.evidence,
            )
        )

    def test_evidence_input_is_trimmed_but_never_drops_important_risks(self) -> None:
        values = [
            evidence(f"minor_{index}", "reviews", "positive", f"次要证据 {index}")
            for index in range(12)
        ]
        values.append(
            evidence(
                "required_risk",
                "constraint",
                "uncertain",
                "硬条件尚未确认",
                important=True,
            )
        )

        selected = select_reason_evidence(values, limit=8)

        self.assertLessEqual(len(selected), 8)
        self.assertIn("required_risk", [item.evidence_id for item in selected])

    def test_positive_evidence_is_selected_in_user_facing_priority_order(self) -> None:
        values = [
            evidence("library", "library", "positive", "命中游戏库偏好"),
            evidence("reviews", "reviews", "positive", "Steam 口碑稳定"),
            evidence("quality", "quality", "positive", "高知名度/大作倾向"),
            evidence("reference", "reference", "positive", "已解析参考游戏"),
            evidence("supporting", "supporting", "positive", "命中辅助标签"),
            evidence("core", "core", "positive", "命中核心标签"),
        ]

        selected = select_reason_evidence(values, limit=4)

        self.assertEqual(
            [item.evidence_id for item in selected],
            ["core", "supporting", "reference", "quality"],
        )

    def test_core_missing_risk_accepts_core_relaxed_or_missing_wording(self) -> None:
        core_missing = evidence(
            "core_missing",
            "core",
            "uncertain",
            "宽松匹配：缺失核心特征 soulslike",
            important=True,
        )
        risk_sentences = (
            "核心特征证据仍不足。",
            "这是宽松匹配结果。",
            "关键玩法存在缺失。",
        )

        for risk_sentence in risk_sentences:
            with self.subTest(risk_sentence=risk_sentence):
                result = validate_reason_response(
                    json.dumps(
                        {
                            "appid": 123,
                            "reason": f"游戏命中部分偏好。{risk_sentence}",
                            "evidence_ids": ["core_missing"],
                        },
                        ensure_ascii=False,
                    ),
                    appid=123,
                    evidence=[core_missing],
                )

                self.assertIsNotNone(result)

    def test_wilson_evidence_uses_statistically_precise_user_facing_wording(self) -> None:
        reviews = evidence(
            "review_confidence",
            "reviews",
            "positive",
            "Steam 好评率 90%，共 1000 条评测；Wilson 置信下界 88%",
        )

        prompt = reason_prompt(123, "Test Game", [reviews])
        fallback = fallback_reason([reviews])

        self.assertIn("95% Wilson 好评率下界 88%", prompt)
        self.assertIn("95% Wilson 好评率下界 88%", fallback)
        self.assertNotIn("Wilson 置信下界", prompt)

    def test_mainstream_prompt_forbids_claiming_aaa_budget(self) -> None:
        mainstream = evidence(
            "mainstream_intent",
            "quality",
            "positive",
            "按高知名度/大作倾向提高成熟口碑在层内的权重",
        )

        prompt = reason_prompt(123, "Test Game", [mainstream])

        self.assertIn("只能表述为高知名度/大作倾向", prompt)
        self.assertIn("不得声称 AAA 制作预算", prompt)

    def test_mainstream_reason_rejects_unverifiable_aaa_budget_claim(self) -> None:
        mainstream = evidence(
            "mainstream_intent",
            "quality",
            "positive",
            "按高知名度/大作倾向提高成熟口碑在层内的权重",
        )

        result = validate_reason_response(
            json.dumps(
                {
                    "appid": 123,
                    "reason": "这是 AAA 制作预算的游戏。整体口碑表现稳定。",
                    "evidence_ids": ["mainstream_intent"],
                },
                ensure_ascii=False,
            ),
            appid=123,
            evidence=[mainstream],
        )

        self.assertIsNone(result)


class ConcurrentReasonGenerationTest(unittest.IsolatedAsyncioTestCase):
    async def test_generates_each_game_independently_with_at_most_five_calls(self) -> None:
        context = ConcurrentReasonContext()
        games = [ranked_game(appid) for appid in range(1, 9)]

        generated = await generate_recommendation_reasons(
            context,
            FakeEvent(),
            "provider-1",
            games,
        )

        self.assertEqual(len(context.calls), 8)
        self.assertEqual(context.max_active, 5)
        self.assertTrue(
            all(
                "匹配合作与解谜偏好" in game.recommendation_reason
                and "Steam 口碑稳定" in game.recommendation_reason
                for game in generated
            )
        )

    async def test_invalid_single_game_response_only_falls_back_for_that_game(self) -> None:
        context = PerGameFailureContext()
        games = [ranked_game(1), ranked_game(2)]

        generated = await generate_recommendation_reasons(
            context,
            FakeEvent(),
            "provider-1",
            games,
        )

        self.assertIn("匹配合作与解谜偏好", generated[0].recommendation_reason)
        self.assertIn("Steam 口碑稳定", generated[0].recommendation_reason)
        self.assertNotEqual(generated[1].recommendation_reason, "格式错误")
        self.assertIn("匹配合作与解谜偏好", generated[1].recommendation_reason)

    async def test_untrusted_reason_text_cannot_add_claims_without_evidence(self) -> None:
        game = RankedGame(
            title="Grounded Game",
            appid=1,
            score=80,
            recommendation_evidence=[
                evidence("tag_match", "preference", "positive", "命中解谜核心玩法")
            ],
        )
        context = StaticReasonContext(
            {
                "appid": 1,
                "recommendation_reason": (
                    "当前仅需 9.99 美元。已确认支持完整中文配音。"
                ),
                "recommendation_evidence_ids": ["tag_match"],
                "caution_reason": None,
                "caution_evidence_ids": [],
            }
        )

        generated = await generate_recommendation_reasons(
            context,
            FakeEvent(),
            "provider-1",
            [game],
        )

        reason = generated[0].recommendation_reason
        self.assertIn("命中解谜核心玩法", reason)
        self.assertNotIn("9.99", reason)
        self.assertNotIn("美元", reason)
        self.assertNotIn("中文配音", reason)


class UnplayedReasonTest(unittest.IsolatedAsyncioTestCase):
    async def test_unplayed_reason_focuses_on_gameplay_reviews_and_popularity(self) -> None:
        game = GameCandidate(
            title="Backlog Game",
            appid=77,
            genres=["Adventure"],
            tags=["Puzzle", "Story Rich"],
            review_total=20_000,
            review_positive_ratio=0.91,
        )
        context = StaticReasonContext(
            {
                "appid": 77,
                "reason": (
                    "它以冒险解谜和剧情体验为主。较高好评率与充足评测量说明口碑和知名度都较稳。"
                ),
                "evidence_ids": ["gameplay", "reviews", "popularity"],
            }
        )

        reason = await generate_unplayed_reason(
            context,
            FakeEvent(),
            "provider-1",
            game,
        )

        self.assertIn("类型", reason)
        self.assertIn("好评率", reason)
        self.assertNotEqual(reason, context.payload["reason"])
        self.assertEqual(
            [item.evidence_id for item in build_unplayed_evidence(game)],
            ["gameplay", "reviews", "popularity"],
        )

    async def test_unplayed_reason_cannot_add_untrusted_price_or_language_claims(
        self,
    ) -> None:
        game = GameCandidate(
            title="Backlog Game",
            appid=77,
            genres=["Adventure"],
            tags=["Puzzle"],
        )
        context = StaticReasonContext(
            {
                "appid": 77,
                "reason": "当前仅需 9.99 美元。已确认支持完整中文配音。",
                "evidence_ids": ["gameplay"],
            }
        )

        reason = await generate_unplayed_reason(
            context,
            FakeEvent(),
            "provider-1",
            game,
        )

        self.assertIn("类型", reason)
        self.assertNotIn("9.99", reason)
        self.assertNotIn("美元", reason)
        self.assertNotIn("中文配音", reason)

    async def test_unplayed_failure_fallback_keeps_gameplay_reviews_and_popularity(self) -> None:
        game = GameCandidate(
            title="Backlog Game",
            appid=77,
            genres=["Adventure"],
            tags=["Puzzle"],
            review_total=20_000,
            review_positive_ratio=0.91,
        )

        reason = await generate_unplayed_reason(
            StaticReasonContext({"appid": 77, "reason": "坏格式", "evidence_ids": []}),
            FakeEvent(),
            "provider-1",
            game,
        )

        self.assertIn("类型", reason)
        self.assertIn("好评率", reason)
        self.assertIn("知名度", reason)


def ranked_game(appid: int) -> RankedGame:
    return RankedGame(
        title=f"Game {appid}",
        appid=appid,
        score=80,
        recommendation_evidence=[
            evidence("tag_match", "preference", "positive", "匹配合作与解谜偏好"),
            evidence("reviews", "reviews", "positive", "Steam 口碑稳定"),
        ],
    )


def evidence(
    evidence_id: str,
    category: str,
    sentiment: str,
    text: str,
    important: bool = False,
) -> RecommendationEvidence:
    return RecommendationEvidence(
        evidence_id=evidence_id,
        category=category,
        sentiment=sentiment,
        text=text,
        important=important,
    )


class FakeEvent:
    unified_msg_origin = "qq:test"


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.completion_text = text


class ConcurrentReasonContext:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[dict] = []

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        appid = int(re.search(r"APPID=(\d+)", kwargs["prompt"]).group(1))
        return FakeResponse(
            json.dumps(
                {
                    "appid": appid,
                    "reason": "这款游戏玩法契合偏好。Steam 口碑表现稳定。",
                    "evidence_ids": ["tag_match", "reviews"],
                },
                ensure_ascii=False,
            )
        )


class PerGameFailureContext(ConcurrentReasonContext):
    async def llm_generate(self, **kwargs):
        appid = int(re.search(r"APPID=(\d+)", kwargs["prompt"]).group(1))
        if appid == 2:
            return FakeResponse(
                json.dumps(
                    {"appid": 2, "reason": "格式错误", "evidence_ids": ["tag_match"]},
                    ensure_ascii=False,
                )
            )
        return FakeResponse(
            json.dumps(
                {
                    "appid": appid,
                    "reason": "这款游戏玩法契合偏好。Steam 口碑稳定。",
                    "evidence_ids": ["tag_match", "reviews"],
                },
                ensure_ascii=False,
            )
        )


class StaticReasonContext:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def llm_generate(self, **_kwargs):
        return FakeResponse(json.dumps(self.payload, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
