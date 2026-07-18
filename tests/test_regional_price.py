from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from astrbot_plugin_steam_game_recommender.services.formatter import (
    format_game_block,
    format_recommendation_messages,
)
from astrbot_plugin_steam_game_recommender.services.region_query import (
    parse_region_query,
    region_currency,
)
from astrbot_plugin_steam_game_recommender.services.steam_price_bridge import (
    SteamPriceBridge,
    attach_price_summary,
)
from astrbot_plugin_steam_game_recommender.storage.models import (
    GamePreference,
    GamePriceSummary,
    RankedGame,
    RecommendationEvidence,
    ScoreBreakdown,
)
from astrbot_plugin_steam_price_heybox.models import (
    GameIdentity,
    PriceHistory,
    PricePoint,
    SaleEvent,
    SteamGameDetails,
    SteamPrice,
)


class RegionQueryTest(unittest.TestCase):
    def test_parses_country_code_and_chinese_region_aliases(self) -> None:
        code = parse_region_query("-US 双人合作，预算 50", default_region="CN")
        alias = parse_region_query("双人合作 日区", default_region="CN")

        self.assertEqual((code.region, code.query), ("US", "双人合作，预算 50"))
        self.assertEqual((alias.region, alias.query), ("JP", "双人合作"))

    def test_uses_default_and_does_not_treat_co_op_as_a_region(self) -> None:
        parsed = parse_region_query("co-op puzzle", default_region="JP")

        self.assertEqual(parsed.region, "JP")
        self.assertEqual(parsed.query, "co-op puzzle")

    def test_maps_supported_regions_to_local_currency(self) -> None:
        self.assertEqual(region_currency("CN"), "CNY")
        self.assertEqual(region_currency("US"), "USD")
        self.assertEqual(region_currency("JP"), "JPY")


class RegionalPriceBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_known_appid_uses_exact_identity_instead_of_title_search(self) -> None:
        service = TrackingPriceService(
            history=active_history(),
            resolved_appid=456,
            details_appid=456,
        )
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "US"},
            service_factory=lambda _config, _client: service,
        )

        summary = await bridge.lookup(
            "Control Ultimate Edition",
            country="US",
            appid=456,
        )

        self.assertEqual(service.resolve_queries, ["456"])
        self.assertEqual(service.detail_appids, [456])
        self.assertIsNotNone(summary)

    async def test_known_appid_rejects_mismatched_resolved_identity(self) -> None:
        service = TrackingPriceService(history=active_history(), resolved_appid=999)
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "US"},
            service_factory=lambda _config, _client: service,
        )

        summary = await bridge.lookup("Control", country="US", appid=456)

        self.assertIsNone(summary)
        self.assertEqual(service.resolve_queries, ["456"])
        self.assertEqual(service.detail_appids, [])
        self.assertEqual(service.history_countries, [])

    async def test_lookup_requests_only_selected_region_and_never_global_prices(self) -> None:
        service = TrackingPriceService(history=active_history())
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "CN"},
            service_factory=lambda _config, _client: service,
            today_provider=lambda: date(2026, 7, 13),
        )

        summary = await bridge.lookup("Test Game", country="US")

        self.assertEqual(service.detail_countries, ["US"])
        self.assertEqual(service.history_countries, ["US"])
        self.assertEqual(service.global_price_calls, 0)
        assert summary is not None
        self.assertEqual(summary.region, "US")
        self.assertEqual(summary.currency, "USD")
        self.assertEqual(summary.current_price, "$60")
        self.assertEqual(summary.historic_low, "$40")
        self.assertEqual(summary.recent_sale_price, "$50")
        self.assertEqual(summary.sale_time_status, "已开始 3 天")

    async def test_completed_sale_reports_days_since_end(self) -> None:
        service = TrackingPriceService(history=completed_history())
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "JP"},
            service_factory=lambda _config, _client: service,
            today_provider=lambda: date(2026, 7, 13),
        )

        summary = await bridge.lookup("Test Game")

        assert summary is not None
        self.assertEqual(summary.recent_sale_price, "$45")
        self.assertEqual(summary.sale_time_status, "结束于 12 天前")

    async def test_missing_history_keeps_only_current_region_price(self) -> None:
        service = TrackingPriceService(history=None)
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "US"},
            service_factory=lambda _config, _client: service,
            today_provider=lambda: date(2026, 7, 13),
        )

        summary = await bridge.lookup("Test Game")

        assert summary is not None
        self.assertEqual(summary.current_price, "$60")
        self.assertIsNone(summary.historic_low)
        self.assertIsNone(summary.recent_sale_price)
        self.assertIsNone(summary.sale_time_status)

    def test_price_summary_tracks_each_component_currency(self) -> None:
        fields = getattr(GamePriceSummary, "model_fields", None) or GamePriceSummary.__fields__

        self.assertEqual(
            set(fields),
            {
                "region",
                "currency",
                "current_price",
                "current_amount",
                "current_currency",
                "historic_low",
                "historic_low_amount",
                "historic_low_currency",
                "recent_sale_price",
                "recent_sale_amount",
                "recent_sale_currency",
                "sale_time_status",
            },
        )


class RegionalBudgetAndFormattingTest(unittest.TestCase):
    def test_historic_low_in_different_currency_is_not_compared_to_budget(self) -> None:
        game = RankedGame(title="Mixed Currency Game", appid=123, score=80)
        summary = GamePriceSummary(
            region="US",
            currency="USD",
            current_price="$60",
            current_amount=60,
            current_currency="USD",
            historic_low="ARS 20",
            historic_low_amount=20,
            historic_low_currency="ARS",
        )

        enriched = attach_price_summary(
            game,
            summary,
            GamePreference(budget=50, budget_currency="USD", region="US"),
        )

        self.assertEqual(enriched.score_breakdown.budget_adjustment, -2)
        evidence = next(
            item for item in enriched.recommendation_evidence if item.category == "budget"
        )
        self.assertEqual(evidence.evidence_id, "budget_currency_mismatch")
        self.assertNotIn("史低曾低于预算", evidence.text)

    def test_currency_mismatch_uses_unknown_budget_penalty(self) -> None:
        game = RankedGame(title="US Game", appid=123, score=80)
        summary = GamePriceSummary(
            region="US",
            currency="USD",
            current_price="$40",
            current_amount=40,
            historic_low="$20",
            historic_low_amount=20,
        )

        enriched = attach_price_summary(
            game,
            summary,
            GamePreference(budget=100, budget_currency="CNY", region="US"),
        )

        self.assertEqual(enriched.score, 78)
        self.assertEqual(enriched.score_breakdown.budget_adjustment, -2)

    def test_region_local_budget_is_used_when_currency_is_implicit(self) -> None:
        game = RankedGame(title="US Game", appid=123, score=80)
        summary = GamePriceSummary(
            region="US",
            currency="USD",
            current_price="$40",
            current_amount=40,
            historic_low="$20",
            historic_low_amount=20,
        )

        enriched = attach_price_summary(
            game,
            summary,
            GamePreference(budget=50, region="US"),
        )

        self.assertEqual(enriched.score, 85)
        self.assertEqual(enriched.score_breakdown.budget_adjustment, 5)

    def test_game_block_has_only_compact_region_price_and_steam_link(self) -> None:
        game = RankedGame(
            title="Test Game",
            appid=123,
            score=86,
            recommendation_reason="玩法契合合作解谜偏好。评测规模较大，口碑稳定。",
            price_summary=GamePriceSummary(
                region="US",
                currency="USD",
                current_price="$60",
                current_amount=60,
                historic_low="$40",
                historic_low_amount=40,
                recent_sale_price="$50",
                recent_sale_amount=50,
                sale_time_status="结束于 12 天前",
            ),
        )

        text = "\n".join(format_game_block(1, game))

        self.assertEqual(
            text,
            "\n".join(
                [
                    "1. 《Test Game》｜推荐分：86/100",
                    "",
                    "推荐理由：玩法契合合作解谜偏好。评测规模较大，口碑稳定。",
                    "",
                    "价格（US）：当前价 $60；历史最低 $40；最近促销 $50（结束于 12 天前）",
                    "",
                    "购买链接：https://store.steampowered.com/app/123/",
                ]
            ),
        )
        self.assertNotIn("小黑盒", text)
        self.assertNotIn("数据来源", text)

    def test_intro_is_the_fixed_short_sentence_with_only_real_warnings_appended(self) -> None:
        game = RankedGame(title="Test Game", appid=123, score=86)

        plain = format_recommendation_messages(
            GamePreference(region="US", result_count=1),
            [game],
        )
        warned = format_recommendation_messages(
            GamePreference(region="US", result_count=1, parse_warnings=["已排除已有游戏"]),
            [game],
        )

        self.assertEqual(plain[0], "找到 1 款 Steam 游戏，按推荐分从高到低排列。")
        self.assertEqual(
            warned[0],
            "找到 1 款 Steam 游戏，按推荐分从高到低排列。\n"
            "偏好解析提示：\n- 已排除已有游戏",
        )

    def test_intro_uses_tier_first_wording_when_ranked_results_have_anchor_tiers(self) -> None:
        tiered = RankedGame(
            title="Tiered Game",
            appid=123,
            score=86,
            score_breakdown=ScoreBreakdown(relevance_tier="A"),
        )

        messages = format_recommendation_messages(
            GamePreference(region="CN", result_count=1),
            [tiered],
        )

        self.assertEqual(
            messages[0],
            "找到 1 款 Steam 游戏，按核心匹配层级及层内推荐分排列。",
        )

    def test_relaxed_tier_reason_names_missing_core_feature(self) -> None:
        relaxed = RankedGame(
            title="Relaxed Game",
            appid=123,
            score=72,
            score_breakdown=ScoreBreakdown(relevance_tier="B"),
            recommendation_reason="Steam 口碑表现较稳定。",
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_missing",
                    category="core",
                    sentiment="uncertain",
                    text="宽松匹配：缺失或证据不足的核心特征为soulslike",
                    important=True,
                )
            ],
        )

        text = "\n".join(format_game_block(1, relaxed))

        self.assertIn("推荐理由：Steam 口碑表现较稳定。", text)
        self.assertIn("不推荐理由：宽松匹配：", text)
        self.assertIn("核心特征为类魂", text)

    def test_relaxed_tier_without_core_evidence_still_discloses_limited_match(self) -> None:
        relaxed = RankedGame(
            title="Relaxed Game",
            appid=123,
            score=60,
            score_breakdown=ScoreBreakdown(relevance_tier="C"),
            recommendation_reason="Steam 口碑表现较稳定。",
        )

        text = "\n".join(format_game_block(1, relaxed))

        self.assertNotIn("宽松匹配", text)
        self.assertNotIn("不推荐理由：", text)

    def test_relaxed_tier_adds_label_when_core_evidence_uses_plain_wording(self) -> None:
        relaxed = RankedGame(
            title="Relaxed Game",
            appid=123,
            score=60,
            score_breakdown=ScoreBreakdown(relevance_tier="C"),
            recommendation_reason="Steam 口碑表现较稳定。",
            recommendation_evidence=[
                RecommendationEvidence(
                    evidence_id="core_missing",
                    category="core",
                    sentiment="uncertain",
                    text="核心特征 soulslike 缺失",
                    important=True,
                )
            ],
        )

        text = "\n".join(format_game_block(1, relaxed))

        self.assertIn("不推荐理由：核心特征 类魂 缺失", text)


class TrackingPriceService:
    default_language = "schinese"

    def __init__(
        self,
        history: PriceHistory | None,
        *,
        resolved_appid: int = 123,
        details_appid: int = 123,
    ) -> None:
        self.history = history
        self.resolved_appid = resolved_appid
        self.details_appid = details_appid
        self.resolve_queries: list[str] = []
        self.detail_appids: list[int] = []
        self.detail_countries: list[str] = []
        self.history_countries: list[str] = []
        self.global_price_calls = 0
        self.steam_client = self
        self.heybox_client = self

    async def resolve_game(self, query: str, country: str) -> tuple[GameIdentity, str]:
        self.resolve_queries.append(query)
        return (
            GameIdentity(
                self.resolved_appid,
                f"Test Game / appid={self.resolved_appid}",
            ),
            country,
        )

    async def details(self, appid: int, country: str, _language: str) -> SteamGameDetails:
        self.detail_appids.append(appid)
        self.detail_countries.append(country)
        return steam_details(appid=self.details_appid)

    async def load_history(self, _appid: int, country: str) -> PriceHistory | None:
        self.history_countries.append(country)
        return self.history

    async def global_prices(self, _appid: int):
        self.global_price_calls += 1
        raise AssertionError("global_prices must not be called")


def steam_details(appid: int = 123) -> SteamGameDetails:
    return SteamGameDetails(
        appid=appid,
        name="Test Game",
        game_type="game",
        is_free=False,
        coming_soon=False,
        release_date="2026-01-01",
        price=SteamPrice(Decimal("60"), Decimal("100"), "USD", 40),
        developers=(),
        publishers=(),
        platforms=("windows",),
        genres=(),
        categories=(),
        languages=("English",),
        controller_support="",
        achievement_count=0,
        dlc_count=0,
        metacritic_score=None,
        recommendation_count=None,
        required_age="",
        content_notes="",
        website="",
    )


def active_history() -> PriceHistory:
    return PriceHistory(
        points=(PricePoint(date(2026, 7, 13), Decimal("60"), "USD", None, 40),),
        events=(SaleEvent(date(2026, 7, 10), None, Decimal("50"), None, "USD", 50),),
        lowest_price=Decimal("40"),
        lowest_currency="USD",
        lowest_date=date(2026, 1, 1),
        lowest_discount=60,
        lowest_occurrences=1,
        maximum_discount=60,
    )


def completed_history() -> PriceHistory:
    return PriceHistory(
        points=(PricePoint(date(2026, 7, 13), Decimal("60"), "USD", None, 0),),
        events=(
            SaleEvent(
                date(2026, 6, 25),
                date(2026, 7, 1),
                Decimal("45"),
                None,
                "USD",
                55,
            ),
        ),
        lowest_price=Decimal("40"),
        lowest_currency="USD",
        lowest_date=date(2026, 1, 1),
        lowest_discount=60,
        lowest_occurrences=1,
        maximum_discount=60,
    )


if __name__ == "__main__":
    unittest.main()
