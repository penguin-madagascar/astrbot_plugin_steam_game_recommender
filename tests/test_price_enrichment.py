from __future__ import annotations

import ast
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from astrbot_plugin_game_recommender.services.steam_price_bridge import (  # noqa: E402
    SteamPriceBridge,
    attach_price_summary,
)
from astrbot_plugin_game_recommender.storage.models import (  # noqa: E402
    GameCandidate,
    GamePreference,
    GamePriceSummary,
    RankedGame,
)

from astrbot_plugin_game_recommender.services.formatter import (  # noqa: E402
    format_game_block,
    format_game_detail,
)
from astrbot_plugin_steam_price_heybox.models import (  # noqa: E402
    GameIdentity,
    RegionPrice,
    SteamGameDetails,
    SteamPrice,
)
from astrbot_plugin_steam_price_heybox.price_analysis import parse_price_history  # noqa: E402


class CommandRegistrationTest(unittest.TestCase):
    def test_english_commands_keep_chinese_aliases(self) -> None:
        tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        commands: dict[str, set[str]] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "command"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    continue
                aliases: set[str] = set()
                for keyword in decorator.keywords:
                    if keyword.arg == "alias" and isinstance(keyword.value, ast.Set):
                        aliases = {
                            item.value
                            for item in keyword.value.elts
                            if isinstance(item, ast.Constant) and isinstance(item.value, str)
                        }
                commands[str(decorator.args[0].value)] = aliases

        self.assertIn("gamerec", commands)
        self.assertIn("游戏推荐", commands["gamerec"])
        self.assertIn("gamedesc", commands)
        self.assertIn("游戏详情", commands["gamedesc"])

    def test_rawg_api_key_is_optional_in_config_schema(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        rawg_config = schema["rawg_api_key"]

        self.assertEqual(rawg_config["default"], "")
        self.assertNotIn("必填", rawg_config["hint"])

    def test_dashboard_hides_price_plugin_runtime_settings(self) -> None:
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))

        self.assertIn("steam_price_heybox_notice", schema)
        for removed_key in (
            "itad" + "_api_key",
            "enable_steam_price_enrichment",
            "steam_price_country",
            "steam_price_history_days",
            "steam_price_lookup_limit",
        ):
            self.assertNotIn(removed_key, schema)

    def test_repository_no_longer_mentions_itad(self) -> None:
        ignored_parts = {".git", ".venv", "__pycache__"}
        searched_suffixes = {".py", ".json", ".md", ".yaml", ".toml", ".txt"}
        markers = ("Itad" + "Client", "IsThereAny" + "Deal", "itad" + "_api_key")
        offenders = []

        for path in ROOT.rglob("*"):
            if any(part in ignored_parts for part in path.parts):
                continue
            if path.is_file() and path.suffix in searched_suffixes:
                text = path.read_text(encoding="utf-8")
                for marker in markers:
                    if marker in text:
                        offenders.append(f"{path.relative_to(ROOT)}:{marker}")

        self.assertEqual(offenders, [])


class PriceFormattingTest(unittest.TestCase):
    def test_price_summary_is_json_serializable(self) -> None:
        game = RankedGame(
            title="Test Game",
            score=10,
            price_summary=price_summary(current_cny=60, lowest_cny=50),
        )

        payload = json.dumps(game.model_dump(), ensure_ascii=False)

        self.assertIn("current_price", payload)
        self.assertIn("¥60", payload)
        self.assertIn("lowest_cny", payload)

    def test_recommendation_block_includes_price_and_links(self) -> None:
        game = RankedGame(
            title="Test Game",
            platforms=["PC"],
            stores=["Steam"],
            score=10,
            price_summary=price_summary(current_cny=60, lowest_cny=50),
        )

        text = "\n".join(format_game_block(1, game))

        self.assertIn("价格：Steam 当前价 ¥60", text)
        self.assertIn("史低 ¥50", text)
        self.assertIn("Steam：https://store.steampowered.com/app/123/", text)
        self.assertIn("小黑盒：https://www.xiaoheihe.cn/app/topic/game/pc/123", text)

    def test_game_detail_appends_price_summary_when_available(self) -> None:
        game = GameCandidate(title="Test Game", platforms=["PC"], stores=["Steam"])

        text = format_game_detail(game, price_summary(current_cny=60, lowest_cny=50))

        self.assertIn("《Test Game》", text)
        self.assertIn("Steam 价格：Steam 当前价 ¥60", text)
        self.assertIn("史低 ¥50", text)


class PriceBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_http_client_leaves_games_unchanged(self) -> None:
        bridge = SteamPriceBridge(client=None, config={})
        games = [RankedGame(title="Test Game", score=10)]

        enriched = await bridge.enrich_ranked_games(games, GamePreference(budget=100))

        self.assertFalse(bridge.is_available())
        self.assertEqual(enriched[0].title, "Test Game")
        self.assertIsNone(enriched[0].price_summary)

    async def test_bridge_uses_default_region_and_internal_price_defaults(self) -> None:
        captured_config = {}

        def factory(price_config, _client):
            captured_config.update(price_config)
            return FakePriceService()

        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "us"},
            service_factory=factory,
        )

        self.assertTrue(bridge.is_available())
        self.assertEqual(bridge.lookup_limit, 5)
        self.assertEqual(captured_config["default_country"], "US")
        self.assertEqual(captured_config["default_history_country"], "US")
        self.assertEqual(captured_config["default_language"], "schinese")
        self.assertEqual(captured_config["history_days"], 720)
        self.assertEqual(captured_config["global_price_limit"], 10)
        self.assertFalse(captured_config["show_api_links"])
        self.assertEqual(captured_config["llm_name_retry_count"], 0)

    async def test_lookup_builds_summary_from_price_service(self) -> None:
        bridge = SteamPriceBridge(
            client=object(),
            config={"default_region": "CN"},
            service_factory=lambda _config, _client: FakePriceService(),
        )

        summary = await bridge.lookup("Test Game")

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.appid, 123)
        self.assertEqual(summary.current_price, "¥60")
        self.assertEqual(summary.lowest_price, "¥50")
        self.assertEqual(summary.lowest_date, "2026-01-10")
        self.assertEqual(summary.current_cny, 60)
        self.assertEqual(summary.lowest_cny, 50)
        self.assertIn("当前促销", summary.sale_status or "")
        self.assertIn("乌克兰 / UA", summary.region_summary or "")


class BudgetScoringTest(unittest.TestCase):
    def test_current_price_inside_budget_adds_score(self) -> None:
        game = RankedGame(title="Budget Game", score=10)
        enriched = attach_price_summary(
            game,
            price_summary(current_cny=60, lowest_cny=50),
            GamePreference(budget=100),
        )

        self.assertGreater(enriched.score, game.score)
        self.assertTrue(any("预算" in reason for reason in enriched.reasons))

    def test_lowest_price_inside_budget_warns_but_keeps_game(self) -> None:
        game = RankedGame(title="Sale Game", score=10)
        enriched = attach_price_summary(
            game,
            price_summary(current_cny=120, lowest_cny=80),
            GamePreference(budget=100),
        )

        self.assertGreaterEqual(enriched.score, game.score)
        self.assertTrue(
            any("史低" in warning and "预算" in warning for warning in enriched.warnings)
        )

    def test_price_over_budget_penalizes_without_filtering(self) -> None:
        game = RankedGame(title="Expensive Game", score=10)
        enriched = attach_price_summary(
            game,
            price_summary(current_cny=120, lowest_cny=110),
            GamePreference(budget=100),
        )

        self.assertEqual(enriched.title, "Expensive Game")
        self.assertLess(enriched.score, game.score)
        self.assertTrue(any("高于预算" in warning for warning in enriched.warnings))


def price_summary(current_cny: float, lowest_cny: float) -> GamePriceSummary:
    return GamePriceSummary(
        source="steam_price_heybox",
        appid=123,
        country="CN",
        current_price=f"¥{current_cny:g}",
        lowest_price=f"¥{lowest_cny:g}",
        lowest_date="2026-01-10",
        lowest_discount=50,
        sale_status="当前促销：2026-01-10 开始，最低 ¥50",
        region_summary="最低价区服：乌克兰 / UA，约 ¥40",
        store_url="https://store.steampowered.com/app/123/",
        heybox_url="https://www.xiaoheihe.cn/app/topic/game/pc/123",
        current_cny=current_cny,
        lowest_cny=lowest_cny,
    )


class FakeSteamClient:
    async def details(self, _appid: int, _country: str, _language: str) -> SteamGameDetails:
        return SteamGameDetails(
            appid=123,
            name="Test Game",
            game_type="game",
            is_free=False,
            coming_soon=False,
            release_date="2026 年 1 月 1 日",
            price=SteamPrice(Decimal("60"), Decimal("100"), "CNY", 40),
            developers=("Developer",),
            publishers=("Publisher",),
            platforms=("windows",),
            genres=("动作",),
            categories=("单人",),
            languages=("简体中文",),
            controller_support="full",
            achievement_count=10,
            dlc_count=0,
            metacritic_score=80,
            recommendation_count=1000,
            required_age="",
            content_notes="",
            website="",
        )


class FakeHeyboxClient:
    async def global_prices(self, _appid: int) -> list[RegionPrice]:
        return [
            RegionPrice("CN", "中国", Decimal("60"), Decimal("100"), 40),
            RegionPrice("UA", "乌克兰", Decimal("40"), Decimal("80"), 50),
        ]


class FakePriceService:
    default_language = "schinese"
    steam_client = FakeSteamClient()
    heybox_client = FakeHeyboxClient()

    async def resolve_game(self, _title: str, country: str) -> tuple[GameIdentity, str]:
        return GameIdentity(123, "Test Game / appid=123"), country

    async def load_history(self, _appid: int, _country: str):
        return parse_price_history(
            {
                "prices": [
                    history_point("2026-01-01", "100", 0),
                    history_point("2026-01-10", "50", 50),
                ],
                "lowest_info": {"date": "2026-01-10", "price": "50", "discount": 50},
                "lowest_info_v2": {"currency": "CNY"},
            }
        )


def history_point(recorded_on: str, price: str, discount: int) -> dict:
    return {
        "date": recorded_on,
        "price": price,
        "rmb_price": price,
        "currency": "CNY",
        "discount": discount,
    }


if __name__ == "__main__":
    unittest.main()
