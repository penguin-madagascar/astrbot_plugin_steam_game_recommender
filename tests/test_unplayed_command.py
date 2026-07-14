from __future__ import annotations

import json
import unittest

try:
    __import__("tests.test_prepare_recommendation")
except ModuleNotFoundError:
    __import__("test_prepare_recommendation")

from astrbot_plugin_steam_game_recommender.main import SteamGameRecommenderPlugin
from astrbot_plugin_steam_game_recommender.storage.models import (
    GameCandidate,
    SteamAccountBinding,
    SteamOwnedGame,
)


class RandomRecommendationCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_command_sends_one_plain_message_without_forward_record_or_extra_fields(
        self,
    ) -> None:
        plugin = object.__new__(SteamGameRecommenderPlugin)
        plugin.cache = BoundCache()
        plugin.steam_client = UnplayedSteamClient()
        plugin.context = ReasonContext()
        plugin.provider_id = "provider-1"
        plugin.recommendation_config = {
            "steam_min_review_count": 50,
            "steam_min_positive_ratio": 0.65,
        }
        plugin.config = {
            "steam_min_review_count": 50_000,
            "steam_min_positive_ratio": 0.99,
        }
        event = PlainEvent()

        results = [item async for item in plugin.recommend_random_game(event)]

        self.assertEqual(len(results), 1)
        self.assertEqual(len(event.plain_messages), 1)
        self.assertEqual(event.chain_calls, 0)
        message = event.plain_messages[0]
        self.assertTrue(message.startswith("《Backlog Game》\n"))
        for excluded in ("推荐分", "价格", "购买链接", "http", "数据来源"):
            self.assertNotIn(excluded, message)


class PlainEvent:
    unified_msg_origin = "qq:test"
    sender_id = "test"
    platform = "qq"

    def __init__(self) -> None:
        self.plain_messages: list[str] = []
        self.chain_calls = 0

    def plain_result(self, text: str):
        self.plain_messages.append(text)
        return ("plain", text)

    def chain_result(self, _chain):
        self.chain_calls += 1
        return ("chain", _chain)


class BoundCache:
    async def get_steam_account_binding(self, _platform: str, _user_id: str):
        return SteamAccountBinding(
            chat_user_id="test",
            steam_id64="76561198000000000",
            account_kind="steamid64",
            display_value="76561198000000000",
        )


class ReviewSummary:
    total_reviews = 20_000
    positive_ratio = 0.91
    recent_positive_ratio = 0.90


class UnplayedSteamClient:
    def has_web_api_key(self) -> bool:
        return True

    async def get_owned_games(self, _steam_id64: str):
        return [SteamOwnedGame(appid=77, name="Backlog Game", playtime_forever=0)]

    async def get_review_summary(self, _appid: int):
        return ReviewSummary()

    async def get_game_detail(self, _appid: int):
        return GameCandidate(
            title="Backlog Game",
            appid=77,
            app_type="game",
            genres=["Adventure"],
            tags=["Puzzle", "Story Rich"],
        )


class ReasonResponse:
    completion_text = json.dumps(
        {
            "appid": 77,
            "reason": "它以冒险解谜和剧情体验为主。较高好评率与充足评测量说明口碑和知名度都较稳。",
            "evidence_ids": ["gameplay", "reviews", "popularity"],
        },
        ensure_ascii=False,
    )


class ReasonContext:
    async def llm_generate(self, **_kwargs):
        return ReasonResponse()


if __name__ == "__main__":
    unittest.main()
