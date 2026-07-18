from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.services.message_delivery import (
    build_forward_message_chain,
    prepare_message_delivery,
)


class MessageDeliveryTest(unittest.TestCase):
    def test_builds_one_forward_chat_record_from_multiple_messages(self) -> None:
        messages = [
            "一句话结论：优先看前 2 款。",
            "1. 《Split Fiction》\n平台：PC、Nintendo Switch 2",
            "2. 《Unravel Two》\n平台：PC、Nintendo Switch",
        ]

        chain = build_forward_message_chain(
            messages,
            components=FakeForwardComponents,
        )

        self.assertEqual(len(chain), 1)
        nodes = chain[0]
        self.assertIsInstance(nodes, FakeNodes)
        self.assertEqual(len(nodes.nodes), 3)
        self.assertEqual([node.name for node in nodes.nodes], ["游戏推荐", "游戏推荐", "游戏推荐"])
        self.assertEqual(nodes.nodes[0].content[0].text, messages[0])
        self.assertEqual(nodes.nodes[1].content[0].text, messages[1])
        self.assertEqual(nodes.nodes[2].content[0].text, messages[2])

    def test_returns_none_when_forward_components_are_unavailable(self) -> None:
        self.assertIsNone(build_forward_message_chain(["hello"], components=None))

    def test_only_aiocqhttp_uses_forward_nodes(self) -> None:
        onebot = prepare_message_delivery(
            PlatformEvent("aiocqhttp"),
            ["first", "second"],
            components=FakeForwardComponents,
        )
        self.assertIsInstance(onebot.forward_chain[0], FakeNodes)
        self.assertEqual(onebot.plain_blocks, [])
        for platform in ("qq_official", "telegram", "discord"):
            with self.subTest(platform=platform):
                delivery = prepare_message_delivery(
                    PlatformEvent(platform),
                    ["first", "second"],
                    components=FakeForwardComponents,
                )
                self.assertIsNone(delivery.forward_chain)
                self.assertEqual(delivery.plain_blocks, ["first", "second"])

    def test_plain_delivery_splits_each_block_at_1800_characters(self) -> None:
        message = "\n".join(["x" * 500] * 5)

        delivery = prepare_message_delivery(
            PlatformEvent("discord"),
            [message, "short"],
            components=FakeForwardComponents,
        )

        self.assertGreater(len(delivery.plain_blocks), 2)
        self.assertTrue(all(0 < len(block) <= 1800 for block in delivery.plain_blocks))
        self.assertEqual(delivery.plain_blocks[-1], "short")

    def test_aiocqhttp_falls_back_to_plain_blocks_without_node_components(self) -> None:
        delivery = prepare_message_delivery(
            PlatformEvent("aiocqhttp"),
            ["hello"],
            components=None,
        )

        self.assertIsNone(delivery.forward_chain)
        self.assertEqual(delivery.plain_blocks, ["hello"])


class FakePlain:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeNode:
    def __init__(self, name: str, content: list[FakePlain]) -> None:
        self.name = name
        self.content = content


class FakeNodes:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self.nodes = nodes


class FakeForwardComponents:
    Plain = FakePlain
    Node = FakeNode
    Nodes = FakeNodes


class PlatformEvent:
    def __init__(self, platform_name: str) -> None:
        self.platform_name = platform_name

    def get_platform_name(self) -> str:
        return self.platform_name


if __name__ == "__main__":
    unittest.main()
