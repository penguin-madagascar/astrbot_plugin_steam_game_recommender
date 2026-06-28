from __future__ import annotations

import unittest

from astrbot_plugin_game_recommender.services.message_delivery import (
    build_forward_message_chain,
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


if __name__ == "__main__":
    unittest.main()
