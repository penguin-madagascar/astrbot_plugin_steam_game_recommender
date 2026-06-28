from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ForwardComponents:
    Plain: type
    Node: type
    Nodes: type


def build_forward_message_chain(
    messages: list[str],
    node_name: str = "游戏推荐",
    components: Any | None = None,
) -> list[Any] | None:
    resolved = components or load_forward_components()
    if resolved is None:
        return None

    nodes = []
    for message in messages:
        text = str(message or "").strip()
        if not text:
            continue
        nodes.append(
            resolved.Node(
                name=node_name,
                content=[resolved.Plain(text)],
            )
        )
    if not nodes:
        return None
    return [resolved.Nodes(nodes=nodes)]


def load_forward_components() -> ForwardComponents | None:
    for module_name in (
        "astrbot.api.message_components",
        "astrbot.core.message.components",
    ):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        plain = getattr(module, "Plain", None)
        node = getattr(module, "Node", None)
        nodes = getattr(module, "Nodes", None)
        if plain and node and nodes:
            return ForwardComponents(Plain=plain, Node=node, Nodes=nodes)
    return None
