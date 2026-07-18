from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .account_binding import platform_name_from_event

PLAIN_MESSAGE_LIMIT = 1800


@dataclass(frozen=True)
class ForwardComponents:
    Plain: type
    Node: type
    Nodes: type


@dataclass(frozen=True)
class MessageDelivery:
    forward_chain: list[Any] | None
    plain_blocks: list[str]


def prepare_message_delivery(
    event: Any,
    messages: list[str],
    *,
    node_name: str = "游戏推荐",
    components: Any | None = None,
) -> MessageDelivery:
    if platform_name_from_event(event) == "aiocqhttp":
        forward_chain = build_forward_message_chain(
            messages,
            node_name=node_name,
            components=components,
        )
        if forward_chain is not None:
            return MessageDelivery(forward_chain=forward_chain, plain_blocks=[])
    return MessageDelivery(
        forward_chain=None,
        plain_blocks=split_plain_blocks(messages),
    )


def split_plain_blocks(
    messages: list[str],
    limit: int = PLAIN_MESSAGE_LIMIT,
) -> list[str]:
    resolved_limit = max(min(int(limit), PLAIN_MESSAGE_LIMIT), 1)
    blocks: list[str] = []
    for message in messages:
        remaining = str(message or "").strip()
        while remaining:
            if len(remaining) <= resolved_limit:
                blocks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, resolved_limit + 1)
            if split_at <= 0:
                split_at = resolved_limit
            blocks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
    return [block for block in blocks if block]


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
