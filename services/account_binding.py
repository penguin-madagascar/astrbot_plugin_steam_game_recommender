from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STEAMID64_BASE = 76561197960265728
STEAM_ACCOUNT_ID_MAX = 2**32 - 1


class AccountBindingError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedSteamAccount:
    steam_id64: str
    account_kind: str
    display_value: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_account_binding_command(text: str) -> ParsedSteamAccount:
    raw = str(text or "").strip()
    if not raw:
        raise AccountBindingError("请输入账号，例如：/accountbind 76561198000000000")
    return parse_steam_account(raw)


def parse_steam_account(value: str) -> ParsedSteamAccount:
    display_value = str(value or "").strip()
    digits = re.sub(r"[\s-]+", "", display_value)
    if not digits or not digits.isdigit():
        raise AccountBindingError("Steam 账号只能填写 SteamID64 或纯数字好友码。")

    number = int(digits)
    if (
        len(digits) == 17
        and STEAMID64_BASE < number <= STEAMID64_BASE + STEAM_ACCOUNT_ID_MAX
    ):
        return ParsedSteamAccount(
            steam_id64=digits,
            account_kind="steam_id64",
            display_value=display_value,
        )

    if len(digits) < 17 and 0 < number <= STEAM_ACCOUNT_ID_MAX:
        return ParsedSteamAccount(
            steam_id64=str(STEAMID64_BASE + number),
            account_kind="steam_friend_code",
            display_value=display_value,
            metadata={"steam_friend_code": digits},
        )

    raise AccountBindingError(
        "SteamID64 应为 17 位数字；好友码应为较短的纯数字。"
    )


def platform_name_from_event(event: Any) -> str:
    platform = _event_text(event, "get_platform_name", "platform_name", "platform")
    return platform.casefold() or "default"


def account_identity_from_event(event: Any) -> tuple[str, str]:
    user_id = sender_id_from_event(event)
    platform_instance = _event_text(event, "get_platform_id", "platform_id")
    if not platform_instance:
        unified_msg_origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
        platform_instance = unified_msg_origin.partition(":")[0].strip()
    return platform_instance or platform_name_from_event(event), user_id


def recommendation_scope_from_event(event: Any) -> tuple[str, str]:
    user_id = sender_id_from_event(event)
    unified_msg_origin = str(getattr(event, "unified_msg_origin", "") or "").strip()
    if not unified_msg_origin:
        raise AccountBindingError("无法识别当前会话。")
    return unified_msg_origin, user_id


def platform_instance_ids_for_name(
    context: Any,
    platform_name: str,
) -> list[str] | None:
    manager = getattr(context, "platform_manager", None)
    if manager is None:
        return None
    getter = getattr(manager, "get_insts", None)
    try:
        instances = (
            getter()
            if callable(getter)
            else getattr(manager, "platform_insts", None)
        )
    except Exception:
        return None
    if not isinstance(instances, (list, tuple)):
        return None

    expected_name = str(platform_name or "").strip().casefold()
    ids: list[str] = []
    for instance in instances:
        meta_getter = getattr(instance, "meta", None)
        if not callable(meta_getter):
            return None
        try:
            meta = meta_getter()
        except Exception:
            return None
        name = str(getattr(meta, "name", "") or "").strip().casefold()
        platform_id = str(getattr(meta, "id", "") or "").strip()
        if name == expected_name and platform_id and platform_id not in ids:
            ids.append(platform_id)
    return ids


def chat_identity_from_event(event: Any) -> tuple[str, str]:
    return account_identity_from_event(event)


def sender_id_from_event(event: Any) -> str:
    user_getter = getattr(event, "get_sender_id", None)
    user_id = str(user_getter() if callable(user_getter) else getattr(event, "sender_id", ""))
    user_id = user_id.strip()
    if not user_id:
        raise AccountBindingError("无法识别当前发送者。")
    return user_id


def _event_text(event: Any, *names: str) -> str:
    for name in names:
        value = getattr(event, name, "")
        if callable(value):
            value = value()
        text = str(value or "").strip()
        if text:
            return text
    return ""
