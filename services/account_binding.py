from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STEAMID64_BASE = 76561197960265728
STEAM_ACCOUNT_ID_MAX = 2**32 - 1


class AccountBindingError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedAccountBinding:
    provider: str
    account_id: str
    account_kind: str
    display_value: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_account_binding_command(text: str) -> ParsedAccountBinding:
    raw = str(text or "").strip()
    if not raw:
        raise AccountBindingError("请输入账号，例如：/accountbind steam 76561198000000000")

    provider, account_text = split_provider(raw)
    if provider != "steam":
        raise AccountBindingError("目前仅支持绑定 Steam 账号。")
    return parse_steam_account(account_text)


def split_provider(text: str) -> tuple[str, str]:
    first, separator, rest = text.partition(" ")
    provider = first.strip().lower()
    if provider in {"steam", "steamid"}:
        account_text = rest.strip()
        if not separator or not account_text:
            raise AccountBindingError("请输入 Steam ID64 或好友码。")
        return "steam", account_text
    if re.fullmatch(r"[0-9\s-]+", text):
        return "steam", text
    return provider, rest.strip()


def parse_steam_account(value: str) -> ParsedAccountBinding:
    display_value = str(value or "").strip()
    digits = re.sub(r"[\s-]+", "", display_value)
    if not digits or not digits.isdigit():
        raise AccountBindingError("Steam 账号只能填写 SteamID64 或纯数字好友码。")

    number = int(digits)
    if len(digits) == 17 and number >= STEAMID64_BASE:
        return ParsedAccountBinding(
            provider="steam",
            account_id=digits,
            account_kind="steam_id64",
            display_value=display_value,
        )

    if len(digits) < 17 and 0 < number <= STEAM_ACCOUNT_ID_MAX:
        return ParsedAccountBinding(
            provider="steam",
            account_id=str(STEAMID64_BASE + number),
            account_kind="steam_friend_code",
            display_value=display_value,
            metadata={"steam_friend_code": digits},
        )

    raise AccountBindingError("SteamID64 应为 17 位数字；好友码应为较短的纯数字。")


def chat_identity_from_event(event: Any) -> tuple[str, str]:
    user_getter = getattr(event, "get_sender_id", None)
    user_id = str(user_getter() if callable(user_getter) else getattr(event, "sender_id", ""))
    user_id = user_id.strip()
    if not user_id:
        raise AccountBindingError("无法识别当前发送者。")

    platform = ""
    for name in ("get_platform_name", "get_platform_id"):
        getter = getattr(event, name, None)
        if callable(getter):
            platform = str(getter() or "").strip()
            if platform:
                break
    if not platform:
        for name in ("platform_name", "platform_id", "platform"):
            platform = str(getattr(event, name, "") or "").strip()
            if platform:
                break
    return platform or "default", user_id
