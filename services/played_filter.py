from __future__ import annotations

from dataclasses import dataclass

from ..storage.models import RankedGame, SteamOwnedGame
from .game_identity import game_family_key

LIBRARY_FILTER_EXCLUDE_OWNED = "exclude_owned"
LIBRARY_FILTER_ONLY_OWNED = "only_owned"
LIBRARY_FILTER_MODES = {LIBRARY_FILTER_EXCLUDE_OWNED, LIBRARY_FILTER_ONLY_OWNED}

PREFIX_ARGUMENTS = (
    ("exclude-owned", LIBRARY_FILTER_EXCLUDE_OWNED),
    ("exclude_owned", LIBRARY_FILTER_EXCLUDE_OWNED),
    ("only-owned", LIBRARY_FILTER_ONLY_OWNED),
    ("only_owned", LIBRARY_FILTER_ONLY_OWNED),
    ("排除已有", LIBRARY_FILTER_EXCLUDE_OWNED),
    ("仅查看已有", LIBRARY_FILTER_ONLY_OWNED),
)

TEXT_ARGUMENTS = (
    *PREFIX_ARGUMENTS,
    ("exclude owned", LIBRARY_FILTER_EXCLUDE_OWNED),
    ("only owned", LIBRARY_FILTER_ONLY_OWNED),
)


class LibraryFilterModeError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "library_filter_invalid",
    ) -> None:
        self.code = code
        super().__init__(message)


LIBRARY_FILTER_USER_MESSAGES = {
    "query_required": "请输入游戏需求后再试。",
    "account_binding_required": (
        "当前用户未绑定 Steam 账号；请先使用 /accountbind 完成绑定。"
    ),
    "account_binding_unavailable": "无法确认当前 Steam 账号绑定，请重新绑定后再试。",
    "steam_api_key_required": "未配置 Steam Web API Key，无法读取个人游戏库。",
    "steam_library_unavailable": "Steam 游戏库暂时不可读，请稍后重试。",
    "steam_library_empty": "Steam 游戏库为空或不可见，无法执行游戏库过滤。",
}


def library_filter_user_message(error: LibraryFilterModeError) -> str:
    return LIBRARY_FILTER_USER_MESSAGES.get(
        error.code,
        "游戏库过滤条件无效，请检查后重试。",
    )


@dataclass(frozen=True)
class LibraryFilterCommand:
    mode: str | None
    query: str


def parse_library_filter_command(text: str) -> LibraryFilterCommand:
    query = str(text or "").strip()
    matched = match_prefix_argument(query)
    if matched is None:
        return LibraryFilterCommand(mode=None, query=query)

    mode, rest = matched
    second_match = match_prefix_argument(rest)
    if second_match is not None and second_match[0] != mode:
        raise LibraryFilterModeError("排除已有和仅查看已有不能同时使用。")
    return LibraryFilterCommand(mode=mode, query=rest.strip())


def match_prefix_argument(text: str) -> tuple[str, str] | None:
    stripped = str(text or "").lstrip()
    lowered = stripped.lower()
    for phrase, mode in PREFIX_ARGUMENTS:
        phrase_lower = phrase.lower()
        if not lowered.startswith(phrase_lower):
            continue
        rest = stripped[len(phrase) :]
        if phrase_lower.isascii() and rest and not rest[0].isspace():
            continue
        return mode, rest
    return None


def detect_library_filter_mode(text: str) -> str | None:
    normalized = " ".join(str(text or "").lower().split())
    hits: set[str] = set()
    for phrase, mode in TEXT_ARGUMENTS:
        if phrase.lower() in normalized:
            hits.add(mode)
    if len(hits) > 1:
        raise LibraryFilterModeError("排除已有和仅查看已有不能同时使用。")
    return next(iter(hits), None)


def resolve_library_filter_mode(*modes: str | None) -> str | None:
    normalized = [mode for mode in modes if mode]
    unknown = [mode for mode in normalized if mode not in LIBRARY_FILTER_MODES]
    if unknown:
        raise LibraryFilterModeError(f"未知游戏库过滤模式：{unknown[0]}")
    if len(set(normalized)) > 1:
        raise LibraryFilterModeError("排除已有和仅查看已有不能同时使用。")
    return normalized[0] if normalized else None


def filter_games_by_library_mode(
    games: list[RankedGame],
    owned_games: list[SteamOwnedGame],
    mode: str,
) -> tuple[list[RankedGame], int]:
    owned_appids = {owned.appid for owned in owned_games if owned.appid}
    if mode == LIBRARY_FILTER_EXCLUDE_OWNED:
        owned_families = {
            game_family_key(owned.name)
            for owned in owned_games
            if owned.name
        }
        filtered = [
            game
            for game in games
            if (game.appid is None or game.appid not in owned_appids)
            and game_family_key(game.title) not in owned_families
        ]
    elif mode == LIBRARY_FILTER_ONLY_OWNED:
        filtered = [game for game in games if game.appid is not None and game.appid in owned_appids]
    else:
        raise LibraryFilterModeError(f"未知游戏库过滤模式：{mode}")
    return filtered, len(games) - len(filtered)
