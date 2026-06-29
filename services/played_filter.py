from __future__ import annotations

from ..storage.models import RankedGame, SteamOwnedGame

PLAYED_EXCLUSION_PHRASES = (
    "排除已玩",
    "过滤已玩",
    "不要已玩",
    "别推荐已玩",
    "排除玩过",
    "过滤玩过",
    "不要玩过",
    "别推荐玩过",
)


def wants_played_game_exclusion(text: str) -> bool:
    normalized = "".join(str(text or "").split())
    return any(phrase in normalized for phrase in PLAYED_EXCLUSION_PHRASES)


def filter_played_games(
    games: list[RankedGame],
    owned_games: list[SteamOwnedGame],
) -> tuple[list[RankedGame], int]:
    played_appids = {
        owned.appid
        for owned in owned_games
        if owned.appid and owned.playtime_forever > 0
    }
    if not played_appids:
        return games, 0
    filtered = [
        game
        for game in games
        if game.appid is None or game.appid not in played_appids
    ]
    return filtered, len(games) - len(filtered)
