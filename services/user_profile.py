from __future__ import annotations

import math
from typing import Any

from ..storage.models import GameCandidate, SteamOwnedGame
from .tag_normalizer import candidate_canonical_tags

PROFILE_IGNORED_TAGS = {"singleplayer", "chinese"}


async def load_bound_user_tag_weights(
    chat_platform: str,
    chat_user_id: str,
    cache: Any,
    steam_client: Any,
    index_entries: list[GameCandidate],
) -> dict[str, float]:
    binding = await cache.get_steam_account_binding(chat_platform, chat_user_id)
    if binding is None:
        return {}
    has_key = getattr(steam_client, "has_web_api_key", None)
    if not callable(has_key) or not has_key():
        return {}
    try:
        owned_games = await steam_client.get_owned_games(
            binding.steam_id64,
            binding_identity=(binding.chat_platform, binding.chat_user_id),
        )
    except Exception:
        return {}
    return build_user_tag_weights(owned_games, index_entries)


def build_user_tag_weights(
    owned_games: list[SteamOwnedGame],
    index_entries: list[GameCandidate],
    max_tags: int = 12,
) -> dict[str, float]:
    by_appid = {entry.appid: entry for entry in index_entries if entry.appid is not None}
    weights: dict[str, float] = {}
    for owned in owned_games:
        if owned.playtime_forever <= 0:
            continue
        candidate = by_appid.get(owned.appid)
        if candidate is None:
            continue
        playtime_weight = min(math.log1p(owned.playtime_forever / 60), 4.0)
        for tag in candidate_canonical_tags(candidate):
            if tag in PROFILE_IGNORED_TAGS:
                continue
            weights[tag] = weights.get(tag, 0.0) + playtime_weight

    if not weights:
        return {}

    max_weight = max(weights.values())
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:max_tags]
    return {tag: round(weight / max_weight, 4) for tag, weight in ranked if weight > 0}
