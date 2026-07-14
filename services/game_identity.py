from __future__ import annotations

from ..storage.models import GameCandidate


def is_confirmed_base_game(candidate: GameCandidate) -> bool:
    return candidate.app_type == "game"
