from __future__ import annotations


class IgdbClient:
    """IGDB client placeholder for future enrichment."""

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def search_games(self, query: str) -> list[dict]:
        # TODO: Implement IGDB recall/enrichment after RAWG MVP is stable.
        return []

