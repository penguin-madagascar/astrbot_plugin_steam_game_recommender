from __future__ import annotations


class ItadClient:
    """IsThereAnyDeal client placeholder for future price history support."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def lookup_prices(self, title: str, region: str = "CN") -> dict | None:
        # TODO: Implement official ITAD API calls for regional current price and lows.
        return None

