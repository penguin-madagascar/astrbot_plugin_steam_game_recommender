from __future__ import annotations

import unittest

from astrbot_plugin_steam_game_recommender.storage.cache_access import (
    set_json_with_ttl,
)


class CacheAccessTest(unittest.IsolatedAsyncioTestCase):
    async def test_passes_explicit_retention_to_policy_aware_cache(self) -> None:
        cache = PolicyAwareCache()

        await set_json_with_ttl(
            cache,
            "key",
            {"value": 1},
            ttl_hours=168,
            owner_scope="owner",
        )

        self.assertEqual(cache.write, ("key", {"value": 1}, 168, "owner"))

    async def test_keeps_test_and_legacy_cache_implementations_compatible(self) -> None:
        cache = LegacyCache()

        await set_json_with_ttl(cache, "key", {"value": 1}, ttl_hours=168)

        self.assertEqual(cache.write, ("key", {"value": 1}))


class PolicyAwareCache:
    def __init__(self) -> None:
        self.write = None

    async def set_json(
        self,
        key,
        payload,
        ttl_hours=24,
        *,
        owner_scope="",
    ) -> None:
        self.write = (key, payload, ttl_hours, owner_scope)


class LegacyCache:
    def __init__(self) -> None:
        self.write = None

    async def set_json(self, key, payload) -> None:
        self.write = (key, payload)


if __name__ == "__main__":
    unittest.main()
