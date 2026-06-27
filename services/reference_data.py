from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReferenceAlias:
    rawg_slug: str
    canonical_title: str
    rawg_id: int | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceProfile:
    rawg_slug: str
    genres_like: tuple[str, ...] = ()
    required_tags: tuple[str, ...] = ()
    excluded_tags: tuple[str, ...] = ()
    seed_titles: tuple[str, ...] = ()
    seed_notes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    seed_warnings: dict[str, tuple[str, ...]] = field(default_factory=dict)


REFERENCE_ALIASES: dict[str, ReferenceAlias] = {
    "it-takes-two": ReferenceAlias(
        rawg_slug="it-takes-two",
        canonical_title="It Takes Two",
        aliases=("It Takes Two", "双人成行", "雙人成行"),
    ),
}

REFERENCE_PROFILES: dict[str, ReferenceProfile] = {
    "it-takes-two": ReferenceProfile(
        rawg_slug="it-takes-two",
        genres_like=(
            "co-op",
            "local co-op",
            "puzzle",
            "adventure",
            "casual",
            "platformer",
        ),
        required_tags=("co-op", "multiplayer"),
        seed_titles=(
            "Split Fiction",
            "Unravel Two",
            "PHOGS!",
            "Moving Out 2",
            "Overcooked! All You Can Eat",
            "KeyWe",
            "Sackboy: A Big Adventure",
        ),
        seed_notes={
            "split fiction": (
                "参考画像种子：双人合作、分屏/在线协作、解谜冒险节奏接近 It Takes Two",
            ),
            "unravel two": (
                "参考画像种子：双人合作、平台跳跃与轻解谜接近 It Takes Two",
            ),
            "phogs": (
                "参考画像种子：双人合作、轻解谜和低压力协作",
            ),
            "moving out 2": (
                "参考画像种子：本地/在线合作、偏轻松的协作闯关",
            ),
            "overcooked all you can eat": (
                "参考画像种子：多人合作和派对协作，节奏更忙乱",
            ),
            "keywe": (
                "参考画像种子：双人合作、沟通解谜和轻量关卡",
            ),
            "sackboy a big adventure": (
                "参考画像种子：合作平台动作，整体难度相对友好",
            ),
        },
        seed_warnings={
            "split fiction": (
                "Nintendo 侧为 Switch 2，不是原版 Switch；请按设备确认版本",
            ),
            "overcooked all you can eat": (
                "节奏偏忙乱，可能比 It Takes Two 更考验配合",
            ),
        },
    ),
}
