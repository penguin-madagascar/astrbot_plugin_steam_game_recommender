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
    "stardew-valley": ReferenceAlias(
        rawg_slug="stardew-valley",
        canonical_title="Stardew Valley",
        aliases=("Stardew Valley", "星露谷物语", "星露谷物語", "星露谷"),
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
    "stardew-valley": ReferenceProfile(
        rawg_slug="stardew-valley",
        genres_like=(
            "simulation",
            "casual",
            "rpg",
            "co-op",
            "multiplayer",
            "farming",
            "crafting",
            "building",
            "management",
            "relaxing",
        ),
        required_tags=("co-op", "farming"),
        seed_titles=(
            "Farm Together 2",
            "Roots of Pacha",
            "Sun Haven",
            "Dinkum",
            "Fae Farm",
            "My Time at Sandrock",
            "Core Keeper",
        ),
        seed_notes={
            "farm together 2": (
                "参考画像种子：多人农场经营、低压力种田循环接近 Stardew Valley",
            ),
            "roots of pacha": (
                "参考画像种子：合作农场生活、采集制作和村落经营接近 Stardew Valley",
            ),
            "sun haven": (
                "参考画像种子：多人农场生活、制作探索和轻 RPG 养成接近 Stardew Valley",
            ),
            "dinkum": (
                "参考画像种子：多人采集、建造、经营和休闲生活模拟",
            ),
            "fae farm": (
                "参考画像种子：合作农场、制作和轻松生活模拟",
            ),
            "my time at sandrock": (
                "参考画像种子：制作经营、城镇委托和生活模拟，节奏更偏工坊建造",
            ),
            "core keeper": (
                "参考画像种子：多人采集、建造和农场养成，探索战斗占比更高",
            ),
        },
        seed_warnings={
            "core keeper": (
                "地下探索和战斗占比较高，可能不如 Stardew Valley 轻松",
            ),
            "my time at sandrock": (
                "多人内容和主线体验需按具体版本确认",
            ),
        },
    ),
}
