from __future__ import annotations

import asyncio
import logging
from datetime import date
from decimal import Decimal
from typing import Any, Callable

from ..storage.models import GamePreference, GamePriceSummary, RankedGame

logger = logging.getLogger(__name__)

try:
    from astrbot_plugin_steam_price_heybox.models import (
        PriceHistory,
        RegionPrice,
        SteamGameDetails,
    )
    from astrbot_plugin_steam_price_heybox.steam_price import (
        PriceLookupError,
        SteamPriceService,
        format_region_summary,
        format_sale_status,
        money_text,
        parse_country,
    )
except Exception as exc:  # pragma: no cover - depends on optional sibling plugin.
    PriceHistory = None  # type: ignore[assignment]
    PriceLookupError = RuntimeError  # type: ignore[assignment]
    RegionPrice = None  # type: ignore[assignment]
    SteamGameDetails = None  # type: ignore[assignment]
    SteamPriceService = None  # type: ignore[assignment]
    format_region_summary = None  # type: ignore[assignment]
    format_sale_status = None  # type: ignore[assignment]
    money_text = None  # type: ignore[assignment]
    parse_country = None  # type: ignore[assignment]
    PRICE_PLUGIN_IMPORT_ERROR: Exception | None = exc
else:
    PRICE_PLUGIN_IMPORT_ERROR = None

ServiceFactory = Callable[[dict[str, Any], Any], Any]
DEFAULT_PRICE_LOOKUP_LIMIT = 5
DEFAULT_HISTORY_DAYS = 720
DEFAULT_GLOBAL_PRICE_LIMIT = 10
DEFAULT_LANGUAGE = "schinese"


class SteamPriceBridge:
    def __init__(
        self,
        client: Any,
        config: Any,
        service_factory: ServiceFactory | None = None,
    ) -> None:
        self.default_country = normalize_country(str(config.get("default_region") or "CN"))
        self.lookup_limit = DEFAULT_PRICE_LOOKUP_LIMIT
        self.service: Any | None = None

        if client is None:
            return

        factory = service_factory or default_service_factory()
        if factory is None:
            if PRICE_PLUGIN_IMPORT_ERROR:
                logger.debug("Steam price bridge disabled: %s", PRICE_PLUGIN_IMPORT_ERROR)
            return

        price_config = {
            "default_country": self.default_country,
            "default_history_country": self.default_country,
            "default_language": DEFAULT_LANGUAGE,
            "history_days": DEFAULT_HISTORY_DAYS,
            "global_price_limit": DEFAULT_GLOBAL_PRICE_LIMIT,
            "show_api_links": False,
            "llm_name_retry_count": 0,
        }
        self.service = factory(price_config, client)

    def is_available(self) -> bool:
        return self.service is not None

    async def enrich_ranked_games(
        self,
        games: list[RankedGame],
        preference: GamePreference,
    ) -> list[RankedGame]:
        if not self.is_available() or self.lookup_limit <= 0:
            return games

        enriched: list[RankedGame] = []
        for index, game in enumerate(games):
            if index >= self.lookup_limit:
                enriched.append(game)
                continue
            summary = await self.lookup(game.title)
            enriched.append(attach_price_summary(game, summary, preference))

        enriched.sort(key=lambda item: item.score, reverse=True)
        return enriched

    async def lookup(self, title: str, country: str | None = None) -> GamePriceSummary | None:
        if not self.is_available() or not title.strip():
            return None

        try:
            resolved_country = normalize_country(country or self.default_country)
            identity, resolved_country = await self.service.resolve_game(title, resolved_country)
            details_result, history_result, regions_result = await asyncio.gather(
                self.service.steam_client.details(
                    identity.appid,
                    resolved_country,
                    self.service.default_language,
                ),
                self.service.load_history(identity.appid, resolved_country),
                self.service.heybox_client.global_prices(identity.appid),
                return_exceptions=True,
            )
        except PriceLookupError as exc:
            logger.debug("Steam price lookup skipped for %s: %s", title, exc)
            return None
        except Exception as exc:
            logger.warning("Steam price lookup failed for %s: %s", title, exc)
            return None

        details = details_result if is_steam_details(details_result) else None
        history = history_result if is_price_history(history_result) else None
        regions = regions_result if is_region_prices(regions_result) else []
        if details is None and history is None and not regions:
            return None

        return build_price_summary(identity.appid, resolved_country, details, history, regions)


def default_service_factory() -> ServiceFactory | None:
    if SteamPriceService is None:
        return None
    return SteamPriceService.from_config


def build_price_summary(
    appid: int,
    country: str,
    details: Any | None,
    history: Any | None,
    regions: list[Any],
) -> GamePriceSummary:
    current_price, current_cny = current_price_fields(details, history, regions, country)
    lowest_price, lowest_cny, lowest_date, lowest_discount = lowest_price_fields(history)
    sale_status = (
        "；".join(format_sale_status(history, service_today(history))) if history else None
    )
    region_summary = format_region_summary(regions) if regions and format_region_summary else None
    return GamePriceSummary(
        source="steam_price_heybox",
        appid=appid,
        country=country,
        current_price=current_price,
        lowest_price=lowest_price,
        lowest_date=lowest_date,
        lowest_discount=lowest_discount,
        sale_status=sale_status,
        region_summary=region_summary,
        store_url=f"https://store.steampowered.com/app/{appid}/",
        heybox_url=f"https://www.xiaoheihe.cn/app/topic/game/pc/{appid}",
        current_cny=current_cny,
        lowest_cny=lowest_cny,
    )


def current_price_fields(
    details: Any | None,
    history: Any | None,
    regions: list[Any],
    country: str,
) -> tuple[str | None, float | None]:
    if details and getattr(details, "is_free", False):
        return "免费", 0.0
    if (
        details
        and getattr(details, "coming_soon", False)
        and getattr(details, "price", None) is None
    ):
        return "尚未发售", None
    if details and getattr(details, "price", None):
        price = details.price
        text = money_text(price.current, price.currency) if money_text else str(price.current)
        return text, cny_value(price.current, price.currency) or region_cny(regions, country)
    if history and getattr(history, "current", None):
        current = history.current
        text = money_text(current.price, current.currency) if money_text else str(current.price)
        return text, (
            decimal_to_float(current.rmb_price)
            or cny_value(current.price, current.currency)
        )
    return None, region_cny(regions, country)


def lowest_price_fields(
    history: Any | None,
) -> tuple[str | None, float | None, str | None, int | None]:
    if not history or history.lowest_price is None:
        return None, None, None, None
    text = (
        money_text(history.lowest_price, history.lowest_currency)
        if money_text
        else str(history.lowest_price)
    )
    lowest_cny = cny_value(history.lowest_price, history.lowest_currency)
    if lowest_cny is None:
        lowest_cny = lowest_history_rmb(history)
    lowest_date = history.lowest_date.isoformat() if history.lowest_date else None
    return text, lowest_cny, lowest_date, int(history.lowest_discount or 0)


def attach_price_summary(
    game: RankedGame,
    summary: GamePriceSummary | None,
    preference: GamePreference,
) -> RankedGame:
    if summary is None:
        return game

    data = dump_model(game)
    data["price_summary"] = summary
    score = float(game.score)
    reasons = list(game.reasons)
    warnings = list(game.warnings)
    budget = preference.budget

    if budget is not None:
        if summary.current_cny is not None:
            if summary.current_cny <= budget:
                score += 8
                append_unique(
                    reasons,
                    (
                        f"当前价 {summary.current_price or format_cny(summary.current_cny)} "
                        f"在预算 {format_budget(budget)} 以内"
                    ),
                )
            elif summary.lowest_cny is not None and summary.lowest_cny <= budget:
                append_unique(
                    warnings,
                    (
                        f"当前价 {summary.current_price or format_cny(summary.current_cny)} "
                        f"高于预算 {format_budget(budget)}，但史低 "
                        f"{summary.lowest_price or format_cny(summary.lowest_cny)} "
                        "进过预算"
                    ),
                )
            else:
                score -= 8
                append_unique(
                    warnings,
                    (
                        f"当前价 {summary.current_price or format_cny(summary.current_cny)} "
                        f"高于预算 {format_budget(budget)}"
                    ),
                )
        elif summary.lowest_cny is not None and summary.lowest_cny <= budget:
            score += 1
            append_unique(
                reasons,
                (
                    f"史低 {summary.lowest_price or format_cny(summary.lowest_cny)} "
                    f"进过预算 {format_budget(budget)}"
                ),
            )

    data["score"] = round(score, 2)
    data["reasons"] = reasons
    data["warnings"] = warnings
    return validate_ranked_game(data)


def normalize_country(value: str) -> str:
    if parse_country:
        return parse_country(value) or "CN"
    text = value.strip()
    return text.upper() if len(text) == 2 and text.isalpha() else "CN"


def is_steam_details(value: Any) -> bool:
    return SteamGameDetails is not None and isinstance(value, SteamGameDetails)


def is_price_history(value: Any) -> bool:
    return PriceHistory is not None and isinstance(value, PriceHistory)


def is_region_prices(value: Any) -> bool:
    return isinstance(value, list) and (
        not value or RegionPrice is None or all(isinstance(item, RegionPrice) for item in value)
    )


def cny_value(value: Decimal, currency: str) -> float | None:
    return decimal_to_float(value) if currency.upper() == "CNY" else None


def region_cny(regions: list[Any], country: str) -> float | None:
    for region in regions:
        if getattr(region, "code", "").upper() == country.upper():
            return decimal_to_float(region.current_rmb)
    return None


def lowest_history_rmb(history: Any) -> float | None:
    for point in getattr(history, "points", ()):
        if point.price == history.lowest_price and point.rmb_price is not None:
            return decimal_to_float(point.rmb_price)
    return None


def decimal_to_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def service_today(history: Any) -> Any:
    active = getattr(history, "active_sale", None)
    if active and getattr(active, "ended_on", None) is None:
        return getattr(active, "started_on")
    current = getattr(history, "current", None)
    return (
        getattr(current, "recorded_on", None)
        or getattr(history, "lowest_date", None)
        or date.today()
    )


def append_unique(values: list[str], text: str) -> None:
    if text and text not in values:
        values.append(text)


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_ranked_game(data: dict[str, Any]) -> RankedGame:
    validator = getattr(RankedGame, "model_validate", None)
    return validator(data) if validator else RankedGame.parse_obj(data)


def format_budget(value: float) -> str:
    return f"¥{value:g}"


def format_cny(value: float) -> str:
    return f"¥{value:g}"
