from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from ..storage.models import GamePreference, GamePriceSummary, RankedGame

logger = logging.getLogger(__name__)

ServiceFactory = Callable[[dict[str, Any], Any], Any]
DEFAULT_PRICE_LOOKUP_LIMIT = 10
PRICE_LOOKUP_CONCURRENCY = 4
DEFAULT_HISTORY_DAYS = 720
DEFAULT_GLOBAL_PRICE_LIMIT = 10
DEFAULT_LANGUAGE = "schinese"
PRICE_PLUGIN_PACKAGE = "astrbot_plugin_steam_price_heybox"
PRICE_PLUGIN_IMPORT_ERROR: Exception | None = None
_PRICE_PLUGIN_SYMBOLS: "PricePluginSymbols | None" = None


@dataclass(frozen=True)
class PricePluginSymbols:
    history_class: type
    region_class: type
    details_class: type
    lookup_error_class: type[Exception]
    service_class: type
    format_region_summary: Callable[[list[Any]], str]
    format_sale_status: Callable[[Any, Any], list[str]]
    money_text: Callable[[Decimal, str], str]
    parse_country: Callable[[str], str]


def load_price_plugin_symbols(
    search_roots: list[Path] | None = None,
) -> PricePluginSymbols:
    roots = list(search_roots or sibling_plugin_search_roots())
    errors: list[Exception] = []
    for root in [None, *roots]:
        try:
            return import_price_plugin_symbols(root)
        except Exception as exc:
            errors.append(exc)
    raise errors[-1] if errors else ModuleNotFoundError(PRICE_PLUGIN_PACKAGE)


def import_price_plugin_symbols(search_root: Path | None) -> PricePluginSymbols:
    inserted = False
    if search_root is not None:
        root_text = str(search_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
    try:
        models = importlib.import_module(f"{PRICE_PLUGIN_PACKAGE}.models")
        steam_price = importlib.import_module(f"{PRICE_PLUGIN_PACKAGE}.steam_price")
        return PricePluginSymbols(
            history_class=models.PriceHistory,
            region_class=models.RegionPrice,
            details_class=models.SteamGameDetails,
            lookup_error_class=steam_price.PriceLookupError,
            service_class=steam_price.SteamPriceService,
            format_region_summary=steam_price.format_region_summary,
            format_sale_status=steam_price.format_sale_status,
            money_text=steam_price.money_text,
            parse_country=steam_price.parse_country,
        )
    finally:
        if inserted:
            try:
                sys.path.remove(str(search_root))
            except ValueError:
                pass


def sibling_plugin_search_roots() -> list[Path]:
    current_plugin = Path(__file__).resolve().parents[1]
    roots = [current_plugin.parent]
    return [root for root in roots if (root / PRICE_PLUGIN_PACKAGE).is_dir()]


def get_price_plugin_symbols() -> PricePluginSymbols | None:
    global PRICE_PLUGIN_IMPORT_ERROR, _PRICE_PLUGIN_SYMBOLS
    if _PRICE_PLUGIN_SYMBOLS is not None:
        return _PRICE_PLUGIN_SYMBOLS
    try:
        _PRICE_PLUGIN_SYMBOLS = load_price_plugin_symbols()
    except Exception as exc:  # pragma: no cover - depends on optional sibling plugin.
        PRICE_PLUGIN_IMPORT_ERROR = exc
        return None
    PRICE_PLUGIN_IMPORT_ERROR = None
    return _PRICE_PLUGIN_SYMBOLS


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

        semaphore = asyncio.Semaphore(PRICE_LOOKUP_CONCURRENCY)

        async def enrich_one(index: int, game: RankedGame) -> RankedGame:
            if index >= self.lookup_limit or not has_steam_purchase_signal(game):
                return game
            async with semaphore:
                summary = await self.lookup(game.title)
            return attach_price_summary(game, summary, preference)

        enriched = list(
            await asyncio.gather(*(enrich_one(index, game) for index, game in enumerate(games)))
        )
        if preference.budget is None:
            return enriched
        original_order = {id(game): index for index, game in enumerate(enriched)}
        return sorted(
            enriched,
            key=lambda game: (
                tier_order(game.tier),
                -float(game.score),
                original_order[id(game)],
            ),
        )

    async def lookup(self, title: str, country: str | None = None) -> GamePriceSummary | None:
        if not self.is_available() or not title.strip():
            return None

        symbols = get_price_plugin_symbols()
        lookup_error = symbols.lookup_error_class if symbols else RuntimeError

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
        except lookup_error as exc:
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
    symbols = get_price_plugin_symbols()
    if symbols is None:
        return None
    return symbols.service_class.from_config


def build_price_summary(
    appid: int,
    country: str,
    details: Any | None,
    history: Any | None,
    regions: list[Any],
) -> GamePriceSummary:
    current_price, current_cny = current_price_fields(details, history, regions, country)
    lowest_price, lowest_cny, lowest_date, lowest_discount = lowest_price_fields(history)
    symbols = get_price_plugin_symbols()
    sale_status = (
        "；".join(symbols.format_sale_status(history, service_today(history)))
        if history and symbols
        else None
    )
    region_summary = symbols.format_region_summary(regions) if regions and symbols else None
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
        text = money_text_value(price.current, price.currency)
        return text, cny_value(price.current, price.currency) or region_cny(regions, country)
    if history and getattr(history, "current", None):
        current = history.current
        text = money_text_value(current.price, current.currency)
        return text, (
            decimal_to_float(current.rmb_price) or cny_value(current.price, current.currency)
        )
    return None, region_cny(regions, country)


def lowest_price_fields(
    history: Any | None,
) -> tuple[str | None, float | None, str | None, int | None]:
    if not history or history.lowest_price is None:
        return None, None, None, None
    text = money_text_value(history.lowest_price, history.lowest_currency)
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
        if preference.budget is not None and has_steam_purchase_signal(game):
            return attach_missing_price_warning(game)
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
                score += 5
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
                score -= 5
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
    symbols = get_price_plugin_symbols()
    if symbols:
        return symbols.parse_country(value) or "CN"
    text = value.strip()
    return text.upper() if len(text) == 2 and text.isalpha() else "CN"


def is_steam_details(value: Any) -> bool:
    symbols = get_price_plugin_symbols()
    return symbols is not None and isinstance(value, symbols.details_class)


def is_price_history(value: Any) -> bool:
    symbols = get_price_plugin_symbols()
    return symbols is not None and isinstance(value, symbols.history_class)


def is_region_prices(value: Any) -> bool:
    symbols = get_price_plugin_symbols()
    return isinstance(value, list) and (
        not value
        or symbols is None
        or all(isinstance(item, symbols.region_class) for item in value)
    )


def money_text_value(value: Decimal, currency: str) -> str:
    symbols = get_price_plugin_symbols()
    return symbols.money_text(value, currency) if symbols else str(value)


def has_steam_purchase_signal(game: RankedGame) -> bool:
    haystack = " | ".join([*game.platforms, *game.stores]).lower()
    return any(term in haystack for term in ("steam", "pc", "windows"))


def attach_missing_price_warning(game: RankedGame) -> RankedGame:
    data = dump_model(game)
    warnings = list(game.warnings)
    append_unique(warnings, "Steam 价格未获取到，预算匹配无法确认")
    data["score"] = round(float(game.score) - 2, 2)
    data["warnings"] = warnings
    return validate_ranked_game(data)


def tier_order(tier: str) -> int:
    return {"strong": 0, "recommended": 1, "backup": 2}.get(tier, 9)


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
        return active.started_on
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
