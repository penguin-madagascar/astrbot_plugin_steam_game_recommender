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

from ..storage.models import (
    GamePreference,
    GamePriceSummary,
    RankedGame,
    RecommendationEvidence,
    ScoreBreakdown,
)
from .region_query import normalize_region, region_currency
from .similarity_ranker import clamp_score, ranked_game_sort_key

logger = logging.getLogger(__name__)

ServiceFactory = Callable[[dict[str, Any], Any], Any]
DEFAULT_PRICE_LOOKUP_LIMIT = 10
PRICE_LOOKUP_CONCURRENCY = 4
DEFAULT_HISTORY_DAYS = 720
DEFAULT_LANGUAGE = "schinese"
PRICE_PLUGIN_PACKAGE = "astrbot_plugin_steam_price_heybox"
PRICE_PLUGIN_IMPORT_ERROR: Exception | None = None
_PRICE_PLUGIN_SYMBOLS: "PricePluginSymbols | None" = None


@dataclass(frozen=True)
class PricePluginSymbols:
    history_class: type
    details_class: type
    lookup_error_class: type[Exception]
    service_class: type
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
            details_class=models.SteamGameDetails,
            lookup_error_class=steam_price.PriceLookupError,
            service_class=steam_price.SteamPriceService,
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
        today_provider: Callable[[], date] = date.today,
    ) -> None:
        self.default_country = normalize_country(str(config.get("default_region") or "CN"))
        self.lookup_limit = DEFAULT_PRICE_LOOKUP_LIMIT
        self.today_provider = today_provider
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
        if self.lookup_limit <= 0:
            return games
        if not self.is_available():
            if preference.budget is None:
                return games
            return [
                attach_missing_price_warning(game) if has_steam_purchase_signal(game) else game
                for game in games
            ]

        country = normalize_country(preference.region or self.default_country)
        semaphore = asyncio.Semaphore(PRICE_LOOKUP_CONCURRENCY)

        async def enrich_one(index: int, game: RankedGame) -> RankedGame:
            if index >= self.lookup_limit or not has_steam_purchase_signal(game):
                return game
            async with semaphore:
                summary = await self.lookup(game.title, country)
            return attach_price_summary(game, summary, preference)

        enriched = list(
            await asyncio.gather(*(enrich_one(index, game) for index, game in enumerate(games)))
        )
        return (
            sorted(enriched, key=ranked_game_sort_key)
            if preference.budget is not None
            else enriched
        )

    async def lookup(self, title: str, country: str | None = None) -> GamePriceSummary | None:
        if not self.is_available() or not title.strip():
            return None

        symbols = get_price_plugin_symbols()
        lookup_error = symbols.lookup_error_class if symbols else RuntimeError
        try:
            resolved_country = normalize_country(country or self.default_country)
            identity, resolved_country = await self.service.resolve_game(title, resolved_country)
            details_result, history_result = await asyncio.gather(
                self.service.steam_client.details(
                    identity.appid,
                    resolved_country,
                    self.service.default_language,
                ),
                self.service.load_history(identity.appid, resolved_country),
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
        if details is None and history is None:
            return None
        return build_price_summary(
            resolved_country,
            details,
            history,
            today=self.today_provider(),
        )


def default_service_factory() -> ServiceFactory | None:
    symbols = get_price_plugin_symbols()
    return symbols.service_class.from_config if symbols else None


def build_price_summary(
    country: str,
    details: Any | None,
    history: Any | None,
    today: date,
) -> GamePriceSummary:
    current_price, current_amount, current_currency = current_price_fields(
        details,
        history,
        country,
    )
    historic_low, historic_low_amount, historic_currency = historic_low_fields(history)
    recent_price, recent_amount, recent_currency, timing = recent_sale_fields(history, today)
    currency = current_currency or historic_currency or recent_currency or region_currency(country)
    return GamePriceSummary(
        region=country,
        currency=currency,
        current_price=current_price,
        current_amount=current_amount,
        historic_low=historic_low,
        historic_low_amount=historic_low_amount,
        recent_sale_price=recent_price,
        recent_sale_amount=recent_amount,
        sale_time_status=timing,
    )


def current_price_fields(
    details: Any | None,
    history: Any | None,
    country: str,
) -> tuple[str | None, float | None, str | None]:
    fallback_currency = region_currency(country)
    if details and getattr(details, "is_free", False):
        return "免费", 0.0, fallback_currency
    if (
        details
        and getattr(details, "coming_soon", False)
        and not getattr(
            details,
            "price",
            None,
        )
    ):
        return "尚未发售", None, fallback_currency
    if details and getattr(details, "price", None):
        price = details.price
        return (
            money_text_value(price.current, price.currency),
            decimal_to_float(price.current),
            normalize_currency(price.currency),
        )
    current = getattr(history, "current", None) if history else None
    if current:
        return (
            money_text_value(current.price, current.currency),
            decimal_to_float(current.price),
            normalize_currency(current.currency),
        )
    return None, None, fallback_currency


def historic_low_fields(
    history: Any | None,
) -> tuple[str | None, float | None, str | None]:
    if not history or getattr(history, "lowest_price", None) is None:
        return None, None, None
    currency = normalize_currency(getattr(history, "lowest_currency", ""))
    value = history.lowest_price
    return money_text_value(value, currency), decimal_to_float(value), currency


def recent_sale_fields(
    history: Any | None,
    today: date,
) -> tuple[str | None, float | None, str | None, str | None]:
    if history is None:
        return None, None, None, None
    active = getattr(history, "active_sale", None)
    if active is not None:
        days = max((today - active.started_on).days, 0)
        currency = normalize_currency(active.currency)
        return (
            money_text_value(active.lowest_price, currency),
            decimal_to_float(active.lowest_price),
            currency,
            f"已开始 {days} 天",
        )
    previous = getattr(history, "last_completed_sale", None)
    if previous is None or previous.ended_on is None:
        return None, None, None, None
    days = max((today - previous.ended_on).days, 0)
    currency = normalize_currency(previous.currency)
    return (
        money_text_value(previous.lowest_price, currency),
        decimal_to_float(previous.lowest_price),
        currency,
        f"结束于 {days} 天前",
    )


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
    adjustment = 0.0
    evidence = [item for item in game.recommendation_evidence if item.category != "budget"]
    budget = preference.budget
    if budget is not None:
        expected_currency = normalize_currency(
            preference.budget_currency or region_currency(preference.region or summary.region) or ""
        )
        summary_currency = normalize_currency(summary.currency or "")
        if not expected_currency or not summary_currency:
            evidence.append(
                budget_evidence(
                    "budget_currency_unknown",
                    "uncertain",
                    "价格币种未确认，未调整预算评分",
                )
            )
        elif expected_currency != summary_currency:
            evidence.append(
                budget_evidence(
                    "budget_currency_mismatch",
                    "uncertain",
                    (
                        f"预算币种 {expected_currency} 与价格币种 {summary_currency} 不一致，"
                        "未调整预算评分"
                    ),
                )
            )
        else:
            adjustment, budget_item = evaluate_budget(summary, budget, expected_currency)
            evidence.append(budget_item)

    data["score"] = clamp_score(game.score + adjustment)
    data["score_breakdown"] = copy_score_breakdown(
        game.score_breakdown,
        budget_adjustment=adjustment,
    )
    data["recommendation_evidence"] = evidence
    return validate_ranked_game(data)


def evaluate_budget(
    summary: GamePriceSummary,
    budget: float,
    currency: str,
) -> tuple[float, RecommendationEvidence]:
    budget_text = format_money(budget, currency)
    if summary.current_amount is None:
        return -2.0, budget_evidence(
            "budget_price_unknown",
            "uncertain",
            "当前价格未获取，预算匹配无法确认",
        )
    if summary.current_amount <= budget:
        return 5.0, budget_evidence(
            "budget_current_within",
            "positive",
            f"当前价 {summary.current_price} 在预算 {budget_text} 以内",
        )
    if summary.historic_low_amount is None:
        return -2.0, budget_evidence(
            "budget_history_unknown",
            "uncertain",
            f"当前价 {summary.current_price} 高于预算 {budget_text}，但史低未知",
        )
    if summary.historic_low_amount <= budget:
        return 0.0, budget_evidence(
            "budget_historic_within",
            "negative",
            (
                f"当前价 {summary.current_price} 高于预算 {budget_text}，"
                f"但史低 {summary.historic_low} 进过预算"
            ),
        )
    return -5.0, budget_evidence(
        "budget_over",
        "negative",
        (f"当前价 {summary.current_price} 与史低 {summary.historic_low} 都高于预算 {budget_text}"),
        important=True,
    )


def attach_missing_price_warning(game: RankedGame) -> RankedGame:
    data = dump_model(game)
    evidence = [item for item in game.recommendation_evidence if item.category != "budget"]
    evidence.append(
        budget_evidence(
            "budget_price_unknown",
            "uncertain",
            "Steam 价格未获取到，预算匹配无法确认",
        )
    )
    data["score"] = clamp_score(game.score - 2)
    data["score_breakdown"] = copy_score_breakdown(
        game.score_breakdown,
        budget_adjustment=-2.0,
    )
    data["recommendation_evidence"] = evidence
    return validate_ranked_game(data)


def budget_evidence(
    evidence_id: str,
    sentiment: str,
    text: str,
    important: bool = False,
) -> RecommendationEvidence:
    return RecommendationEvidence(
        evidence_id=evidence_id,
        category="budget",
        sentiment=sentiment,
        text=text,
        important=important,
    )


def copy_score_breakdown(
    breakdown: ScoreBreakdown,
    budget_adjustment: float,
) -> ScoreBreakdown:
    copier = getattr(breakdown, "model_copy", None)
    if copier:
        return copier(update={"budget_adjustment": budget_adjustment})
    return breakdown.copy(update={"budget_adjustment": budget_adjustment})


def normalize_country(value: str) -> str:
    symbols = get_price_plugin_symbols()
    if symbols:
        return symbols.parse_country(value) or normalize_region(value)
    return normalize_region(value)


def normalize_currency(value: str) -> str:
    return str(value or "").strip().upper()


def is_steam_details(value: Any) -> bool:
    symbols = get_price_plugin_symbols()
    return symbols is not None and isinstance(value, symbols.details_class)


def is_price_history(value: Any) -> bool:
    symbols = get_price_plugin_symbols()
    return symbols is not None and isinstance(value, symbols.history_class)


def money_text_value(value: Decimal, currency: str) -> str:
    symbols = get_price_plugin_symbols()
    return symbols.money_text(value, currency) if symbols else format_money(float(value), currency)


def format_money(value: float, currency: str) -> str:
    symbols = {"CNY": "¥", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    amount = f"{float(value):.2f}".rstrip("0").rstrip(".")
    code = normalize_currency(currency)
    symbol = symbols.get(code)
    return f"{symbol}{amount}" if symbol else f"{amount} {code}".strip()


def has_steam_purchase_signal(game: RankedGame) -> bool:
    return game.appid is not None or any(
        term in " | ".join([*game.platforms, *game.stores]).lower()
        for term in ("steam", "pc", "windows")
    )


def decimal_to_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def dump_model(model: Any) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    return dumper() if dumper else model.dict()


def validate_ranked_game(data: dict[str, Any]) -> RankedGame:
    validator = getattr(RankedGame, "model_validate", None)
    return validator(data) if validator else RankedGame.parse_obj(data)
