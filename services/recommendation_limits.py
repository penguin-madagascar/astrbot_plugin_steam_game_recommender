from __future__ import annotations

DEFAULT_RECOMMENDATION_COUNT = 10
MAX_RECOMMENDATION_COUNT = 10


def effective_result_limit(
    configured_max_results: int | None,
    requested_count: int | None,
) -> int:
    configured = _bounded_count(configured_max_results, DEFAULT_RECOMMENDATION_COUNT)
    requested = _bounded_count(requested_count, DEFAULT_RECOMMENDATION_COUNT)
    return min(configured, requested)


def _bounded_count(value: int | None, default: int) -> int:
    try:
        count = default if value is None else int(value)
    except (TypeError, ValueError):
        count = default
    return min(max(count, 1), MAX_RECOMMENDATION_COUNT)
