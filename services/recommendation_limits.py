from __future__ import annotations


def effective_result_limit(configured_max_results: int, requested_count: int | None) -> int:
    configured = min(max(int(configured_max_results or 5), 1), 10)
    if requested_count is None:
        return configured
    requested = min(max(int(requested_count), 1), 10)
    return min(configured, requested)
