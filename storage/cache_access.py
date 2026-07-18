from __future__ import annotations

import inspect
from typing import Any


async def set_json_with_ttl(
    cache: Any,
    key: str,
    payload: Any,
    *,
    ttl_hours: int | float,
    owner_scope: str = "",
) -> None:
    setter = cache.set_json
    parameters = inspect.signature(setter).parameters
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "ttl_hours" in parameters or accepts_keywords:
        kwargs: dict[str, Any] = {"ttl_hours": ttl_hours}
        if owner_scope and ("owner_scope" in parameters or accepts_keywords):
            kwargs["owner_scope"] = owner_scope
        await setter(key, payload, **kwargs)
        return
    await setter(key, payload)
