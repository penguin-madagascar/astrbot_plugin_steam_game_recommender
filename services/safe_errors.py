from __future__ import annotations

import hashlib
import logging
from typing import Any


def safe_error_id(stage: str, exc: BaseException) -> str:
    identity = f"{stage}:{type(exc).__module__}.{type(exc).__qualname__}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]


def log_external_failure(
    logger: Any,
    event: str,
    *,
    stage: str,
    exc: BaseException,
    level: int = logging.WARNING,
) -> str:
    error_id = safe_error_id(stage, exc)
    values = (event, stage, type(exc).__name__, error_id)
    log_method = getattr(logger, "log", None)
    if callable(log_method):
        log_method(
            level,
            "%s stage=%s error_type=%s error_id=%s",
            *values,
        )
    else:
        method_name = (
            "debug"
            if level <= logging.DEBUG
            else "info"
            if level <= logging.INFO
            else "warning"
            if level <= logging.WARNING
            else "error"
            if level <= logging.ERROR
            else "critical"
        )
        getattr(logger, method_name)(
            "%s stage=%s error_type=%s error_id=%s",
            *values,
        )
    return error_id
