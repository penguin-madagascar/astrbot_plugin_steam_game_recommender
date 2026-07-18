from __future__ import annotations

import hashlib
import logging
import re
import secrets
from typing import Any

SAFE_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_:-]{0,63}")


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
    correlation_id = secrets.token_hex(6)
    error_code = safe_error_code(exc)
    status_code = safe_status_code(exc)
    values = (
        event,
        stage,
        type(exc).__name__,
        error_code,
        status_code,
        error_id,
        correlation_id,
    )
    template = (
        "%s stage=%s error_type=%s error_code=%s status_code=%s "
        "error_id=%s correlation_id=%s"
    )
    log_method = getattr(logger, "log", None)
    if callable(log_method):
        log_method(
            level,
            template,
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
            template,
            *values,
        )
    return error_id


def safe_error_code(exc: BaseException) -> str:
    code = str(getattr(exc, "code", "") or "").strip().lower()
    return code if SAFE_ERROR_CODE_PATTERN.fullmatch(code) else "unknown"


def safe_status_code(exc: BaseException) -> int | str:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, bool):
        return "none"
    try:
        parsed = int(status_code)
    except (TypeError, ValueError):
        return "none"
    return parsed if 100 <= parsed <= 599 else "none"
