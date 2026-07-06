from __future__ import annotations

import re
from dataclasses import dataclass

RETRY_PREFIXES = (
    "重新推荐",
    "换一批",
    "再推荐",
    "gamerec_retry",
    "retry",
)


@dataclass(frozen=True)
class RetryRequest:
    is_retry: bool
    supplement: str = ""


def parse_retry_request(text: str) -> RetryRequest:
    stripped = str(text or "").strip()
    lowered = stripped.lower()
    for prefix in RETRY_PREFIXES:
        key = prefix.lower()
        if lowered == key:
            return RetryRequest(is_retry=True)
        if lowered.startswith(key):
            rest = stripped[len(prefix):]
            if key.isascii() and rest and not rest[0].isspace():
                continue
            supplement = re.sub(r"^[\s,，、;；:：]+", "", rest).strip()
            return RetryRequest(is_retry=True, supplement=supplement)
    return RetryRequest(is_retry=False)
