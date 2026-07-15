from __future__ import annotations

from dataclasses import dataclass

RUN_NOTICE_SEVERITIES = frozenset({"info", "warning", "error"})


@dataclass(frozen=True)
class RunNotice:
    code: str
    severity: str
    text: str

    def __post_init__(self) -> None:
        code = str(self.code or "").strip()
        severity = str(self.severity or "").strip().lower()
        text = str(self.text or "").strip()
        if not code or not text:
            raise ValueError("run notice code and text are required")
        if severity not in RUN_NOTICE_SEVERITIES:
            raise ValueError("invalid run notice severity")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "text", text)


def dedupe_run_notices(values: list[RunNotice] | tuple[RunNotice, ...]) -> tuple[RunNotice, ...]:
    result: list[RunNotice] = []
    seen: set[str] = set()
    for notice in values:
        if notice.code not in seen:
            result.append(notice)
            seen.add(notice.code)
    return tuple(result)
