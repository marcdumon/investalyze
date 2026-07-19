"""The `Fix` record every fix-type module and the registry share."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fix:
    """One cleaning fix instance from `cleaning.toml`; `column`/`value` are used by set_value."""

    fix_type: str
    table: str
    tickers: list[str]
    start: date | None
    end: date | None
    reason: str
    column: str | None = None
    value: float | None = None
