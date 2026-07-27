"""The `Fix` record every fix-type module and the registry share."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Fix:
    """One cleaning fix instance from `cleaning.toml`; `column`/`value` are used by set_value,
    `new_ticker` by rename_ticker."""

    fix_type: str
    table: str | None
    tickers: list[str]
    start: date | None
    end: date | None
    reason: str
    column: str | None = None
    value: float | None = None
    new_ticker: str | None = None
