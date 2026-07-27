"""Registry of persistent data-cleaning fixes.

`cleaning.toml` lists fix instances; each fix *type* is a module in this package exposing
`detect(con, fix) -> int` and `apply(con, fix) -> int`. `FIX_TYPES` maps each TOML section
name to its module. Adding a new instance of a known problem is a TOML entry; adding a new
kind of problem is one new module plus a `FIX_TYPES` line.
"""

import tomllib
from pathlib import Path

from investalyze.cleaning import delete_date_range, rebuild_adjusted_close, rename_ticker, repair_zero_low, repair_zero_open, set_value
from investalyze.cleaning.fix import Fix

FIX_TYPES = {
    'delete_date_range': delete_date_range,
    'rebuild_adjusted_close': rebuild_adjusted_close,
    'rename_ticker': rename_ticker,
    'repair_zero_low': repair_zero_low,
    'repair_zero_open': repair_zero_open,
    'set_value': set_value,
}

_REQUIRED = ('tickers', 'reason')
_REQUIRED_EXTRA = {
    'delete_date_range': ('table',),
    'rebuild_adjusted_close': ('table',),
    'repair_zero_low': ('table',),
    'repair_zero_open': ('table',),
    'set_value': ('table', 'column'),
    'rename_ticker': ('new_ticker',),
}


def parse_fixes(text: str) -> list[Fix]:
    """Parse cleaning TOML text into `Fix` records.

    Every top-level array-of-tables section must be a known fix type. All entries require
    `tickers` and `reason`. Additional required fields depend on fix type: `table` for most
    types (delete_date_range, rebuild_adjusted_close, repair_zero_low, repair_zero_open),
    `column` for set_value, and `new_ticker` for rename_ticker. `start`/`end` are optional
    inclusive TOML dates. Raises ValueError on unknown section, missing required field, or
    validation error (e.g. rename_ticker with != 1 ticker).
    """
    raw = tomllib.loads(text)
    fixes: list[Fix] = []
    for section, entries in raw.items():
        if section not in FIX_TYPES:
            raise ValueError(f'unknown fix type {section!r}, known: {sorted(FIX_TYPES)}')
        if not isinstance(entries, list):
            raise ValueError(f'{section!r} must be an array of tables ([[{section}]]), got {type(entries).__name__}')
        required = _REQUIRED + _REQUIRED_EXTRA.get(section, ())
        for entry in entries:
            missing = [field for field in required if field not in entry]
            if missing:
                raise ValueError(f'[[{section}]] entry missing required field(s): {", ".join(missing)}')
            if section == 'rename_ticker' and len(entry['tickers']) != 1:
                raise ValueError(f'[[{section}]] entry needs exactly one ticker, got {len(entry["tickers"])}')
            fixes.append(
                Fix(
                    fix_type=section,
                    table=entry.get('table'),
                    tickers=list(entry['tickers']),
                    start=entry.get('start'),
                    end=entry.get('end'),
                    reason=entry['reason'],
                    column=entry.get('column'),
                    value=entry.get('value'),
                    new_ticker=entry.get('new_ticker'),
                )
            )
    return fixes


def read_fixes(path: Path) -> list[Fix]:
    """Parse the cleaning TOML file at `path` into `Fix` records."""
    return parse_fixes(path.read_text())
