"""Registry of persistent data-cleaning fixes.

`cleaning.toml` lists fix instances; each fix *type* is a module in this package exposing
`detect(con, fix) -> int` and `apply(con, fix) -> int`. `FIX_TYPES` maps each TOML section
name to its module. Adding a new instance of a known problem is a TOML entry; adding a new
kind of problem is one new module plus a `FIX_TYPES` line.
"""

import tomllib
from pathlib import Path

from investalyze.cleaning import delete_date_range, rebuild_adjusted_close, repair_zero_low, repair_zero_open, set_value
from investalyze.cleaning.fix import Fix

FIX_TYPES = {
    'delete_date_range': delete_date_range,
    'rebuild_adjusted_close': rebuild_adjusted_close,
    'repair_zero_low': repair_zero_low,
    'repair_zero_open': repair_zero_open,
    'set_value': set_value,
}

_REQUIRED = ('table', 'tickers', 'reason')
_REQUIRED_EXTRA = {'set_value': ('column',)}


def parse_fixes(text: str) -> list[Fix]:
    """Parse cleaning TOML text into `Fix` records.

    Every top-level array-of-tables section must be a known fix type, and every entry must
    carry `table`, `tickers` and `reason` (plus `column` for set_value); `start`/`end` are
    optional inclusive TOML dates. Raises ValueError on an unknown section or a missing field.
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
            fixes.append(
                Fix(
                    fix_type=section,
                    table=entry['table'],
                    tickers=list(entry['tickers']),
                    start=entry.get('start'),
                    end=entry.get('end'),
                    reason=entry['reason'],
                    column=entry.get('column'),
                    value=entry.get('value'),
                )
            )
    return fixes


def read_fixes(path: Path) -> list[Fix]:
    """Parse the cleaning TOML file at `path` into `Fix` records."""
    return parse_fixes(path.read_text())
