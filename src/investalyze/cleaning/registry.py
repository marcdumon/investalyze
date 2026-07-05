"""Registry of persistent data-cleaning fixes.

`cleaning.toml` lists fix instances; each fix *type* is a module in this package exposing
`detect(con, fix) -> int` and `apply(con, fix) -> int`. `FIX_TYPES` maps each TOML section
name to its module. Adding a new instance of a known problem is a TOML entry; adding a new
kind of problem is one new module plus a `FIX_TYPES` line.
"""

import tomllib
from pathlib import Path

from investalyze.cleaning import delete_date_range
from investalyze.cleaning.fix import Fix

FIX_TYPES = {
    'delete_date_range': delete_date_range,
}

_REQUIRED = ('table', 'tickers', 'reason')


def read_fixes(path: Path) -> list[Fix]:
    """Parse `cleaning.toml` into `Fix` records.

    Every top-level array-of-tables section must be a known fix type, and every entry must
    carry `table`, `tickers` and `reason`; `start`/`end` are optional inclusive TOML dates.
    Raises ValueError on an unknown section or a missing required field.
    """
    raw = tomllib.loads(path.read_text())
    fixes: list[Fix] = []
    for section, entries in raw.items():
        if section not in FIX_TYPES:
            raise ValueError(f'unknown fix type {section!r}, known: {sorted(FIX_TYPES)}')
        if not isinstance(entries, list):
            raise ValueError(f'{section!r} must be an array of tables ([[{section}]]), got {type(entries).__name__}')
        for entry in entries:
            missing = [field for field in _REQUIRED if field not in entry]
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
                )
            )
    return fixes
