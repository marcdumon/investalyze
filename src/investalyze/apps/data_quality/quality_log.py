"""Triage log: which anomalies have been reviewed, with a tag and a comment.

`quality_log.toml` lists `[[log]]` entries, each keyed by `(check, ticker, date?, key?)` plus a tag
and free-text comment (severity/table/details are recorded too, so the file reads as a standalone
log of the problems seen). The data-quality page hides any anomaly that has a log entry, so the
browser always shows the not-yet-reviewed findings. The log is a non-destructive overlay: it never
touches raw tables or the anomalies table. `date`/`key` are optional (a finding may have neither),
so matching treats a missing value as its own value.
"""

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from investalyze.apps.data_quality import toml_io

# canned classifications offered in the GUI; a free-text comment always accompanies the choice
STANDARD_TAGS = ['real-problem', 'false-alarm', 'investigate', 'known']

_REQUIRED = ('check', 'ticker', 'tag')


@dataclass(frozen=True)
class LogEntry:
    """One reviewed-anomaly entry from `quality_log.toml`."""

    check: str
    ticker: str
    date: date | None
    key: str | None
    tag: str
    comment: str
    severity: str
    src_table: str
    details: str


def parse_log(text: str) -> list[LogEntry]:
    """Parse quality_log.toml text into `LogEntry` records.

    The only allowed section is `[[log]]`; each entry needs `check`, `ticker` and `tag`, with optional
    `date`, `key`, `comment`, `severity`, `table` and `details`. Raises ValueError on an unknown
    section, a non-array section, or a missing required field.
    """
    raw = tomllib.loads(text)
    entries: list[LogEntry] = []
    for section, rows in raw.items():
        if section != 'log':
            raise ValueError(f'unknown log section {section!r}, expected [[log]]')
        if not isinstance(rows, list):
            raise ValueError(f'{section!r} must be an array of tables ([[log]]), got {type(rows).__name__}')
        for entry in rows:
            missing = [name for name in _REQUIRED if name not in entry]
            if missing:
                raise ValueError(f'[[log]] entry missing required field(s): {", ".join(missing)}')
            entries.append(
                LogEntry(
                    check=entry['check'],
                    ticker=entry['ticker'],
                    date=entry.get('date'),
                    key=entry.get('key'),
                    tag=entry['tag'],
                    comment=entry.get('comment', ''),
                    severity=entry.get('severity', ''),
                    src_table=entry.get('table', ''),
                    details=entry.get('details', ''),
                )
            )
    return entries


def read_log(path: Path) -> list[LogEntry]:
    """Parse the `quality_log.toml` at `path`, returning an empty list when the file does not exist."""
    if not path.exists():
        return []
    return parse_log(path.read_text())


def log_keys_frame(entries: list[LogEntry]) -> pd.DataFrame:
    """Frame of the reviewed findings' identity columns, for anti-joining against the anomalies table.

    Dates become ISO strings (cast back to DATE in the query); a missing date/key is null (None/NaN),
    which DuckDB reads as SQL NULL.
    """
    frame = pd.DataFrame(
        [{'check_name': e.check, 'ticker': e.ticker,
          'log_date': e.date.isoformat() if isinstance(e.date, date) else None, 'log_key': e.key}
         for e in entries],
        columns=['check_name', 'ticker', 'log_date', 'log_key'],
    )
    return frame.where(frame.notna(), None)


def append_log(path: Path, block: str) -> str:
    """Append a pre-serialized `[[log]]` block to quality_log.toml, validating the whole file first."""
    return toml_io.append_block(path, block, parse_log)
