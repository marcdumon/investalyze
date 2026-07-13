"""Pure helpers for the data-quality page: normalize anomaly rows and build the log-entry field dict.

An anomaly row is a plain dict as AG Grid hands it back (CheckName, Severity, SrcTable, Ticker,
Date, Key, Details); Date arrives as an ISO string, Key/Date may be missing. These helpers coerce
those cells and build the ordered field dict that `toml_io.serialize_block` turns into a
quality_log.toml `[[log]]` entry. No Dash, no DB.
"""

from datetime import date, datetime

import pandas as pd


def _is_missing(value: object) -> bool:
    """True for None, NaN or NaT cells."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_date(value: object) -> date | None:
    """Coerce a cell (date, Timestamp, 'YYYY-MM-DD', None/NaN) to a date, or None."""
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return date.fromisoformat(text[:10]) if text else None


def _clean_str(value: object) -> str | None:
    """Return a stripped string, or None for missing/empty cells."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def parse_key(key: object) -> dict | None:
    """Split a fundamentals Key 'Market|Period|Fiscal Year|Fiscal Period|IsRestated' into its parts.

    Returns None when the key is missing or is not the 5-part fundamentals form.
    """
    text = _clean_str(key)
    if text is None:
        return None
    parts = text.split('|')
    if len(parts) != 5:
        return None
    market, period, fiscal_year, fiscal_period, is_restated = parts
    return {
        'market': market,
        'period': period,
        'fiscal_year': int(fiscal_year) if fiscal_year.isdigit() else None,
        'fiscal_period': fiscal_period,
        'is_restated': is_restated,
    }


def log_fields(row: dict, tag: str, comment: str) -> dict:
    """Ordered field dict for a quality_log.toml [[log]] block from an anomaly row."""
    return {
        'check': row['CheckName'],
        'ticker': row['Ticker'],
        'date': _to_date(row.get('Date')),
        'key': _clean_str(row.get('Key')),
        'tag': tag,
        'severity': _clean_str(row.get('Severity')),
        'table': _clean_str(row.get('SrcTable')),
        'details': _clean_str(row.get('Details')),
        'comment': _clean_str(comment),
    }
