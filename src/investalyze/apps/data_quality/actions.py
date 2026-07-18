"""Pure helpers for the data-quality page: normalize anomaly rows and build the log-entry field dict.

An anomaly row is a plain dict as AG Grid hands it back (CheckName, Severity, SrcTable, Ticker,
Date, Key, Details); Date arrives as an ISO string, Key/Date may be missing. These helpers coerce
those cells and build the ordered field dict that `toml_io.serialize_block` turns into a
quality_log.toml `[[log]]` entry. No Dash, no DB.
"""

import re
from datetime import date, datetime

import pandas as pd

_INT_RUN = re.compile(r'(?<![\d.])(\d{5,})(\.\d+)?')

# Details prefix of the tolerance-based identity checks -> the line items its formula uses
_IDENTITY_ITEMS = {
    'liab+equity': ('Total Liabilities', 'Total Equity', 'Total Assets'),
    'cur+noncur assets': ('Total Current Assets', 'Total Noncurrent Assets', 'Total Assets'),
    'cur+noncur liab': ('Total Current Liabilities', 'Total Noncurrent Liabilities', 'Total Liabilities'),
    'rev+cogs': ('Revenue', 'Cost of Revenue', 'Gross Profit'),
    'gp+opex': ('Gross Profit', 'Operating Expenses', 'Other Operating Income', 'Operating Income (Loss)'),
    'oi+nonop': ('Operating Income (Loss)', 'Non-Operating Income (Loss)', 'Pretax Income (Loss), Adj.'),
    'op+inv+fin': ('Net Cash from Operating Activities', 'Net Cash from Investing Activities',
                   'Net Cash from Financing Activities', 'Effect of Foreign Exchange Rates',
                   'Change in Cash from Disc. Operations and Other', 'Net Change in Cash'),
}


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


def format_details_numbers(text: str) -> str:
    """Insert thousand separators into every number with 5+ integer digits in `text`.

    Shorter runs (day counts, percentages, the 4-digit years inside dates) stay untouched;
    a decimal fraction is kept as written.
    """
    return _INT_RUN.sub(lambda m: f'{int(m.group(1)):,}{m.group(2) or ""}', text)


def involved_items(check: str, details: str) -> set[str]:
    """The line items a fundamentals check's calculation uses, derived from its Details prefix."""
    if check == 'quarters_vs_fy':
        return {details.split(':', 1)[0].strip()}
    if check in ('balance_identity', 'balance_subtotals', 'income_chain', 'cashflow_identity'):
        return set(_IDENTITY_ITEMS.get(details.split(':', 1)[0].strip(), ()))
    if check == 'hard_invariants':
        return {'Shares (Basic)', 'Shares (Diluted)'} if details.startswith('Shares') else {'Total Assets'}
    if check == 'negative_revenue':
        return {'Revenue'}
    return set()


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
