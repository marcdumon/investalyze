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
_DECIMAL_RUN = re.compile(r'(?<![\d.])(-?\d+\.\d{5,})(?!\d)')

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
    """Round runs of 5+ decimal digits to 4 places, then add thousand separators to 5+ digit integer runs.

    Vendor prices arrive as float32 cast to double and print as noise (0.2800000011920929
    instead of 0.28) unless rounded here; shorter decimals (percentages) and integer runs
    (day counts, the 4-digit years inside dates) stay untouched.
    """
    text = _DECIMAL_RUN.sub(lambda m: f'{float(m.group(1)):.4f}'.rstrip('0').rstrip('.'), text)
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


# tables keyed by (Ticker, Date), where date-scoped fixes and set_value can address a row
DATE_KEYED_TABLES = ('prices', 'market_data', 'dividends')

FIX_ACTIONS = {
    'delete_rows': 'delete rows (each selected date)',
    'delete_span': 'delete date range (selection span)',
    'delete_ticker': 'delete entire ticker history',
    'set_value': 'correct a value',
    'clear_value': 'clear a value (set NULL)',
    'rebuild_adjusted_close': 'rebuild adjusted close',
}

_DATE_SCOPED = ('delete_rows', 'delete_span', 'set_value', 'clear_value')


def fix_entries(action: str, rows: list[dict], column: str | None, value: float | None,
                reason: str) -> list[tuple[str, dict]]:
    """Build the (section, fields) cleaning.toml entries `action` produces for the selected anomaly rows.

    Rows are grouped so each entry covers as many rows as the fix type allows (tickers merged per
    table/date, spans merged per ticker). Raises ValueError with a user-facing message when the
    selection or inputs cannot express the action.
    """
    if action not in FIX_ACTIONS:
        raise ValueError('choose an action')
    if not rows:
        raise ValueError('check or click at least one anomaly row first')
    reason = reason.strip()
    if not reason:
        raise ValueError('a reason is required (it is stored with the fix)')

    tables = sorted({str(row.get('SrcTable') or '') for row in rows})
    if '' in tables:
        raise ValueError('every selected row needs a source table')

    if action in _DATE_SCOPED:
        off_key = [table for table in tables if table not in DATE_KEYED_TABLES]
        if off_key:
            raise ValueError(f"{FIX_ACTIONS[action]} needs date-keyed tables {', '.join(DATE_KEYED_TABLES)}; "
                             f"selection includes {', '.join(off_key)}")
        if any(_to_date(row.get('Date')) is None for row in rows):
            raise ValueError('every selected row needs a date for this action')
    if action in ('set_value', 'clear_value') and not column:
        raise ValueError('pick the column to change')
    if action == 'set_value':
        if len(rows) != 1:
            raise ValueError('correct a value works on exactly one selected row')
        if value is None:
            raise ValueError('enter the corrected value')
    if action == 'rebuild_adjusted_close' and tables != ['prices']:
        raise ValueError('rebuild adjusted close works on prices rows only')

    entries: list[tuple[str, dict]] = []

    if action == 'delete_rows' or action == 'clear_value':
        grouped: dict[tuple[str, date], set[str]] = {}
        for row in rows:
            key = (row['SrcTable'], _to_date(row['Date']))
            grouped.setdefault(key, set()).add(row['Ticker'])
        for (table, day), tickers in sorted(grouped.items()):
            if action == 'delete_rows':
                entries.append(('delete_date_range', {'table': table, 'tickers': sorted(tickers),
                                                      'start': day, 'end': day, 'reason': reason}))
            else:
                entries.append(('set_value', {'table': table, 'tickers': sorted(tickers), 'start': day,
                                              'end': day, 'column': column, 'reason': reason}))

    elif action == 'delete_span':
        spans: dict[tuple[str, str], list[date]] = {}
        for row in rows:
            spans.setdefault((row['SrcTable'], row['Ticker']), []).append(_to_date(row['Date']))
        for (table, ticker), days in sorted(spans.items()):
            entries.append(('delete_date_range', {'table': table, 'tickers': [ticker],
                                                  'start': min(days), 'end': max(days), 'reason': reason}))

    elif action == 'delete_ticker' or action == 'rebuild_adjusted_close':
        by_table: dict[str, set[str]] = {}
        for row in rows:
            by_table.setdefault(row['SrcTable'], set()).add(row['Ticker'])
        section = 'delete_date_range' if action == 'delete_ticker' else 'rebuild_adjusted_close'
        for table, tickers in sorted(by_table.items()):
            entries.append((section, {'table': table, 'tickers': sorted(tickers), 'reason': reason}))

    elif action == 'set_value':
        row = rows[0]
        day = _to_date(row['Date'])
        entries.append(('set_value', {'table': row['SrcTable'], 'tickers': [row['Ticker']],
                                      'start': day, 'end': day, 'column': column, 'value': value, 'reason': reason}))

    return entries


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
