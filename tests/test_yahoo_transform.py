"""Tests for Yahoo per-ticker transforms."""
from datetime import date

import pandas as pd

from investalyze.ingest.providers.yahoo.provider import _to_dividends, _to_prices, _to_splits


def _frame() -> pd.DataFrame:
    """Two-day per-ticker yahoo frame with one dividend and one split."""
    idx = pd.DatetimeIndex([pd.Timestamp('2024-03-20'), pd.Timestamp('2024-03-21')], name='Date')
    return pd.DataFrame({
        'Open': [10.0, 11.0], 'High': [10.5, 11.5], 'Low': [9.5, 10.5],
        'Close': [10.0, 11.0], 'Adj Close': [9.0, 11.0], 'Volume': [100, 200],
        'Dividends': [0.0, 0.5], 'Stock Splits': [2.0, 0.0],
    }, index=idx)


def test_to_prices_canonical_columns_and_values():
    out = _to_prices('AAA', _frame())
    assert list(out.columns) == ['Ticker', 'Date', 'O', 'H', 'L', 'C', 'V']
    row = out.iloc[1]
    assert row['Ticker'] == 'AAA'
    assert row['Date'] == date(2024, 3, 21)
    assert (row['O'], row['H'], row['L'], row['C'], row['V']) == (11.0, 11.5, 10.5, 11.0, 200)


def test_to_dividends_keeps_only_event_rows():
    out = _to_dividends('AAA', _frame())
    assert list(out.columns) == ['Ticker', 'Date', 'Dividend']
    assert out['Date'].tolist() == [date(2024, 3, 21)]
    assert out['Dividend'].tolist() == [0.5]


def test_to_splits_keeps_only_event_rows():
    out = _to_splits('AAA', _frame())
    assert list(out.columns) == ['Ticker', 'Date', 'Ratio']
    assert out['Date'].tolist() == [date(2024, 3, 20)]
    assert out['Ratio'].tolist() == [2.0]
