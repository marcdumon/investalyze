"""Tests for the Stooq transform: raw OHLCV -> canonical market_data rows."""
from datetime import date

import pandas as pd

from investalyze.ingest.providers.stooq.market_data import _transform

# Raw Stooq columns, as they appear in the .txt files.
_RAW_COLS = ['<TICKER>', '<PER>', '<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>', '<OPENINT>']


def _raw(**over: object) -> pd.DataFrame:
    """One raw Stooq row (a bond yield), with optional field overrides."""
    row = {
        '<TICKER>': '10YUSY.B', '<PER>': 'D', '<DATE>': 18710101, '<TIME>': 0,
        '<OPEN>': 5.32, '<HIGH>': 5.33, '<LOW>': 5.31, '<CLOSE>': 5.32, '<VOL>': 0, '<OPENINT>': 0,
    }
    row.update(over)
    return pd.DataFrame([row], columns=_RAW_COLS)


def test_transform_produces_canonical_market_data_columns():
    out = _transform(_raw(), asset_class='bonds')
    assert list(out.columns) == ['Ticker', 'Date', 'O', 'H', 'L', 'C', 'AssetClass']


def test_transform_maps_ohlc_values():
    row = _transform(_raw(), asset_class='bonds').iloc[0]
    assert (row['O'], row['H'], row['L'], row['C']) == (5.32, 5.33, 5.31, 5.32)


def test_transform_strips_exchange_suffix_from_ticker():
    row = _transform(_raw(**{'<TICKER>': '10YUSY.B'}), asset_class='bonds').iloc[0]
    assert row['Ticker'] == '10YUSY'


def test_transform_tags_asset_class():
    row = _transform(_raw(), asset_class='currencies').iloc[0]
    assert row['AssetClass'] == 'currencies'


def test_transform_parses_yyyymmdd_date():
    row = _transform(_raw(**{'<DATE>': 20240321}), asset_class='bonds').iloc[0]
    assert row['Date'] == date(2024, 3, 21)
