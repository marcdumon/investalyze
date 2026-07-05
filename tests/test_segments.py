"""Unit tests for the price-segment windowing layer."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import segments
from investalyze.ingest import storage

_DB = Path('data') / 'investalyze.duckdb'


def _frame(ticker: str, values: list[float], asset_class: str = 'stocks') -> pd.DataFrame:
    """Synthetic long frame for one instrument with consecutive daily dates."""
    dates = pd.date_range('2024-01-01', periods=len(values), freq='D').date
    return pd.DataFrame({'Ticker': ticker, 'Date': dates, 'AssetClass': asset_class, 'Price': values})


# --- build_segments: raw windows + shape --------------------------------------
def test_raw_window_shape_and_values():
    series = _frame('AAA', [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    W, meta = segments.build_segments(series, window_length=3, stride=3)
    assert W.shape == (3, 2)  # window_length rows, one column per window (offsets 0 and 3)
    assert np.allclose(W[:, 0], [10.0, 11.0, 12.0])  # raw, not rebased
    assert np.allclose(W[:, 1], [13.0, 14.0, 15.0])
    assert list(meta['start_idx']) == [0, 3]


def test_window_count_follows_stride():
    series = _frame('AAA', list(map(float, range(1, 9))))  # 8 rows
    W, _ = segments.build_segments(series, window_length=3, stride=1)
    assert W.shape[1] == 6  # (8 - 3)//1 + 1 windows, one per column


def test_drop_short_instrument():
    series = _frame('AAA', [1.0, 2.0, 3.0])  # window_length 4 does not fit
    W, meta = segments.build_segments(series, window_length=4, stride=1)
    assert W.shape == (4, 0)
    assert meta.empty


def test_drop_window_with_nan_or_nonpositive():
    # window at offset 0 has a NaN, at offset 3 has a 0 -> both dropped, offset 6 survives
    series = _frame('AAA', [10.0, np.nan, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 18.0])
    W, meta = segments.build_segments(series, window_length=3, stride=3)
    assert W.shape == (3, 1)
    assert meta['start_idx'].tolist() == [6]
    assert np.allclose(W[:, 0], [16.0, 17.0, 18.0])


@pytest.mark.parametrize('window_length, stride', [(0, 1), (-1, 1), (3, 0), (3, -1)])
def test_rejects_nonpositive_params(window_length, stride):
    series = _frame('AAA', [1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        segments.build_segments(series, window_length=window_length, stride=stride)


# --- build_segments: meta + multi-ticker --------------------------------------
def test_meta_window_dates():
    series = _frame('AAA', list(map(float, range(1, 7))))  # 6 rows, dates 2024-01-01..06
    _, meta = segments.build_segments(series, window_length=5, stride=1)
    row = meta.iloc[0]  # offset 0: window rows 0..4
    assert row['start_date'] == pd.Timestamp('2024-01-01').date()
    assert row['end_date'] == pd.Timestamp('2024-01-05').date()


def test_does_not_cross_ticker():
    a = _frame('AAA', [1.0, 2.0, 3.0])
    b = _frame('BBB', [5.0, 6.0, 7.0])
    series = pd.concat([a, b], ignore_index=True)
    W, meta = segments.build_segments(series, window_length=3, stride=1)
    # each ticker yields exactly one window; no leakage across tickers
    assert meta['Ticker'].tolist() == ['AAA', 'BBB']
    assert np.allclose(W[:, 0], [1.0, 2.0, 3.0])
    assert np.allclose(W[:, 1], [5.0, 6.0, 7.0])


# --- list_tickers / get_series / build_segments against the real DB -----------
def test_list_tickers_rejects_unknown_class():
    con = duckdb.connect()
    with pytest.raises(ValueError, match='unknown class'):
        segments.list_tickers(con, classes=['stock'])  # typo: should be 'stocks'


@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_list_tickers_stocks():
    con = storage.connect(Path('data'), read_only=True)
    try:
        tickers = segments.list_tickers(con, classes=['stocks'])
    finally:
        con.close()
    assert 'AAPL' in tickers
    assert tickers == sorted(tickers)


@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_list_tickers_mixed_classes():
    con = storage.connect(Path('data'), read_only=True)
    try:
        tickers = segments.list_tickers(con, classes=['stocks', 'indices'])
    finally:
        con.close()
    assert 'AAPL' in tickers
    assert '^DJI' in tickers


@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_get_series_stocks_uses_ac():
    con = storage.connect(Path('data'), read_only=True)
    try:
        df = segments.get_series(con, ['AAPL'])
    finally:
        con.close()
    assert not df.empty
    assert set(df['AssetClass'].unique()) == {'stocks'}
    assert df['Price'].notna().all()
    assert df['Ticker'].is_monotonic_increasing


@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_get_series_market_and_mixed():
    con = storage.connect(Path('data'), read_only=True)
    try:
        idx = segments.get_series(con, ['^DJI'])
        mixed = segments.get_series(con, ['AAPL', '^DJI'])
    finally:
        con.close()
    assert set(idx['AssetClass'].unique()) == {'indices'}
    assert set(mixed['AssetClass'].unique()) == {'stocks', 'indices'}


@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_build_segments_returns_raw_values_from_db():
    con = storage.connect(Path('data'), read_only=True)
    try:
        series = segments.get_series(con, ['AAPL'])
    finally:
        con.close()
    W, meta = segments.build_segments(series, window_length=25, stride=20)
    assert W.shape[0] == 25  # window_length rows
    assert not meta.empty
    raw_first_25 = series.sort_values('Date')['Price'].to_numpy(dtype=float)[:25]
    assert np.allclose(W[:, 0], raw_first_25)
