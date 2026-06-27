"""Unit tests for the price-segment vectorization layer."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import segments
from investalyze.ingest import storage

_DB = Path('data') / 'investalyze.duckdb'


def _frame(ticker: str, values: list[float], asset_class: str = 'stocks') -> pd.DataFrame:
    """Synthetic long frame for one instrument with consecutive daily dates."""
    dates = pd.date_range('2024-01-01', periods=len(values), freq='D').date
    return pd.DataFrame({'Ticker': ticker, 'Date': dates, 'AssetClass': asset_class, 'Value': values})


# --- build_segments: shape + rebase -------------------------------------------
def test_rebase_to_100_and_shape():
    series = _frame('AAA', [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    X, meta, _ = segments.build_segments(series, length=3, stride=3)
    assert X.shape == (2, 3)
    assert np.allclose(X[:, 0], 100.0)
    assert np.allclose(X[0], [100.0, 110.0, 120.0])
    assert np.allclose(X[1], [100.0, 14.0 / 13.0 * 100, 15.0 / 13.0 * 100])
    assert list(meta['start_idx']) == [0, 3]


def test_window_count_follows_stride():
    series = _frame('AAA', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    X, _, _ = segments.build_segments(series, length=3, stride=1)
    assert X.shape[0] == 4  # offsets 0,1,2,3


def test_drop_short_instrument():
    series = _frame('AAA', [1.0, 2.0])
    X, meta, succ = segments.build_segments(series, length=3, stride=1)
    assert X.shape == (0, 3)
    assert meta.empty
    assert succ.shape == (0,)


def test_drop_window_with_nan_or_nonpositive():
    # window 0 has a NaN, window 1 (offset 3) has a 0 -> both dropped, only offset 6 survives
    series = _frame('AAA', [10.0, np.nan, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 18.0])
    X, meta, _ = segments.build_segments(series, length=3, stride=3)
    assert X.shape == (1, 3)
    assert meta['start_idx'].tolist() == [6]
    assert np.allclose(X[0], [100.0, 17.0 / 16.0 * 100, 18.0 / 16.0 * 100])


@pytest.mark.parametrize('length, stride', [(0, 1), (-1, 1), (2, 0), (2, -1)])
def test_rejects_nonpositive_length_or_stride(length, stride):
    series = _frame('AAA', [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        segments.build_segments(series, length=length, stride=stride)


# --- build_segments: successor map --------------------------------------------
def test_successor_chain():
    series = _frame('AAA', list(map(float, range(1, 9))))  # 8 rows
    _, _, succ = segments.build_segments(series, length=2, stride=2)
    # offsets 0,2,4,6 -> seg 0,1,2,3; each successor is the block 2 rows later
    assert succ.tolist() == [1, 2, 3, -1]


def test_successor_does_not_cross_ticker():
    a = _frame('AAA', [1.0, 2.0, 3.0, 4.0])
    b = _frame('BBB', [5.0, 6.0, 7.0, 8.0])
    series = pd.concat([a, b], ignore_index=True)
    _, meta, succ = segments.build_segments(series, length=2, stride=2)
    # each ticker: seg at offset0 -> successor offset2; offset2 -> -1 (no leak into other ticker)
    by_ticker = meta.groupby('Ticker')['segment_id'].apply(list).to_dict()
    for seg_ids in by_ticker.values():
        assert succ[seg_ids[0]] == seg_ids[1]
        assert succ[seg_ids[1]] == -1


def test_successor_minus_one_when_stride_does_not_divide_length():
    series = _frame('AAA', list(map(float, range(1, 10))))  # 9 rows
    _, meta, succ = segments.build_segments(series, length=3, stride=2)
    # offsets 0,2,4,6 ; successor needs offset+3 in the set -> 3,5,7,9 none present
    assert meta['start_idx'].tolist() == [0, 2, 4, 6]
    assert succ.tolist() == [-1, -1, -1, -1]


# --- load_series against the real DB ------------------------------------------
@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_load_series_stocks_uses_ac():
    con = storage.connect(Path('data'), read_only=True)
    try:
        df = segments.load_series(con, classes=['stocks'], tickers=['AAPL'])
    finally:
        con.close()
    assert not df.empty
    assert set(df['AssetClass'].unique()) == {'stocks'}
    assert df['Value'].notna().all()
    assert df['Ticker'].is_monotonic_increasing


@pytest.mark.skipif(not _DB.exists(), reason='requires the live investalyze.duckdb')
def test_load_series_market_and_mixed():
    con = storage.connect(Path('data'), read_only=True)
    try:
        idx = segments.load_series(con, classes=['indices'], tickers=['^DJI'])
        mixed = segments.load_series(con, classes=['stocks', 'indices'], tickers=['AAPL', '^DJI'])
    finally:
        con.close()
    assert set(idx['AssetClass'].unique()) == {'indices'}
    assert set(mixed['AssetClass'].unique()) == {'stocks', 'indices'}
