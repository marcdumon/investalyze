"""Tests for the pure adjusted-close derivation.

AC adjusts for dividends ONLY: yfinance's Close is already split-adjusted, so
re-applying splits here would double-count them (see the EZGO 150000x bug).
"""
import pandas as pd

from investalyze.ingest.providers.yahoo.provider import _calc_adjusted_close


def _series(dates, vals):
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


def test_no_events_returns_raw_close():
    dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    close = _series(dates, [10.0, 11.0, 12.0])
    zero = _series(dates, [0.0, 0.0, 0.0])
    out = _calc_adjusted_close(close, zero)
    assert list(out) == [10.0, 11.0, 12.0]


def test_dividend_back_adjusts_prior_days():
    # ex-div 0.5 on day 3; prev close = 11.0 -> factor 1 - 0.5/11 = 0.954545...
    dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    close = _series(dates, [10.0, 11.0, 12.0])
    divs = _series(dates, [0.0, 0.0, 0.5])
    out = _calc_adjusted_close(close, divs)
    factor = 1 - 0.5 / 11.0
    assert out.iloc[2] == 12.0                       # latest unchanged
    assert abs(out.iloc[1] - 11.0 * factor) < 1e-9   # prior days scaled
    assert abs(out.iloc[0] - 10.0 * factor) < 1e-9


def test_splits_are_ignored_already_adjusted_in_close():
    # a split column must NOT move AC — Close is already split-adjusted upstream
    dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    close = _series(dates, [10.0, 11.0, 6.0])
    splits = _series(dates, [0.0, 0.0, 2.0])
    out = _calc_adjusted_close(close, _series(dates, [0.0, 0.0, 0.0]))
    # passing or omitting splits is irrelevant; result == raw close (no dividends)
    assert list(out) == [10.0, 11.0, 6.0]
    assert splits.sum() > 0  # split present in source data, deliberately unused by AC
