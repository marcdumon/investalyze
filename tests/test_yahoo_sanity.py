"""Tests for the adjusted-close sanity comparison."""
import pandas as pd

from investalyze.ingest.providers.yahoo.provider import _calc_ac_max_diff


def _s(vals):
    return pd.Series(vals, index=pd.DatetimeIndex(['2024-01-01', '2024-01-02']))


def test_identical_series_zero_diff():
    assert _calc_ac_max_diff(_s([9.0, 10.0]), _s([9.0, 10.0])) == 0.0


def test_reports_max_relative_difference():
    # 10 vs 10.02 -> 0.2% ; 9 vs 9 -> 0
    assert abs(_calc_ac_max_diff(_s([9.0, 10.02]), _s([9.0, 10.0])) - 0.002) < 1e-9
