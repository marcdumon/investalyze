"""Tests for row-local price/market checks: sign, OHLC consistency, volume, bond bounds."""

import duckdb
import pytest

from investalyze.quality import prices_ohlc, writer


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with tiny prices and market_data tables."""
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT, AC DOUBLE)')
    price_rows = [
        ('OK', '2026-01-02', 10.0, 11.0, 9.0, 10.5, 1000, 10.5),
        ('NEGL', '2026-01-02', 10.0, 11.0, -1.0, 10.5, 1000, 10.5),      # negative Low
        ('NEGAC', '2026-01-02', 10.0, 11.0, 9.0, 10.5, 1000, -0.5),     # negative adjusted close
        ('BADRANGE', '2026-01-02', 12.0, 11.0, 9.0, 10.5, 1000, 10.5),  # Open above High
        ('NEGV', '2026-01-02', 10.0, 11.0, 9.0, 10.5, -5, 10.5),        # negative volume
        ('NULLO', '2026-01-02', None, 11.0, 9.0, 15.0, 1000, 15.0),     # Close above High, Open missing
        ('NULLH', '2026-01-02', 10.0, None, 9.0, -1.0, 1000, 10.0),     # negative Close, High missing
    ]
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', price_rows)
    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, AssetClass VARCHAR)')
    market_rows = [
        ('EURUSD', '2026-01-02', -1.1, -1.0, -1.2, -1.05, 'currencies'),  # negative but OHLC-consistent
        ('10YCHY', '2026-01-02', -0.5, -0.4, -0.6, -0.5, 'bonds'),        # negative yield: genuine
        ('10YBAD', '2026-01-02', 0.7, 0.5, 0.9, 0.7, 'bonds'),            # High below Low
        ('PRBOND', '2026-01-02', 99.0, 100.0, 98.0, 99.5, 'bonds'),       # yield-implausible level
        ('^SPX', '2026-01-02', 5000.0, 5100.0, 4950.0, 5050.0, 'indices'),
    ]
    con.executemany('INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?)', market_rows)
    return con


def test_nonpositive_price_flags_prices_and_nonbond_market_rows(con):
    df = prices_ohlc.nonpositive_price(con)
    assert list(df.columns) == writer.FINDING_COLS
    assert sorted(df['Ticker']) == ['EURUSD', 'NEGAC', 'NEGL', 'NULLH']


def test_nonpositive_price_details_carry_the_offending_values(con):
    df = prices_ohlc.nonpositive_price(con)
    details = df.loc[df['Ticker'] == 'NEGL', 'Details'].iloc[0]
    assert 'L=-1' in details


def test_nonpositive_price_details_survive_null_fields(con):
    df = prices_ohlc.nonpositive_price(con)
    details = df.loc[df['Ticker'] == 'NULLH', 'Details'].iloc[0]
    assert 'H=null' in details


def test_ohlc_inconsistent_flags_all_asset_classes(con):
    df = prices_ohlc.ohlc_inconsistent(con)
    assert sorted(df['Ticker']) == ['10YBAD', 'BADRANGE', 'NULLH', 'NULLO']


def test_ohlc_inconsistent_details_survive_null_fields(con):
    df = prices_ohlc.ohlc_inconsistent(con)
    details = df.loc[df['Ticker'] == 'NULLO', 'Details'].iloc[0]
    assert 'O=null' in details


def test_negative_volume_flags_only_negative_volume(con):
    df = prices_ohlc.negative_volume(con)
    assert sorted(df['Ticker']) == ['NEGV']


def test_bond_yield_bound_flags_only_bonds_beyond_bound(con):
    df = prices_ohlc.bond_yield_bound(con)
    assert sorted(df['Ticker']) == ['PRBOND']


def test_bond_yield_bound_threshold_is_tunable(con):
    assert prices_ohlc.bond_yield_bound(con, max_abs_yield=1000.0).empty
