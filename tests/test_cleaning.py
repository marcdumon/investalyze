"""Tests for the cleaning package: cleaning.toml parsing and each fix type's detect/apply."""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from investalyze.cleaning import delete_date_range, rebuild_adjusted_close, registry, repair_zero_low, repair_zero_open

SEED_TOML = """\
[[delete_date_range]]
table = 'market_data'
tickers = ['^NDX']
end = 1985-10-01
reason = 'pre-launch proxy data'
"""


@pytest.fixture
def con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Writable temp DuckDB with a small market_data table."""
    con = duckdb.connect(str(tmp_path / 'test.duckdb'))
    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, C DOUBLE)')
    rows = [
        ('^NDX', '1980-01-02', 100.0),
        ('^NDX', '1985-10-01', 110.0),
        ('^NDX', '1985-10-02', 120.0),
        ('^NDQ', '1980-01-02', 250.0),
    ]
    con.executemany('INSERT INTO market_data VALUES (?, ?, ?)', rows)
    return con


def make_fix(*, tickers: list[str] | None = None, start: date | None = None, end: date | None = date(1985, 10, 1)) -> registry.Fix:
    """Fix record with the seed entry's shape, fields overridable per test."""
    if tickers is None:
        tickers = ['^NDX']
    return registry.Fix(fix_type='delete_date_range', table='market_data', tickers=tickers, start=start, end=end, reason='test')


# --- read_fixes -----------------------------------------------------------------


def test_read_fixes_parses_seed_entry(tmp_path: Path):
    path = tmp_path / 'cleaning.toml'
    path.write_text(SEED_TOML)
    fixes = registry.read_fixes(path)
    assert len(fixes) == 1
    fix = fixes[0]
    assert fix.fix_type == 'delete_date_range'
    assert fix.table == 'market_data'
    assert fix.tickers == ['^NDX']
    assert fix.start is None
    assert fix.end == date(1985, 10, 1)
    assert fix.reason == 'pre-launch proxy data'


def test_read_fixes_rejects_unknown_fix_type(tmp_path: Path):
    path = tmp_path / 'cleaning.toml'
    path.write_text("[[frobnicate]]\ntable = 't'\ntickers = ['X']\nreason = 'r'\n")
    with pytest.raises(ValueError, match='unknown fix type'):
        registry.read_fixes(path)


def test_read_fixes_rejects_non_array_section(tmp_path: Path):
    path = tmp_path / 'cleaning.toml'
    path.write_text("delete_date_range = 'oops'\n")
    with pytest.raises(ValueError, match='array of tables'):
        registry.read_fixes(path)


def test_read_fixes_rejects_missing_required_field(tmp_path: Path):
    path = tmp_path / 'cleaning.toml'
    path.write_text("[[delete_date_range]]\ntable = 'market_data'\ntickers = ['^NDX']\n")
    with pytest.raises(ValueError, match='reason'):
        registry.read_fixes(path)


# --- delete_date_range ----------------------------------------------------------


def test_detect_counts_only_rows_in_range_for_listed_tickers(con):
    assert delete_date_range.detect(con, make_fix()) == 2


def test_detect_open_ended_range_matches_all_ticker_rows(con):
    assert delete_date_range.detect(con, make_fix(start=None, end=None)) == 3


def test_detect_start_bound_is_inclusive(con):
    fix = make_fix(start=date(1985, 10, 1), end=None)
    assert delete_date_range.detect(con, fix) == 2


def test_apply_deletes_only_matching_rows(con):
    deleted = delete_date_range.apply(con, make_fix())
    assert deleted == 2
    remaining = con.execute('SELECT Ticker, Date FROM market_data ORDER BY Ticker, Date').fetchall()
    assert remaining == [('^NDQ', date(1980, 1, 2)), ('^NDX', date(1985, 10, 2))]


def test_apply_is_idempotent(con):
    assert delete_date_range.apply(con, make_fix()) == 2
    assert delete_date_range.apply(con, make_fix()) == 0

# --- repair_zero_low / repair_zero_open -----------------------------------------


@pytest.fixture
def prices_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Writable temp DuckDB with a small prices table covering the broken-bar patterns."""
    con = duckdb.connect(str(tmp_path / 'prices.duckdb'))
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, AC DOUBLE)')
    rows = [
        ('ATLX', '2015-07-10', 375.0, 375.0, 0.0, 375.0, 375.0),    # only L bad
        ('ATLX', '2015-08-05', 0.0, 375.0, 0.0, 375.0, 375.0),      # O and L both bad
        ('ATLX', '2015-07-11', 380.0, 390.0, 370.0, 385.0, 385.0),  # healthy
        ('CXE', '2000-01-03', 0.0, 5.2, 5.0, 5.1, 4.8),             # only O bad
        ('CXE', '2000-01-04', 0.0, 0.0, 0.0, 0.0, 0.0),             # everything bad: untouched
    ]
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
    return con


def price_fix(fix_type: str, tickers: list[str] | None = None) -> registry.Fix:
    """Fix record for the prices repairs; empty tickers = every ticker."""
    return registry.Fix(fix_type=fix_type, table='prices', tickers=tickers or [], start=None, end=None, reason='test')


def test_repair_zero_low_sets_min_of_open_close(prices_con):
    fix = price_fix('repair_zero_low')
    assert repair_zero_low.detect(prices_con, fix) == 2
    assert repair_zero_low.apply(prices_con, fix) == 2
    low = prices_con.execute("SELECT L FROM prices WHERE Ticker = 'ATLX' AND Date = DATE '2015-07-10'").fetchone()[0]
    assert low == 375.0
    assert repair_zero_low.detect(prices_con, fix) == 0


def test_repair_zero_low_falls_back_to_close_when_open_is_broken_too(prices_con):
    repair_zero_low.apply(prices_con, price_fix('repair_zero_low'))
    low = prices_con.execute("SELECT L FROM prices WHERE Ticker = 'ATLX' AND Date = DATE '2015-08-05'").fetchone()[0]
    assert low == 375.0   # LEAST(O, C) would have been the broken 0 open


def test_repair_zero_open_sets_close(prices_con):
    fix = price_fix('repair_zero_open')
    assert repair_zero_open.detect(prices_con, fix) == 2
    assert repair_zero_open.apply(prices_con, fix) == 2
    opened = prices_con.execute("SELECT O FROM prices WHERE Ticker = 'CXE' AND Date = DATE '2000-01-03'").fetchone()[0]
    assert opened == 5.1
    assert repair_zero_open.detect(prices_con, fix) == 0


def test_repairs_heal_double_broken_bar(prices_con):
    repair_zero_low.apply(prices_con, price_fix('repair_zero_low'))
    repair_zero_open.apply(prices_con, price_fix('repair_zero_open'))
    row = prices_con.execute("SELECT O, H, L, C FROM prices WHERE Date = DATE '2015-08-05'").fetchone()
    assert row == (375.0, 375.0, 375.0, 375.0)


def test_repairs_leave_fully_broken_bars_alone(prices_con):
    repair_zero_low.apply(prices_con, price_fix('repair_zero_low'))
    repair_zero_open.apply(prices_con, price_fix('repair_zero_open'))
    row = prices_con.execute("SELECT O, L FROM prices WHERE Date = DATE '2000-01-04'").fetchone()
    assert row == (0.0, 0.0)


# --- rebuild_adjusted_close ------------------------------------------------------


def test_rebuild_series_normal_dividend_back_adjusts():
    close = pd.Series([100.0, 100.0, 100.0], index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
    dividends = pd.Series([10.0], index=pd.to_datetime(['2020-01-02']))
    rebuilt, excluded = rebuild_adjusted_close.rebuild_series(close, dividends)
    assert excluded == []
    assert rebuilt.iloc[0] == pytest.approx(90.0)   # 1 - 10/100 applied to earlier days
    assert rebuilt.iloc[1] == pytest.approx(100.0)
    assert rebuilt.iloc[2] == pytest.approx(100.0)


def test_rebuild_series_excludes_distribution_above_previous_close():
    close = pd.Series([100.0, 100.0, 100.0], index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']))
    dividends = pd.Series([150.0], index=pd.to_datetime(['2020-01-02']))
    rebuilt, excluded = rebuild_adjusted_close.rebuild_series(close, dividends)
    assert len(excluded) == 1
    assert (rebuilt == close).all()   # the poison event contributes no factor


def test_rebuild_adjusted_close_apply_fixes_negative_prefix(tmp_path):
    con = duckdb.connect(str(tmp_path / 'ac.duckdb'))
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE, AC DOUBLE)')
    con.execute('CREATE TABLE dividends (Ticker VARCHAR, Date DATE, Dividend DOUBLE)')
    # stored AC carries a negative prefix from the poison event on day 2 (factor 1 - 150/100 = -0.5)
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?)', [
        ('X', '2020-01-01', 100.0, -45.0),
        ('X', '2020-01-02', 100.0, 90.0),
        ('X', '2020-01-03', 100.0, 90.0),
        ('X', '2020-01-04', 100.0, 100.0),
    ])
    con.executemany('INSERT INTO dividends VALUES (?, ?, ?)', [
        ('X', '2020-01-02', 150.0),   # poison: above previous close
        ('X', '2020-01-04', 10.0),    # normal
    ])
    fix = registry.Fix(fix_type='rebuild_adjusted_close', table='prices', tickers=['X'],
                       start=None, end=None, reason='test')
    assert rebuild_adjusted_close.detect(con, fix) == 1   # only the poisoned first day changes
    assert rebuild_adjusted_close.apply(con, fix) == 1
    fixed = [r[0] for r in con.execute('SELECT AC FROM prices ORDER BY Date').fetchall()]
    assert fixed == pytest.approx([90.0, 90.0, 90.0, 100.0])
    assert rebuild_adjusted_close.detect(con, fix) == 0
