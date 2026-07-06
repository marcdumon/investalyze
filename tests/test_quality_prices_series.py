"""Tests for per-series price checks: extreme returns, stale runs, date gaps."""

from datetime import date, timedelta

import duckdb
import pytest

from investalyze.quality import prices_series, writer


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with empty prices and splits tables; tests seed their own series."""
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT, AC DOUBLE)')
    con.execute('CREATE TABLE splits (Ticker VARCHAR, Date DATE, Ratio DOUBLE)')
    return con


def seed_series(con: duckdb.DuckDBPyConnection, ticker: str, closes: list[float], start: date = date(2026, 1, 1)) -> None:
    """Insert one price row per close, on consecutive days from `start`, with O=H=L=AC=C."""
    rows = []
    for i, c in enumerate(closes):
        rows.append((ticker, start + timedelta(days=i), c, c, c, c, 100, c))
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)


# --- extreme_return -------------------------------------------------------------


def test_extreme_return_flags_jump_without_split(con):
    seed_series(con, 'JUMP', [10.0, 10.0, 30.0, 30.0])
    df = prices_series.extreme_return(con)
    assert list(df.columns) == writer.FINDING_COLS
    assert len(df) == 1
    assert df['Ticker'].iloc[0] == 'JUMP'
    assert df['Date'].iloc[0].date() == date(2026, 1, 3)
    assert 'ret=200' in df['Details'].iloc[0]


def test_extreme_return_skips_jump_on_split_day(con):
    seed_series(con, 'SPLITOK', [40.0, 40.0, 10.0, 10.0])
    con.execute("INSERT INTO splits VALUES ('SPLITOK', DATE '2026-01-03', 4.0)")
    assert prices_series.extreme_return(con).empty


def test_extreme_return_tags_spike_and_revert(con):
    seed_series(con, 'SPIKE', [10.0, 30.0, 10.0, 10.0])
    df = prices_series.extreme_return(con)
    by_date = {d.date(): details for d, details in zip(df['Date'], df['Details'])}
    assert '(spike-and-revert)' in by_date[date(2026, 1, 2)]
    assert '(spike-and-revert)' not in by_date[date(2026, 1, 3)]


def test_extreme_return_threshold_is_tunable(con):
    seed_series(con, 'JUMP', [10.0, 10.0, 30.0, 30.0])
    assert prices_series.extreme_return(con, max_abs_log_return=5.0).empty


# --- stale_run ------------------------------------------------------------------


def test_stale_run_flags_runs_at_default_threshold(con):
    seed_series(con, 'S20', [5.0] * 20)
    seed_series(con, 'S19', [5.0] * 19)
    df = prices_series.stale_run(con)
    assert df['Ticker'].tolist() == ['S20']
    assert df['Date'].iloc[0].date() == date(2026, 1, 20)
    assert '20 identical closes' in df['Details'].iloc[0]


def test_stale_run_counts_only_consecutive_identical_closes(con):
    seed_series(con, 'BROKEN', [5.0] * 10 + [6.0] + [5.0] * 10)
    assert prices_series.stale_run(con).empty


def test_stale_run_min_stale_run_is_tunable(con):
    seed_series(con, 'S3', [5.0, 5.0, 5.0, 7.0])
    df = prices_series.stale_run(con, min_stale_run=3)
    assert df['Ticker'].tolist() == ['S3']
    assert '3 identical closes' in df['Details'].iloc[0]


# --- date_gap -------------------------------------------------------------------


def test_date_gap_flags_gap_beyond_threshold(con):
    seed_series(con, 'GAP', [10.0])
    con.execute("INSERT INTO prices VALUES ('GAP', DATE '2026-02-20', 10, 10, 10, 10, 100, 10)")
    df = prices_series.date_gap(con)
    assert df['Ticker'].tolist() == ['GAP']
    assert 'gap 50 days' in df['Details'].iloc[0]


def test_date_gap_ignores_gap_at_threshold(con):
    seed_series(con, 'G30', [10.0])
    con.execute("INSERT INTO prices VALUES ('G30', DATE '2026-01-31', 10, 10, 10, 10, 100, 10)")
    assert prices_series.date_gap(con).empty


def test_date_gap_threshold_is_tunable(con):
    seed_series(con, 'GAP', [10.0])
    con.execute("INSERT INTO prices VALUES ('GAP', DATE '2026-02-20', 10, 10, 10, 10, 100, 10)")
    assert prices_series.date_gap(con, max_gap_days=60).empty
