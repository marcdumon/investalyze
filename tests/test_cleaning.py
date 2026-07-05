"""Tests for the cleaning package: cleaning.toml parsing and delete_date_range detect/apply."""

from datetime import date
from pathlib import Path

import duckdb
import pytest

from investalyze.cleaning import delete_date_range, registry

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
