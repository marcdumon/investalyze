"""Tests for the aligned return panel: trading calendar, wide close matrix, returns, standardization.

The synthetic fixture plants the failure mode the calendar exists to survive: a holiday that carries a
couple of stray rows and would otherwise make every ticker in the panel look incomplete.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.ingest import storage
from notebooks import panels

_DB = Path('data') / 'investalyze.duckdb'
DATES = [f'2026-01-{day:02d}' for day in range(5, 10)]  # five consecutive business days
HOLIDAY = DATES[2]
SESSIONS = [date for date in DATES if date != HOLIDAY]  # the four dates a real ticker trades on


def _price_rows(ticker: str, closes: dict[str, float], volume: float = 1e6) -> list[tuple]:
    """Price rows for one ticker; AC follows C, and a date left out of `closes` is simply absent."""
    return [(ticker, date, close, close, close, close, volume, close) for date, close in closes.items()]


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB holding a tiny prices table, plus companies and market_data.

    `2026-01-07` is the planted holiday: only THIN trades that day, so the calendar must drop it and
    every other ticker must still count as complete.
    """
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT, AC DOUBLE)')
    rows = (
        _price_rows('AAA', dict(zip(SESSIONS, [100.0, 110.0, 121.0, 133.1])), volume=5e6)
        + _price_rows('BBB', {date: 50.0 + 2 * i for i, date in enumerate(SESSIONS)}, volume=3e6)
        + _price_rows('CCC', {date: 20.0 + i for i, date in enumerate(SESSIONS)}, volume=1e6)
        + _price_rows('GAPPY', {date: 30.0 + i for i, date in enumerate(SESSIONS) if date != SESSIONS[2]})
        + _price_rows('THIN', {HOLIDAY: 7.0})
        + [('BADAC', SESSIONS[0], 5.0, 5.0, 5.0, 5.0, 1000, -1.0), ('BADAC', SESSIONS[1], 5.0, 5.0, 5.0, 5.0, 1000, None)]
    )
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)

    # share counts chosen so the market-cap order (BBB, CCC, AAA) reverses the liquidity order
    con.execute('CREATE TABLE income (Ticker VARCHAR, Period VARCHAR, "Report Date" DATE, '
                '"Shares (Basic)" BIGINT, "Shares (Diluted)" BIGINT)')
    con.executemany('INSERT INTO income VALUES (?, ?, ?, ?, ?)',
                    [('AAA', 'Q', '2025-12-31', 1_000_000, 1_000_000),
                     ('BBB', 'Q', '2025-12-31', 100_000_000, 100_000_000),
                     ('CCC', 'Q', '2025-12-31', 10_000_000, 10_000_000),
                     ('GAPPY', 'Q', '2025-12-31', 5_000_000, 5_000_000)])

    con.execute('CREATE TABLE companies (Ticker VARCHAR, Sector VARCHAR)')
    con.executemany('INSERT INTO companies VALUES (?, ?)',
                    [('AAA', 'Technology'), ('BBB', 'Energy'), ('CCC', None), ('GAPPY', 'Utilities')])

    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, C DOUBLE)')
    con.executemany('INSERT INTO market_data VALUES (?, ?, ?)', [('^SPX', date, 1000.0 * (1.01**i)) for i, date in enumerate(DATES)])
    return con


@pytest.fixture
def live_db() -> duckdb.DuckDBPyConnection:
    """Read-only connection to the real DB, skipped when the file is absent or a writer holds it."""
    if not _DB.exists():
        pytest.skip('requires the live investalyze.duckdb')
    try:
        connection = storage.connect(Path('data'), read_only=True)
    except duckdb.IOException as exc:
        pytest.skip(f'investalyze.duckdb is locked by another process: {exc}')
    try:
        yield connection
    finally:
        connection.close()


# --- trading calendar ---


def test_calendar_drops_a_holiday_carrying_only_stray_rows(con):
    assert pd.Timestamp(HOLIDAY) not in panels.trading_calendar(con, min_coverage=0.5)


def test_calendar_keeps_every_genuine_session(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert list(calendar.strftime('%Y-%m-%d')) == SESSIONS


def test_calendar_keeps_the_thin_date_when_the_threshold_is_low_enough(con):
    # the busiest session has 4 tickers and the holiday has 1, so a 0.25 bar admits it
    assert pd.Timestamp(HOLIDAY) in panels.trading_calendar(con, min_coverage=0.25)


def test_calendar_honours_the_start_bound(con):
    assert panels.trading_calendar(con, start=SESSIONS[2], min_coverage=0.5).min() == pd.Timestamp(SESSIONS[2])


def test_calendar_rejects_a_window_with_no_priced_sessions(con):
    with pytest.raises(ValueError, match='no priced sessions'):
        panels.trading_calendar(con, start='2030-01-01')


def test_calendar_rejects_a_non_positive_local_window(con):
    with pytest.raises(ValueError, match='local_window'):
        panels.trading_calendar(con, local_window=0)


def _growing_universe_db() -> duckdb.DuckDBPyConnection:
    """Prices whose ticker count doubles across the window, with one holiday planted in the busy half.

    Reproduces the live table, where roughly 2,600 securities are priced on a 2010 session against 6,300
    in 2026: a threshold taken against the busiest session would discard the whole early half.
    """
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, C DOUBLE, V BIGINT, AC DOUBLE)')
    dates = pd.bdate_range('2026-01-05', periods=60).strftime('%Y-%m-%d').tolist()
    holiday = dates[45]
    rows = []
    for i, date in enumerate(dates):
        listed = 20 if i < 30 else 40
        traded = 1 if date == holiday else listed
        for ticker in range(traded):
            rows.append((f'T{ticker:03d}', date, 10.0, 10.0, 10.0, 10.0, 1000, 10.0))
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)
    return con, dates, holiday


def test_calendar_keeps_early_sessions_when_the_listed_universe_grows_over_time():
    con, dates, holiday = _growing_universe_db()
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert pd.Timestamp(dates[0]) in calendar


def test_calendar_still_drops_a_holiday_when_the_listed_universe_grows_over_time():
    con, dates, holiday = _growing_universe_db()
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert pd.Timestamp(holiday) not in calendar


def test_calendar_keeps_every_session_but_the_holiday_when_the_universe_grows():
    con, dates, holiday = _growing_universe_db()
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert len(calendar) == len(dates) - 1


# --- universe and close panel ---


def test_complete_universe_keeps_only_tickers_priced_on_every_session(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar, require_sector=False) == ['AAA', 'BBB', 'CCC']


def test_complete_universe_ranks_by_dollar_volume(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar, n_top=2, require_sector=False) == ['AAA', 'BBB']


def test_complete_universe_can_require_a_known_sector(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar) == ['AAA', 'BBB']


def test_complete_universe_ranks_by_market_cap_when_asked(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar, rank_by='mcap', require_sector=False) == ['BBB', 'CCC', 'AAA']


def test_complete_universe_ranks_alphabetically_when_no_preference_is_wanted(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar, rank_by='ticker', require_sector=False) == ['AAA', 'BBB', 'CCC']


def test_complete_universe_restricts_to_the_supplied_candidates(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar, require_sector=False, candidates=['BBB', 'CCC']) == ['BBB', 'CCC']


def test_complete_universe_still_drops_an_incomplete_candidate(con):
    # naming a ticker explicitly does not exempt it from needing an unbroken history
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert panels.complete_universe(con, calendar, require_sector=False, candidates=['AAA', 'GAPPY']) == ['AAA']


def test_complete_universe_rejects_an_unknown_ranking(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    with pytest.raises(ValueError, match='rank_by'):
        panels.complete_universe(con, calendar, rank_by='momentum')


def test_close_panel_is_indexed_by_the_calendar(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    panel = panels.close_panel(con, ['AAA', 'BBB'], calendar)
    assert panel.index.equals(calendar)


def test_close_panel_drops_a_ticker_below_the_coverage_floor(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert 'GAPPY' not in panels.close_panel(con, ['AAA', 'GAPPY'], calendar).columns


def test_close_panel_keeps_a_gappy_ticker_once_the_floor_is_relaxed(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    assert 'GAPPY' in panels.close_panel(con, ['AAA', 'GAPPY'], calendar, min_obs_frac=0.7).columns


def test_close_panel_excludes_nonpositive_and_missing_adjusted_closes(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    with pytest.raises(ValueError, match='no priced rows'):
        panels.close_panel(con, ['BADAC'], calendar)


def test_close_panel_rejects_an_empty_ticker_list(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    with pytest.raises(ValueError, match='tickers is empty'):
        panels.close_panel(con, [], calendar)


# --- returns ---


def test_log_returns_match_the_closed_form(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    returns = panels.returns_panel(panels.close_panel(con, ['AAA'], calendar))
    assert returns['AAA'].iloc[0] == pytest.approx(np.log(1.1))


def test_simple_returns_match_the_closed_form(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    returns = panels.returns_panel(panels.close_panel(con, ['AAA'], calendar), kind='simple')
    assert returns['AAA'].iloc[0] == pytest.approx(0.1)


def test_a_one_day_hole_produces_two_missing_returns_rather_than_one_spliced_move(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    returns = panels.returns_panel(panels.close_panel(con, ['GAPPY'], calendar, min_obs_frac=0.7))
    assert returns['GAPPY'].isna().sum() == 2


def test_returns_panel_rejects_an_unknown_kind(con):
    calendar = panels.trading_calendar(con, min_coverage=0.5)
    with pytest.raises(ValueError, match='kind'):
        panels.returns_panel(panels.close_panel(con, ['AAA'], calendar), kind='pct')


def test_standardized_columns_have_zero_mean_and_unit_deviation():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.standard_normal((200, 3)) * [0.01, 0.05, 0.2], columns=['A', 'B', 'C'])
    standardized = panels.standardize(returns)
    assert standardized.z.std(ddof=1).to_numpy() == pytest.approx(np.ones(3))


def test_standardize_reports_the_volatility_that_rebuilds_a_tradable_weight():
    returns = pd.DataFrame({'A': [0.01, -0.01, 0.02, -0.02]})
    assert panels.standardize(returns).vol['A'] == pytest.approx(returns['A'].std(ddof=1))


def test_standardize_rejects_a_ticker_that_never_moves():
    returns = pd.DataFrame({'A': [0.01, -0.01, 0.02], 'FLAT': [0.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match='no return variation'):
        panels.standardize(returns)


# --- benchmark, sectors, rolling windows ---


def test_benchmark_returns_come_back_as_a_dated_series(con):
    returns = panels.benchmark_returns(con)
    assert len(returns) == len(DATES) - 1


def test_benchmark_returns_reject_an_unknown_ticker(con):
    with pytest.raises(ValueError, match='no market_data rows'):
        panels.benchmark_returns(con, '^NOPE')


def test_sector_labels_skip_tickers_with_no_recorded_sector(con):
    assert panels.sector_labels(con, ['AAA', 'BBB', 'CCC']).index.tolist() == ['AAA', 'BBB']


def test_rolling_windows_advance_by_the_requested_step():
    calendar = pd.DatetimeIndex(pd.bdate_range('2026-01-01', periods=10))
    assert panels.rolling_windows(calendar, window=4, step=3) == [
        (calendar[0], calendar[3]), (calendar[3], calendar[6]), (calendar[6], calendar[9])
    ]


def test_rolling_windows_reject_a_window_longer_than_the_calendar():
    calendar = pd.DatetimeIndex(pd.bdate_range('2026-01-01', periods=5))
    with pytest.raises(ValueError, match='exceeds'):
        panels.rolling_windows(calendar, window=10, step=1)


# --- against the live database ---


def test_live_calendar_excludes_the_juneteenth_holiday(live_db):
    # 2026-06-19 carries 3 rows out of ~6,000, the quirk the coverage rule exists to absorb
    calendar = panels.trading_calendar(live_db, start='2026-06-01', end='2026-07-01')
    assert pd.Timestamp('2026-06-19') not in calendar


def test_live_complete_case_panel_has_no_gaps(live_db):
    calendar = panels.trading_calendar(live_db, start='2025-01-01')
    tickers = panels.complete_universe(live_db, calendar, n_top=50)
    panel = panels.close_panel(live_db, tickers, calendar)
    assert not panel.isna().to_numpy().any()
