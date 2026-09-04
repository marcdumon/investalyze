"""Tests for how a cross-sectional signal reads the stored factor panel.

The panel is built on its own grid of as-of dates, and a run rebalances on its own schedule. The two
need not line up, so the rule under test is that a session between panel dates reads the most recent
one before it, and that a run the panel cannot reach is refused rather than quietly traded flat.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import factors_pit
from investalyze.backtest.market import MarketData
from investalyze.backtest.signals import cross_sectional

TICKERS = np.array(['AAA', 'BBB', 'CCC', 'DDD'])
SESSIONS = pd.bdate_range('2020-01-01', periods=60)


def _market(dates: pd.DatetimeIndex = SESSIONS) -> MarketData:
    """A flat-priced market over the given sessions, enough for a factory to build a step."""
    shape = (len(dates), len(TICKERS))
    prices = np.full(shape, 100.0)
    return MarketData(
        dates=dates, tickers=TICKERS,
        open_=prices, high=prices, low=prices, close=prices,
        volume=np.full(shape, 1e6, dtype=np.float32),
        dividend=np.zeros(shape),
        adv=np.full(shape, 1e8, dtype=np.float32),
        tradable=np.ones(shape, dtype=bool),
        adjusted_close=prices,
    )


def _panel_on(dates: list[str], *, factors: list[str] | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory database whose factor panel holds rows only on the named dates.

    Every factor listed gets a clean per-ticker ordering; the rest stay empty, which is what a
    factor added to the library after the last build looks like.
    """
    filled = factors_pit.FACTORS if factors is None else factors
    columns = factors_pit.PANEL_KEY + factors_pit.IDENTITY_COLS + factors_pit.FACTORS
    rows = []
    for date in dates:
        for position, ticker in enumerate(TICKERS):
            row = {column: np.nan for column in columns}
            row |= {'Ticker': ticker, 'Date': pd.Timestamp(date).date(),
                    'Version': factors_pit.AS_REPORTED, 'name': ticker, 'sector': 'tech',
                    'industry': 'software', 'last_close': 100.0, 'mcap': 1e9, 'mcap_bn': 1.0}
            for factor in filled:
                row[factor] = float(position + 1)
            rows.append(row)
    connection = duckdb.connect()
    factors_pit.store_factor_panel(connection, pd.DataFrame(rows, columns=columns))
    return connection


def _scores(con: duckdb.DuckDBPyConnection, market: MarketData,
            factors: list[str] = ('roe',)) -> pd.DataFrame:
    """The composite score matrix a run would act on."""
    panel = cross_sectional.read_panel(con, market.tickers, market.dates)
    return cross_sectional.composite_scores(panel, list(factors), market.tickers, market.dates)


# --- reading between panel dates ---

def test_a_session_between_panel_dates_reads_the_most_recent_one_before_it():
    market = _market()
    con = _panel_on(['2020-01-31', '2020-02-28'])

    scores = _scores(con, market)
    mid_february = pd.Timestamp('2020-02-14')

    assert scores.loc[mid_february].notna().all()
    assert scores.loc[mid_february].equals(scores.loc[pd.Timestamp('2020-01-31')])


def test_sessions_before_the_first_panel_date_have_nothing_to_read():
    """Reading forward from a date the panel does not reach would be inventing values."""
    market = _market()
    con = _panel_on(['2020-02-28'])

    scores = _scores(con, market)

    assert scores.loc[pd.Timestamp('2020-01-15')].isna().all()
    assert scores.loc[pd.Timestamp('2020-03-02')].notna().all()


def test_a_run_starting_after_a_panel_date_still_reads_it():
    """The panel query reaches back past the run's first session, or the first rebalance sees nothing."""
    market = _market(pd.bdate_range('2020-02-10', periods=20))
    con = _panel_on(['2020-01-31'])

    scores = _scores(con, market)

    assert scores.iloc[0].notna().all()


def test_a_monthly_panel_covers_every_rebalance_of_a_weekly_run():
    market = _market()
    con = _panel_on(['2020-01-31', '2020-02-28'])
    sessions = cross_sectional.rebalance_sessions(market.dates, 'W')

    scores = _scores(con, market)
    covered = cross_sectional.covered_sessions(scores, sessions)

    assert covered == sum(1 for session in sessions if market.dates[session] >= pd.Timestamp('2020-01-31'))
    assert covered > len(sessions) / 2


# --- refusing what it cannot read ---

def test_a_factor_with_no_values_in_the_panel_is_named_rather_than_ranked():
    market = _market()
    con = _panel_on(['2020-01-31'], factors=['roe'])

    with pytest.raises(ValueError, match='no values for fcf_yield'):
        _scores(con, market, factors=['roe', 'fcf_yield'])


def test_a_run_the_panel_barely_reaches_is_refused_with_the_counts():
    market = _market()
    con = _panel_on(['2020-03-20'])   # only the last week of the market

    with pytest.raises(ValueError, match=r'reaches only \d+ of this run'):
        cross_sectional.factor_composite(con, market, {'rebalance': 'W', 'factors': ['roe']})


def test_a_run_the_panel_covers_builds_its_step():
    market = _market()
    con = _panel_on(['2020-01-31', '2020-02-28'])

    step = cross_sectional.factor_composite(con, market, {'rebalance': 'M', 'factors': ['roe']})

    assert callable(step)


def test_covered_sessions_counts_only_the_rebalances_with_a_name_to_rank():
    market = _market()
    con = _panel_on(['2020-02-28'])
    scores = _scores(con, market)

    early = cross_sectional.covered_sessions(scores, [0, 1, 2])
    late = cross_sectional.covered_sessions(scores, [len(market.dates) - 1])

    assert early == 0
    assert late == 1
