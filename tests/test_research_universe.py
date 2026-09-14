"""Tests for the point-in-time liquidity universe."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from investalyze.research import universe


@pytest.fixture
def market() -> tuple:
    con = duckdb.connect()
    rows = []
    sessions = pd.bdate_range('2020-01-01', periods=80)
    for position, session in enumerate(sessions):
        # AAA trades heavily early and dries up; BBB starts quiet and becomes the liquid name
        rows.append(('AAA', session.date(), 10.0, 1000.0 if position < 40 else 1.0))
        rows.append(('BBB', session.date(), 10.0, 1.0 if position < 40 else 1000.0))
        rows.append(('CCC', session.date(), 10.0, 10.0))
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE, V DOUBLE)')
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?)', rows)
    return con, sessions


def test_membership_uses_liquidity_known_on_each_date(market):
    con, sessions = market
    early, late = sessions[35], sessions[79]
    members = universe.liquid_universe(con, pd.DatetimeIndex([early, late]), size=1, lookback=20)
    by_date = members.groupby('Date')['Ticker'].apply(list).to_dict()
    assert by_date[pd.Timestamp(early)] == ['AAA']
    assert by_date[pd.Timestamp(late)] == ['BBB']


def test_a_later_liquid_name_is_not_selected_early():
    """Selecting by liquidity measured after the date is look-ahead bias."""
    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE, V DOUBLE)')
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?)', [
        ('OLD', date(2020, 1, 2), 10.0, 100.0), ('NEW', date(2020, 1, 2), 10.0, 1.0),
        ('OLD', date(2021, 1, 4), 10.0, 1.0), ('NEW', date(2021, 1, 4), 10.0, 1_000_000.0),
    ])
    members = universe.liquid_universe(con, pd.DatetimeIndex(['2020-01-02']), size=1, lookback=5)
    assert members['Ticker'].tolist() == ['OLD']


def test_the_size_caps_each_date(market):
    con, sessions = market
    members = universe.liquid_universe(con, pd.DatetimeIndex([sessions[50]]), size=2, lookback=20)
    assert len(members) == 2


def test_a_name_not_trading_on_the_date_is_excluded(market):
    con, sessions = market
    con.execute("DELETE FROM prices WHERE Ticker = 'CCC' AND Date = ?", [sessions[50].date()])
    members = universe.liquid_universe(con, pd.DatetimeIndex([sessions[50]]), size=3, lookback=20)
    assert 'CCC' not in members['Ticker'].tolist()
