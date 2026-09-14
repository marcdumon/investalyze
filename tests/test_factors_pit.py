"""Tests for the point-in-time factor panel.

The fixture plants the cases point-in-time exists to get right: a quarter published months after the
one that follows it, a ticker with too few quarters to form a window, and two fundamentals versions
that disagree. `OK` and `LATE` carry identical revenue and differ only in when one filing was
published, so any test comparing the two isolates publication timing from the numbers themselves.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import factors_pit, panels
from investalyze.ingest import storage

# twelve quarters ending 2018-03-31 through 2020-12-31, revenue rising 100 by 10 a quarter
REPORT_DATES = ['2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31',
                '2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31',
                '2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31']
PUBLISH_DATES = ['2018-05-15', '2018-08-14', '2018-11-14', '2019-02-14',
                 '2019-05-15', '2019-08-14', '2019-11-14', '2020-02-14',
                 '2020-05-15', '2020-08-14', '2020-11-14', '2021-02-14']
REVENUES = [100.0 + 10 * i for i in range(12)]

LATE_INDEX = 8              # the 2020-03-31 quarter, the one LATE publishes out of order
LATE_PUBLISH = '2020-11-15'  # after both the 2020-06-30 and 2020-09-30 filings

_INCOME_DDL = """
    CREATE TABLE income (
        Ticker VARCHAR, Period VARCHAR, IsRestated BOOLEAN,
        "Fiscal Year" BIGINT, "Fiscal Period" VARCHAR,
        "Report Date" TIMESTAMP, "Publish Date" TIMESTAMP,
        "Shares (Basic)" BIGINT, "Shares (Diluted)" BIGINT,
        Revenue BIGINT, "Gross Profit" BIGINT, "Operating Income (Loss)" BIGINT,
        "Depreciation & Amortization" BIGINT, "Interest Expense" BIGINT,
        "Net Income" BIGINT, "Net Income (Common)" BIGINT
    )
"""
_BALANCE_DDL = """
    CREATE TABLE balance (
        Ticker VARCHAR, Period VARCHAR, IsRestated BOOLEAN,
        "Report Date" TIMESTAMP, "Publish Date" TIMESTAMP,
        "Total Equity" BIGINT, "Total Assets" BIGINT,
        "Short Term Debt" BIGINT, "Long Term Debt" BIGINT, "Cash & Cash Equivalents" BIGINT
    )
"""
_CASHFLOW_DDL = """
    CREATE TABLE cashflow (
        Ticker VARCHAR, Period VARCHAR, IsRestated BOOLEAN,
        "Report Date" TIMESTAMP, "Publish Date" TIMESTAMP,
        "Net Cash from Operating Activities" BIGINT, "Change in Fixed Assets & Intangibles" BIGINT
    )
"""


def _income_rows(ticker: str, *, quarters: int = 12, revenue_scale: float = 1.0, restated: bool = False,
                 late_index: int | None = None, shares: int = 1_000) -> list[tuple]:
    """Quarterly income rows for one ticker, optionally publishing one quarter out of order."""
    rows = []
    for i in range(quarters):
        publish = LATE_PUBLISH if i == late_index else PUBLISH_DATES[i]
        revenue = int(REVENUES[i] * revenue_scale)
        rows.append((ticker, 'Q', restated, 2018 + i // 4, f'Q{i % 4 + 1}', REPORT_DATES[i], publish,
                     shares, shares, revenue, revenue // 2, revenue // 4, -revenue // 20, -revenue // 50,
                     revenue // 5, revenue // 5))
    return rows


def _balance_rows(ticker: str, *, quarters: int = 12) -> list[tuple]:
    """Quarterly balance rows for one ticker, with equity and assets rising alongside revenue."""
    rows = []
    for i in range(quarters):
        rows.append((ticker, 'Q', False, REPORT_DATES[i], PUBLISH_DATES[i],
                     int(REVENUES[i] * 5), int(REVENUES[i] * 10), 10, 40, 25))
    return rows


def _cashflow_rows(ticker: str, *, quarters: int = 12) -> list[tuple]:
    """Quarterly cashflow rows for one ticker, with operating cash flow tracking revenue."""
    rows = []
    for i in range(quarters):
        rows.append((ticker, 'Q', False, REPORT_DATES[i], PUBLISH_DATES[i],
                     int(REVENUES[i] // 4), -int(REVENUES[i] // 20)))
    return rows


@pytest.fixture
def fundamentals() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB holding only the three statement tables, for the interval-join tests."""
    con = duckdb.connect()
    for ddl in (_INCOME_DDL, _BALANCE_DDL, _CASHFLOW_DDL):
        con.execute(ddl)

    income = (
        _income_rows('OK')                                        # every filing on time
        + _income_rows('LATE', late_index=LATE_INDEX)             # one filing months out of order
        + _income_rows('THIN', quarters=3)                        # too few quarters to form a window
        + _income_rows('VINT')                                    # as-reported revenue
        + _income_rows('VINT', revenue_scale=2.0, restated=True)  # restated revenue, doubled
    )
    con.executemany(f'INSERT INTO income VALUES ({", ".join("?" * 16)})', income)
    con.executemany(f'INSERT INTO balance VALUES ({", ".join("?" * 10)})',
                    _balance_rows('OK') + _balance_rows('LATE'))
    con.executemany(f'INSERT INTO cashflow VALUES ({", ".join("?" * 7)})',
                    _cashflow_rows('OK') + _cashflow_rows('LATE'))
    return con


def _ttm(con: duckdb.DuckDBPyConnection, dates: list[str], **kwargs) -> pd.DataFrame:
    """Revenue TTM panel over the given as-of dates, indexed by (Date, Ticker) for easy lookup."""
    grid = pd.DatetimeIndex(dates)
    df = factors_pit.quarterly_ttm(con, 'income', {'Revenue': 'revenue'}, as_of=grid, **kwargs)
    return df.set_index([pd.DatetimeIndex(df['Date']), 'Ticker'])


# --- publication timing ---

def test_panel_excludes_a_quarter_whose_publish_date_is_after_the_as_of_date(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-13'])
    # the 2020-12-31 quarter publishes on 2021-02-14, so the window still ends at 2020-09-30
    assert ttm.loc[(pd.Timestamp('2021-02-13'), 'OK'), 'revenue_ttm'] == pytest.approx(170 + 180 + 190 + 200)


def test_panel_includes_that_quarter_once_the_as_of_date_passes_its_publish_date(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'])
    assert ttm.loc[(pd.Timestamp('2021-02-15'), 'OK'), 'revenue_ttm'] == pytest.approx(180 + 190 + 200 + 210)


def test_the_prior_year_window_trails_the_current_one_by_four_quarters(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'])
    assert ttm.loc[(pd.Timestamp('2021-02-15'), 'OK'), 'revenue_prior'] == pytest.approx(140 + 150 + 160 + 170)


def test_a_late_filing_holds_back_every_window_that_needs_it(fundamentals):
    """OK and LATE carry identical revenue, so the gap between them is publication timing alone."""
    ttm = _ttm(fundamentals, ['2020-09-01'])
    on_time = ttm.loc[(pd.Timestamp('2020-09-01'), 'OK'), 'revenue_ttm']
    delayed = ttm.loc[(pd.Timestamp('2020-09-01'), 'LATE'), 'revenue_ttm']
    assert on_time == pytest.approx(160 + 170 + 180 + 190)   # window through 2020-06-30
    assert delayed == pytest.approx(140 + 150 + 160 + 170)   # window through 2019-12-31, one year older


def test_the_late_filing_becomes_visible_the_day_it_publishes(fundamentals):
    ttm = _ttm(fundamentals, ['2020-11-14', '2020-11-15'])
    before = ttm.loc[(pd.Timestamp('2020-11-14'), 'LATE'), 'revenue_ttm']
    after = ttm.loc[(pd.Timestamp('2020-11-15'), 'LATE'), 'revenue_ttm']
    assert before == pytest.approx(140 + 150 + 160 + 170)
    assert after == pytest.approx(170 + 180 + 190 + 200)     # jumps straight to the 2020-09-30 window


def test_validity_intervals_never_overlap_when_a_late_filing_arrives_out_of_order(fundamentals):
    dates = pd.date_range('2018-01-31', '2021-06-30', freq='D').strftime('%Y-%m-%d').tolist()
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(dates))
    assert df.groupby(['Date', 'Ticker']).size().max() == 1


# --- window completeness ---

def test_a_ticker_with_three_quarters_has_a_null_ttm(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'])
    assert np.isnan(ttm.loc[(pd.Timestamp('2021-02-15'), 'THIN'), 'revenue_ttm'])


def test_the_prior_window_is_null_before_eight_quarters_have_published(fundamentals):
    ttm = _ttm(fundamentals, ['2019-06-01'])
    row = ttm.loc[(pd.Timestamp('2019-06-01'), 'OK')]
    assert not np.isnan(row['revenue_ttm'])
    assert np.isnan(row['revenue_prior'])


def test_a_date_before_any_filing_yields_no_rows(fundamentals):
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(['2017-01-01']))
    assert df.empty


# --- versions ---

def test_as_reported_version_never_reads_a_restated_row(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'], version=factors_pit.AS_REPORTED)
    assert ttm.loc[(pd.Timestamp('2021-02-15'), 'VINT'), 'revenue_ttm'] == pytest.approx(180 + 190 + 200 + 210)


def test_the_restated_version_reads_the_restated_numbers(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'], version=factors_pit.RESTATED)
    assert ttm.loc[(pd.Timestamp('2021-02-15'), 'VINT'), 'revenue_ttm'] == pytest.approx(2 * (180 + 190 + 200 + 210))


def test_the_restated_version_holds_only_the_tickers_that_have_restated_rows(fundamentals):
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(['2021-02-15']), version=factors_pit.RESTATED)
    assert sorted(df['Ticker'].unique()) == ['VINT']


def test_an_unknown_version_is_rejected(fundamentals):
    with pytest.raises(ValueError, match='version must be one of'):
        factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                  as_of=pd.DatetimeIndex(['2021-02-15']), version='whatever')


# --- dtypes and filtering ---

def test_bigint_sums_come_back_as_float_not_nullable_int(fundamentals):
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(['2021-02-15']))
    assert df['revenue_ttm'].dtype == np.float64
    assert df['revenue_prior'].dtype == np.float64


def test_a_ticker_filter_narrows_the_panel(fundamentals):
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(['2021-02-15']), tickers=['OK'])
    assert sorted(df['Ticker'].unique()) == ['OK']


# --- balance sheet ---

def test_the_balance_panel_uses_the_latest_published_quarter(fundamentals):
    df = factors_pit.balance_panel(fundamentals, as_of=pd.DatetimeIndex(['2021-02-15']))
    row = df.set_index('Ticker').loc['OK']
    assert row['equity'] == pytest.approx(210 * 5)      # the 2020-12-31 quarter
    assert row['debt'] == pytest.approx(50)             # short-term 10 plus long-term 40


def test_the_balance_panel_waits_for_the_publish_date(fundamentals):
    df = factors_pit.balance_panel(fundamentals, as_of=pd.DatetimeIndex(['2021-02-13']))
    assert df.set_index('Ticker').loc['OK', 'equity'] == pytest.approx(200 * 5)


# --- prices, identity and the assembled panel ---

_SESSIONS = pd.bdate_range('2019-01-01', periods=300)
_LAST_SESSION = _SESSIONS[-1]
_PRICED_TICKERS = ['OK', 'LATE', 'GAPPY']


def _price_rows(ticker: str, *, skip: set[pd.Timestamp] = frozenset()) -> list[tuple]:
    """A rising price series for one ticker, leaving out any session in `skip`."""
    rows = []
    for i, session in enumerate(_SESSIONS):
        if session in skip:
            continue
        close = 100.0 + i
        rows.append((ticker, session.date().isoformat(), close, close, close, close, 1_000_000, close))
    return rows


@pytest.fixture
def priced(fundamentals: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """The statement tables plus prices, dividends, companies and a benchmark index."""
    con = fundamentals
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, '
                'C DOUBLE, V BIGINT, AC DOUBLE)')
    con.execute('CREATE TABLE dividends (Ticker VARCHAR, Date DATE, Dividend DOUBLE)')
    con.execute('CREATE TABLE companies (Ticker VARCHAR, CompanyName VARCHAR, Sector VARCHAR, Industry VARCHAR)')
    con.execute('CREATE TABLE market_data (Ticker VARCHAR, Date DATE, O DOUBLE, H DOUBLE, L DOUBLE, '
                'C DOUBLE, AssetClass VARCHAR)')

    gap_session = _SESSIONS[len(_SESSIONS) - 1 - 126]  # the session ret_6m reaches back to
    rows = _price_rows('OK') + _price_rows('LATE') + _price_rows('GAPPY', skip={gap_session})
    con.executemany('INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows)
    con.executemany('INSERT INTO dividends VALUES (?, ?, ?)', [
        # anchored to the last session so one lands in its trailing year and one the year before
        ('OK', (_LAST_SESSION - pd.Timedelta(days=100)).date().isoformat(), 1.0),
        ('OK', (_LAST_SESSION - pd.Timedelta(days=400)).date().isoformat(), 0.5),
    ])
    con.executemany('INSERT INTO companies VALUES (?, ?, ?, ?)', [
        ('OK', 'Ok Corp', 'Technology', 'Software'),
        ('LATE', 'Late Corp', 'Energy', 'Oil'),
    ])
    con.executemany('INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [('^SPX', s.date().isoformat(), 1.0, 1.0, 1.0, 1000.0 + i, 'indices')
                     for i, s in enumerate(_SESSIONS)])
    return con


def test_identity_panel_takes_the_last_close_on_or_before_the_as_of_date(priced):
    as_of = pd.DatetimeIndex([_SESSIONS[10]])
    df = factors_pit.identity_panel(priced, as_of=as_of, tickers=['OK'])
    assert df.iloc[0]['last_close'] == pytest.approx(110.0)


def test_identity_panel_labels_a_ticker_with_no_company_row_unknown(priced):
    df = factors_pit.identity_panel(priced, as_of=pd.DatetimeIndex([_LAST_SESSION]), tickers=['GAPPY'])
    assert df.iloc[0]['sector'] == 'unknown'
    assert df.iloc[0]['name'] == ''


# --- against the live database ---

_DB = Path('data') / 'investalyze.duckdb'


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


def test_every_rebalance_date_is_a_real_trading_session(live_db):
    grid = factors_pit.rebalance_dates(live_db, start='2020-01-01', end='2020-12-31', freq='M')
    sessions = set(panels.trading_calendar(live_db, start='2020-01-01', end='2020-12-31'))
    assert len(grid) == 12
    assert set(grid) <= sessions


# --- reading back what the stored panel covers ---


# --- an empty universe is not every universe ---

def test_no_ticker_list_means_no_filter():
    assert factors_pit._ticker_clause(None) == 'TRUE'
    assert factors_pit._ticker_params(None) == []


def test_an_empty_ticker_list_means_no_names_rather_than_all_of_them():
    """A universe that resolved to nothing used to build over the whole database."""
    assert factors_pit._ticker_clause([]) != 'TRUE'
    assert factors_pit._ticker_params([]) == [[]]


# --- a rank is not a property of a ticker and a date ---


# --- dates a name could not have had a row on ---
