"""Tests for the point-in-time factor panel.

The fixture plants the cases point-in-time exists to get right: a quarter published months after the
one that follows it, a ticker with too few quarters to form a window, and two fundamentals vintages
that disagree. `OK` and `LATE` carry identical revenue and differ only in when one filing was
published, so any test comparing the two isolates publication timing from the numbers themselves.
"""

import time
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


# --- vintages ---

def test_as_reported_vintage_never_reads_a_restated_row(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'], vintage=factors_pit.AS_REPORTED)
    assert ttm.loc[(pd.Timestamp('2021-02-15'), 'VINT'), 'revenue_ttm'] == pytest.approx(180 + 190 + 200 + 210)


def test_the_restated_vintage_reads_the_restated_numbers(fundamentals):
    ttm = _ttm(fundamentals, ['2021-02-15'], vintage=factors_pit.RESTATED)
    assert ttm.loc[(pd.Timestamp('2021-02-15'), 'VINT'), 'revenue_ttm'] == pytest.approx(2 * (180 + 190 + 200 + 210))


def test_the_restated_vintage_holds_only_the_tickers_that_have_restated_rows(fundamentals):
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(['2021-02-15']), vintage=factors_pit.RESTATED)
    assert sorted(df['Ticker'].unique()) == ['VINT']


def test_an_unknown_vintage_is_rejected(fundamentals):
    with pytest.raises(ValueError, match='vintage must be one of'):
        factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                  as_of=pd.DatetimeIndex(['2021-02-15']), vintage='whatever')


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


def test_the_share_extras_carry_the_current_and_year_earlier_counts(fundamentals):
    df = factors_pit.quarterly_ttm(fundamentals, 'income', {'Revenue': 'revenue'},
                                   as_of=pd.DatetimeIndex(['2021-02-15']),
                                   extras=factors_pit._INCOME_EXTRAS)
    row = df.set_index('Ticker').loc['OK']
    assert row['shares_latest'] == 1_000
    assert row['shares_prior'] == 1_000
    assert row['shares_mcap'] == 1_000


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


def test_market_cap_multiplies_the_reported_share_count_by_that_close(priced):
    as_of = pd.DatetimeIndex([_LAST_SESSION])
    panel = factors_pit.build_factor_panel(priced, as_of, tickers=['OK'])
    row = panel.iloc[0]
    assert row['mcap'] == pytest.approx(row['last_close'] * 1_000)


def test_momentum_is_null_when_the_calendar_offset_row_is_missing(priced):
    as_of = pd.DatetimeIndex([_LAST_SESSION])
    panel = factors_pit.build_factor_panel(priced, as_of, tickers=_PRICED_TICKERS).set_index('Ticker')
    assert not np.isnan(panel.loc['OK', 'ret_6m'])
    assert np.isnan(panel.loc['GAPPY', 'ret_6m'])


def test_dividend_growth_compares_the_trailing_year_with_the_one_before(priced):
    panel = factors_pit.build_factor_panel(priced, pd.DatetimeIndex([_LAST_SESSION]),
                                           tickers=['OK']).set_index('Ticker')
    assert panel.loc['OK', 'div_growth'] == pytest.approx(1.0 / 0.5 - 1)


def test_build_factor_panel_returns_the_declared_columns_and_vintage(priced):
    panel = factors_pit.build_factor_panel(priced, pd.DatetimeIndex([_LAST_SESSION]), tickers=['OK'])
    assert list(panel.columns) == factors_pit.PANEL_KEY + factors_pit.IDENTITY_COLS + factors_pit.FACTORS
    assert set(panel['Vintage']) == {factors_pit.AS_REPORTED}


def test_build_factor_panel_rejects_an_empty_grid(priced):
    with pytest.raises(ValueError, match='as_of is empty'):
        factors_pit.build_factor_panel(priced, pd.DatetimeIndex([]), tickers=['OK'])


def test_the_stored_panel_is_keyed_so_a_rerun_replaces_rather_than_duplicates(priced):
    as_of = pd.DatetimeIndex([_LAST_SESSION])
    panel = factors_pit.build_factor_panel(priced, as_of, tickers=['OK'])
    factors_pit.store_factor_panel(priced, panel)
    first = priced.execute('SELECT count(*) FROM factor_panel').fetchone()[0]
    factors_pit.store_factor_panel(priced, panel)
    assert priced.execute('SELECT count(*) FROM factor_panel').fetchone()[0] == first


def test_store_factor_panel_rejects_a_frame_with_the_wrong_columns(priced):
    with pytest.raises(ValueError, match='panel columns must be'):
        factors_pit.store_factor_panel(priced, pd.DataFrame({'Ticker': ['OK']}))


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


def test_market_cap_is_continuous_across_a_real_four_for_one_split(live_db):
    """Guards the assumption the panel rests on: vendor share counts are already in current split units.

    NVDA split 4:1 on 2021-07-20. Prices are back-adjusted into current units, so if share counts were
    in the units of their own filing date, market cap would jump by the split ratio here.
    """
    grid = pd.DatetimeIndex(['2021-06-30', '2021-08-31'])
    panel = factors_pit.build_factor_panel(live_db, grid, tickers=['NVDA']).set_index('Date')
    before, after = panel.loc['2021-06-30', 'mcap'], panel.loc['2021-08-31', 'mcap']
    assert 0.5 < after / before < 2.0, f'market cap moved {after / before:.2f}x across the split'


def test_the_panel_builds_for_a_hundred_tickers_over_five_years(live_db):
    tickers = [row[0] for row in live_db.execute("""
        SELECT Ticker FROM income WHERE Period = 'Q' AND NOT IsRestated
        GROUP BY Ticker HAVING count(*) >= 20 ORDER BY Ticker LIMIT 100
    """).fetchall()]
    grid = factors_pit.rebalance_dates(live_db, start='2019-01-01', end='2023-12-31', freq='M')

    started = time.perf_counter()
    panel = factors_pit.build_factor_panel(live_db, grid, tickers=tickers)
    elapsed = time.perf_counter() - started

    assert list(panel.columns) == factors_pit.PANEL_KEY + factors_pit.IDENTITY_COLS + factors_pit.FACTORS
    assert panel['Date'].nunique() == len(grid)
    assert elapsed < 30, f'building the panel took {elapsed:.1f}s'


def test_every_rebalance_date_is_a_real_trading_session(live_db):
    grid = factors_pit.rebalance_dates(live_db, start='2020-01-01', end='2020-12-31', freq='M')
    sessions = set(panels.trading_calendar(live_db, start='2020-01-01', end='2020-12-31'))
    assert len(grid) == 12
    assert set(grid) <= sessions


def test_a_point_in_time_value_factor_never_uses_a_filing_published_later(live_db):
    """AAPL's fiscal Q1 2021 published on 2021-01-28, so a 2021-01-15 panel must predate it."""
    published = live_db.execute("""
        SELECT "Publish Date"::DATE FROM income
        WHERE Ticker = 'AAPL' AND Period = 'Q' AND NOT IsRestated AND "Report Date" = DATE '2020-12-31'
    """).fetchone()
    if published is None:
        pytest.skip('the live DB has no as-reported AAPL quarter ending 2020-12-31')
    assert published[0] > pd.Timestamp('2021-01-15').date()

    grid = pd.DatetimeIndex(['2021-01-15', '2021-02-15'])
    panel = factors_pit.build_factor_panel(live_db, grid, tickers=['AAPL']).set_index('Date')
    assert panel.loc['2021-01-15', 'earnings_yield'] != panel.loc['2021-02-15', 'earnings_yield']


# --- reading back what the stored panel covers ---

def test_the_frequency_is_read_off_the_spacing_of_the_dates():
    monthly = pd.DatetimeIndex(['2020-01-31', '2020-02-28', '2020-03-31', '2020-04-30'])
    weekly = pd.bdate_range('2020-01-03', periods=8, freq='W-FRI')
    assert factors_pit.infer_frequency(monthly) == 'M'
    assert factors_pit.infer_frequency(weekly) == 'W'
    assert factors_pit.infer_frequency(pd.bdate_range('2020-01-01', periods=10)) == 'D'


def test_a_single_date_says_nothing_about_frequency():
    assert factors_pit.infer_frequency(pd.DatetimeIndex(['2020-01-31'])) is None


def test_coverage_of_a_database_with_no_panel_is_nothing():
    import duckdb

    assert factors_pit.panel_coverage(duckdb.connect()) is None


# --- an empty universe is not every universe ---

def test_no_ticker_list_means_no_filter():
    assert factors_pit._ticker_clause(None) == 'TRUE'
    assert factors_pit._ticker_params(None) == []


def test_an_empty_ticker_list_means_no_names_rather_than_all_of_them():
    """A universe that resolved to nothing used to build over the whole database."""
    assert factors_pit._ticker_clause([]) != 'TRUE'
    assert factors_pit._ticker_params([]) == [[]]


# --- a rank is not a property of a ticker and a date ---

def _customs(**expressions):
    """Expression definitions parsed the way the file would give them."""
    from investalyze.analysis import custom_factors
    from investalyze.analysis.factors import BUILTIN_FACTORS

    text = '\n'.join(custom_factors.serialize_factor(
        name, {'expression': expression, 'higher_is_better': True, 'description': ''})
        for name, expression in expressions.items())
    return custom_factors.parse(text, BUILTIN_FACTORS)


def test_a_factor_referencing_a_rank_is_not_storable():
    """Its value depends on the names it was ranked against, which the key cannot record."""
    from investalyze.analysis import custom_factors

    customs = _customs(blend='0.5 * rank_roe + 0.5 * rank_roa')
    assert custom_factors.population_dependent(customs) == ['blend']


def test_an_expression_without_ranks_stays_stored():
    """`ret_12_1 / vol_252` is an absolute quantity, so it means the same whoever else is present."""
    from investalyze.analysis import custom_factors

    customs = _customs(rar='ret_12_1 / vol_252')
    assert custom_factors.population_dependent(customs) == []


def test_every_factor_is_either_stored_or_derived_at_read_time():
    assert set(factors_pit.STORED_FACTORS) | set(factors_pit.POPULATION_DEPENDENT) == set(factors_pit.FACTORS)
    assert not set(factors_pit.STORED_FACTORS) & set(factors_pit.POPULATION_DEPENDENT)


def test_a_reader_ranks_over_its_own_cross_section(monkeypatch):
    """The same ticker and date score differently against one peer than against four."""
    from investalyze.analysis import custom_factors
    from investalyze.analysis.factors import BUILTIN_FACTORS, BUILTIN_HIGHER_IS_BETTER

    customs = _customs(blend='0.5 * rank_roe + 0.5 * rank_roa')
    wide = pd.DataFrame({'Date': [pd.Timestamp('2020-01-31')] * 5, 'Ticker': list('ABCDE'),
                         'roe': [0.1, 0.2, 0.3, 0.4, 0.5], 'roa': [0.1, 0.2, 0.3, 0.4, 0.5]})
    narrow = wide.iloc[:2].reset_index(drop=True)

    def ranked(frame):
        return custom_factors.apply_expressions(frame, customs, BUILTIN_FACTORS,
                                                BUILTIN_HIGHER_IS_BETTER, by='Date')

    assert ranked(wide).set_index('Ticker')['blend']['A'] != \
        ranked(narrow).set_index('Ticker')['blend']['A']


def test_dropping_a_rank_column_leaves_the_absolute_ones_alone():
    """An older table holds ranks from whatever universe wrote them; they are not comparable."""
    import duckdb

    con = duckdb.connect()
    con.execute('CREATE TABLE factor_panel (Ticker VARCHAR, Date DATE, Vintage VARCHAR, '
                'roe DOUBLE, quality_blend DOUBLE)')
    con.execute("INSERT INTO factor_panel VALUES ('AAA', DATE '2020-01-31', 'as_reported', 0.2, 37.5)")

    monkeypatch_dependent = ['quality_blend']
    original = factors_pit.POPULATION_DEPENDENT
    factors_pit.POPULATION_DEPENDENT = monkeypatch_dependent
    try:
        factors_pit._widen_stored_panel(con, pd.DataFrame(columns=['Ticker', 'Date', 'Vintage', 'roe']))
    finally:
        factors_pit.POPULATION_DEPENDENT = original

    columns = {row[0] for row in con.execute('DESCRIBE factor_panel').fetchall()}
    assert 'roe' in columns
    assert 'quality_blend' not in columns


# --- dates a name could not have had a row on ---

def test_dates_before_a_name_first_traded_are_not_buildable():
    """A panel row comes from a close, so a date before the name listed has nothing to hold."""
    import duckdb

    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE)')
    con.execute("INSERT INTO prices VALUES ('FSLR', DATE '2006-11-17', 24.0)")
    grid = pd.DatetimeIndex(['2000-01-31', '2006-10-31', '2006-11-30', '2007-01-31'])

    buildable = factors_pit.buildable_dates(con, grid, ['FSLR'])

    assert list(buildable) == [pd.Timestamp('2006-11-30'), pd.Timestamp('2007-01-31')]


def test_a_ticker_list_with_no_prices_can_build_nothing():
    import duckdb

    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE)')
    grid = pd.DatetimeIndex(['2020-01-31'])

    assert len(factors_pit.buildable_dates(con, grid, ['NOPE'])) == 0


def test_the_whole_database_keeps_every_date_it_was_asked_for():
    import duckdb

    con = duckdb.connect()
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE)')
    con.execute("INSERT INTO prices VALUES ('AAA', DATE '1990-01-02', 1.0)")
    grid = pd.DatetimeIndex(['2000-01-31', '2020-01-31'])

    assert list(factors_pit.buildable_dates(con, grid, None)) == list(grid)
