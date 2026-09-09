"""Regression test: the Screener's live factors and the backtest's point-in-time factors must agree.

`factors.fundamental_factors` and `factors_pit._derive_factors` are two hand-kept-in-sync copies of the
same formula block (see the backlog: converging them into one code path was deliberately rejected, since
it would give the always-working Screener a hard dependency on `factor_panel`). This test is the
mechanical check that stands in for that decision: it does not converge the two paths, it just catches
the day one of them drifts from the other.
"""

import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import factors, factors_pit

REPORT_DATES = pd.to_datetime(['2024-12-31', '2024-09-30', '2024-06-30', '2024-03-31',
                               '2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31'])
PUBLISH_DATES = REPORT_DATES + pd.Timedelta(days=30)
AS_OF = pd.Timestamp('2025-02-15')   # after every quarter's Publish Date, so the whole window is visible

SHARED_COLUMNS = ['earnings_yield', 'fcf_yield', 'sales_yield', 'book_to_market', 'ebitda_ev',
                  'roe', 'roa', 'gross_margin', 'op_margin', 'margin_trend', 'accruals',
                  'debt_to_equity', 'interest_coverage', 'revenue_yoy', 'eps_yoy', 'fcf_yoy',
                  'share_change_1y']


def _income_rows() -> pd.DataFrame:
    return pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'Q', 'IsRestated': True,
        'Report Date': REPORT_DATES, 'Publish Date': PUBLISH_DATES,
        'Shares (Basic)': 1000.0, 'Shares (Diluted)': 1000.0,
        'Revenue': [1100.0] * 4 + [1000.0] * 4,
        'Gross Profit': [550.0] * 4 + [500.0] * 4,
        'Operating Income (Loss)': [220.0] * 4 + [200.0] * 4,
        'Depreciation & Amortization': [-55.0] * 4 + [-50.0] * 4,
        'Interest Expense': [-11.0] * 4 + [-10.0] * 4,
        'Net Income': [110.0] * 4 + [100.0] * 4,
        'Net Income (Common)': [110.0] * 4 + [100.0] * 4,
    })


def _balance_rows() -> pd.DataFrame:
    return pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'Q', 'IsRestated': True,
        'Report Date': REPORT_DATES[:2], 'Publish Date': PUBLISH_DATES[:2],
        'Total Equity': [2000.0, 1500.0], 'Total Assets': [5000.0, 4000.0],
        'Short Term Debt': [300.0, 250.0], 'Long Term Debt': [700.0, 650.0],
        'Cash & Cash Equivalents': [500.0, 400.0],
    })


def _cashflow_rows() -> pd.DataFrame:
    return pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'Q', 'IsRestated': True,
        'Report Date': REPORT_DATES, 'Publish Date': PUBLISH_DATES,
        'Net Cash from Operating Activities': [130.0] * 4 + [120.0] * 4,
        'Change in Fixed Assets & Intangibles': [-30.0] * 8,
    })


@pytest.fixture
def con():
    """In-memory DuckDB holding only the three statement tables both paths read."""
    import duckdb

    con = duckdb.connect()
    con.register('_i', _income_rows())
    con.execute('CREATE TABLE income AS SELECT * FROM _i')
    con.register('_b', _balance_rows())
    con.execute('CREATE TABLE balance AS SELECT * FROM _b')
    con.register('_c', _cashflow_rows())
    con.execute('CREATE TABLE cashflow AS SELECT * FROM _c')
    yield con
    con.close()


def test_fundamental_factors_matches_the_point_in_time_derivation(con):
    """`version='restated'` reads the same rows `fundamental_factors` reads, per `_derive_factors`'s
    own docstring; as of a date after every quarter published, the two should compute identical ratios
    for the same company from the same statement rows.
    """
    identity = pd.DataFrame({'Ticker': ['AAA'], 'mcap': [2_000_000.0]})
    live = factors.fundamental_factors(con, identity)

    grid = pd.DatetimeIndex([AS_OF])
    income = factors_pit.quarterly_ttm(con, 'income', factors.INCOME_TTM, as_of=grid, version='restated',
                                       tickers=['AAA'], extras=factors_pit._INCOME_EXTRAS)
    cashflow = factors_pit.quarterly_ttm(con, 'cashflow', factors.CASHFLOW_TTM, as_of=grid,
                                         version='restated', tickers=['AAA'])
    balance = factors_pit.balance_panel(con, as_of=grid, version='restated', tickers=['AAA'])
    raw = income.merge(cashflow, on=['Date', 'Ticker']).merge(balance, on=['Date', 'Ticker'])
    raw['mcap'] = 2_000_000.0
    raw['div_ttm'] = np.nan
    raw['div_prior'] = np.nan
    pit = factors_pit._derive_factors(raw.copy())

    live_row = live.set_index('Ticker').loc['AAA']
    pit_row = pit.set_index('Ticker').loc['AAA']
    for column in SHARED_COLUMNS:
        assert pit_row[column] == pytest.approx(live_row[column]), column
