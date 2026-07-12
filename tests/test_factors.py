"""Tests for the per-ticker factor library (seeded throwaway DuckDB, deterministic values)."""
import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import factors

DATES = pd.bdate_range(end='2026-07-10', periods=300)
QUARTERS = pd.to_datetime(['2026-03-31', '2025-12-31', '2025-09-30', '2025-06-30',
                           '2025-03-31', '2024-12-31', '2024-09-30', '2024-06-30'])


def price_series(step: float, start: float) -> pd.Series:
    """Deterministic series: +step on odd days, flat on even days."""
    returns = np.where(np.arange(len(DATES)) % 2 == 1, step, 0.0)
    returns[0] = 0.0
    return pd.Series(start * np.cumprod(1 + returns), index=DATES)


AAA = price_series(0.04, 100.0)   # stock: alternating +4% / 0%
SPX = price_series(0.02, 50.0)    # market: alternating +2% / 0% on the same days -> beta exactly 2


def _price_rows(ticker: str, series: pd.Series) -> pd.DataFrame:
    values = series.to_numpy()
    return pd.DataFrame({'Ticker': ticker, 'Date': series.index.date, 'O': values, 'H': values,
                         'L': values, 'C': values, 'V': 1000, 'AC': values})


def _seed_prices(con: duckdb.DuckDBPyConnection) -> None:
    bbb = pd.Series(10.0, index=DATES[-50:])
    prices = pd.concat([_price_rows('AAA', AAA), _price_rows('BBB', bbb)], ignore_index=True)
    con.register('_p', prices)
    con.execute('CREATE TABLE prices AS SELECT * FROM _p')
    spx = _price_rows('^SPX', SPX).drop(columns=['V', 'AC'])
    spx['AssetClass'] = 'indices'
    con.register('_m', spx)
    con.execute('CREATE TABLE market_data AS SELECT * FROM _m')


def _seed_fundamentals(con: duckdb.DuckDBPyConnection) -> None:
    income_q = pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'Q', 'IsRestated': True, 'Report Date': QUARTERS.date,
        'Shares (Basic)': 1000.0, 'Shares (Diluted)': 1000.0,
        'Revenue': [1100.0] * 4 + [1000.0] * 4,
        'Gross Profit': [550.0] * 4 + [500.0] * 4,
        'Operating Income (Loss)': [220.0] * 4 + [200.0] * 4,
        'Depreciation & Amortization': [-55.0] * 4 + [-50.0] * 4,   # stored negative, like real fundamentals data
        'Interest Expense': [-11.0] * 4 + [-10.0] * 4,               # stored negative, like real fundamentals data
        'Net Income': [110.0] * 4 + [100.0] * 4,
        'Net Income (Common)': [110.0] * 4 + [100.0] * 4,
    })
    income_a = pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'A', 'IsRestated': True,
        'Report Date': pd.to_datetime(['2025-12-31', '2024-12-31', '2023-12-31', '2022-12-31']).date,
        'Shares (Basic)': 1000.0, 'Shares (Diluted)': 1000.0,
        'Revenue': [4000.0, 3000.0, 2500.0, 2000.0],
        'Gross Profit': np.nan, 'Operating Income (Loss)': np.nan,
        'Depreciation & Amortization': np.nan, 'Interest Expense': np.nan,
        'Net Income': np.nan, 'Net Income (Common)': np.nan,
    })
    con.register('_i', pd.concat([income_q, income_a], ignore_index=True))
    con.execute('CREATE TABLE income AS SELECT * FROM _i')

    balance = pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'Q', 'IsRestated': True,
        'Report Date': pd.to_datetime(['2026-03-31', '2025-12-31']).date,
        'Total Equity': [2000.0, 1500.0], 'Total Assets': [5000.0, 4000.0],
        'Short Term Debt': [300.0, 250.0], 'Long Term Debt': [700.0, 650.0],
        'Cash & Cash Equivalents': [500.0, 400.0],
    })
    con.register('_b', balance)
    con.execute('CREATE TABLE balance AS SELECT * FROM _b')

    cashflow = pd.DataFrame({
        'Ticker': 'AAA', 'Period': 'Q', 'IsRestated': True, 'Report Date': QUARTERS.date,
        'Net Cash from Operating Activities': [130.0] * 4 + [120.0] * 4,
        'Change in Fixed Assets & Intangibles': [-30.0] * 8,
    })
    con.register('_c', cashflow)
    con.execute('CREATE TABLE cashflow AS SELECT * FROM _c')


def _seed_events(con: duckdb.DuckDBPyConnection) -> None:
    dividends = pd.DataFrame({
        'Ticker': 'AAA',
        'Date': pd.to_datetime(['2026-06-01', '2026-03-01', '2025-12-01', '2025-09-01',
                                '2025-06-01', '2025-03-01', '2024-12-01', '2024-09-01']).date,
        'Dividend': [1.0] * 4 + [0.8] * 4,
    })
    con.register('_d', dividends)
    con.execute('CREATE TABLE dividends AS SELECT * FROM _d')
    companies = pd.DataFrame([{'Ticker': 'AAA', 'CompanyName': 'Alpha Corp',
                               'Sector': 'Technology', 'Industry': 'Software'}])
    con.register('_co', companies)
    con.execute('CREATE TABLE companies AS SELECT * FROM _co')


@pytest.fixture
def con(tmp_path):
    con = duckdb.connect(str(tmp_path / 'factors.duckdb'))
    _seed_prices(con)
    _seed_fundamentals(con)
    _seed_events(con)
    yield con
    con.close()


def _row(df: pd.DataFrame, ticker: str) -> pd.Series:
    return df.loc[df['Ticker'] == ticker].iloc[0]


# ---------- building blocks ----------

def test_ttm_sums_and_prior_window(con):
    df = factors._ttm(con, 'income', factors._INCOME_TTM)
    row = _row(df, 'AAA')
    assert row['revenue_ttm'] == 4400.0
    assert row['revenue_prior'] == 4000.0
    assert row['ni_common_ttm'] == 440.0
    assert row['interest_expense_ttm'] == -44.0


def test_ttm_masks_incomplete_windows(con):
    con.execute("DELETE FROM income WHERE Period = 'Q' AND \"Report Date\" < DATE '2025-06-01'")
    df = factors._ttm(con, 'income', factors._INCOME_TTM)   # 4 recent quarters left, prior window incomplete
    row = _row(df, 'AAA')
    assert row['revenue_ttm'] == 4400.0
    assert pd.isna(row['revenue_prior'])


def test_latest_balance_picks_latest_quarter(con):
    row = _row(factors._latest_balance(con), 'AAA')
    assert row['equity'] == 2000.0
    assert row['assets'] == 5000.0
    assert row['debt'] == 1000.0
    assert row['cash'] == 500.0


def test_shares_by_rank(con):
    row = _row(factors._shares_by_rank(con), 'AAA')
    assert row['shares_latest'] == 1000.0
    assert row['shares_prior'] == 1000.0


def test_revenue_cagr_3y(con):
    row = _row(factors._revenue_cagr_3y(con), 'AAA')
    assert row['revenue_3y'] == pytest.approx((4000.0 / 2000.0) ** (1 / 3) - 1)


def test_identity(con):
    df = factors._identity(con)
    aaa = _row(df, 'AAA')
    assert aaa['name'] == 'Alpha Corp'
    assert aaa['sector'] == 'Technology'
    assert aaa['last_close'] == pytest.approx(AAA.iloc[-1])
    assert aaa['mcap'] == pytest.approx(1000.0 * AAA.iloc[-1])
    bbb = _row(df, 'BBB')
    assert bbb['sector'] == 'unknown'
    assert pd.isna(bbb['mcap'])


# ---------- fundamental factors ----------

@pytest.fixture
def fundamentals(con):
    return factors.fundamental_factors(con, factors._identity(con))


def test_value_factors(fundamentals):
    row = _row(fundamentals, 'AAA')
    mcap = 1000.0 * AAA.iloc[-1]
    assert row['earnings_yield'] == pytest.approx(440.0 / mcap)
    assert row['fcf_yield'] == pytest.approx((520.0 - 120.0) / mcap)   # TTM CFO 520 + capex -120
    assert row['sales_yield'] == pytest.approx(4400.0 / mcap)
    assert row['book_to_market'] == pytest.approx(2000.0 / mcap)
    assert row['ebitda_ev'] == pytest.approx((880.0 + 220.0) / (mcap + 1000.0 - 500.0))


def test_quality_factors(fundamentals):
    row = _row(fundamentals, 'AAA')
    assert row['roe'] == pytest.approx(440.0 / 2000.0)
    assert row['roa'] == pytest.approx(440.0 / 5000.0)
    assert row['gross_margin'] == pytest.approx(0.5)
    assert row['op_margin'] == pytest.approx(0.2)
    assert row['margin_trend'] == pytest.approx(0.0)     # 880/4400 now vs 800/4000 a year ago
    assert row['accruals'] == pytest.approx((440.0 - 520.0) / 5000.0)
    assert row['debt_to_equity'] == pytest.approx(1000.0 / 2000.0)
    assert row['interest_coverage'] == pytest.approx(880.0 / 44.0)


def test_growth_factors(fundamentals):
    row = _row(fundamentals, 'AAA')
    assert row['revenue_yoy'] == pytest.approx(0.1)
    assert row['revenue_3y'] == pytest.approx(2.0 ** (1 / 3) - 1)
    assert row['eps_yoy'] == pytest.approx(0.1)          # shares constant, NI 440 vs 400
    assert row['fcf_yoy'] == pytest.approx(400.0 / 360.0 - 1)
    assert row['share_change_1y'] == pytest.approx(0.0)


def test_missing_fundamentals_are_nan(fundamentals):
    row = _row(fundamentals, 'BBB')
    for factor in ['earnings_yield', 'roe', 'revenue_yoy', 'ebitda_ev', 'interest_coverage']:
        assert pd.isna(row[factor]), factor


# ---------- price factors ----------

def test_momentum_factors(con):
    df = factors.price_factors(con)
    row = _row(df, 'AAA')
    assert row['ret_6m'] == pytest.approx(AAA.iloc[-1] / AAA.iloc[-127] - 1)
    assert row['ret_12_1'] == pytest.approx(AAA.iloc[-22] / AAA.iloc[-253] - 1)
    assert row['high_52w'] == pytest.approx(1.0)          # monotonic series: last is the high


def test_vol_and_beta(con):
    df = factors.price_factors(con)
    row = _row(df, 'AAA')
    expected_vol = AAA.iloc[-253:].pct_change().dropna().std() * np.sqrt(252)
    assert row['vol_252'] == pytest.approx(expected_vol)
    assert row['beta_252'] == pytest.approx(2.0)          # stock returns are exactly 2x market returns


def test_insufficient_history_is_nan(con):
    row = _row(factors.price_factors(con), 'BBB')         # 50 sessions only
    for factor in ['ret_12_1', 'ret_6m', 'vol_252', 'beta_252']:
        assert pd.isna(row[factor]), factor
    assert row['high_52w'] == pytest.approx(1.0)          # computable from any history


def test_dividend_growth(con):
    df = factors.dividend_growth(con)
    row = _row(df, 'AAA')
    assert row['div_ttm'] == pytest.approx(4.0)
    assert row['div_prior'] == pytest.approx(3.2)


# ---------- assembly ----------

def test_build_factors_shape_and_content(con):
    df = factors.build_factors(con)
    assert list(df['Ticker']) == ['AAA', 'BBB']
    for column in ['name', 'sector', 'mcap_bn'] + factors.FACTORS:
        assert column in df.columns, column
    aaa = _row(df, 'AAA')
    assert aaa['div_growth'] == pytest.approx(4.0 / 3.2 - 1)
    assert aaa['roe'] == pytest.approx(0.22)
    bbb = _row(df, 'BBB')
    assert pd.isna(bbb['div_growth'])
