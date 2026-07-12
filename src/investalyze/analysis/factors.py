"""Per-ticker factor library for stock screening: today-snapshot values from the shared DuckDB.

Pure functions over an open read-only connection, so notebooks and app pages can share one
implementation. Fundamentals use restated quarterly rows: TTM is the
sum of the latest 4 quarters, the prior-year TTM is quarters 5 to 8, balance items come from
the latest quarter. Missing inputs or nonsensical denominators yield NaN. Factor columns hold
natural raw values; consumers orient ranks with `HIGHER_IS_BETTER` so rank 100 always means best.
"""

import duckdb
import numpy as np
import pandas as pd

MARKET_TICKER = '^SPX'
MIN_RISK_OBS = 200  # minimum daily return observations for vol_252 / beta_252

FAMILIES: dict[str, list[str]] = {
    'value': ['earnings_yield', 'fcf_yield', 'sales_yield', 'book_to_market', 'ebitda_ev'],
    'momentum': ['ret_12_1', 'ret_6m', 'high_52w'],
    'risk': ['vol_252', 'beta_252'],
    'quality': ['roe', 'roa', 'gross_margin', 'op_margin', 'margin_trend', 'accruals',
                'debt_to_equity', 'interest_coverage'],
    'growth': ['revenue_yoy', 'revenue_3y', 'eps_yoy', 'fcf_yoy', 'share_change_1y', 'div_growth'],
}
FACTORS: list[str] = [factor for family in FAMILIES.values() for factor in family]
_LOWER_IS_BETTER = {'vol_252', 'beta_252', 'accruals', 'debt_to_equity', 'share_change_1y'}
HIGHER_IS_BETTER: dict[str, bool] = {factor: factor not in _LOWER_IS_BETTER for factor in FACTORS}

# source column -> short alias for the TTM aggregates
_INCOME_TTM = {
    'Revenue': 'revenue', 'Gross Profit': 'gross_profit', 'Operating Income (Loss)': 'op_income',
    'Depreciation & Amortization': 'dep_amort', 'Interest Expense': 'interest_expense',
    'Net Income': 'net_income', 'Net Income (Common)': 'ni_common',
}
_CASHFLOW_TTM = {'Net Cash from Operating Activities': 'cfo', 'Change in Fixed Assets & Intangibles': 'capex'}


def _growth(now: pd.Series, prior: pd.Series) -> np.ndarray:
    """Relative change now vs prior; NaN unless prior is strictly positive."""
    return np.where(prior > 0, now / prior - 1, np.nan)


def _ttm(con: duckdb.DuckDBPyConnection, table: str, columns: dict[str, str]) -> pd.DataFrame:
    """Per ticker: TTM (latest 4 restated quarters) and prior-year TTM (quarters 5 to 8) sums.

    Output columns `<alias>_ttm` / `<alias>_prior`; a window with fewer than 4 quarters is NaN.
    """
    terms = ',\n'.join(
        f'sum(CASE WHEN rn <= 4 THEN "{src}" END) AS {alias}_ttm, '
        f'sum(CASE WHEN rn BETWEEN 5 AND 8 THEN "{src}" END) AS {alias}_prior'
        for src, alias in columns.items()
    )
    df = con.execute(f"""
        WITH q AS (
            SELECT *, row_number() OVER (PARTITION BY Ticker ORDER BY "Report Date" DESC) AS rn
            FROM {table} WHERE Period = 'Q' AND IsRestated
        )
        SELECT Ticker,
               count(CASE WHEN rn <= 4 THEN 1 END) AS n_recent,
               count(CASE WHEN rn BETWEEN 5 AND 8 THEN 1 END) AS n_prior,
               {terms}
        FROM q WHERE rn <= 8 GROUP BY Ticker
    """).df()
    numeric = [c for c in df.columns if c != 'Ticker']
    df[numeric] = df[numeric].astype('float64')   # DuckDB returns nullable Int64 when a sum is NULL
    recent = [f'{alias}_ttm' for alias in columns.values()]
    prior = [f'{alias}_prior' for alias in columns.values()]
    df.loc[df['n_recent'] < 4, recent] = np.nan
    df.loc[df['n_prior'] < 4, prior] = np.nan
    return df.drop(columns=['n_recent', 'n_prior'])


def _latest_balance(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Latest restated quarterly balance items per ticker: equity, assets, total debt, cash."""
    df = con.execute("""
        SELECT Ticker,
               arg_max("Total Equity", "Report Date") AS equity,
               arg_max("Total Assets", "Report Date") AS assets,
               arg_max(COALESCE("Short Term Debt", 0) + COALESCE("Long Term Debt", 0), "Report Date") AS debt,
               arg_max(COALESCE("Cash & Cash Equivalents", 0), "Report Date") AS cash
        FROM balance WHERE Period = 'Q' AND IsRestated
        GROUP BY Ticker
    """).df()
    numeric = ['equity', 'assets', 'debt', 'cash']
    df[numeric] = df[numeric].astype('float64')   # equity/assets can be NULL -> nullable Int64 from DuckDB
    return df


def _shares_by_rank(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Diluted shares at the latest quarter and 4 quarters earlier (for EPS and dilution)."""
    df = con.execute("""
        WITH q AS (
            SELECT Ticker, "Shares (Diluted)" AS shares,
                   row_number() OVER (PARTITION BY Ticker ORDER BY "Report Date" DESC) AS rn
            FROM income WHERE Period = 'Q' AND IsRestated
        )
        SELECT Ticker,
               max(CASE WHEN rn = 1 THEN shares END) AS shares_latest,
               max(CASE WHEN rn = 5 THEN shares END) AS shares_prior
        FROM q WHERE rn IN (1, 5) GROUP BY Ticker
    """).df()
    numeric = ['shares_latest', 'shares_prior']
    df[numeric] = df[numeric].astype('float64')   # shares_prior is NULL for tickers with <5 quarters
    return df


def _revenue_cagr_3y(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """3-year revenue CAGR from restated fiscal-year rows (latest FY vs 3 FYs earlier)."""
    df = con.execute("""
        WITH a AS (
            SELECT Ticker, Revenue,
                   row_number() OVER (PARTITION BY Ticker ORDER BY "Report Date" DESC) AS rn
            FROM income WHERE Period = 'A' AND IsRestated AND Revenue IS NOT NULL
        )
        SELECT Ticker,
               max(CASE WHEN rn = 1 THEN Revenue END) AS rev_fy0,
               max(CASE WHEN rn = 4 THEN Revenue END) AS rev_fy3
        FROM a WHERE rn IN (1, 4) GROUP BY Ticker
    """).df()
    df[['rev_fy0', 'rev_fy3']] = df[['rev_fy0', 'rev_fy3']].astype('float64')   # rev_fy3 NULL for younger tickers
    valid = (df['rev_fy0'] > 0) & (df['rev_fy3'] > 0)
    df['revenue_3y'] = np.where(valid, (df['rev_fy0'] / df['rev_fy3']) ** (1 / 3) - 1, np.nan)
    return df[['Ticker', 'revenue_3y']]


def _identity(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Company identity, last close and market cap for every ticker with price data."""
    base = con.execute("""
        SELECT Ticker, arg_max(C, Date) AS last_close
        FROM prices GROUP BY Ticker
    """).df()
    shares = con.execute("""
        SELECT Ticker, arg_max(COALESCE("Shares (Basic)", "Shares (Diluted)"), "Report Date") AS shares
        FROM income
        WHERE Period = 'Q' AND IsRestated AND COALESCE("Shares (Basic)", "Shares (Diluted)") IS NOT NULL
        GROUP BY Ticker
    """).df()
    companies = con.execute(
        'SELECT Ticker, CompanyName AS name, Sector AS sector, Industry AS industry FROM companies'
    ).df()
    df = base.merge(shares, on='Ticker', how='left').merge(companies, on='Ticker', how='left')
    df['mcap'] = df['shares'] * df['last_close']
    df['mcap_bn'] = (df['mcap'] / 1e9).round(3)
    df['name'] = df['name'].fillna('')
    for column in ('sector', 'industry'):
        df[column] = df[column].fillna('unknown')
    return df[['Ticker', 'name', 'sector', 'industry', 'last_close', 'mcap', 'mcap_bn']]


def fundamental_factors(con: duckdb.DuckDBPyConnection, identity: pd.DataFrame) -> pd.DataFrame:
    """Value, quality and growth factors joined onto the identity frame (div_growth comes separately)."""
    df = identity.merge(_ttm(con, 'income', _INCOME_TTM), on='Ticker', how='left')
    df = df.merge(_ttm(con, 'cashflow', _CASHFLOW_TTM), on='Ticker', how='left')
    df = df.merge(_latest_balance(con), on='Ticker', how='left')
    df = df.merge(_shares_by_rank(con), on='Ticker', how='left')
    df = df.merge(_revenue_cagr_3y(con), on='Ticker', how='left')

    mcap = df['mcap']
    fcf_ttm = df['cfo_ttm'] + df['capex_ttm']            # capex is stored negative
    fcf_prior = df['cfo_prior'] + df['capex_prior']
    ev = mcap + df['debt'] - df['cash']
    ebitda = df['op_income_ttm'] - df['dep_amort_ttm']   # D&A is stored negative; subtract to add it back
    positive_equity = df['equity'] > 0

    df['earnings_yield'] = df['ni_common_ttm'] / mcap
    df['fcf_yield'] = fcf_ttm / mcap
    df['sales_yield'] = df['revenue_ttm'] / mcap
    df['book_to_market'] = df['equity'] / mcap
    df['ebitda_ev'] = np.where((ev > 0) & (ebitda > 0), ebitda / ev, np.nan)

    df['roe'] = np.where(positive_equity, df['net_income_ttm'] / df['equity'], np.nan)
    df['roa'] = np.where(df['assets'] > 0, df['net_income_ttm'] / df['assets'], np.nan)
    df['gross_margin'] = np.where(df['revenue_ttm'] > 0, df['gross_profit_ttm'] / df['revenue_ttm'], np.nan)
    df['op_margin'] = np.where(df['revenue_ttm'] > 0, df['op_income_ttm'] / df['revenue_ttm'], np.nan)
    op_margin_prior = np.where(df['revenue_prior'] > 0, df['op_income_prior'] / df['revenue_prior'], np.nan)
    df['margin_trend'] = df['op_margin'] - op_margin_prior
    df['accruals'] = np.where(df['assets'] > 0, (df['net_income_ttm'] - df['cfo_ttm']) / df['assets'], np.nan)
    df['debt_to_equity'] = np.where(positive_equity, df['debt'] / df['equity'], np.nan)
    df['interest_coverage'] = np.where(df['interest_expense_ttm'] != 0,   # interest expense is stored negative
                                       df['op_income_ttm'] / df['interest_expense_ttm'].abs(), np.nan)

    df['revenue_yoy'] = _growth(df['revenue_ttm'], df['revenue_prior'])
    eps = pd.Series(np.where(df['shares_latest'] > 0, df['ni_common_ttm'] / df['shares_latest'], np.nan))
    eps_prior = pd.Series(np.where(df['shares_prior'] > 0, df['ni_common_prior'] / df['shares_prior'], np.nan))
    df['eps_yoy'] = _growth(eps, eps_prior)
    df['fcf_yoy'] = _growth(fcf_ttm, fcf_prior)
    df['share_change_1y'] = _growth(df['shares_latest'], df['shares_prior'])
    return df


def price_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Momentum (12-1, 6m, 52w-high proximity) and risk (volatility, beta vs MARKET_TICKER).

    Works on each ticker's trailing 253 sessions. Momentum offsets need the exact row (rn 22,
    127, 253) to exist; vol/beta need at least MIN_RISK_OBS daily returns. Otherwise NaN.
    """
    prices = con.execute("""
        WITH r AS (
            SELECT Ticker, Date, AC, row_number() OVER (PARTITION BY Ticker ORDER BY Date DESC) AS rn
            FROM prices
        )
        SELECT Ticker, Date, AC, rn FROM r WHERE rn <= 253 ORDER BY Ticker, Date
    """).df()
    market = con.execute(
        'SELECT Date, C FROM market_data WHERE Ticker = ? ORDER BY Date DESC LIMIT 253', [MARKET_TICKER]
    ).df().sort_values('Date')
    market_returns = pd.Series(market['C'].to_numpy(), index=pd.DatetimeIndex(market['Date'])).pct_change().dropna()

    rows = []
    for ticker, group in prices.groupby('Ticker'):
        ac = pd.Series(group['AC'].to_numpy(), index=pd.DatetimeIndex(group['Date']))
        by_rn = dict(zip(group['rn'], group['AC']))
        returns = ac.pct_change().dropna()
        row: dict[str, object] = {'Ticker': ticker}
        row['ret_12_1'] = by_rn[22] / by_rn[253] - 1 if 22 in by_rn and 253 in by_rn else np.nan
        row['ret_6m'] = by_rn[1] / by_rn[127] - 1 if 127 in by_rn else np.nan
        row['high_52w'] = by_rn[1] / ac.max() if len(ac) and ac.max() > 0 else np.nan
        if len(returns) >= MIN_RISK_OBS:
            row['vol_252'] = float(returns.std() * np.sqrt(252))
            joint = pd.concat([returns.rename('r'), market_returns.rename('rm')], axis=1, join='inner').dropna()
            market_var = joint['rm'].var()
            beta_ok = len(joint) >= MIN_RISK_OBS and market_var > 0
            row['beta_252'] = float(joint['r'].cov(joint['rm']) / market_var) if beta_ok else np.nan
        else:
            row['vol_252'] = np.nan
            row['beta_252'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dividend_growth(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Dividends per ticker over the trailing 365 days vs the 365 days before, anchored at the latest price date."""
    return con.execute("""
        WITH anchor AS (SELECT max(Date) AS d FROM prices)
        SELECT dv.Ticker,
               sum(CASE WHEN dv.Date > a.d - INTERVAL 365 DAY THEN dv.Dividend END) AS div_ttm,
               sum(CASE WHEN dv.Date <= a.d - INTERVAL 365 DAY
                         AND dv.Date > a.d - INTERVAL 730 DAY THEN dv.Dividend END) AS div_prior
        FROM dividends dv CROSS JOIN anchor a
        GROUP BY dv.Ticker
    """).df()


def build_factors(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per ticker with identity columns and every factor in FACTORS."""
    df = fundamental_factors(con, _identity(con))
    df = df.merge(price_factors(con), on='Ticker', how='left')
    df = df.merge(dividend_growth(con), on='Ticker', how='left')
    df['div_growth'] = _growth(df['div_ttm'], df['div_prior'])
    keep = ['Ticker', 'name', 'sector', 'industry', 'last_close', 'mcap', 'mcap_bn'] + FACTORS
    return df[keep].sort_values('Ticker', ignore_index=True)
