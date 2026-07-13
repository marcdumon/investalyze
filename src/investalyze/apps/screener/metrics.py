"""Build the per-ticker metrics table the ticker selector filters on."""

import duckdb
import numpy as np
import pandas as pd

# lower bounds in dollars for the usual market-cap bucket labels
MCAP_BINS = [-np.inf, 50e6, 300e6, 2e9, 10e9, 200e9, np.inf]
MCAP_LABELS = ['nano', 'micro', 'small', 'mid', 'large', 'mega']


def build_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per ticker that has price data, with every filterable angle as a column.

    Joins price-history stats, liquidity, market cap (latest reported shares x last close),
    fundamentals coverage, company identity and anomaly counts.
    """
    price_stats = con.execute("""
        SELECT Ticker,
               MIN(Date) AS first_date,
               MAX(Date) AS last_date,
               COUNT(*) AS n_days,
               arg_max(C, Date) AS last_close
        FROM prices
        GROUP BY Ticker
    """).df()

    # liquidity over each ticker's own most recent 252 sessions, so delisted tickers get their final year
    dollar_volume = con.execute("""
        SELECT Ticker, median(C * V) AS dollar_vol
        FROM (
            SELECT Ticker, C, V, row_number() OVER (PARTITION BY Ticker ORDER BY Date DESC) AS rn
            FROM prices
        )
        WHERE rn <= 252
        GROUP BY Ticker
    """).df()

    shares = con.execute("""
        SELECT Ticker, arg_max(shares, "Report Date") AS shares
        FROM (
            SELECT Ticker, COALESCE("Shares (Basic)", "Shares (Diluted)") AS shares, "Report Date"
            FROM income
            WHERE COALESCE("Shares (Basic)", "Shares (Diluted)") IS NOT NULL
        )
        GROUP BY Ticker
    """).df()

    n_periods = con.execute('SELECT Ticker, COUNT(*) AS n_periods FROM income GROUP BY Ticker').df()
    anomalies = con.execute('SELECT Ticker, COUNT(*) AS n_anomalies FROM anomalies GROUP BY Ticker').df()
    companies = con.execute("""
        SELECT Ticker, CompanyName AS name, Sector AS sector, Industry AS industry,
               Country AS country, NrEmployees AS employees
        FROM companies
    """).df()

    df = price_stats
    for other in (dollar_volume, shares, n_periods, anomalies, companies):
        df = df.merge(other, on='Ticker', how='left')

    df['years'] = ((df['last_date'] - df['first_date']).dt.days / 365.25).round(1)
    df['active'] = df['last_date'] >= df['last_date'].max() - pd.Timedelta(days=7)
    df['mcap'] = df['shares'] * df['last_close']
    df['mcap_bucket'] = pd.cut(df['mcap'], bins=MCAP_BINS, labels=MCAP_LABELS).astype(object)

    df['n_periods'] = df['n_periods'].fillna(0).astype(int)
    df['n_anomalies'] = df['n_anomalies'].fillna(0).astype(int)
    for column in ('sector', 'industry', 'country'):
        df[column] = df[column].fillna('unknown')
    df['name'] = df['name'].fillna('')

    # display-friendly units for the grid; filters use the raw columns
    df['mcap_bn'] = (df['mcap'] / 1e9).round(3)
    df['dvol_mn'] = (df['dollar_vol'] / 1e6).round(3)
    df['last_close'] = df['last_close'].round(2)
    df['first_date'] = df['first_date'].dt.date.astype(str)
    df['last_date'] = df['last_date'].dt.date.astype(str)

    return df.sort_values('Ticker', ignore_index=True)
