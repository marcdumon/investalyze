"""Data access and shaping for the ticker analysis page.

Snapshot metrics and factor columns come from the screener's cached pool so values match the
screener exactly. Time series (prices, benchmark, quarterly fundamentals) are queried per
selection via short-lived read-only connections. Shaping helpers are pure frame-in/frame-out
functions, testable without a database.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from investalyze.analysis.factors import FACTORS, FAMILIES
from investalyze.apps.screener.logic import compute_ranks
from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'
MARKET_TICKER = '^SPX'
MIN_PEERS = 5
MAX_PEERS = 15
MAX_UNIVERSE_PEERS = 100   # keeps the strip charts and peer list responsive for large universes
MAX_LOG_GAP = 2.5   # a peer must be within ~300x of the ticker's market cap
TRAILING_WINDOWS = {'1m': 21, '3m': 63, '6m': 126, '1y': 252, '3y': 756, '5y': 1260}
RANGE_SESSIONS = {'1y': 252, '3y': 756, '5y': 1260, '10y': 2520, 'max': None}

_INCOME_ITEMS = ['Revenue', 'Gross Profit', 'Operating Income (Loss)', 'Net Income (Common)', 'Shares (Diluted)']
_CASHFLOW_ITEMS = ['Net Cash from Operating Activities', 'Change in Fixed Assets & Intangibles']
_BALANCE_ITEMS = ['Total Equity', 'Short Term Debt', 'Long Term Debt', 'Cash, Cash Equivalents & Short Term Investments']


def _mcap_gap(peers: pd.DataFrame, mcap: float) -> pd.Series:
    """Absolute log10 market-cap distance to `mcap`; NaN for peers without a positive mcap."""
    return (np.log10(peers['mcap'].where(peers['mcap'] > 0)) - np.log10(mcap)).abs()


def peer_group(pool: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """The ticker's comparison group: itself (first row) plus size-comparable industry mates.

    A peer must be within MAX_LOG_GAP orders of magnitude in market cap; fewer than MIN_PEERS
    such industry mates widens to the sector. The group is capped to the MAX_PEERS closest by
    market cap. A ticker without its own market cap falls back to plain industry/sector mates.
    """
    me = pool[pool['Ticker'] == ticker]
    if me.empty:
        return me
    industry, sector, mcap = me.iloc[0]['industry'], me.iloc[0]['sector'], me.iloc[0]['mcap']
    if industry != 'unknown':
        candidates = pool[(pool['industry'] == industry) & (pool['Ticker'] != ticker)]
    else:
        candidates = pool.iloc[0:0]
    if pd.isna(mcap) or mcap <= 0:
        if len(candidates) < MIN_PEERS and sector != 'unknown':
            candidates = pool[(pool['sector'] == sector) & (pool['Ticker'] != ticker)]
        return pd.concat([me, candidates.head(MAX_PEERS)], ignore_index=True)
    gap = _mcap_gap(candidates, mcap)
    if (gap <= MAX_LOG_GAP).sum() < MIN_PEERS and sector != 'unknown':
        candidates = pool[(pool['sector'] == sector) & (pool['Ticker'] != ticker)]
        gap = _mcap_gap(candidates, mcap)
    within = gap <= MAX_LOG_GAP
    peers = candidates[within].loc[gap[within].sort_values().index].head(MAX_PEERS)
    return pd.concat([me, peers], ignore_index=True)


def universe_peer_group(pool: pd.DataFrame, ticker: str, members: list[str],
                        cap: int | None = MAX_UNIVERSE_PEERS) -> pd.DataFrame:
    """The ticker's comparison group from an explicit member list: itself (first row) plus the members
    found in the pool, capped to the `cap` closest by market cap (None keeps them all)."""
    me = pool[pool['Ticker'] == ticker]
    if me.empty:
        return me
    mates = pool[pool['Ticker'].isin(set(members) - {ticker})]
    if cap is not None and len(mates) > cap:
        mcap = me.iloc[0]['mcap']
        if pd.notna(mcap) and mcap > 0:
            mates = mates.loc[_mcap_gap(mates, mcap).sort_values().index]
        mates = mates.head(cap)
    return pd.concat([me, mates], ignore_index=True)


def price_history(ticker: str) -> pd.DataFrame:
    """Daily adjusted closes joined to the benchmark on shared dates: columns Date, AC, market."""
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        prices = con.execute("SELECT Date, AC FROM prices WHERE Ticker = ? AND AC IS NOT NULL ORDER BY Date", [ticker]).df()
        market = con.execute("SELECT Date, C AS market FROM market_data WHERE Ticker = ? ORDER BY Date", [MARKET_TICKER]).df()
    finally:
        con.close()
    return prices.merge(market, on='Date', how='inner')


def rebased(history: pd.DataFrame, sessions: int | None) -> pd.DataFrame:
    """The last `sessions` rows (all when None) with AC and market rebased to 100 at the window start."""
    window = history.tail(sessions) if sessions else history
    if window.empty:
        return window
    out = window.copy()
    out['AC'] = out['AC'] / out['AC'].iloc[0] * 100
    out['market'] = out['market'] / out['market'].iloc[0] * 100
    return out


def drawdown(series: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak: 0 at new highs, negative below."""
    return series / series.cummax() - 1


def trailing_returns(history: pd.DataFrame) -> pd.DataFrame:
    """Ticker vs market fractional return over each TRAILING_WINDOWS window; NaN when history is shorter."""
    rows = []
    for label, sessions in TRAILING_WINDOWS.items():
        if len(history) > sessions:
            start, end = history.iloc[-sessions - 1], history.iloc[-1]
            rows.append({'window': label, 'ticker': end['AC'] / start['AC'] - 1,
                         'market': end['market'] / start['market'] - 1})
        else:
            rows.append({'window': label, 'ticker': np.nan, 'market': np.nan})
    return pd.DataFrame(rows)


def fundamentals_history(ticker: str) -> pd.DataFrame:
    """Restated quarterly income, cashflow and balance items merged on Report Date, oldest first."""
    frames = []
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        for table, items in (('income', _INCOME_ITEMS), ('cashflow', _CASHFLOW_ITEMS), ('balance', _BALANCE_ITEMS)):
            quoted = ', '.join(f'"{item}"' for item in items)
            frames.append(con.execute(
                f"SELECT \"Report Date\", {quoted} FROM {table} "
                f"WHERE Ticker = ? AND Period = 'Q' AND IsRestated ORDER BY \"Report Date\"", [ticker],
            ).df())
    finally:
        con.close()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on='Report Date', how='outer')
    return merged.sort_values('Report Date', ignore_index=True)


def ttm_history(quarters: pd.DataFrame) -> pd.DataFrame:
    """Rolling 4-quarter view per Report Date: flow sums, margins, per-share values, balance levels.

    Flows are 4-quarter sums (NaN unless all 4 quarters are present); margins are ratios of those
    sums; per-share values divide by that quarter's diluted shares; balance items stay
    point-in-time. The first 3 rows (no full TTM window) are dropped.
    """
    df = quarters.sort_values('Report Date').reset_index(drop=True)
    flows = {'Revenue': 'revenue', 'Gross Profit': 'gross_profit', 'Operating Income (Loss)': 'op_income',
             'Net Income (Common)': 'ni_common', 'Net Cash from Operating Activities': 'cfo',
             'Change in Fixed Assets & Intangibles': 'capex'}
    out = pd.DataFrame({'Report Date': df['Report Date']})
    for source, alias in flows.items():
        out[alias] = df[source].rolling(4).sum()
    out['fcf'] = out['cfo'] + out['capex']   # capex is stored negative
    out['gross_margin'] = np.where(out['revenue'] > 0, out['gross_profit'] / out['revenue'], np.nan)
    out['op_margin'] = np.where(out['revenue'] > 0, out['op_income'] / out['revenue'], np.nan)
    out['net_margin'] = np.where(out['revenue'] > 0, out['ni_common'] / out['revenue'], np.nan)
    shares = df['Shares (Diluted)']
    out['eps'] = np.where(shares > 0, out['ni_common'] / shares, np.nan)
    out['fcf_ps'] = np.where(shares > 0, out['fcf'] / shares, np.nan)
    out['shares'] = shares
    out['equity'] = df['Total Equity']
    out['cash'] = df['Cash, Cash Equivalents & Short Term Investments']
    st, lt = df['Short Term Debt'], df['Long Term Debt']
    out['debt'] = np.where(st.notna() | lt.notna(), st.fillna(0) + lt.fillna(0), np.nan)
    return out.iloc[3:].reset_index(drop=True)


def peer_percentiles(peers: pd.DataFrame, ticker: str) -> tuple[pd.Series, pd.Series]:
    """(family scores, factor ranks) for `ticker` inside `peers`: 0-100 percentiles, 100 best.

    Factor ranks come from compute_ranks over the peer group; a family score is the mean of its
    members' ranks with NaN members skipped.
    """
    ranked = compute_ranks(peers, FACTORS)
    row = ranked[ranked['Ticker'] == ticker].iloc[0]
    factor_ranks = pd.Series({factor: row[f'rank_{factor}'] for factor in FACTORS})
    family_scores = pd.Series({family: factor_ranks[members].mean() for family, members in FAMILIES.items()})
    return family_scores, factor_ranks
