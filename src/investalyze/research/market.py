"""Market data fields: price, volume, returns, risk, market capitalisation and dividend yield.

Computed from daily bars on the trading calendar, then sampled on each as-of date from the last
session on or before it. A value is known at the close of its session, so sampling never reads a
later price. Returns use the dividend-adjusted close (total return); price level, volume and market
capitalisation use the split-adjusted close.
"""

import numpy as np
import pandas as pd

from investalyze.analysis import factors_pit, panels

TRADING_DAYS = 252
MIN_RISK_OBS = 200
STALE_DAYS = factors_pit.CLOSE_LOOKBACK_DAYS

FIELDS = {
    'open': 'Split-adjusted open.',
    'high': 'Split-adjusted high.',
    'low': 'Split-adjusted low.',
    'close': 'Split-adjusted close.',
    'adj_close': 'Close adjusted for splits and dividends, the basis of every return field.',
    'volume': 'Shares traded on the as-of session.',
    'adv20': 'Average daily dollar volume over 20 trading days.',
    'market_cap': 'Split-adjusted close times basic shares outstanding (MRQ).',
    'ret_1m': 'Total return over 21 trading days.',
    'ret_3m': 'Total return over 63 trading days.',
    'ret_6m': 'Total return over 126 trading days.',
    'ret_12m': 'Total return over 252 trading days.',
    'ret_12_1': 'Total return from 252 to 21 trading days ago (12-1 momentum).',
    'vol_63': 'Annualised volatility of daily returns over 63 trading days.',
    'vol_252': 'Annualised volatility of daily returns over 252 trading days.',
    'beta_252': 'Beta of daily returns to the S&P 500 over 252 trading days.',
    'high_52w': 'Close relative to its 52-week high.',
    'div_yield': 'Dividends per share over the trailing 365 days, divided by close.',
}

# fields computed from the daily matrices; market_cap and div_yield join other tables instead
_DAILY = ('open', 'high', 'low', 'close', 'adj_close', 'volume', 'adv20', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m', 'ret_12_1',
          'vol_63', 'vol_252', 'beta_252', 'high_52w')


def compute(names: list[str], *, adjusted: pd.DataFrame, raw: pd.DataFrame, volume: pd.DataFrame,
            benchmark: pd.Series | None = None, open_: pd.DataFrame | None = None, high: pd.DataFrame | None = None,
            low: pd.DataFrame | None = None) -> dict:
    """Wide daily matrices for the requested fields, sessions down the rows and tickers across."""
    unknown = sorted(name for name in names if name not in _DAILY)
    if unknown:
        raise ValueError(f'not daily market fields: {", ".join(unknown)}')

    returns = adjusted.pct_change(fill_method=None)
    result = {}
    for name in names:
        if name == 'close':
            result[name] = raw
        elif name == 'adj_close':
            result[name] = adjusted
        elif name == 'open':
            result[name] = open_
        elif name == 'high':
            result[name] = high
        elif name == 'low':
            result[name] = low
        elif name == 'volume':
            result[name] = volume
        elif name == 'adv20':
            result[name] = (raw * volume).rolling(20, min_periods=15).mean()
        elif name == 'ret_1m':
            result[name] = adjusted / adjusted.shift(21) - 1.0
        elif name == 'ret_3m':
            result[name] = adjusted / adjusted.shift(63) - 1.0
        elif name == 'ret_6m':
            result[name] = adjusted / adjusted.shift(126) - 1.0
        elif name == 'ret_12m':
            result[name] = adjusted / adjusted.shift(252) - 1.0
        elif name == 'ret_12_1':
            result[name] = adjusted.shift(21) / adjusted.shift(252) - 1.0
        elif name == 'vol_63':
            result[name] = returns.rolling(63, min_periods=50).std() * np.sqrt(TRADING_DAYS)
        elif name == 'vol_252':
            result[name] = returns.rolling(252, min_periods=MIN_RISK_OBS).std() * np.sqrt(TRADING_DAYS)
        elif name == 'beta_252':
            market = benchmark.reindex(returns.index)
            covariance = returns.rolling(252, min_periods=MIN_RISK_OBS).cov(market)
            result[name] = covariance.div(market.rolling(252, min_periods=MIN_RISK_OBS).var(), axis=0)
        elif name == 'high_52w':
            result[name] = raw / raw.rolling(253, min_periods=MIN_RISK_OBS).max()
    return result


def sample(matrices: dict, as_of: pd.DatetimeIndex) -> pd.DataFrame:
    """Long rows of `Date`, `Ticker` and each field, read from the last session on or before each date.

    A session more than `STALE_DAYS` calendar days before the as-of date is not used, so a name that
    stopped trading does not keep its last value.
    """
    if not matrices:
        return pd.DataFrame(columns=['Date', 'Ticker'])
    sessions = next(iter(matrices.values())).index
    positions = sessions.searchsorted(as_of, side='right') - 1
    rows = []
    for when, position in zip(as_of, positions):
        if position < 0 or (when - sessions[position]).days > STALE_DAYS:
            continue
        rows.append((when, sessions[position]))
    if not rows:
        return pd.DataFrame(columns=['Date', 'Ticker'] + list(matrices))

    stacked = []
    for name, matrix in matrices.items():
        picked = matrix.loc[[session for _, session in rows]]
        picked.index = pd.DatetimeIndex([when for when, _ in rows], name='Date')
        stacked.append(picked.stack(future_stack=True).rename(name))
    frame = pd.concat(stacked, axis=1).reset_index()
    frame.columns = ['Date', 'Ticker'] + list(matrices)
    return frame


def _daily_bars(con, tickers: list[str], calendar: pd.DatetimeIndex, columns: tuple[str, ...]) -> dict:
    """The requested price columns (`O`, `H`, `L`, `C`, `AC`, `V`) as wide matrices on the calendar."""
    selected = ', '.join(columns)
    bars = con.execute(f"""
        SELECT Date, Ticker, {selected} FROM prices
        WHERE Ticker = ANY(?) AND Date BETWEEN ? AND ? AND C > 0 AND AC > 0
    """, [tickers, calendar.min().date(), calendar.max().date()]).df()
    bars['Date'] = pd.to_datetime(bars['Date'])
    wide = {}
    for column in columns:
        wide[column] = bars.pivot_table(index='Date', columns='Ticker', values=column, aggfunc='last').reindex(calendar)
    return wide


def panel(con, names: set[str], *, as_of: pd.DatetimeIndex, tickers: list[str] | None = None) -> pd.DataFrame:
    """Long panel of the requested market fields on the as-of dates."""
    wanted = sorted(name for name in names if name in FIELDS)
    if not wanted:
        return pd.DataFrame(columns=['Date', 'Ticker'])
    grid = pd.DatetimeIndex(as_of)
    if tickers is None:
        tickers = [row[0] for row in con.execute('SELECT DISTINCT Ticker FROM prices').fetchall()]

    start = (grid.min() - pd.Timedelta(days=factors_pit.HISTORY_DAYS)).date().isoformat()
    end = grid.max().date().isoformat()
    calendar = panels.trading_calendar(con, start=start, end=end)
    wanted_bars = ['AC', 'C', 'V'] + [column for name, column in (('open', 'O'), ('high', 'H'), ('low', 'L')) if name in wanted]
    bars = _daily_bars(con, tickers, calendar, tuple(wanted_bars))
    adjusted, raw, volume = bars['AC'], bars['C'], bars['V']

    daily = [name for name in wanted if name in _DAILY]
    if 'market_cap' in wanted and 'close' not in daily:
        daily.append('close')
    benchmark = None
    if 'beta_252' in daily:
        benchmark = panels.benchmark_returns(con, panels.MARKET_TICKER, start=start, end=end, kind='simple')
    frame = sample(compute(daily, adjusted=adjusted, raw=raw, volume=volume, benchmark=benchmark,
                           open_=bars.get('O'), high=bars.get('H'), low=bars.get('L')), grid)

    if 'market_cap' in wanted:
        shares = factors_pit.balance_panel(con, as_of=grid, tickers=tickers,
                                           extras={'Shares (Basic)': 'shares_basic', 'Shares (Diluted)': 'shares_diluted'})
        shares['shares'] = shares['shares_basic'].fillna(shares['shares_diluted'])
        shares['Date'] = pd.to_datetime(shares['Date'])
        frame = frame.merge(shares[['Date', 'Ticker', 'shares']], on=['Date', 'Ticker'], how='left')
        frame['market_cap'] = frame['close'] * frame['shares']
        frame = frame.drop(columns=['shares'])

    if 'div_yield' in wanted:
        dividends = con.execute("""
            SELECT g.as_of AS Date, d.Ticker, sum(d.Dividend) AS dividends
            FROM (SELECT unnest(?::DATE[]) AS as_of) g
            JOIN dividends d ON d.Date <= g.as_of AND d.Date > g.as_of - INTERVAL 365 DAY
            WHERE d.Ticker = ANY(?)
            GROUP BY g.as_of, d.Ticker
        """, [[when.date() for when in grid], tickers]).df()
        dividends['Date'] = pd.to_datetime(dividends['Date'])
        if 'close' not in frame.columns:
            closes = sample({'close': raw}, grid)
            frame = frame.merge(closes, on=['Date', 'Ticker'], how='outer')
        frame = frame.merge(dividends, on=['Date', 'Ticker'], how='left')
        frame['div_yield'] = frame['dividends'].fillna(0.0) / frame['close']
        frame = frame.drop(columns=['dividends'])

    keep = ['Date', 'Ticker'] + wanted
    return frame[keep].dropna(subset=wanted, how='all').reset_index(drop=True)
