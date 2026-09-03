"""Performance statistics for an equity curve or a return series.

Pure functions over pandas objects, so the engine, the notebooks and the app pages share one
implementation and nothing here needs a database. Series are indexed by date and ordered ascending.
Ratios are annualized from TRADING_DAYS sessions a year unless a function says otherwise.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DOWNSIDE_FLOOR = 1e-12  # guards the ratios whose denominator can collapse to zero


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak: 0 at new highs, negative below."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """The deepest fractional fall from a peak over the whole curve."""
    if equity.empty:
        return np.nan
    return float(drawdown_series(equity).min())


def drawdown_table(equity: pd.Series, *, top_n: int = 5) -> pd.DataFrame:
    """The deepest drawdowns, with the peak they started from, the trough, and the recovery date."""
    if equity.empty:
        return pd.DataFrame(columns=['peak', 'trough', 'recovered', 'depth', 'sessions'])
    drawdown = drawdown_series(equity)
    underwater = drawdown < 0

    episodes = []
    start = None
    for position, (date, below) in enumerate(underwater.items()):
        if below and start is None:
            start = position
        elif not below and start is not None:
            episodes.append((start, position))
            start = None
    if start is not None:
        episodes.append((start, len(underwater)))

    rows = []
    for first, stop in episodes:
        window = drawdown.iloc[first:stop]
        trough = window.idxmin()
        recovered = equity.index[stop] if stop < len(equity) else pd.NaT
        rows.append({'peak': equity.index[first - 1] if first else equity.index[0], 'trough': trough,
                     'recovered': recovered, 'depth': float(window.min()), 'sessions': stop - first})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=['peak', 'trough', 'recovered', 'depth', 'sessions'])
    return table.sort_values('depth').head(top_n).reset_index(drop=True)


def _years(equity: pd.Series) -> float:
    """Elapsed years between the first and last session of a curve."""
    if len(equity) < 2:
        return np.nan
    span = (equity.index[-1] - equity.index[0]).days
    return span / 365.25 if span > 0 else np.nan


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate implied by the first and last points of a curve."""
    years = _years(equity)
    if not np.isfinite(years) or years <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    """Standard deviation of per-session returns, scaled to a year."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, *, risk_free: float = 0.0) -> float:
    """Annualized excess return over annualized volatility."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    excess = clean - risk_free / TRADING_DAYS
    spread = excess.std(ddof=1)
    if spread < DOWNSIDE_FLOOR:
        return np.nan
    return float(excess.mean() / spread * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, *, risk_free: float = 0.0) -> float:
    """Annualized excess return over the volatility of the losing sessions alone."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    excess = clean - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    spread = np.sqrt(np.square(downside).mean())
    if spread < DOWNSIDE_FLOOR:
        return np.nan
    return float(excess.mean() / spread * np.sqrt(TRADING_DAYS))


def calmar(equity: pd.Series) -> float:
    """Compound growth rate over the depth of the worst drawdown."""
    depth = max_drawdown(equity)
    growth = cagr(equity)
    if not np.isfinite(depth) or abs(depth) < DOWNSIDE_FLOOR:
        return np.nan
    return float(growth / abs(depth))


def hit_rate(returns: pd.Series) -> float:
    """Share of sessions that made money."""
    clean = returns.dropna()
    return float((clean > 0).mean()) if len(clean) else np.nan


def turnover(fills: pd.DataFrame, equity: pd.Series) -> pd.Series:
    """Traded notional as a share of the book, per session."""
    if fills.empty or equity.empty:
        return pd.Series(dtype=float, name='turnover')
    traded = fills.groupby('Date')['Notional'].sum()
    aligned = traded.reindex(equity.index).fillna(0.0)
    return (aligned / equity).rename('turnover')


def annual_turnover(fills: pd.DataFrame, equity: pd.Series) -> float:
    """Traded notional over a year, as a multiple of the book."""
    per_session = turnover(fills, equity)
    if per_session.empty:
        return np.nan
    return float(per_session.mean() * TRADING_DAYS)


def exposures(gross: pd.Series, net: pd.Series, equity: pd.Series) -> dict[str, float]:
    """Average and peak gross and net exposure, as a share of the book."""
    if equity.empty:
        return {'mean_gross': np.nan, 'max_gross': np.nan, 'mean_net': np.nan, 'max_net': np.nan}
    invested_gross = (gross / equity).replace([np.inf, -np.inf], np.nan).dropna()
    invested_net = (net / equity).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        'mean_gross': float(invested_gross.mean()) if len(invested_gross) else np.nan,
        'max_gross': float(invested_gross.max()) if len(invested_gross) else np.nan,
        'mean_net': float(invested_net.mean()) if len(invested_net) else np.nan,
        'max_net': float(invested_net.max()) if len(invested_net) else np.nan,
    }


def _paired(returns: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """The two series on the sessions they share, with either side's gaps dropped."""
    return pd.DataFrame({'strategy': returns, 'benchmark': benchmark}).dropna()


def beta_and_alpha(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Sensitivity to the benchmark and the annualized return left over after it."""
    paired = _paired(returns, benchmark)
    if len(paired) < 2:
        return np.nan, np.nan
    variance = paired['benchmark'].var(ddof=1)
    if variance < DOWNSIDE_FLOOR:
        return np.nan, np.nan
    beta = paired['strategy'].cov(paired['benchmark']) / variance
    alpha = (paired['strategy'].mean() - beta * paired['benchmark'].mean()) * TRADING_DAYS
    return float(beta), float(alpha)


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> float:
    """Annualized mean of the return difference over its own volatility."""
    paired = _paired(returns, benchmark)
    if len(paired) < 2:
        return np.nan
    active = paired['strategy'] - paired['benchmark']
    spread = active.std(ddof=1)
    if spread < DOWNSIDE_FLOOR:
        return np.nan
    return float(active.mean() / spread * np.sqrt(TRADING_DAYS))


def capture_ratios(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Share of the benchmark's up-session and down-session moves the strategy captured."""
    paired = _paired(returns, benchmark)
    if paired.empty:
        return np.nan, np.nan
    captured = []
    for rising in (True, False):
        side = paired[paired['benchmark'] > 0] if rising else paired[paired['benchmark'] < 0]
        reference = side['benchmark'].mean()
        captured.append(float(side['strategy'].mean() / reference)
                        if len(side) and abs(reference) > DOWNSIDE_FLOOR else np.nan)
    return captured[0], captured[1]


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Compounded returns by calendar year and month, years down the rows."""
    clean = returns.dropna()
    if clean.empty:
        return pd.DataFrame()
    compounded = (1 + clean).groupby([clean.index.year, clean.index.month]).prod() - 1
    table = compounded.unstack()
    table.index.name = 'year'
    table.columns.name = 'month'
    return table


def summary(equity: pd.Series, fills: pd.DataFrame, benchmark: pd.Series | None = None) -> dict[str, float]:
    """The headline statistics of one run, as a flat mapping the run table stores."""
    returns = equity.pct_change()
    stats = {
        'cagr': cagr(equity),
        'vol': annualized_volatility(returns),
        'sharpe': sharpe(returns),
        'sortino': sortino(returns),
        'max_drawdown': max_drawdown(equity),
        'calmar': calmar(equity),
        'hit_rate': hit_rate(returns),
        'turnover': annual_turnover(fills, equity),
        'n_trades': int(len(fills)),
        'final_equity': float(equity.iloc[-1]) if len(equity) else np.nan,
        'total_costs': float(fills['Cost'].sum()) if not fills.empty else 0.0,
    }
    if benchmark is not None:
        beta, alpha = beta_and_alpha(returns, benchmark)
        up_capture, down_capture = capture_ratios(returns, benchmark)
        stats |= {'beta': beta, 'alpha': alpha, 'information_ratio': information_ratio(returns, benchmark),
                  'up_capture': up_capture, 'down_capture': down_capture}
    return stats
