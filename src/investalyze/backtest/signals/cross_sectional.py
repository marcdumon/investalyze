"""Cross-sectional factor signals: rank the universe on a rebalance date, hold the best bucket.

The factory does all the work once. It reads the stored point-in-time factor panel, turns each
factor into an orientation-adjusted percentile rank so a high rank always means a good name, averages
the selected factors into one composite score, and forms target weights for every rebalance session.
The step function then hands the engine a prepared row, so a monthly strategy does no work on the
sessions between rebalances.
"""

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import factors_pit
from investalyze.analysis.factors import FACTORS, HIGHER_IS_BETTER
from investalyze.backtest import portfolio
from investalyze.backtest.book import Book
from investalyze.backtest.market import MarketData
from investalyze.backtest.engine import HOLD, Decision, SignalStep
from investalyze.ingest import storage

DEFAULT_FACTORS = ['earnings_yield', 'fcf_yield', 'roe', 'ret_12_1']
DEFAULT_QUANTILES = 5
DEFAULT_REBALANCE = 'M'
DEFAULT_VINTAGE = 'as_reported'
MIN_COVERED_SESSIONS = 0.5   # below this share of rebalances carrying a score, the run is refused


def rebalance_sessions(dates: pd.DatetimeIndex, freq: str) -> list[int]:
    """Index of the last session of each period in the calendar, the sessions a run rebalances on."""
    if freq == 'D':
        return list(range(len(dates)))
    positions = pd.Series(range(len(dates)), index=dates)
    return sorted(positions.groupby(dates.to_period(freq)).max().tolist())


def read_panel(con: duckdb.DuckDBPyConnection, tickers: np.ndarray, dates: pd.DatetimeIndex,
               *, vintage: str = DEFAULT_VINTAGE) -> pd.DataFrame:
    """The stored factor panel for one vintage, from the last row on or before the run's first session.

    The lower bound reaches back past the run's start rather than sitting on it, so the first
    rebalance reads the most recent row that existed by then instead of finding nothing because the
    panel's grid does not land on that session.
    """
    if not storage.table_exists(con, 'factor_panel'):
        raise ValueError('factor_panel does not exist; build it with `python -m investalyze.backtest factors`')
    start, end = dates[0].date().isoformat(), dates[-1].date().isoformat()
    earlier = con.execute('SELECT max(Date) FROM factor_panel WHERE Vintage = ? AND Date <= ?',
                          [vintage, start]).fetchone()[0]
    lower = earlier.isoformat() if earlier is not None else start
    panel = con.execute("""
        SELECT * FROM factor_panel
        WHERE Vintage = ? AND Ticker IN (SELECT unnest(?::VARCHAR[])) AND Date BETWEEN ? AND ?
    """, [vintage, sorted(tickers.tolist()), lower, end]).df()
    if panel.empty:
        raise ValueError(f'factor_panel holds no {vintage} rows for this universe and date range')
    panel['Date'] = pd.to_datetime(panel['Date'])
    # rank-based factors are ranked here, over this run's names, since that is the
    # cross-section the signal actually chooses from
    return factors_pit.apply_derived(panel)


def composite_scores(panel: pd.DataFrame, factors: list[str], tickers: np.ndarray,
                     dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Average orientation-adjusted percentile rank of the selected factors, per session and ticker.

    Each factor is ranked within its own cross-section, then flipped where a lower raw value is the
    better one, so every column reads the same way before they are averaged.

    A session between two panel dates reads the earlier one, the most recent values a reader could
    have had. That is what lets a monthly panel drive a weekly run: the score is stale between panel
    dates, which is the truth of the matter, rather than absent.
    """
    unknown = [factor for factor in factors if factor not in FACTORS]
    if unknown:
        raise ValueError(f'not factors of the library: {", ".join(unknown)}')

    ranked, empty = [], []
    for factor in factors:
        wide = panel.pivot_table(index='Date', columns='Ticker', values=factor, aggfunc='first')
        wide = wide.reindex(columns=tickers).reindex(index=dates, method='ffill')
        if not wide.notna().to_numpy().any():
            empty.append(factor)
            continue
        rank = wide.rank(axis=1, pct=True)
        ranked.append(rank if HIGHER_IS_BETTER[factor] else 1 - rank)
    if empty:
        raise ValueError(f'the factor panel holds no values for {", ".join(empty)} over this range '
                         'and universe; rebuild it so the factor has values to rank')
    return pd.concat(ranked).groupby(level=0).mean().reindex(index=dates, columns=tickers)


def covered_sessions(scores: pd.DataFrame, sessions: list[int]) -> int:
    """How many of a run's rebalance sessions have at least one name to rank."""
    values = scores.to_numpy(dtype=np.float64)
    return sum(1 for session in sessions if np.isfinite(values[session]).any())


def _target_weights(scores: pd.DataFrame, sessions: list[int], *, n_quantiles: int,
                    max_weight: float) -> np.ndarray:
    """A (session, asset) weight matrix with rows filled in only on the rebalance sessions."""
    weights = np.full(scores.shape, np.nan, dtype=np.float64)
    values = scores.to_numpy(dtype=np.float64)
    for session in sessions:
        row = portfolio.quantile_weights(values[session], n_quantiles=n_quantiles)
        if np.count_nonzero(row):
            row = portfolio.cap_weights(row, max_weight=max_weight, max_gross=1.0, max_net=1.0)
        weights[session] = row
    return weights


def factor_composite(con: duckdb.DuckDBPyConnection, market: MarketData, params: dict) -> SignalStep:
    """Hold the top bucket of an equal-weighted composite of the selected factors, rebalanced on a schedule."""
    factors = list(params.get('factors', DEFAULT_FACTORS))
    n_quantiles = int(params.get('n_quantiles', DEFAULT_QUANTILES))
    max_weight = float(params.get('max_weight', 1.0))
    vintage = params.get('vintage', DEFAULT_VINTAGE)
    sessions = rebalance_sessions(market.dates, params.get('rebalance', DEFAULT_REBALANCE))

    panel = read_panel(con, market.tickers, market.dates, vintage=vintage)
    scores = composite_scores(panel, factors, market.tickers, market.dates)

    covered = covered_sessions(scores, sessions)
    if covered < MIN_COVERED_SESSIONS * len(sessions):
        raise ValueError(f'the factor panel reaches only {covered} of this run\'s {len(sessions)} '
                         f'rebalance dates; build it over {market.dates[0].date()} to '
                         f'{market.dates[-1].date()} before running this signal')

    weights = _target_weights(scores, sessions, n_quantiles=n_quantiles, max_weight=max_weight)
    rebalancing = set(sessions)

    def step(session: int, book: Book) -> Decision:
        """Hand over the prepared weights on a rebalance session and hold otherwise."""
        if session not in rebalancing:
            return HOLD
        return Decision(targets=weights[session], orders=None)

    return step


def single_factor(con: duckdb.DuckDBPyConnection, market: MarketData, params: dict) -> SignalStep:
    """Hold the top bucket of one factor, the composite signal narrowed to a single column."""
    factor = params.get('factor', 'ret_12_1')
    return factor_composite(con, market, params | {'factors': [factor]})
