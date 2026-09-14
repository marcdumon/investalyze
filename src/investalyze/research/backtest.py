"""Factor portfolio backtest: hold the top quantile of one factor or a composite of several.

With several factors the score is the average of their percentile ranks, each oriented by the factor's
direction, over names that have a value for every factor. The portfolio is long the top quantile, equal
weight, capped at `max_weight` per name with the rest in cash. Quantiles come from
`evaluation.quantile_buckets`, the function IC analysis uses, so a single-factor backtest holds the
names the quantile returns table measured, on the same point-in-time universe.

The engine decides at the close of a rebalance session and fills at the next open, with the slippage,
commission, market impact and participation limits from `backtest.toml`.

A run is stored like any other backtest, in `backtest_runs`, `backtest_equity` and `backtest_trades`.
Its run id hashes the factor fingerprints with every setting that changes the result, so redefining a
factor produces a new run rather than overwriting the old one.
"""

import duckdb
import numpy as np
import pandas as pd

from investalyze.backtest import config as backtest_config
from investalyze.backtest import engine, evaluation, performance, report, writer
from investalyze.backtest import market as market_data
from investalyze.backtest.book import Book
from investalyze.research import factors

SIGNAL_KIND = engine.CROSS_SECTIONAL
DEFAULT_QUANTILES = 5
REBALANCE = {'M': 'monthly', 'Q': 'quarterly'}


def composite_scores(frame: pd.DataFrame, names: list[str], library: dict) -> pd.DataFrame:
    """(dates, tickers) score: one factor's values, or the mean oriented percentile rank of several."""
    if len(names) == 1:
        scores = evaluation.factor_scores(frame, names[0])
        return scores if library[names[0]].higher_is_better else -scores
    ranks = []
    for name in names:
        scores = evaluation.factor_scores(frame, name)
        oriented = scores if library[name].higher_is_better else -scores
        ranks.append(oriented.rank(axis=1, pct=True))
    stacked = pd.concat(ranks, keys=range(len(ranks)))
    # a name missing any factor is left out rather than scored on the factors it happens to have
    return stacked.groupby(level=1).mean().where(stacked.groupby(level=1).count() == len(names))


def target_weights(scores: pd.DataFrame, sessions: pd.DatetimeIndex, tickers: np.ndarray, *,
                   n_quantiles: int, max_weight: float, higher_is_better: bool = True) -> np.ndarray:
    """(sessions, tickers) target weights: the top quantile on each scored session, NaN elsewhere."""
    weights = np.full((len(sessions), len(tickers)), np.nan)
    oriented = scores if higher_is_better else -scores
    quantiles = evaluation.quantile_buckets(oriented, n_quantiles=n_quantiles)
    positions = sessions.get_indexer(pd.DatetimeIndex(quantiles.index))
    for row, position in zip(quantiles.itertuples(index=False), positions):
        if position < 0:
            continue
        held = pd.Series(row, index=quantiles.columns) == n_quantiles - 1
        if not held.any():
            continue
        target = pd.Series(0.0, index=tickers)
        chosen = held[held].index.intersection(target.index)
        target.loc[chosen] = min(1.0 / len(chosen), max_weight) if len(chosen) else 0.0
        weights[position] = target.to_numpy()
    return weights


def benchmark_curve(con: duckdb.DuckDBPyConnection, ticker: str, sessions: pd.DatetimeIndex, starting_cash: float) -> pd.Series | None:
    """The benchmark close rebased to the starting cash, on the run's sessions."""
    rows = con.execute('SELECT Date, C FROM market_data WHERE Ticker = ? ORDER BY Date', [ticker]).df()
    if rows.empty:
        return None
    series = pd.Series(rows['C'].to_numpy(), index=pd.DatetimeIndex(rows['Date'])).reindex(sessions).ffill()
    if series.dropna().empty:
        return None
    return series / series.dropna().iloc[0] * starting_cash


def period_stats(equity: pd.Series, benchmark: pd.Series | None, *, holdout_start: pd.Timestamp | None) -> pd.DataFrame:
    """CAGR, volatility, Sharpe and max drawdown for the full sample, in-sample and out-of-sample."""
    periods = [('Full sample', equity)]
    if holdout_start is not None:
        periods.append(('In-sample', equity[equity.index < holdout_start]))
        periods.append(('Out-of-sample', equity[equity.index >= holdout_start]))
    rows = []
    for label, curve in periods:
        curve = curve.dropna()
        if len(curve) < 2:
            continue
        returns = curve.pct_change()
        row = {'period': label, 'start': curve.index[0].date().isoformat(), 'end': curve.index[-1].date().isoformat(),
               'cagr': performance.cagr(curve), 'vol': performance.annualized_volatility(returns),
               'sharpe': performance.sharpe(returns), 'max_drawdown': performance.max_drawdown(curve)}
        if benchmark is not None:
            bench = benchmark.reindex(curve.index).dropna()
            row['benchmark_cagr'] = performance.cagr(bench) if len(bench) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def simulate(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, names: list[str], library: dict, *,
             n_quantiles: int = DEFAULT_QUANTILES, cfg: backtest_config.Config | None = None) -> dict:
    """Run the top-quantile portfolio through the engine and return the book's curves, fills and stats."""
    cfg = cfg or backtest_config.read()
    scores = composite_scores(frame, names, library)
    tickers = sorted(scores.columns[scores.notna().any()])
    if not tickers:
        raise ValueError('no factor values over this range, so there is nothing to trade')
    start, end = scores.index.min().date().isoformat(), scores.index.max().date().isoformat()
    market = market_data.build_market(con, tickers, start=start, end=end, exclude_checks=cfg.exclude_checks)
    weights = target_weights(scores[list(market.tickers)], market.dates, market.tickers, n_quantiles=n_quantiles,
                             max_weight=cfg.limits.max_weight)
    scheduled = {session for session in range(len(market.dates)) if not np.isnan(weights[session]).all()}

    def step(session: int, book: Book) -> engine.Decision:
        """Send the target weights on the sessions where holdings change, and hold otherwise."""
        if session not in scheduled:
            return engine.HOLD
        return engine.Decision(targets=weights[session], orders=None)

    book = engine.run_backtest(market, step, starting_cash=cfg.starting_cash, costs=cfg.costs, limits=cfg.limits)
    equity = pd.Series(book.equity, index=market.dates)
    gross = pd.Series(book.gross, index=market.dates)
    net = pd.Series(book.net, index=market.dates)
    fills = book.blotter()
    bench = benchmark_curve(con, cfg.benchmark, market.dates, cfg.starting_cash)
    stats = performance.summary(equity, fills, bench.pct_change() if bench is not None else None)
    held = (weights > 0).sum(axis=1)
    return {'market': market, 'equity': equity, 'gross': gross, 'net': net, 'fills': fills,
            'benchmark_equity': bench, 'stats': stats, 'tickers': list(market.tickers),
            'names_median': float(np.median(held[sorted(scheduled)])) if scheduled else 0.0}


def run_params(names: list[str], library: dict, *, n_quantiles: int, rebalance: str, universe_size: int,
               start: str, end: str, holdout: str | None) -> dict:
    """The settings of a factor portfolio run, including each factor's expression and fingerprint."""
    return {'factors': list(names), 'n_quantiles': n_quantiles, 'rebalance': rebalance,
            'universe_size': universe_size, 'start': start, 'end': end, 'holdout': holdout,
            'expressions': {name: library[name].expression for name in names},
            'fingerprints': {name: factors.fingerprint(name, library) for name in names}}


def run_id(params: dict, cfg: backtest_config.Config) -> str:
    """Stable run id over the factor fingerprints and every setting that changes the result."""
    return writer.run_id({'signal': 'factor_portfolio', 'params': {k: v for k, v in params.items() if k != 'expressions'},
                          'costs': vars(cfg.costs), 'limits': vars(cfg.limits), 'starting_cash': cfg.starting_cash,
                          'benchmark': cfg.benchmark, 'exclude_checks': cfg.exclude_checks})


def signal_label(names: list[str]) -> str:
    """How a run is named in the stored runs table: its factors."""
    return ' + '.join(names)


def build_report(result: dict, params: dict, cfg: backtest_config.Config) -> report.BacktestReport:
    """The stored-run report for a simulated factor portfolio."""
    equity = result['equity']
    return report.BacktestReport(
        run_id=run_id(params, cfg), signal=signal_label(params['factors']), kind=SIGNAL_KIND, config=cfg,
        params=params, start=equity.index[0], end=equity.index[-1], tickers=result['tickers'],
        benchmark=cfg.benchmark, caveats=[], stats=result['stats'], equity=equity,
        gross=result['gross'], net=result['net'], cash=equity - result['net'], fills=result['fills'],
        benchmark_equity=result['benchmark_equity'],
        exposure=performance.exposures(result['gross'], result['net'], equity),
    )
