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

import argparse
from dataclasses import dataclass
from typing import Self

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import factors_pit
from investalyze.backtest import config as backtest_config
from investalyze.backtest import engine, evaluation, performance, report, writer
from investalyze.backtest import market as market_data
from investalyze.backtest.book import Book
from investalyze.research import factors

SIGNAL_KIND = engine.CROSS_SECTIONAL
DEFAULT_QUANTILES = 5
REBALANCE_CHOICES = (factors_pit.Rebalance.MONTHLY, factors_pit.Rebalance.QUARTERLY)


@dataclass(frozen=True)
class FactorPortfolio:
    """A factor portfolio: the top quantile of one factor, or of the average rank of several."""

    names: tuple[str, ...]
    n_quantiles: int = DEFAULT_QUANTILES
    rebalance: factors_pit.Rebalance = factors_pit.Rebalance.MONTHLY

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError('Tick at least one factor in the factor library table.')
        if self.n_quantiles < 2:
            raise ValueError('A factor portfolio needs at least 2 quantiles.')
        object.__setattr__(self, 'names', tuple(self.names))
        object.__setattr__(self, 'rebalance', factors_pit.Rebalance(self.rebalance))

    @property
    def label(self) -> str:
        """How a run is named in the stored runs table: its factors."""
        return ' + '.join(self.names)

    def arguments(self) -> list[str]:
        """The `python -m investalyze.research backtest` arguments that describe this portfolio."""
        return [','.join(self.names), '--rebalance', str(self.rebalance), '--quantiles', str(self.n_quantiles)]

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> Self:
        """The portfolio a parsed `backtest` command describes."""
        names = tuple(name.strip() for name in (args.name or '').split(',') if name.strip())
        return cls(names=names, n_quantiles=args.quantiles, rebalance=args.rebalance)


@dataclass(frozen=True)
class Simulation:
    """What a simulated run produced: its curves, fills, headline statistics and the names it traded."""

    equity: pd.Series
    gross: pd.Series
    net: pd.Series
    fills: pd.DataFrame
    benchmark_equity: pd.Series | None
    stats: dict
    tickers: list[str]
    names_median: float


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


def trade_weights(con: duckdb.DuckDBPyConnection, market: market_data.MarketData, weights: np.ndarray,
                  cfg: backtest_config.Config, *, names_median: float) -> Simulation:
    """Trade target weights through the engine: weights on a session's row are sent, all-NaN rows hold."""
    scheduled = {session for session in range(len(market.dates)) if not np.isnan(weights[session]).all()}

    def step(session: int, book: Book) -> engine.Decision:
        """Send the target weights on the sessions where holdings change, and hold otherwise."""
        if session not in scheduled:
            return engine.HOLD
        return engine.Decision(targets=weights[session], orders=None)

    book = engine.run_backtest(market, step, starting_cash=cfg.starting_cash, costs=cfg.costs, limits=cfg.limits)
    equity = pd.Series(book.equity, index=market.dates)
    fills = book.blotter()
    bench = benchmark_curve(con, cfg.benchmark, market.dates, cfg.starting_cash)
    return Simulation(equity=equity, gross=pd.Series(book.gross, index=market.dates),
                      net=pd.Series(book.net, index=market.dates), fills=fills, benchmark_equity=bench,
                      stats=performance.summary(equity, fills, bench.pct_change() if bench is not None else None),
                      tickers=list(market.tickers), names_median=names_median)


def simulate(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, portfolio: FactorPortfolio, library: dict,
             cfg: backtest_config.Config | None = None) -> Simulation:
    """Run the top-quantile portfolio through the engine."""
    cfg = cfg or backtest_config.read()
    scores = composite_scores(frame, list(portfolio.names), library)
    tickers = sorted(scores.columns[scores.notna().any()])
    if not tickers:
        raise ValueError('no factor values over this range, so there is nothing to trade')
    start, end = scores.index.min().date().isoformat(), scores.index.max().date().isoformat()
    market = market_data.build_market(con, tickers, start=start, end=end, exclude_checks=cfg.exclude_checks)
    weights = target_weights(scores[list(market.tickers)], market.dates, market.tickers,
                             n_quantiles=portfolio.n_quantiles, max_weight=cfg.limits.max_weight)
    rebalances = ~np.isnan(weights).all(axis=1)
    held = (weights[rebalances] > 0).sum(axis=1)
    return trade_weights(con, market, weights, cfg, names_median=float(np.median(held)) if held.size else 0.0)


def run_params(portfolio: FactorPortfolio, library: dict, *, universe_size: int, start: str, end: str,
               holdout: str | None) -> dict:
    """The settings of a factor portfolio run, including each factor's expression and fingerprint."""
    names = list(portfolio.names)
    return {'factors': names, 'n_quantiles': portfolio.n_quantiles, 'rebalance': portfolio.rebalance,
            'universe_size': universe_size, 'start': start, 'end': end, 'holdout': holdout,
            'expressions': {name: library[name].expression for name in names},
            'fingerprints': {name: factors.fingerprint(name, library) for name in names}}


def run_id(params: dict, cfg: backtest_config.Config, *, signal: str = 'factor_portfolio') -> str:
    """Stable run id over the factor fingerprints and every setting that changes the result."""
    return writer.run_id({'signal': signal, 'params': {k: v for k, v in params.items() if k != 'expressions'},
                          'costs': vars(cfg.costs), 'limits': vars(cfg.limits), 'starting_cash': cfg.starting_cash,
                          'benchmark': cfg.benchmark, 'exclude_checks': cfg.exclude_checks})


def stored_report(simulation: Simulation, params: dict, cfg: backtest_config.Config, *, run_id: str, label: str,
                  kind: str) -> report.BacktestReport:
    """The stored-run report for a simulated run."""
    equity = simulation.equity
    return report.BacktestReport(
        run_id=run_id, signal=label, kind=kind, config=cfg, params=params, start=equity.index[0], end=equity.index[-1],
        tickers=simulation.tickers, benchmark=cfg.benchmark, caveats=[], stats=simulation.stats, equity=equity,
        gross=simulation.gross, net=simulation.net, cash=equity - simulation.net, fills=simulation.fills,
        benchmark_equity=simulation.benchmark_equity,
        exposure=performance.exposures(simulation.gross, simulation.net, equity),
    )


def build_report(simulation: Simulation, portfolio: FactorPortfolio, params: dict,
                 cfg: backtest_config.Config) -> report.BacktestReport:
    """The stored-run report for a simulated factor portfolio."""
    return stored_report(simulation, params, cfg, run_id=run_id(params, cfg), label=portfolio.label, kind=SIGNAL_KIND)
