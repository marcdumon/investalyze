"""Timing rule backtest: each name's own entry and exit rules decide whether it is held.

A timing rule is built from factors in the factor library. A name is entered on a session where its
entry factor is true, meaning above zero. It is exited on the first session any chosen exit applies:
the entry factor is no longer true, an exit factor is true, or its close has fallen `trailing_stop` below
the highest close since entry. At most `max_positions` names are held, equal weight `1 / max_positions`
each; when more names qualify than there are free positions, a priority factor picks the highest values
first, and trailing liquidity is used when none is given.

Decisions are made at each close on the daily grid and filled at the next open by the engine, with the
costs and limits from `backtest.toml`. Target weights are sent only on sessions where holdings change.
"""

import duckdb
import numpy as np
import pandas as pd

from investalyze.backtest import config as backtest_config
from investalyze.backtest import engine, performance, report, writer
from investalyze.backtest import market as market_data
from investalyze.backtest.book import Book
from investalyze.research import backtest as factor_backtest
from investalyze.research import factors

SIGNAL_KIND = engine.TIMING
DEFAULT_MAX_POSITIONS = 20


def holdings(entry: np.ndarray, close: np.ndarray, *, exit_signal: np.ndarray | None, exit_on_entry_false: bool,
             trailing_stop: float | None, max_positions: int, priority: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(sessions, names) held flags, and target weights on the sessions where holdings change (NaN elsewhere)."""
    sessions, names = entry.shape
    held_by_session = np.zeros((sessions, names), dtype=bool)
    weights = np.full((sessions, names), np.nan)
    held = np.zeros(names, dtype=bool)
    peak = np.full(names, np.nan)
    for session in range(sessions):
        previous = held.copy()
        priced = np.isfinite(close[session])
        exits = np.zeros(names, dtype=bool)
        if exit_on_entry_false:
            exits |= held & ~entry[session]
        if exit_signal is not None:
            exits |= held & exit_signal[session]
        if trailing_stop is not None:
            peak = np.where(held & priced, np.fmax(peak, close[session]), peak)
            exits |= held & priced & (close[session] <= peak * (1.0 - trailing_stop))
        held &= ~exits

        free = max_positions - int(held.sum())
        candidates = np.flatnonzero(entry[session] & priced & ~held & ~exits)
        if free > 0 and candidates.size:
            if priority is not None:
                scores = np.nan_to_num(priority[session, candidates], nan=-np.inf)
                candidates = candidates[np.argsort(-scores, kind='stable')]
            chosen = candidates[:free]
            held[chosen] = True
            peak[chosen] = close[session, chosen]

        held_by_session[session] = held
        if not np.array_equal(held, previous):
            weights[session] = np.where(held, 1.0 / max_positions, 0.0)
    return held_by_session, weights


def signal_label(entry_factor: str, *, exit_factor: str | None, exit_on_entry_false: bool,
                 trailing_stop: float | None) -> str:
    """How a timing run is named in the stored runs table: its entry factor and exits."""
    exits = []
    if exit_on_entry_false:
        exits.append('entry false')
    if exit_factor:
        exits.append(exit_factor)
    parts = [entry_factor]
    if exits:
        parts.append('exit on ' + ' or '.join(exits))
    if trailing_stop:
        parts.append(f'trailing stop {trailing_stop:.0%}')
    return ', '.join(parts)


def _wide(frame: pd.DataFrame, name: str, market: market_data.MarketData) -> pd.DataFrame:
    """One factor as a (sessions, tickers) matrix aligned to the run's market data."""
    wide = frame.pivot_table(index='Date', columns='Ticker', values=name, aggfunc='first')
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.reindex(index=market.dates, columns=list(market.tickers))


def simulate(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, library: dict, *, entry_factor: str, exit_factor: str | None,
             exit_on_entry_false: bool, trailing_stop: float | None, max_positions: int, priority_factor: str | None,
             start: str, end: str, cfg: backtest_config.Config | None = None) -> dict:
    """Run a timing rule through the engine over `start` to `end` and return the book's curves, fills and stats."""
    cfg = cfg or backtest_config.read()
    if not (exit_on_entry_false or exit_factor or trailing_stop):
        raise ValueError('a timing rule needs at least one exit: entry false, an exit factor, or a trailing stop')
    tickers = sorted(frame.loc[frame[entry_factor].notna(), 'Ticker'].unique())
    if not tickers:
        raise ValueError(f'{entry_factor!r} has no values over this range, so there is nothing to trade')
    market = market_data.build_market(con, tickers, start=start, end=end, exclude_checks=cfg.exclude_checks)
    entry = (_wide(frame, entry_factor, market).fillna(0) > 0).to_numpy()
    exit_signal = (_wide(frame, exit_factor, market).fillna(0) > 0).to_numpy() if exit_factor else None
    priority = _wide(frame, priority_factor, market).to_numpy() if priority_factor else market.adv.astype(float)
    close = np.where(market.tradable, market.close, np.nan)
    held, weights = holdings(entry, close, exit_signal=exit_signal, exit_on_entry_false=exit_on_entry_false,
                             trailing_stop=trailing_stop, max_positions=max_positions, priority=priority)
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
    bench = factor_backtest.benchmark_curve(con, cfg.benchmark, market.dates, cfg.starting_cash)
    stats = performance.summary(equity, fills, bench.pct_change() if bench is not None else None)
    return {'equity': equity, 'gross': gross, 'net': net, 'fills': fills, 'benchmark_equity': bench, 'stats': stats,
            'tickers': list(market.tickers), 'names_median': float(np.median(held.sum(axis=1)))}


def run_params(library: dict, *, entry_factor: str, exit_factor: str | None, exit_on_entry_false: bool,
               trailing_stop: float | None, max_positions: int, priority_factor: str | None, universe: str | None,
               universe_size: int, start: str, end: str, holdout: str | None) -> dict:
    """The settings of a timing run, including each factor's expression and fingerprint."""
    used = [name for name in (entry_factor, exit_factor, priority_factor) if name]
    return {'entry_factor': entry_factor, 'exit_factor': exit_factor, 'exit_on_entry_false': exit_on_entry_false,
            'trailing_stop': trailing_stop, 'max_positions': max_positions, 'priority_factor': priority_factor,
            'universe': universe, 'universe_size': None if universe else universe_size,
            'start': start, 'end': end, 'holdout': holdout, 'rebalance': 'D',
            'expressions': {name: library[name].expression for name in used},
            'fingerprints': {name: factors.fingerprint(name, library) for name in used}}


def run_id(params: dict, cfg: backtest_config.Config) -> str:
    """Stable run id over the factor fingerprints and every setting that changes the result."""
    return writer.run_id({'signal': 'timing_rule', 'params': {k: v for k, v in params.items() if k != 'expressions'},
                          'costs': vars(cfg.costs), 'limits': vars(cfg.limits), 'starting_cash': cfg.starting_cash,
                          'benchmark': cfg.benchmark, 'exclude_checks': cfg.exclude_checks})


def build_report(result: dict, params: dict, cfg: backtest_config.Config) -> report.BacktestReport:
    """The stored-run report for a simulated timing rule."""
    equity = result['equity']
    label = signal_label(params['entry_factor'], exit_factor=params['exit_factor'],
                         exit_on_entry_false=params['exit_on_entry_false'], trailing_stop=params['trailing_stop'])
    return report.BacktestReport(
        run_id=run_id(params, cfg), signal=label, kind=SIGNAL_KIND, config=cfg, params=params,
        start=equity.index[0], end=equity.index[-1], tickers=result['tickers'], benchmark=cfg.benchmark, caveats=[],
        stats=result['stats'], equity=equity, gross=result['gross'], net=result['net'], cash=equity - result['net'],
        fills=result['fills'], benchmark_equity=result['benchmark_equity'],
        exposure=performance.exposures(result['gross'], result['net'], equity),
    )
