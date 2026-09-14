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

import argparse
from dataclasses import dataclass
from typing import Self

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import factors_pit
from investalyze.backtest import config as backtest_config
from investalyze.backtest import engine, report
from investalyze.backtest import market as market_data
from investalyze.research import backtest as factor_backtest
from investalyze.research import factors

SIGNAL_KIND = engine.TIMING
DEFAULT_MAX_POSITIONS = 20


@dataclass(frozen=True)
class TimingRule:
    """A timing rule: the factors that enter and exit each name, the trailing stop and the position cap."""

    entry_factor: str
    exit_factor: str | None = None
    exit_on_entry_false: bool = True
    trailing_stop: float | None = None
    max_positions: int = DEFAULT_MAX_POSITIONS
    priority_factor: str | None = None

    def __post_init__(self) -> None:
        if not self.entry_factor:
            raise ValueError('Pick an entry factor.')
        if not (self.exit_on_entry_false or self.exit_factor or self.trailing_stop):
            raise ValueError('A timing rule needs at least one exit: the entry factor turning false, an exit factor, '
                             'or a trailing stop.')
        if self.trailing_stop is not None and not 0 < self.trailing_stop < 1:
            raise ValueError('The trailing stop is a fraction between 0 and 1.')
        if self.max_positions < 1:
            raise ValueError('A timing rule holds at least one position.')

    @property
    def factors(self) -> list[str]:
        """The library factors the rule reads: entry, exit and priority."""
        return [name for name in (self.entry_factor, self.exit_factor, self.priority_factor) if name]

    @property
    def label(self) -> str:
        """How a timing run is named in the stored runs table: its entry factor and exits."""
        exits = []
        if self.exit_on_entry_false:
            exits.append('entry false')
        if self.exit_factor:
            exits.append(self.exit_factor)
        parts = [self.entry_factor]
        if exits:
            parts.append('exit on ' + ' or '.join(exits))
        if self.trailing_stop:
            parts.append(f'trailing stop {self.trailing_stop:.0%}')
        return ', '.join(parts)

    def arguments(self) -> list[str]:
        """The `python -m investalyze.research timing` arguments that describe this rule."""
        arguments = [self.entry_factor, '--max-positions', str(self.max_positions)]
        if self.exit_factor:
            arguments += ['--exit-factor', self.exit_factor]
        if not self.exit_on_entry_false:
            arguments.append('--keep-on-entry-false')
        if self.trailing_stop:
            arguments += ['--trailing-stop', f'{self.trailing_stop:g}']
        if self.priority_factor:
            arguments += ['--priority-factor', self.priority_factor]
        return arguments

    @classmethod
    def from_arguments(cls, args: argparse.Namespace) -> Self:
        """The rule a parsed `timing` command describes."""
        return cls(entry_factor=args.name, exit_factor=args.exit_factor, exit_on_entry_false=not args.keep_on_entry_false,
                   trailing_stop=args.trailing_stop or None, max_positions=args.max_positions,
                   priority_factor=args.priority_factor)


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


def _wide(frame: pd.DataFrame, name: str, market: market_data.MarketData) -> pd.DataFrame:
    """One factor as a (sessions, tickers) matrix aligned to the run's market data."""
    wide = frame.pivot_table(index='Date', columns='Ticker', values=name, aggfunc='first')
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.reindex(index=market.dates, columns=list(market.tickers))


def simulate(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame, rule: TimingRule, *, start: str, end: str,
             cfg: backtest_config.Config | None = None) -> factor_backtest.Simulation:
    """Run a timing rule through the engine over `start` to `end`."""
    cfg = cfg or backtest_config.read()
    tickers = sorted(frame.loc[frame[rule.entry_factor].notna(), 'Ticker'].unique())
    if not tickers:
        raise ValueError(f'{rule.entry_factor!r} has no values over this range, so there is nothing to trade')
    market = market_data.build_market(con, tickers, start=start, end=end, exclude_checks=cfg.exclude_checks)
    entry = (_wide(frame, rule.entry_factor, market).fillna(0) > 0).to_numpy()
    exit_signal = (_wide(frame, rule.exit_factor, market).fillna(0) > 0).to_numpy() if rule.exit_factor else None
    if rule.priority_factor:
        priority = _wide(frame, rule.priority_factor, market).to_numpy()
    else:
        priority = market.adv.astype(float)
    close = np.where(market.tradable, market.close, np.nan)
    held, weights = holdings(entry, close, exit_signal=exit_signal, exit_on_entry_false=rule.exit_on_entry_false,
                             trailing_stop=rule.trailing_stop, max_positions=rule.max_positions, priority=priority)
    return factor_backtest.trade_weights(con, market, weights, cfg, names_median=float(np.median(held.sum(axis=1))))


def run_params(rule: TimingRule, library: dict, *, universe: str | None, universe_size: int, start: str, end: str,
               holdout: str | None) -> dict:
    """The settings of a timing run, including each factor's expression and fingerprint."""
    return {'entry_factor': rule.entry_factor, 'exit_factor': rule.exit_factor,
            'exit_on_entry_false': rule.exit_on_entry_false, 'trailing_stop': rule.trailing_stop,
            'max_positions': rule.max_positions, 'priority_factor': rule.priority_factor,
            'universe': universe, 'universe_size': None if universe else universe_size,
            'start': start, 'end': end, 'holdout': holdout, 'rebalance': factors_pit.Rebalance.DAILY,
            'expressions': {name: library[name].expression for name in rule.factors},
            'fingerprints': {name: factors.fingerprint(name, library) for name in rule.factors}}


def run_id(params: dict, cfg: backtest_config.Config) -> str:
    """Stable run id over the factor fingerprints and every setting that changes the result."""
    return factor_backtest.run_id(params, cfg, signal='timing_rule')


def build_report(simulation: factor_backtest.Simulation, rule: TimingRule, params: dict,
                 cfg: backtest_config.Config) -> report.BacktestReport:
    """The stored-run report for a simulated timing rule."""
    return factor_backtest.stored_report(simulation, params, cfg, run_id=run_id(params, cfg), label=rule.label,
                                         kind=SIGNAL_KIND)
