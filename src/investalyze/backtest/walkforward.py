"""Walk-forward evaluation: fit on one span, trade the next, and step both forward.

Two things here are load bearing. The universe is chosen from each fold's training span alone, since
`panels.complete_universe` asks which names were priced across a whole calendar and handing it the
full history would pick the survivors of the entire sample. And the book carries across fold
boundaries, so the seams show the turnover and cost a real transition would pay instead of restarting
flat at every one.

The embargo sits between the training span and the test span. A signal fitted on forward-return
labels of horizon h has, in its last h training sessions, labels drawn from the test span; leaving h
sessions empty between them is what keeps those apart.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from investalyze.analysis import panels
from investalyze.backtest import engine, market as market_data, performance
from investalyze.backtest.book import Book
from investalyze.backtest.config import Config
from investalyze.backtest.registry import SignalFactory


@dataclass(frozen=True)
class Window:
    """One fold: a fitting span, then an embargo gap, then the out-of-sample span that follows it."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        """The fold's out-of-sample span, as it appears in the per-fold table."""
        return f'{self.test_start.date()}..{self.test_end.date()}'


@dataclass
class WalkForwardResult:
    """Every fold's out-of-sample track, concatenated, plus the per-fold statistics."""

    windows: list[Window]
    equity: pd.Series
    gross: pd.Series
    net: pd.Series
    fills: pd.DataFrame
    folds: pd.DataFrame


def walk_forward_windows(calendar: pd.DatetimeIndex, *, train: int, test: int, step: int,
                         embargo: int = 0, anchored: bool = False) -> list[Window]:
    """Split a calendar into folds of `train` fitting sessions and `test` out-of-sample sessions.

    Built on `panels.rolling_windows`, so the fold boundaries are the same ones the rolling stability
    study uses. `anchored` pins every fitting span to the first session, growing it each fold.
    """
    for name, value in (('train', train), ('test', test), ('step', step)):
        if value < 1:
            raise ValueError(f'{name} must be at least 1 session, got {value}')
    if embargo < 0:
        raise ValueError(f'embargo must not be negative, got {embargo}')

    span = train + embargo + test
    if span > len(calendar):
        raise ValueError(f'a fold needs {span} sessions (train {train} + embargo {embargo} + test {test}) '
                         f'but the calendar holds {len(calendar)}')
    windows = []
    for start, _ in panels.rolling_windows(calendar, window=span, step=step):
        first = calendar.get_loc(start)
        train_end = first + train - 1
        test_start = first + train + embargo
        test_end = first + span - 1
        if test_end >= len(calendar):
            continue
        windows.append(Window(
            train_start=calendar[0] if anchored else calendar[first],
            train_end=calendar[train_end],
            test_start=calendar[test_start],
            test_end=calendar[test_end],
        ))
    return windows


def _seed_book(cash: float, tickers: list[str], sessions: int, holdings: dict[str, float]) -> Book:
    """A book opening with the carried cash and whatever of the carried positions this fold tracks."""
    book = Book.start(cash, len(tickers), sessions)
    for position, ticker in enumerate(tickers):
        book.shares[position] = holdings.get(ticker, 0.0)
    return book


def _liquidate_dropped(holdings: dict[str, float], marks: dict[str, float], tickers: list[str],
                       config: Config) -> tuple[float, list[dict]]:
    """Sell, at the previous fold's last mark, whatever the new fold's universe no longer tracks."""
    tracked = set(tickers)
    proceeds = 0.0
    sold = []
    for ticker, shares in list(holdings.items()):
        if ticker in tracked or shares == 0:
            continue
        mark = marks.get(ticker, 0.0)
        if mark <= 0:
            holdings.pop(ticker)
            continue
        notional = shares * mark * (1 - np.sign(shares) * config.costs.slippage_bps / 1e4)
        fee = max(config.costs.commission_bps / 1e4 * abs(notional), config.costs.min_commission)
        proceeds += notional - fee
        sold.append({'Ticker': ticker, 'Side': 'sell' if shares > 0 else 'buy',
                     'Shares': abs(shares), 'Price': mark, 'Notional': abs(notional),
                     'Cost': fee, 'Kind': 'market'})
        holdings.pop(ticker)
    return proceeds, sold


def run_walk_forward(con: duckdb.DuckDBPyConnection, factory: SignalFactory, windows: list[Window],
                     config: Config, params: dict) -> WalkForwardResult:
    """Run the engine over each fold's out-of-sample span, carrying the book across the boundaries."""
    if not windows:
        raise ValueError('no walk-forward windows to run')

    cash = config.starting_cash
    holdings: dict[str, float] = {}
    marks: dict[str, float] = {}
    curves, grosses, nets, blotters, rows = [], [], [], [], []

    for window in windows:
        training = panels.trading_calendar(con, start=window.train_start.date().isoformat(),
                                           end=window.train_end.date().isoformat())
        tickers = panels.complete_universe(con, training, n_top=config.universe.n_top,
                                           rank_by=config.universe.rank_by,
                                           require_sector=config.universe.require_sector)
        if not tickers:
            continue

        proceeds, sold = _liquidate_dropped(holdings, marks, tickers, config)
        cash += proceeds
        if sold:
            for fill in sold:
                fill['Date'] = window.test_start
            blotters.append(pd.DataFrame(sold))

        market = market_data.build_market(con, tickers, start=window.test_start.date().isoformat(),
                                          end=window.test_end.date().isoformat(),
                                          exclude_checks=config.exclude_checks)
        book = _seed_book(cash, list(market.tickers), len(market.dates), holdings)
        step = factory(con, market, params)
        book = _run_fold(market, step, book, config)

        equity = pd.Series(book.equity, index=market.dates)
        curves.append(equity)
        grosses.append(pd.Series(book.gross, index=market.dates))
        nets.append(pd.Series(book.net, index=market.dates))
        fills = book.blotter()
        if not fills.empty:
            blotters.append(fills)

        cash = book.cash
        holdings = {ticker: float(shares) for ticker, shares in zip(market.tickers, book.shares)
                    if shares != 0}
        closing = pd.DataFrame(market.close).ffill().to_numpy()[-1]
        marks = {ticker: float(price) for ticker, price in zip(market.tickers, closing)
                 if np.isfinite(price)}

        summary = performance.summary(equity, fills)
        rows.append({'fold': window.label, 'train_start': window.train_start,
                     'train_end': window.train_end, 'n_tickers': len(tickers)} | summary)

    if not curves:
        raise ValueError('every walk-forward fold was empty')
    return WalkForwardResult(
        windows=windows,
        equity=pd.concat(curves),
        gross=pd.concat(grosses),
        net=pd.concat(nets),
        fills=pd.concat(blotters, ignore_index=True) if blotters else pd.DataFrame(),
        folds=pd.DataFrame(rows),
    )


def _run_fold(market: market_data.MarketData, step, book: Book, config: Config) -> Book:
    """Walk one fold's sessions with a book that is already holding the previous fold's positions."""
    seeded = book.shares.copy()
    result = engine.run_backtest(market, step, starting_cash=book.cash, costs=config.costs,
                                limits=config.limits, opening_shares=seeded)
    return result
