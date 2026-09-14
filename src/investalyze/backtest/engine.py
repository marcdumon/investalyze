"""The daily loop: signals decide at a close, orders fill at the next open.

That one sentence is the whole look-ahead guard. A signal sees the market through session `t` and
returns what it wants; the resulting orders reach the book at session `t + 1`'s open, at a price the
signal could not have known. Nothing in the loop reads a row beyond the session it is on.

Positions are carried in the split-adjusted share units the price matrices use, so a split changes no
share count and the loop needs no corporate-action step. Dividends arrive as cash on their ex-date.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd

from investalyze.backtest import orders as order_rules
from investalyze.backtest.book import Book
from investalyze.backtest.config import Costs, Limits
from investalyze.backtest.market import MarketData
from investalyze.backtest.orders import OrderBatch

TRADING_DAYS = 252
CASH_TOLERANCE = 1e-6  # relative slack the debug accounting check allows on a session's cash

CROSS_SECTIONAL = 'cross_sectional'
TIMING = 'timing'
KINDS = (CROSS_SECTIONAL, TIMING)


class Decision(NamedTuple):
    """What a signal wants on one session: target weights, explicit orders, or neither.

    `targets` holds a weight per asset, where NaN leaves that name's position alone; `None` means
    this session is not a rebalance. `orders` carries limit or stop orders for the next session.
    """

    targets: np.ndarray | None
    orders: OrderBatch | None


HOLD = Decision(targets=None, orders=None)

SignalStep = Callable[[int, Book], Decision]


class LookAheadError(RuntimeError):
    """Raised when a signal reads a session later than the one it is deciding on."""


class MarketView:
    """A read-only view of the market that refuses any row later than the session it is set to.

    A signal that computes inside its step function can take this instead of the raw matrices, which
    turns a look-ahead bug from something a reviewer has to notice into something a test catches.
    """

    def __init__(self, market: MarketData, session: int) -> None:
        self._market = market
        self.session = session

    def rows(self, name: str, through: int | None = None) -> np.ndarray:
        """Every session of one matrix up to and including the current one."""
        last = self.session if through is None else through
        if last > self.session:
            raise LookAheadError(f'session {last} is later than the current session {self.session}')
        return getattr(self._market, name)[:last + 1]

    def row(self, name: str, session: int | None = None) -> np.ndarray:
        """One session of one matrix, refused when it lies in the future."""
        wanted = self.session if session is None else session
        if wanted > self.session:
            raise LookAheadError(f'session {wanted} is later than the current session {self.session}')
        return getattr(self._market, name)[wanted]


def _marks(market: MarketData) -> np.ndarray:
    """Closing prices with gaps carried forward, so a session without a print holds its last value."""
    frame = pd.DataFrame(market.close).ffill()
    return frame.to_numpy(dtype=np.float64)


def _accrue_dividends(book: Book, market: MarketData, session: int) -> float:
    """Credit the cash a session's ex-dates pay on the positions held into them."""
    paid = float(np.nansum(book.shares * market.dividend[session]))
    book.cash += paid
    return paid


def _accrue_financing(book: Book, marks: np.ndarray, session: int, costs: Costs) -> float:
    """Charge borrow on the short book and credit interest on positive cash, for one session."""
    values = book.shares * np.nan_to_num(marks[session], nan=0.0)
    short_notional = float(-values[values < 0].sum())
    borrow = short_notional * costs.borrow_bps_annual / 1e4 / TRADING_DAYS
    interest = max(book.cash, 0.0) * costs.cash_bps_annual / 1e4 / TRADING_DAYS
    book.cash += interest - borrow
    return interest - borrow


def _apply_fill(book: Book, asset: int, shares: float, price: float) -> None:
    """Move shares and cost basis for one fill, resetting the basis when a position flips sign."""
    held = book.shares[asset]
    updated = held + shares
    if held == 0 or np.sign(updated) != np.sign(held):
        book.cost_basis[asset] = price
    elif abs(updated) > abs(held):
        book.cost_basis[asset] = (abs(held) * book.cost_basis[asset] + abs(shares) * price) / abs(updated)
    book.shares[asset] = updated


def _execute(pending: OrderBatch, book: Book, market: MarketData, session: int, costs: Costs,
             limits: Limits) -> tuple[OrderBatch, float, float, float]:
    """Fill what the session allows, carry the rest, and return the carried orders plus the cash moved."""
    carried: list[tuple] = []
    bought = sold = charged = 0.0

    for index in range(len(pending)):
        asset = int(pending.asset[index])
        wanted = float(pending.qty[index])
        kind = int(pending.kind[index])
        age = int(pending.age[index])
        side = np.sign(wanted)
        if wanted == 0:
            continue

        if not market.tradable[session, asset]:
            _carry(carried, asset, wanted, kind, float(pending.trigger[index]), age, limits)
            continue

        reference = order_rules.reference_price(kind, float(pending.trigger[index]),
                                                float(market.open_[session, asset]),
                                                float(market.high[session, asset]),
                                                float(market.low[session, asset]), side)
        if not np.isfinite(reference) or reference <= 0:
            _carry(carried, asset, wanted, kind, float(pending.trigger[index]), age, limits)
            continue

        allowed = limits.participation * float(market.volume[session, asset])
        shares = side * min(abs(wanted), allowed)
        if shares == 0:
            _carry(carried, asset, wanted, kind, float(pending.trigger[index]), age, limits)
            continue

        gross = abs(shares) * reference
        price = order_rules.fill_price(reference, side, gross, float(market.adv[session, asset]),
                                       slippage_bps=costs.slippage_bps, impact_coef=costs.impact_coef)
        notional = shares * price
        fee = order_rules.commission(notional, commission_bps=costs.commission_bps,
                                     min_commission=costs.min_commission)
        book.cash -= notional + fee
        _apply_fill(book, asset, shares, price)
        charged += fee
        if shares > 0:
            bought += notional
        else:
            sold += -notional

        book.fills.append({
            'Date': market.dates[session], 'Ticker': market.tickers[asset],
            'Side': 'buy' if shares > 0 else 'sell', 'Shares': abs(shares), 'Price': price,
            'Notional': abs(notional), 'Cost': fee, 'Kind': order_rules.KIND_NAMES[kind],
        })

        remainder = wanted - shares
        if abs(remainder) > 0:
            _carry(carried, asset, remainder, kind, float(pending.trigger[index]), age, limits)

    return _batch(carried), bought, sold, charged


def _carry(carried: list[tuple], asset: int, qty: float, kind: int, trigger: float, age: int,
           limits: Limits) -> None:
    """Hold an unfilled order over to the next session until it reaches the maximum age."""
    if age + 1 < limits.max_order_age:
        carried.append((asset, qty, kind, trigger, age + 1))


def _batch(rows: list[tuple]) -> OrderBatch:
    """An OrderBatch from a list of order tuples."""
    if not rows:
        return OrderBatch.empty()
    columns = list(zip(*rows))
    return OrderBatch(
        asset=np.array(columns[0], dtype=np.int32),
        qty=np.array(columns[1], dtype=np.float64),
        kind=np.array(columns[2], dtype=np.int8),
        trigger=np.array(columns[3], dtype=np.float64),
        age=np.array(columns[4], dtype=np.int16),
    )


def _mark_to_market(book: Book, marks: np.ndarray, session: int) -> None:
    """Value the book at the session's closing prices."""
    values = book.shares * np.nan_to_num(marks[session], nan=0.0)
    book.equity[session] = book.cash + float(values.sum())
    book.gross[session] = float(np.abs(values).sum())
    book.net[session] = float(values.sum())


def _decide_orders(decision: Decision, book: Book, market: MarketData, marks: np.ndarray,
                   session: int, limits: Limits) -> OrderBatch:
    """Turn a decision into the orders that will reach the next session's open."""
    rows: list[tuple] = []
    if decision.targets is not None:
        equity = book.equity[session]
        prices = marks[session]
        for asset in range(len(market.tickers)):
            target = decision.targets[asset]
            price = prices[asset]
            if not np.isfinite(target) or not np.isfinite(price) or price <= 0:
                continue
            capped = float(np.clip(target, -limits.max_weight, limits.max_weight))
            delta_value = capped * equity - book.shares[asset] * price
            shares = delta_value / price
            cap = limits.max_position_adv * float(market.adv[session, asset]) / price
            if np.isfinite(cap) and cap > 0:
                shares = float(np.clip(shares, -cap, cap))
            if abs(shares) > 0:
                rows.append((asset, shares, order_rules.MARKET, np.nan, 0))

    batch = _batch(rows)
    if decision.orders is not None and len(decision.orders):
        extra = decision.orders
        batch = _batch(list(zip(batch.asset, batch.qty, batch.kind, batch.trigger, batch.age))
                       + list(zip(extra.asset, extra.qty, extra.kind, extra.trigger, extra.age)))
    return order_rules.netted(batch.asset, batch.qty, batch.kind, batch.trigger, batch.age)


def _merge_pending(carried: OrderBatch, fresh: OrderBatch, decision: Decision) -> OrderBatch:
    """Combine orders carried over with the ones this session decided on.

    A target weight is a complete statement of what a name should hold, so it supersedes anything
    still outstanding in that name. Names the decision has no opinion on keep carrying their orders.
    """
    keep = carried
    if decision.targets is not None and len(carried):
        superseded = np.isfinite(decision.targets)
        keep = carried.select(~superseded[carried.asset])
    return OrderBatch.concat(keep, fresh)


def last_tradable_session(market: MarketData) -> np.ndarray:
    """The final session each name could be traded on, or -1 for one that never could.

    A name whose prices stop is delisted as far as this data is concerned: Yahoo removes the symbol,
    the history already stored keeps its last bar, and nothing arrives after it.
    """
    tradable = market.tradable
    ever = tradable.any(axis=0)
    last = np.where(ever, tradable.shape[0] - 1 - np.argmax(tradable[::-1], axis=0), -1)
    return last.astype(np.int64)


def _liquidate_delisted(book: Book, market: MarketData, marks: np.ndarray, last_tradable: np.ndarray,
                        session: int) -> float:
    """Close out any position whose name stopped trading, at the last price it traded at.

    Without this a delisted holding is marked from its final close straight to nothing, because the
    close is NaN and the mark treats NaN as zero. That writes off the whole position overnight with
    no fill, no cash and no record, and leaves the shares on the book contributing nothing for the
    rest of the run.

    The last close is the honest figure to use when the reason for the delisting is unknown: a
    company taken over paid something close to it, and one that failed had already fallen to near
    nothing by then. No costs are charged, because this is a corporate event rather than a trade.
    """
    proceeds = 0.0
    gone = np.flatnonzero((book.shares != 0) & (last_tradable == session - 1))
    for asset in gone:
        shares = float(book.shares[asset])
        price = float(marks[last_tradable[asset], asset])
        if not np.isfinite(price):
            continue
        notional = shares * price
        book.cash += notional
        book.shares[asset] = 0.0
        book.cost_basis[asset] = 0.0
        proceeds += notional
        book.fills.append({
            'Date': market.dates[session], 'Ticker': market.tickers[asset],
            'Side': 'sell' if shares > 0 else 'buy', 'Shares': abs(shares), 'Price': price,
            'Notional': abs(notional), 'Cost': 0.0, 'Kind': 'delisting',
        })
    return proceeds


def run_backtest(market: MarketData, step, *, starting_cash: float, costs: Costs, limits: Limits,
                 check_accounting: bool = False) -> Book:
    """Walk the calendar, filling each session's orders at its open and marking the book at its close.

    `step` is a signal's per-session function, taking the session index and the book and returning a
    `Decision`. Set `check_accounting` to assert cash and equity reconcile on every session, which is
    slower and is what the accounting tests run with.
    """
    sessions, assets = market.shape
    if sessions < 2:
        raise ValueError(f'a backtest needs at least 2 sessions, got {sessions}')
    marks = _marks(market)
    book = Book.start(starting_cash, assets, sessions)
    _mark_to_market(book, marks, 0)
    pending = _decide_orders(step(0, book), book, market, marks, 0, limits)

    last_tradable = last_tradable_session(market)

    for session in range(1, sessions):
        opening_cash = book.cash
        paid = _accrue_dividends(book, market, session)
        financed = _accrue_financing(book, marks, session, costs)
        delisted = _liquidate_delisted(book, market, marks, last_tradable, session)
        carried, bought, sold, charged = _execute(pending, book, market, session, costs, limits)
        _mark_to_market(book, marks, session)

        if check_accounting:
            expected = opening_cash + paid + financed + delisted - bought + sold - charged
            _assert_close(book.cash, expected, f'cash at session {session}')
            values = book.shares * np.nan_to_num(marks[session], nan=0.0)
            _assert_close(book.equity[session], book.cash + values.sum(), f'equity at session {session}')

        decision = step(session, book)
        fresh = _decide_orders(decision, book, market, marks, session, limits)
        pending = _merge_pending(carried, fresh, decision)

    return book


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Raise when two figures that must reconcile differ by more than the tolerance."""
    scale = max(1.0, abs(expected))
    if abs(actual - expected) / scale > CASH_TOLERANCE:
        raise AssertionError(f'{label}: {actual!r} does not reconcile with {expected!r}')
