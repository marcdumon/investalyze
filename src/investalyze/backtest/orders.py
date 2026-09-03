"""Orders the engine carries from one session to the next.

One day's orders are parallel arrays rather than per-order objects: a run over 16,274 sessions and
several hundred names would otherwise allocate millions of small objects. Quantities are signed, so
one array covers both sides.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MARKET = 0
LIMIT = 1
STOP = 2

KIND_NAMES: dict[int, str] = {MARKET: 'market', LIMIT: 'limit', STOP: 'stop'}


@dataclass(frozen=True)
class OrderBatch:
    """One day's orders as parallel arrays: asset index, signed quantity, kind, trigger and age."""

    asset: np.ndarray     # (K,) int32, index into MarketData.tickers
    qty: np.ndarray       # (K,) float64, positive buys, negative sells
    kind: np.ndarray      # (K,) int8, one of MARKET / LIMIT / STOP
    trigger: np.ndarray   # (K,) float64, the limit or stop price; NaN on a market order
    age: np.ndarray       # (K,) int16, sessions this order has been carried unfilled

    def __len__(self) -> int:
        """Number of orders in the batch."""
        return len(self.asset)

    def select(self, keep: np.ndarray) -> 'OrderBatch':
        """The orders a boolean mask keeps, as a new batch."""
        return OrderBatch(asset=self.asset[keep], qty=self.qty[keep], kind=self.kind[keep],
                          trigger=self.trigger[keep], age=self.age[keep])

    @classmethod
    def concat(cls, first: 'OrderBatch', second: 'OrderBatch') -> 'OrderBatch':
        """One batch holding both batches' orders."""
        return cls(
            asset=np.concatenate([first.asset, second.asset]),
            qty=np.concatenate([first.qty, second.qty]),
            kind=np.concatenate([first.kind, second.kind]),
            trigger=np.concatenate([first.trigger, second.trigger]),
            age=np.concatenate([first.age, second.age]),
        )

    @classmethod
    def empty(cls) -> 'OrderBatch':
        """A batch holding no orders, with every array correctly typed."""
        return cls(
            asset=np.empty(0, dtype=np.int32),
            qty=np.empty(0, dtype=np.float64),
            kind=np.empty(0, dtype=np.int8),
            trigger=np.empty(0, dtype=np.float64),
            age=np.empty(0, dtype=np.int16),
        )


def netted(asset: np.ndarray, qty: np.ndarray, kind: np.ndarray, trigger: np.ndarray,
           age: np.ndarray) -> 'OrderBatch':
    """Reduce a batch to one order per asset, side and kind.

    Market orders in the same name and direction add up, so a name fills at most once a side and
    (run, date, ticker, side) stays a valid key for the blotter. A resting limit or stop is replaced
    by the newest one in that name and direction, which is what lets a signal re-state a trailing
    stop every session and still have exactly one working.
    """
    if len(asset) == 0:
        return OrderBatch.empty()
    frame = pd.DataFrame({'asset': asset, 'qty': qty, 'kind': kind, 'trigger': trigger, 'age': age})
    frame['side'] = np.sign(frame['qty'])

    market_rows = frame['kind'] == MARKET
    combined = (frame[market_rows]
                .groupby(['asset', 'side'], as_index=False)
                .agg(qty=('qty', 'sum'), kind=('kind', 'first'), trigger=('trigger', 'first'),
                     age=('age', 'max')))
    resting = (frame[~market_rows]
               .groupby(['asset', 'side', 'kind'], as_index=False)
               .agg(qty=('qty', 'last'), trigger=('trigger', 'last'), age=('age', 'min')))

    columns = ['asset', 'qty', 'kind', 'trigger', 'age']
    stacked = pd.concat([combined[columns], resting[columns]], ignore_index=True)
    stacked = stacked[stacked['qty'] != 0]
    return OrderBatch(
        asset=stacked['asset'].to_numpy(dtype=np.int32),
        qty=stacked['qty'].to_numpy(dtype=np.float64),
        kind=stacked['kind'].to_numpy(dtype=np.int8),
        trigger=stacked['trigger'].to_numpy(dtype=np.float64),
        age=stacked['age'].to_numpy(dtype=np.int16),
    )


def reference_price(kind: int, trigger: float, open_price: float, high: float, low: float,
                    side: float) -> float:
    """The price an order transacts at on this session, or NaN when it does not trigger.

    A market order takes the open. A limit order fills only once the session trades through its
    price, and takes the better of the open and the limit. A stop fills once the session trades
    through its trigger, and takes the worse of the open and the trigger, so a gap through the stop
    fills at the open.
    """
    if kind == MARKET:
        return open_price
    if kind == LIMIT:
        if side > 0:
            return min(open_price, trigger) if low <= trigger else np.nan
        return max(open_price, trigger) if high >= trigger else np.nan
    if kind == STOP:
        if side > 0:
            return max(open_price, trigger) if high >= trigger else np.nan
        return min(open_price, trigger) if low <= trigger else np.nan
    raise ValueError(f'unknown order kind {kind}')


def fill_price(reference: float, side: float, notional: float, adv: float, *, slippage_bps: float,
               impact_coef: float) -> float:
    """The reference price moved against the order by fixed slippage plus a size-dependent impact."""
    impact = impact_coef * np.sqrt(notional / adv) if adv > 0 and notional > 0 else 0.0
    return reference * (1 + side * (slippage_bps / 1e4 + impact))


def commission(notional: float, *, commission_bps: float, min_commission: float) -> float:
    """What a fill costs in fees: a rate on notional, floored at a per-fill minimum."""
    return max(commission_bps / 1e4 * abs(notional), min_commission)
