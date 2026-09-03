"""Tests for the per-ticker timing signals and the engine path they exercise.

A timing rule and a cross-sectional rule go through the same engine, so the test that matters most is
that both run to a finite equity curve from the same loop. The rest pin down when a crossover acts
and that a restated trailing stop stays a single working order rather than accumulating.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.backtest import engine, orders, registry
from investalyze.backtest.book import Book
from investalyze.backtest.config import Costs, Limits
from investalyze.backtest.market import MarketData
from investalyze.backtest.signals import timing

FREE = Costs(slippage_bps=0.0, commission_bps=0.0, min_commission=0.0, impact_coef=0.0,
             borrow_bps_annual=0.0, cash_bps_annual=0.0)
LIMITS = Limits(max_weight=1.0, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                participation=1e12, max_order_age=5)


def _market(closes: np.ndarray, tickers: list[str] | None = None) -> MarketData:
    """A synthetic market whose opens equal its closes, so a fill lands on the printed price."""
    closes = closes.reshape(len(closes), -1)
    sessions, assets = closes.shape
    return MarketData(
        dates=pd.bdate_range('2020-01-01', periods=sessions),
        tickers=np.array(tickers if tickers else [f'T{i}' for i in range(assets)]),
        open_=closes.astype(float), high=closes.astype(float), low=closes.astype(float),
        close=closes.astype(float), volume=np.full((sessions, assets), 1e9, dtype=np.float32),
        dividend=np.zeros((sessions, assets)), adv=np.full((sessions, assets), 1e12, dtype=np.float32),
        tradable=np.ones((sessions, assets), dtype=bool), adjusted_close=closes.astype(float),
    )


# --- moving-average crossover ---

def test_sma_cross_holds_nothing_until_the_slow_window_fills():
    market = _market(np.linspace(100, 200, 30))
    step = timing.sma_cross(None, market, {'fast': 3, 'slow': 10})
    book = Book.start(10_000.0, 1, 30)
    assert step(0, book).targets.tolist() == [0.0]


def test_sma_cross_takes_a_position_once_the_fast_average_leads():
    market = _market(np.linspace(100, 200, 40))
    step = timing.sma_cross(None, market, {'fast': 3, 'slow': 10, 'max_positions': 1})
    book = Book.start(10_000.0, 1, 40)
    decisions = [step(session, book) for session in range(40)]
    entered = [i for i, decision in enumerate(decisions)
               if decision.targets is not None and decision.targets[0] > 0]
    assert entered, 'a rising series should eventually put the fast average above the slow one'
    assert entered[0] >= 10 - 1   # the slow average first exists on the tenth session, index 9


def test_sma_cross_stays_quiet_on_the_sessions_its_membership_does_not_change():
    market = _market(np.linspace(100, 200, 40))
    step = timing.sma_cross(None, market, {'fast': 3, 'slow': 10})
    book = Book.start(10_000.0, 1, 40)
    decisions = [step(session, book) for session in range(40)]
    assert sum(decision.targets is None for decision in decisions) > 20


def test_a_fast_window_at_least_as_long_as_the_slow_one_is_rejected():
    market = _market(np.linspace(100, 200, 30))
    with pytest.raises(ValueError, match='must be shorter than slow window'):
        timing.sma_cross(None, market, {'fast': 10, 'slow': 10})


# --- breakout with a trailing stop ---

def test_breakout_enters_a_name_making_a_new_high():
    market = _market(np.linspace(100, 200, 30))
    step = timing.breakout_52w(None, market, {'lookback': 5, 'max_positions': 1})
    book = Book.start(10_000.0, 1, 30)
    decision = step(10, book)
    assert decision.targets is not None
    assert decision.targets[0] > 0


def test_breakout_rests_a_stop_under_a_held_position():
    market = _market(np.linspace(100, 200, 30))
    step = timing.breakout_52w(None, market, {'lookback': 5, 'stop_fraction': 0.1})
    book = Book.start(10_000.0, 1, 30)
    book.shares[0] = 10.0
    decision = step(10, book)
    assert decision.orders is not None
    assert decision.orders.kind[0] == orders.STOP
    assert decision.orders.qty[0] == pytest.approx(-10.0)
    assert decision.orders.trigger[0] == pytest.approx(market.close[10, 0] * 0.9)


def test_the_breakout_stop_ratchets_up_with_the_running_high():
    market = _market(np.linspace(100, 200, 30))
    step = timing.breakout_52w(None, market, {'lookback': 5, 'stop_fraction': 0.1})
    book = Book.start(10_000.0, 1, 30)
    book.shares[0] = 10.0
    triggers = [step(session, book).orders.trigger[0] for session in (10, 20)]
    assert triggers[1] > triggers[0]


def test_a_stop_fraction_outside_zero_to_one_is_rejected():
    market = _market(np.linspace(100, 200, 30))
    with pytest.raises(ValueError, match='stop_fraction must be in'):
        timing.breakout_52w(None, market, {'stop_fraction': 1.5})


# --- how the restated stop reaches the engine ---

def test_restating_a_stop_leaves_exactly_one_working_order():
    first = orders.OrderBatch(asset=np.array([0], dtype=np.int32), qty=np.array([-10.0]),
                              kind=np.array([orders.STOP], dtype=np.int8), trigger=np.array([90.0]),
                              age=np.array([1], dtype=np.int16))
    second = orders.OrderBatch(asset=np.array([0], dtype=np.int32), qty=np.array([-10.0]),
                               kind=np.array([orders.STOP], dtype=np.int8), trigger=np.array([95.0]),
                               age=np.array([0], dtype=np.int16))
    both = orders.OrderBatch.concat(first, second)
    netted = orders.netted(both.asset, both.qty, both.kind, both.trigger, both.age)
    assert len(netted) == 1
    assert netted.trigger[0] == pytest.approx(95.0)   # the newest level, not the first


def test_market_orders_in_the_same_name_and_direction_still_add_up():
    batch = orders.OrderBatch(asset=np.array([0, 0], dtype=np.int32), qty=np.array([5.0, 7.0]),
                              kind=np.array([orders.MARKET, orders.MARKET], dtype=np.int8),
                              trigger=np.array([np.nan, np.nan]), age=np.array([0, 0], dtype=np.int16))
    netted = orders.netted(batch.asset, batch.qty, batch.kind, batch.trigger, batch.age)
    assert len(netted) == 1
    assert netted.qty[0] == pytest.approx(12.0)


def test_opposite_sides_of_one_name_stay_separate_orders():
    batch = orders.OrderBatch(asset=np.array([0, 0], dtype=np.int32), qty=np.array([5.0, -7.0]),
                              kind=np.array([orders.MARKET, orders.MARKET], dtype=np.int8),
                              trigger=np.array([np.nan, np.nan]), age=np.array([0, 0], dtype=np.int16))
    netted = orders.netted(batch.asset, batch.qty, batch.kind, batch.trigger, batch.age)
    assert len(netted) == 2


# --- both shapes through one engine ---

def test_a_timing_signal_and_a_cross_sectional_signal_run_through_the_same_engine():
    rising = np.linspace(100, 200, 60)
    falling = np.linspace(200, 150, 60)
    market = _market(np.column_stack([rising, falling]), tickers=['AAA', 'BBB'])

    timing_step = timing.sma_cross(None, market, {'fast': 3, 'slow': 10, 'max_positions': 2})
    timing_book = engine.run_backtest(market, timing_step, starting_cash=100_000.0, costs=FREE,
                                      limits=LIMITS, check_accounting=True)

    def cross_sectional(session: int, book: Book):
        """Hold the first name outright, a stand-in for a ranked cross-section."""
        return registry.Decision(targets=np.array([1.0, 0.0]), orders=None)

    ranked_book = engine.run_backtest(market, cross_sectional, starting_cash=100_000.0, costs=FREE,
                                      limits=LIMITS, check_accounting=True)

    assert np.isfinite(timing_book.equity).all()
    assert np.isfinite(ranked_book.equity).all()
    assert timing_book.equity[-1] > 0 and ranked_book.equity[-1] > 0


def test_a_breakout_run_places_and_fills_stop_orders():
    """A series that climbs then collapses must trip the trailing stop and leave the position."""
    climb = np.linspace(100, 200, 40)
    collapse = np.linspace(200, 80, 20)
    market = _market(np.concatenate([climb, collapse]))
    step = timing.breakout_52w(None, market, {'lookback': 5, 'stop_fraction': 0.1, 'max_positions': 1})
    book = engine.run_backtest(market, step, starting_cash=100_000.0, costs=FREE, limits=LIMITS,
                               check_accounting=True)
    blotter = book.blotter()
    assert not blotter.empty
    assert 'stop' in set(blotter['Kind'])
    assert book.shares[0] == pytest.approx(0.0)


def test_every_registered_timing_signal_produces_a_finite_curve():
    market = _market(np.linspace(100, 300, 80))
    for name, (kind, factory) in registry.SIGNALS.items():
        if kind != registry.TIMING:
            continue
        step = factory(duckdb.connect(), market, {'fast': 3, 'slow': 10, 'lookback': 5})
        book = engine.run_backtest(market, step, starting_cash=10_000.0, costs=FREE, limits=LIMITS)
        assert np.isfinite(book.equity).all(), name
