"""Tests for order netting: a restated stop replaces the working one, market orders add up."""

import numpy as np
import pytest

from investalyze.backtest import orders


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
