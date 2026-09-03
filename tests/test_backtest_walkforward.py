"""Tests for walk-forward folds: where the boundaries fall and what carries across them.

The embargo and the carried book are the two properties worth guarding. Without the embargo a signal
fitted on forward-return labels trains on labels drawn from its own test span; without the carried
book every fold starts flat, which invents a costless, turnover-free seam at exactly the transitions
walk-forward exists to examine.
"""

import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import panels
from investalyze.backtest import engine, walkforward
from investalyze.backtest.book import Book
from investalyze.backtest.config import Costs, Limits
from investalyze.backtest.market import MarketData
from investalyze.backtest.registry import Decision

CALENDAR = pd.bdate_range('2020-01-01', periods=120)
FREE = Costs(slippage_bps=0.0, commission_bps=0.0, min_commission=0.0, impact_coef=0.0,
             borrow_bps_annual=0.0, cash_bps_annual=0.0)
LIMITS = Limits(max_weight=1.0, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                participation=1e12, max_order_age=5)


def _market(closes: np.ndarray) -> MarketData:
    """A single-name market whose opens equal its closes."""
    grid = closes.reshape(-1, 1)
    sessions = len(closes)
    return MarketData(
        dates=pd.bdate_range('2020-01-01', periods=sessions), tickers=np.array(['AAA']),
        open_=grid, high=grid, low=grid, close=grid,
        volume=np.full((sessions, 1), 1e9, dtype=np.float32), dividend=np.zeros((sessions, 1)),
        adv=np.full((sessions, 1), 1e12, dtype=np.float32),
        tradable=np.ones((sessions, 1), dtype=bool), adjusted_close=grid,
    )


# --- where the boundaries fall ---

def test_walk_forward_windows_reuse_the_rolling_window_boundaries():
    windows = walkforward.walk_forward_windows(CALENDAR, train=40, test=10, step=10)
    rolling = panels.rolling_windows(CALENDAR, window=50, step=10)
    assert windows[0].train_start == rolling[0][0]
    assert len(windows) <= len(rolling)


def test_the_embargo_leaves_a_gap_between_the_training_end_and_the_test_start():
    plain = walkforward.walk_forward_windows(CALENDAR, train=40, test=10, step=10, embargo=0)
    embargoed = walkforward.walk_forward_windows(CALENDAR, train=40, test=10, step=10, embargo=5)
    plain_gap = CALENDAR.get_loc(plain[0].test_start) - CALENDAR.get_loc(plain[0].train_end)
    embargoed_gap = CALENDAR.get_loc(embargoed[0].test_start) - CALENDAR.get_loc(embargoed[0].train_end)
    assert plain_gap == 1
    assert embargoed_gap == 6   # the session after the training end, plus five embargoed


def test_anchored_mode_pins_every_training_span_to_the_first_session():
    windows = walkforward.walk_forward_windows(CALENDAR, train=30, test=10, step=10, anchored=True)
    assert len({window.train_start for window in windows}) == 1
    assert windows[0].train_start == CALENDAR[0]


def test_rolling_mode_moves_the_training_start_forward_each_fold():
    windows = walkforward.walk_forward_windows(CALENDAR, train=30, test=10, step=10)
    starts = [window.train_start for window in windows]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_the_test_spans_do_not_overlap():
    windows = walkforward.walk_forward_windows(CALENDAR, train=30, test=10, step=10)
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.test_end < later.test_start


def test_every_fold_ends_inside_the_calendar():
    windows = walkforward.walk_forward_windows(CALENDAR, train=80, test=20, step=10)
    assert windows
    assert all(window.test_end <= CALENDAR[-1] for window in windows)


def test_a_fold_longer_than_the_calendar_is_rejected_by_name():
    with pytest.raises(ValueError, match='a fold needs 130 sessions'):
        walkforward.walk_forward_windows(CALENDAR, train=100, test=30, step=10)


def test_a_window_size_below_one_session_is_rejected():
    with pytest.raises(ValueError, match='train must be at least 1 session'):
        walkforward.walk_forward_windows(CALENDAR, train=0, test=10, step=10)


def test_a_negative_embargo_is_rejected():
    with pytest.raises(ValueError, match='embargo must not be negative'):
        walkforward.walk_forward_windows(CALENDAR, train=10, test=5, step=5, embargo=-1)


def test_a_fold_label_names_its_out_of_sample_span():
    window = walkforward.walk_forward_windows(CALENDAR, train=30, test=10, step=10)[0]
    assert window.label == f'{window.test_start.date()}..{window.test_end.date()}'


# --- what carries across a boundary ---

def test_a_book_can_open_a_run_already_holding_positions():
    market = _market(np.full(10, 100.0))

    def hold(session: int, book: Book) -> Decision:
        """Never trade, so the opening position is what the run ends with."""
        return Decision(targets=None, orders=None)

    book = engine.run_backtest(market, hold, starting_cash=1_000.0, costs=FREE, limits=LIMITS,
                               opening_shares=np.array([5.0]))
    assert book.shares[0] == pytest.approx(5.0)
    assert book.equity[0] == pytest.approx(1_000.0 + 5 * 100.0)


def test_the_seam_between_folds_pays_for_the_position_it_carries():
    """A carried book starts invested, so its first session is marked on the position, not on cash."""
    market = _market(np.linspace(100.0, 110.0, 10))

    def hold(session: int, book: Book) -> Decision:
        return Decision(targets=None, orders=None)

    flat = engine.run_backtest(market, hold, starting_cash=1_000.0, costs=FREE, limits=LIMITS)
    carried = engine.run_backtest(market, hold, starting_cash=1_000.0, costs=FREE, limits=LIMITS,
                                  opening_shares=np.array([5.0]))
    assert flat.equity[-1] == pytest.approx(1_000.0)
    assert carried.equity[-1] > flat.equity[-1]


def test_opening_shares_must_cover_the_whole_universe():
    market = _market(np.full(10, 100.0))
    with pytest.raises(ValueError, match='opening_shares covers'):
        engine.run_backtest(market, lambda session, book: Decision(None, None), starting_cash=1.0,
                            costs=FREE, limits=LIMITS, opening_shares=np.array([1.0, 2.0]))


def test_names_the_next_fold_stops_tracking_are_liquidated_at_the_last_mark():
    from investalyze.backtest.config import read as read_config

    holdings = {'GONE': 10.0, 'KEPT': 5.0}
    marks = {'GONE': 20.0, 'KEPT': 30.0}
    proceeds, sold = walkforward._liquidate_dropped(holdings, marks, ['KEPT'], read_config())
    assert 'GONE' not in holdings
    assert holdings['KEPT'] == 5.0
    assert len(sold) == 1
    assert sold[0]['Ticker'] == 'GONE'
    assert proceeds == pytest.approx(10 * 20.0 * (1 - 5 / 1e4) - max(2 / 1e4 * 10 * 20.0 * (1 - 5 / 1e4), 1.0))


def test_a_name_with_no_last_mark_is_dropped_without_proceeds():
    from investalyze.backtest.config import read as read_config

    holdings = {'GHOST': 10.0}
    proceeds, sold = walkforward._liquidate_dropped(holdings, {}, ['OTHER'], read_config())
    assert holdings == {}
    assert sold == []
    assert proceeds == 0.0


def test_running_with_no_windows_is_rejected():
    from investalyze.backtest.config import read as read_config

    with pytest.raises(ValueError, match='no walk-forward windows'):
        walkforward.run_walk_forward(None, None, [], read_config(), {})
