"""Tests for the event-driven engine: fills, costs, cash accounting and the look-ahead guard.

Two tests carry most of the weight. The buy-and-hold identity checks the price space, the dividend
cash path and share accounting in one assertion, by holding one name with dividends reinvested and
requiring the equity curve to reproduce the adjusted-close total return exactly. The hand-worked
example pins slippage direction, the commission base and the sign of cash with numbers a reader can
check on paper.

Markets here are built in memory rather than from the database, so a session's open, high, low,
volume and dividend are exactly what the case needs.
"""

import numpy as np
import pandas as pd
import pytest

from investalyze.backtest import engine
from investalyze.backtest.book import Book
from investalyze.backtest.config import Costs, Limits
from investalyze.backtest.market import MarketData
from investalyze.backtest.orders import LIMIT, MARKET, STOP, OrderBatch
from investalyze.backtest.registry import Decision

FREE = Costs(slippage_bps=0.0, commission_bps=0.0, min_commission=0.0, impact_coef=0.0,
             borrow_bps_annual=0.0, cash_bps_annual=0.0)
OPEN_LIMITS = Limits(max_weight=1.0, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                     participation=1e12, max_order_age=5)


def _market(*, closes: np.ndarray, opens: np.ndarray | None = None, highs: np.ndarray | None = None,
            lows: np.ndarray | None = None, volume: float = 1e9, dividends: np.ndarray | None = None,
            adjusted: np.ndarray | None = None, tradable: np.ndarray | None = None,
            tickers: list[str] | None = None) -> MarketData:
    """A synthetic market from column arrays, defaulting to deep, always-tradable single-name data."""
    closes = np.atleast_2d(closes.T).T if closes.ndim > 1 else closes.reshape(-1, 1)
    sessions, assets = closes.shape
    opens = closes if opens is None else opens.reshape(sessions, assets)
    return MarketData(
        dates=pd.bdate_range('2020-01-01', periods=sessions),
        tickers=np.array(tickers if tickers else [f'T{i}' for i in range(assets)]),
        open_=opens.astype(float),
        high=(closes if highs is None else highs.reshape(sessions, assets)).astype(float),
        low=(closes if lows is None else lows.reshape(sessions, assets)).astype(float),
        close=closes.astype(float),
        volume=np.full((sessions, assets), volume, dtype=np.float32),
        dividend=(np.zeros((sessions, assets)) if dividends is None
                  else dividends.reshape(sessions, assets)).astype(float),
        adv=np.full((sessions, assets), 1e12, dtype=np.float32),
        tradable=(np.ones((sessions, assets), dtype=bool) if tradable is None
                  else tradable.reshape(sessions, assets)),
        adjusted_close=(closes if adjusted is None else adjusted.reshape(sessions, assets)).astype(float),
    )


def _fully_invested(session: int, book: Book) -> Decision:
    """Hold every name at full weight, which reinvests any cash the session produced."""
    return Decision(targets=np.array([1.0]), orders=None)


def _hold(session: int, book: Book) -> Decision:
    """Never trade."""
    return Decision(targets=None, orders=None)


def _order_once(qty: float, kind: int = MARKET, trigger: float = np.nan):
    """A signal that places one order on the first session and nothing afterwards."""
    def step(session: int, book: Book) -> Decision:
        if session != 0:
            return Decision(targets=None, orders=None)
        batch = OrderBatch(asset=np.array([0], dtype=np.int32), qty=np.array([qty]),
                           kind=np.array([kind], dtype=np.int8), trigger=np.array([trigger]),
                           age=np.array([0], dtype=np.int16))
        return Decision(targets=None, orders=batch)
    return step


# --- the accounting identities ---

def test_a_zero_cost_buy_and_hold_reproduces_the_adjusted_close_total_return():
    """Holding one name with dividends reinvested must track its adjusted close exactly.

    Each ex-date close is the prior close less the dividend, the condition under which reinvesting
    the cash at that close multiplies the share count by the same factor the back-adjustment divides
    by. Any error in the dividend path, the share count or the marking breaks the equality.
    """
    n = 12
    dividends = np.zeros(n)
    dividends[4], dividends[9] = 2.0, 3.0
    moves = [0.0, 2.0, -1.0, 4.0, 0.0, 3.0, 4.0, 2.0, -1.0, 0.0, 4.0, 3.0]
    closes = np.empty(n)
    closes[0] = 100.0
    for i in range(1, n):
        closes[i] = closes[i - 1] + moves[i] - dividends[i]

    factor = np.ones(n)
    paid = dividends > 0
    factor[paid] = 1 - dividends[paid] / closes[np.flatnonzero(paid) - 1]
    after = np.concatenate([np.cumprod(factor[::-1])[::-1][1:], [1.0]])
    adjusted = closes * after

    opens = np.concatenate([[closes[0]], closes[:-1]])   # no overnight gap, so a fill lands on the prior close
    market = _market(closes=closes, opens=opens, dividends=dividends, adjusted=adjusted)
    book = engine.run_backtest(market, _fully_invested, starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS, check_accounting=True)

    assert book.equity[-1] / book.equity[1] == pytest.approx(adjusted[-1] / adjusted[1], rel=1e-12)
    assert book.equity[-1] / book.equity[1] != pytest.approx(closes[-1] / closes[1])


def test_the_hand_worked_two_day_example_matches_the_arithmetic():
    """Cash 10,000, buy 50 shares filling at an open of 100.00, closing at 110.00.

    Fill 100.00 * 1.0001 = 100.01, notional 5,000.50, commission 5bps of that = 2.50025,
    cash 10,000 - 5,000.50 - 2.50025 = 4,996.99975, equity 4,996.99975 + 50 * 110 = 10,496.99975.
    """
    market = _market(closes=np.array([100.0, 110.0, 110.0]),
                     opens=np.array([100.0, 100.0, 110.0]))
    costs = Costs(slippage_bps=1.0, commission_bps=5.0, min_commission=0.0, impact_coef=0.0,
                  borrow_bps_annual=0.0, cash_bps_annual=0.0)
    book = engine.run_backtest(market, _order_once(50.0), starting_cash=10_000.0, costs=costs,
                               limits=OPEN_LIMITS, check_accounting=True)

    fill = book.fills[0]
    assert fill['Price'] == pytest.approx(100.01)
    assert fill['Notional'] == pytest.approx(5_000.50)
    assert fill['Cost'] == pytest.approx(2.50025)
    assert book.cash == pytest.approx(4_996.99975)
    assert book.equity[1] == pytest.approx(10_496.99975)


def test_cash_and_equity_reconcile_on_every_session():
    rng = np.random.default_rng(7)
    closes = 100 + rng.normal(scale=2.0, size=40).cumsum()
    dividends = np.zeros(40)
    dividends[[5, 20, 33]] = 0.5
    market = _market(closes=np.abs(closes) + 50, dividends=dividends)
    costs = Costs(slippage_bps=5.0, commission_bps=2.0, min_commission=1.0, impact_coef=0.1,
                  borrow_bps_annual=50.0, cash_bps_annual=100.0)
    book = engine.run_backtest(market, _fully_invested, starting_cash=100_000.0, costs=costs,
                               limits=OPEN_LIMITS, check_accounting=True)
    assert np.isfinite(book.equity).all()


# --- when orders fill ---

def test_the_engine_fills_orders_at_the_next_open_not_todays_close():
    market = _market(closes=np.array([100.0, 200.0, 200.0]), opens=np.array([100.0, 150.0, 200.0]))
    book = engine.run_backtest(market, _order_once(10.0), starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS)
    assert book.fills[0]['Price'] == pytest.approx(150.0)   # session 1's open, not session 0's close
    assert book.fills[0]['Date'] == market.dates[1]


def test_a_large_order_fills_only_the_participation_share_of_the_days_volume():
    market = _market(closes=np.array([100.0, 100.0, 100.0]), volume=1_000.0)
    limits = Limits(max_weight=1.0, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                    participation=0.1, max_order_age=5)
    book = engine.run_backtest(market, _order_once(1_000_000.0), starting_cash=1e9, costs=FREE, limits=limits)
    assert book.fills[0]['Shares'] == pytest.approx(100.0)   # 10% of the session's 1,000 shares


def test_the_unfilled_remainder_carries_to_the_following_session():
    market = _market(closes=np.array([100.0] * 4), volume=1_000.0)
    limits = Limits(max_weight=1.0, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                    participation=0.1, max_order_age=5)
    book = engine.run_backtest(market, _order_once(500.0), starting_cash=1e9, costs=FREE, limits=limits)
    assert len(book.fills) == 3                             # 100 shares on each remaining session
    assert sum(fill['Shares'] for fill in book.fills) == pytest.approx(300.0)


def test_an_order_is_cancelled_once_it_reaches_the_maximum_age():
    market = _market(closes=np.array([100.0] * 8), volume=1_000.0)
    limits = Limits(max_weight=1.0, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                    participation=0.1, max_order_age=2)
    book = engine.run_backtest(market, _order_once(500.0), starting_cash=1e9, costs=FREE, limits=limits)
    assert len(book.fills) == 2                             # cancelled before the third session


def test_a_zero_volume_session_never_fills():
    market = _market(closes=np.array([100.0, 100.0, 100.0]), volume=0.0)
    market = MarketData(**{**market.__dict__, 'tradable': np.zeros((3, 1), dtype=bool)})
    book = engine.run_backtest(market, _order_once(10.0), starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS)
    assert book.fills == []
    assert book.cash == pytest.approx(10_000.0)


def test_a_stop_gapping_through_its_trigger_fills_at_the_open():
    market = _market(closes=np.array([100.0, 120.0, 120.0]), opens=np.array([100.0, 118.0, 120.0]),
                     highs=np.array([100.0, 122.0, 120.0]), lows=np.array([100.0, 117.0, 120.0]))
    book = engine.run_backtest(market, _order_once(10.0, kind=STOP, trigger=105.0),
                               starting_cash=10_000.0, costs=FREE, limits=OPEN_LIMITS)
    assert book.fills[0]['Price'] == pytest.approx(118.0)   # the open, worse than the 105 trigger


def test_a_limit_never_fills_when_the_days_low_stays_above_it():
    market = _market(closes=np.array([100.0, 100.0, 100.0]), lows=np.array([100.0, 99.0, 99.0]))
    book = engine.run_backtest(market, _order_once(10.0, kind=LIMIT, trigger=50.0),
                               starting_cash=10_000.0, costs=FREE, limits=OPEN_LIMITS)
    assert book.fills == []


# --- holding positions through trouble ---

def test_a_price_gap_does_not_silently_liquidate_a_position():
    """A position established before a halt is still held after it, marked throughout."""
    closes = np.array([100.0, 100.0, np.nan, np.nan, 130.0])
    tradable = np.array([True, True, False, False, True]).reshape(-1, 1)
    market = _market(closes=closes, opens=np.array([100.0, 100.0, 100.0, 100.0, 130.0]),
                     tradable=tradable)
    book = engine.run_backtest(market, _order_once(10.0), starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS)
    assert book.shares[0] == pytest.approx(10.0)
    assert np.isfinite(book.equity).all()
    assert book.equity[3] == pytest.approx(book.cash + 10 * 100.0)   # held at the last print


def test_an_order_placed_into_a_halt_waits_for_the_session_that_reopens():
    closes = np.array([100.0, np.nan, np.nan, 130.0])
    tradable = np.array([True, False, False, True]).reshape(-1, 1)
    market = _market(closes=closes, opens=np.array([100.0, 100.0, 100.0, 130.0]), tradable=tradable)
    book = engine.run_backtest(market, _order_once(10.0), starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS)
    assert len(book.fills) == 1
    assert book.fills[0]['Date'] == market.dates[3]
    assert book.fills[0]['Price'] == pytest.approx(130.0)


def test_a_position_is_marked_at_the_last_known_price_across_a_gap():
    closes = np.array([100.0, 110.0, np.nan, 130.0])
    market = _market(closes=closes, opens=np.array([100.0, 100.0, 110.0, 130.0]))
    book = engine.run_backtest(market, _order_once(10.0), starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS)
    assert book.equity[2] == pytest.approx(book.cash + 10 * 110.0)


# --- limits ---

def test_gross_exposure_never_exceeds_the_configured_maximum():
    closes = np.column_stack([np.full(5, 100.0), np.full(5, 50.0)])
    market = _market(closes=closes, tickers=['AAA', 'BBB'])

    def greedy(session: int, book: Book) -> Decision:
        """Ask for three times the book in each of two names."""
        return Decision(targets=np.array([1.5, 1.5]), orders=None)

    limits = Limits(max_weight=0.4, max_gross=1.0, max_net=1.0, max_position_adv=1e12,
                    participation=1e12, max_order_age=5)
    book = engine.run_backtest(market, greedy, starting_cash=10_000.0, costs=FREE, limits=limits)
    invested = book.gross[1:] / book.equity[1:]
    assert invested.max() <= 0.4 * 2 + 1e-9


def test_dividends_are_credited_as_cash_to_a_held_position():
    dividends = np.zeros(4)
    dividends[2] = 5.0
    market = _market(closes=np.array([100.0, 100.0, 100.0, 100.0]), dividends=dividends)
    book = engine.run_backtest(market, _order_once(10.0), starting_cash=10_000.0, costs=FREE,
                               limits=OPEN_LIMITS)
    assert book.cash == pytest.approx(10_000.0 - 10 * 100.0 + 10 * 5.0)


def test_borrow_is_charged_on_a_short_position():
    market = _market(closes=np.array([100.0] * 5))
    costs = Costs(slippage_bps=0.0, commission_bps=0.0, min_commission=0.0, impact_coef=0.0,
                  borrow_bps_annual=252.0, cash_bps_annual=0.0)
    book = engine.run_backtest(market, _order_once(-10.0), starting_cash=10_000.0, costs=costs,
                               limits=OPEN_LIMITS)
    charged = 10_000.0 + 1_000.0 - book.cash
    assert charged == pytest.approx(3 * 1_000.0 * 252.0 / 1e4 / 252)   # three sessions short


# --- the look-ahead guard ---

def test_a_market_view_refuses_a_session_later_than_the_one_it_is_on():
    market = _market(closes=np.array([100.0, 110.0, 120.0]))
    view = engine.MarketView(market, session=1)
    assert view.row('close')[0] == pytest.approx(110.0)
    with pytest.raises(engine.LookAheadError, match='later than the current session'):
        view.row('close', 2)


def test_a_market_view_allows_every_session_up_to_the_current_one():
    market = _market(closes=np.array([100.0, 110.0, 120.0]))
    view = engine.MarketView(market, session=1)
    assert len(view.rows('close')) == 2
    with pytest.raises(engine.LookAheadError):
        view.rows('close', 2)


def test_a_backtest_needs_more_than_one_session():
    market = _market(closes=np.array([100.0]))
    with pytest.raises(ValueError, match='at least 2 sessions'):
        engine.run_backtest(market, _hold, starting_cash=1.0, costs=FREE, limits=OPEN_LIMITS)


# --- a name that stops trading ---

def _delisting_market(sessions: int = 6, last: int = 2) -> MarketData:
    """One name priced through `last`, then nothing: what a delisting looks like in stored data."""
    closes = np.array([10.0, 10.0, 12.0, np.nan, np.nan, np.nan])[:sessions]
    tradable = np.array([True] * (last + 1) + [False] * (sessions - last - 1))
    return _market(closes=closes, tradable=tradable, tickers=['GONE'])


def test_the_final_tradable_session_of_each_name_is_found():
    market = _delisting_market()
    assert list(engine.last_tradable_session(market)) == [2]


def test_a_name_that_never_trades_has_no_final_session():
    market = _market(closes=np.array([10.0, 10.0, 10.0]),
                     tradable=np.zeros((3, 1), dtype=bool), tickers=['NEVER'])
    assert list(engine.last_tradable_session(market)) == [-1]


def test_a_delisted_holding_becomes_cash_at_its_last_traded_price():
    """Otherwise the position is frozen at that price and can never be sold or redeployed."""
    market = _delisting_market()
    book = engine.run_backtest(market, _fully_invested, starting_cash=10_000.0,
                               costs=Costs(0, 0, 0, 0, 0, 0), limits=Limits(1, 1, 1, 1, 1, 5))

    assert book.shares[0] == 0
    assert book.cash == pytest.approx(book.equity[-1])


def test_the_delisting_is_recorded_as_a_fill_rather_than_happening_in_silence():
    market = _delisting_market()
    book = engine.run_backtest(market, _fully_invested, starting_cash=10_000.0,
                               costs=Costs(0, 0, 0, 0, 0, 0), limits=Limits(1, 1, 1, 1, 1, 5))

    delistings = [fill for fill in book.fills if fill['Kind'] == 'delisting']
    assert len(delistings) == 1
    assert delistings[0]['Ticker'] == 'GONE'
    assert delistings[0]['Price'] == pytest.approx(12.0)
    assert delistings[0]['Cost'] == 0.0


def test_the_liquidation_price_is_the_last_close_not_a_later_one():
    """A corporate event has no market price to fill against, so the last print is what there is."""
    market = _delisting_market()
    book = engine.run_backtest(market, _fully_invested, starting_cash=10_000.0,
                               costs=Costs(0, 0, 0, 0, 0, 0), limits=Limits(1, 1, 1, 1, 1, 5))

    delisting = next(fill for fill in book.fills if fill['Kind'] == 'delisting')
    assert delisting['Price'] == pytest.approx(market.close[2, 0])


def test_equity_does_not_jump_when_a_name_delists():
    """The position was already marked at the forward-filled close, so turning it to cash is neutral."""
    market = _delisting_market()
    book = engine.run_backtest(market, _hold, starting_cash=10_000.0,
                               costs=Costs(0, 0, 0, 0, 0, 0), limits=Limits(1, 1, 1, 1, 1, 5))

    assert book.equity[3] == pytest.approx(book.equity[2])


def test_a_name_still_trading_is_left_alone():
    market = _market(closes=np.array([10.0, 11.0, 12.0, 13.0]), tickers=['ALIVE'])
    book = engine.run_backtest(market, _fully_invested, starting_cash=10_000.0,
                               costs=Costs(0, 0, 0, 0, 0, 0), limits=Limits(1, 1, 1, 1, 1, 5))

    assert not [fill for fill in book.fills if fill['Kind'] == 'delisting']
    assert book.shares[0] > 0


def test_cash_still_reconciles_across_a_delisting():
    market = _delisting_market()
    engine.run_backtest(market, _fully_invested, starting_cash=10_000.0,
                        costs=Costs(0, 0, 0, 0, 0, 0), limits=Limits(1, 1, 1, 1, 1, 5),
                        check_accounting=True)
