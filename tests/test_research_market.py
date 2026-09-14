"""Tests for market data fields computed from daily prices and sampled on as-of dates."""

import numpy as np
import pandas as pd
import pytest

from investalyze.research import market


def _wide(values: dict, periods: int) -> pd.DataFrame:
    index = pd.bdate_range('2020-01-01', periods=periods)
    return pd.DataFrame(values, index=index)


def test_ret_1m_is_the_return_over_21_trading_days():
    prices = _wide({'AAA': np.linspace(100, 121, 22)}, 22)
    fields = market.compute(['ret_1m'], adjusted=prices, raw=prices, volume=prices)
    assert fields['ret_1m']['AAA'].iloc[-1] == pytest.approx(0.21)


def test_ret_12_1_skips_the_most_recent_month():
    prices = pd.Series(100.0, index=range(253))
    prices.iloc[252 - 20:] = 200.0
    wide = _wide({'AAA': prices.to_numpy()}, 253)
    fields = market.compute(['ret_12_1', 'ret_12m'], adjusted=wide, raw=wide, volume=wide)
    # the jump sits inside the last 21 days, so 12-1 momentum excludes it and 12m includes it
    assert fields['ret_12_1']['AAA'].iloc[-1] == pytest.approx(0.0)
    assert fields['ret_12m']['AAA'].iloc[-1] == pytest.approx(1.0)


def test_volatility_is_annualised():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 300)
    prices = _wide({'AAA': 100 * np.cumprod(1 + returns)}, 300)
    fields = market.compute(['vol_252'], adjusted=prices, raw=prices, volume=prices)
    assert fields['vol_252']['AAA'].iloc[-1] == pytest.approx(0.01 * np.sqrt(252), rel=0.15)


def test_high_52w_is_one_at_a_new_high():
    prices = _wide({'AAA': np.linspace(50, 100, 260)}, 260)
    fields = market.compute(['high_52w'], adjusted=prices, raw=prices, volume=prices)
    assert fields['high_52w']['AAA'].iloc[-1] == pytest.approx(1.0)


def test_adv20_averages_dollar_volume_over_20_days():
    raw = _wide({'AAA': [10.0] * 20}, 20)
    volume = _wide({'AAA': [1000.0] * 20}, 20)
    fields = market.compute(['adv20'], adjusted=raw, raw=raw, volume=volume)
    assert fields['adv20']['AAA'].iloc[-1] == pytest.approx(10_000.0)


def test_only_requested_fields_are_computed():
    prices = _wide({'AAA': np.linspace(100, 121, 22)}, 22)
    assert set(market.compute(['ret_1m'], adjusted=prices, raw=prices, volume=prices)) == {'ret_1m'}


def test_an_unknown_market_field_is_refused():
    prices = _wide({'AAA': [1.0, 2.0]}, 2)
    with pytest.raises(ValueError, match='wibble'):
        market.compute(['wibble'], adjusted=prices, raw=prices, volume=prices)


def test_sampling_reads_the_last_session_on_or_before_each_as_of_date():
    frame = _wide({'AAA': [1.0, 2.0, 3.0, 4.0, 5.0]}, 5)
    saturday = frame.index[4] + pd.Timedelta(days=1)
    sampled = market.sample({'close': frame}, pd.DatetimeIndex([frame.index[2], saturday]))
    assert sampled['close'].tolist() == [3.0, 5.0]


def test_sampling_does_not_carry_a_price_forward_indefinitely():
    """A name that stopped trading must not keep its last price for months."""
    frame = _wide({'AAA': [1.0, 2.0]}, 2)
    much_later = frame.index[-1] + pd.Timedelta(days=90)
    sampled = market.sample({'close': frame}, pd.DatetimeIndex([much_later]))
    assert sampled.empty or np.isnan(sampled['close'].iloc[0])


def test_every_market_field_has_a_description():
    for name, description in market.FIELDS.items():
        assert description.endswith('.'), name


def test_the_bar_fields_pass_through_unchanged():
    prices = _wide({'AAA': [10.0, 11.0]}, 2)
    fields = market.compute(['open', 'high', 'low', 'adj_close'], adjusted=prices * 2, raw=prices, volume=prices,
                            open_=prices - 1, high=prices + 1, low=prices - 2)
    assert fields['open']['AAA'].tolist() == [9.0, 10.0]
    assert fields['high']['AAA'].tolist() == [11.0, 12.0]
    assert fields['low']['AAA'].tolist() == [8.0, 9.0]
    assert fields['adj_close']['AAA'].tolist() == [20.0, 22.0]
