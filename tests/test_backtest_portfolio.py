"""Tests for portfolio construction: scores in, target weights out.

Every case is built so the expected weights can be read off by hand. The cap tests matter most: a
single clip-and-renormalize silently breaches the cap it just applied, so the property under test is
that no weight exceeds the cap after the function returns, not that it was clipped once.
"""

import numpy as np
import pytest

from investalyze.backtest import portfolio

SCORES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
SECTORS = np.array(['tech', 'tech', 'tech', 'energy', 'energy'])


# --- standardizing ---

def test_zscore_centres_a_cross_section_on_zero():
    z = portfolio.zscore(SCORES)
    assert z.mean() == pytest.approx(0.0)
    assert z.std(ddof=1) == pytest.approx(1.0)


def test_zscore_leaves_nan_where_the_score_was_missing():
    z = portfolio.zscore(np.array([1.0, np.nan, 3.0]))
    assert np.isnan(z[1])
    assert np.isfinite(z[[0, 2]]).all()


def test_zscore_of_a_constant_cross_section_is_all_zero():
    assert portfolio.zscore(np.array([5.0, 5.0, 5.0])).tolist() == [0.0, 0.0, 0.0]


def test_winsorize_clips_the_tails_without_moving_the_median():
    scores = np.array([-100.0, 1.0, 2.0, 3.0, 100.0])
    clipped = portfolio.winsorize(scores, limit=0.25)
    assert np.median(clipped) == pytest.approx(np.median(scores))
    assert clipped.min() > -100.0
    assert clipped.max() < 100.0


def test_winsorize_with_a_zero_limit_changes_nothing():
    assert portfolio.winsorize(SCORES, limit=0.0).tolist() == SCORES.tolist()


def test_a_winsorize_limit_of_a_half_or_more_is_rejected():
    with pytest.raises(ValueError, match='limit must be in'):
        portfolio.winsorize(SCORES, limit=0.5)


def test_percentile_rank_spans_zero_to_one():
    assert portfolio.percentile_rank(SCORES).tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


# --- neutralizing ---

def test_neutralize_groups_leaves_every_group_mean_at_zero():
    neutral = portfolio.neutralize_groups(SCORES, SECTORS)
    assert neutral[:3].mean() == pytest.approx(0.0)
    assert neutral[3:].mean() == pytest.approx(0.0)


def test_neutralize_linear_removes_the_exposure_it_is_given():
    exposure = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    residual = portfolio.neutralize_linear(2.0 * exposure + 1.0, exposure)
    assert residual == pytest.approx(np.zeros(5), abs=1e-9)


def test_neutralize_linear_keeps_the_part_the_exposure_does_not_explain():
    exposure = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scores = np.array([1.0, 5.0, 2.0, 6.0, 3.0])
    residual = portfolio.neutralize_linear(scores, exposure)
    assert np.abs(residual).sum() > 0


# --- forming weights ---

def test_quantile_weights_puts_equal_weight_on_every_name_in_the_top_bucket():
    weights = portfolio.quantile_weights(SCORES, n_quantiles=2)
    chosen = weights > 0
    assert weights[chosen].tolist() == pytest.approx([1 / chosen.sum()] * chosen.sum())
    assert weights.sum() == pytest.approx(1.0)


def test_quantile_weights_leaves_the_middle_buckets_flat():
    weights = portfolio.quantile_weights(SCORES, n_quantiles=5)
    assert weights[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert weights[4] == pytest.approx(1.0)


def test_quantile_weights_can_add_a_short_side_summing_to_minus_one():
    weights = portfolio.quantile_weights(SCORES, n_quantiles=5, long_quantile=4, short_quantile=0)
    assert weights[weights > 0].sum() == pytest.approx(1.0)
    assert weights[weights < 0].sum() == pytest.approx(-1.0)
    assert weights.sum() == pytest.approx(0.0)


def test_weights_of_an_all_nan_score_row_are_all_zero():
    weights = portfolio.quantile_weights(np.full(5, np.nan))
    assert weights.tolist() == [0.0] * 5


def test_fewer_than_two_quantiles_is_rejected():
    with pytest.raises(ValueError, match='n_quantiles must be at least 2'):
        portfolio.quantile_weights(SCORES, n_quantiles=1)


def test_score_weights_with_power_zero_is_equal_weight():
    weights = portfolio.score_weights(SCORES, top_n=3, power=0.0)
    assert weights[2:].tolist() == pytest.approx([1 / 3] * 3)
    assert weights[:2].tolist() == [0.0, 0.0]


def test_score_weights_with_a_positive_power_favours_the_best_name():
    weights = portfolio.score_weights(SCORES, top_n=3, power=2.0)
    assert weights[4] > weights[3] > weights[2]
    assert weights.sum() == pytest.approx(1.0)


def test_score_weights_selects_at_most_the_names_that_have_scores():
    weights = portfolio.score_weights(np.array([1.0, np.nan, np.nan]), top_n=3)
    assert np.count_nonzero(weights) == 1


# --- caps and limits ---

def test_cap_weights_never_leaves_a_position_above_the_maximum():
    weights = portfolio.cap_weights(np.array([0.6, 0.3, 0.1]), max_weight=0.4)
    assert weights.max() <= 0.4 + 1e-12
    assert weights.sum() == pytest.approx(1.0)


def test_cap_weights_holds_the_cap_when_several_positions_breach_it():
    weights = portfolio.cap_weights(np.array([0.5, 0.4, 0.05, 0.05]), max_weight=0.3)
    assert weights.max() <= 0.3 + 1e-12


def test_cap_weights_raises_when_the_cap_makes_full_investment_impossible():
    with pytest.raises(ValueError, match='cannot reach full investment'):
        portfolio.cap_weights(np.array([0.5, 0.5]), max_weight=0.1)


def test_cap_weights_scales_gross_exposure_down_to_its_limit():
    weights = portfolio.cap_weights(np.array([1.0, 1.0]), max_weight=1.0, max_gross=1.0)
    assert np.abs(weights).sum() == pytest.approx(1.0)


def test_cap_weights_scales_net_exposure_down_to_its_limit():
    weights = portfolio.cap_weights(np.array([0.5, 0.5]), max_weight=1.0, max_gross=2.0, max_net=0.4)
    assert weights.sum() == pytest.approx(0.4)


def test_cap_weights_treats_a_nan_weight_as_no_position():
    weights = portfolio.cap_weights(np.array([0.5, np.nan, 0.5]), max_weight=1.0)
    assert weights[1] == 0.0


# --- sizing ---

def test_vol_target_scales_gross_to_hit_the_requested_volatility():
    weights = np.array([0.5, 0.5])
    vol = np.array([0.2, 0.2])
    scaled = portfolio.vol_target(weights, vol, target_annual_vol=0.1)
    achieved = np.sqrt(np.square(scaled * vol).sum())
    assert achieved == pytest.approx(0.1)


def test_vol_target_leaves_an_empty_book_alone():
    scaled = portfolio.vol_target(np.zeros(3), np.array([0.2, 0.2, 0.2]), target_annual_vol=0.1)
    assert scaled.tolist() == [0.0, 0.0, 0.0]


def test_a_non_positive_volatility_target_is_rejected():
    with pytest.raises(ValueError, match='target_annual_vol must be positive'):
        portfolio.vol_target(np.array([1.0]), np.array([0.2]), target_annual_vol=0.0)


def test_sector_neutral_gives_each_sector_the_same_gross_weight():
    weights = portfolio.sector_neutral(np.array([0.5, 0.3, 0.2]), np.array(['a', 'a', 'b']))
    assert np.abs(weights[:2]).sum() == pytest.approx(np.abs(weights[2:]).sum())
    assert np.abs(weights).sum() == pytest.approx(1.0)


def test_sector_neutral_leaves_an_empty_book_alone():
    assert portfolio.sector_neutral(np.zeros(3), SECTORS[:3]).tolist() == [0.0, 0.0, 0.0]


def test_the_functions_chain_from_raw_scores_to_capped_weights():
    """The path a cross-sectional signal actually walks, ending inside every stated limit."""
    raw = np.array([1.0, np.nan, -50.0, 4.0, 5.0, 2.0, 3.0, 50.0])
    sectors = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
    scores = portfolio.neutralize_groups(portfolio.zscore(portfolio.winsorize(raw, limit=0.1)), sectors)
    weights = portfolio.cap_weights(portfolio.quantile_weights(scores, n_quantiles=4),
                                    max_weight=0.5, max_gross=1.0, max_net=1.0)
    assert np.abs(weights).max() <= 0.5 + 1e-12
    assert np.abs(weights).sum() <= 1.0 + 1e-12
    assert np.isfinite(weights).all()
