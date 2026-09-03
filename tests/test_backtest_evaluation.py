"""Tests for factor evaluation: rank correlation, quantile spreads, decay and turnover.

The cases are built so the right answer is known without computing it: a score that orders the
forward returns perfectly must give an IC of 1, its reverse must give -1, and a score that is
constant through time must imply no turnover. The hand-computed Spearman pins the convention down on
a cross-section small enough to check on paper.
"""

import numpy as np
import pandas as pd
import pytest

from investalyze.backtest import evaluation

DATES = pd.DatetimeIndex(['2020-01-31', '2020-02-28', '2020-03-31'], name='Date')
TICKERS = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']


def _frame(rows: list[list[float]], dates: pd.DatetimeIndex = DATES) -> pd.DataFrame:
    """A wide sessions-by-tickers frame from literal rows."""
    return pd.DataFrame(rows, index=dates, columns=TICKERS, dtype=float)


@pytest.fixture
def scores() -> pd.DataFrame:
    """Scores rising left to right on every session, so AAA is always worst and EEE always best."""
    return _frame([[1.0, 2.0, 3.0, 4.0, 5.0]] * 3)


@pytest.fixture
def aligned_forwards() -> pd.DataFrame:
    """Forward returns ordered exactly like the scores."""
    return _frame([[0.01, 0.02, 0.03, 0.04, 0.05]] * 3)


# --- rank correlation ---

def test_rank_ic_is_one_for_a_score_that_perfectly_orders_the_forward_returns(scores, aligned_forwards):
    assert evaluation.rank_ic(scores, aligned_forwards).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_rank_ic_is_minus_one_for_a_perfectly_inverted_score(scores, aligned_forwards):
    assert evaluation.rank_ic(scores, aligned_forwards[TICKERS[::-1]].set_axis(TICKERS, axis=1)).tolist() \
        == pytest.approx([-1.0, -1.0, -1.0])


def test_rank_ic_matches_a_hand_computed_spearman_on_five_names():
    """Ranks 1..5 against ranks 1,2,4,3,5: two adjacent names swapped gives rho = 1 - 6*2/(5*24) = 0.9."""
    score = _frame([[1.0, 2.0, 3.0, 4.0, 5.0]], dates=DATES[:1])
    forward = _frame([[0.01, 0.02, 0.04, 0.03, 0.05]], dates=DATES[:1])
    assert evaluation.rank_ic(score, forward).iloc[0] == pytest.approx(0.9)


def test_rank_ic_ignores_sessions_with_fewer_than_two_names():
    score = _frame([[1.0, np.nan, np.nan, np.nan, np.nan]], dates=DATES[:1])
    forward = _frame([[0.01, 0.02, 0.03, 0.04, 0.05]], dates=DATES[:1])
    assert np.isnan(evaluation.rank_ic(score, forward).iloc[0])


def test_rank_ic_pairs_only_names_present_on_both_sides(scores):
    forward = _frame([[0.05, 0.04, 0.03, np.nan, np.nan]] * 3)
    # the three paired names are inverted, so the correlation is -1 despite the two gaps
    assert evaluation.rank_ic(scores, forward).tolist() == pytest.approx([-1.0, -1.0, -1.0])


def test_names_per_date_counts_the_paired_observations(scores):
    forward = _frame([[0.05, 0.04, 0.03, np.nan, np.nan]] * 3)
    assert evaluation.names_per_date(scores, forward).tolist() == [3, 3, 3]


# --- summarizing an IC series ---

def test_ic_summary_t_statistic_matches_mean_over_standard_error():
    ic = pd.Series([0.1, 0.2, 0.3, 0.4])
    summary = evaluation.ic_summary(ic)
    expected = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))
    assert summary['t_stat'] == pytest.approx(expected)
    assert summary['information_ratio'] == pytest.approx(ic.mean() / ic.std(ddof=1))
    assert summary['hit_rate'] == pytest.approx(1.0)
    assert summary['n_periods'] == 4


def test_ic_summary_of_an_empty_series_reports_no_periods():
    summary = evaluation.ic_summary(pd.Series(dtype=float))
    assert summary['n_periods'] == 0
    assert np.isnan(summary['mean'])


def test_ic_summary_hit_rate_counts_only_positive_periods():
    assert evaluation.ic_summary(pd.Series([0.1, -0.2, 0.3, -0.4]))['hit_rate'] == pytest.approx(0.5)


# --- forward returns ---

def test_forward_returns_measure_the_move_after_the_session_they_are_dated_on():
    closes = _frame([[100.0] * 5, [110.0] * 5, [121.0] * 5])
    forward = evaluation.forward_returns(closes, horizon=1)
    assert forward.iloc[0].tolist() == pytest.approx([0.1] * 5)
    assert forward.iloc[1].tolist() == pytest.approx([0.1] * 5)
    assert forward.iloc[2].isna().all()


def test_forward_returns_compound_over_a_longer_horizon():
    closes = _frame([[100.0] * 5, [110.0] * 5, [121.0] * 5])
    assert evaluation.forward_returns(closes, horizon=2).iloc[0].tolist() == pytest.approx([0.21] * 5)


def test_a_horizon_below_one_session_is_rejected():
    with pytest.raises(ValueError, match='horizon must be at least 1'):
        evaluation.forward_returns(_frame([[1.0] * 5]), horizon=0)


# --- quantiles ---

def test_quantile_returns_puts_the_highest_scores_in_the_top_bucket(scores, aligned_forwards):
    quantiles = evaluation.quantile_returns(scores, aligned_forwards, n_quantiles=5)
    assert quantiles.iloc[0][4] == pytest.approx(0.05)   # EEE, the best score
    assert quantiles.iloc[0][0] == pytest.approx(0.01)   # AAA, the worst


def test_quantile_returns_averages_the_names_that_share_a_bucket(scores, aligned_forwards):
    quantiles = evaluation.quantile_returns(scores, aligned_forwards, n_quantiles=2)
    assert quantiles.iloc[0][0] == pytest.approx(np.mean([0.01, 0.02]))
    assert quantiles.iloc[0][1] == pytest.approx(np.mean([0.03, 0.04, 0.05]))


def test_quantile_spread_is_the_top_bucket_minus_the_bottom(scores, aligned_forwards):
    quantiles = evaluation.quantile_returns(scores, aligned_forwards, n_quantiles=5)
    assert evaluation.quantile_spread(quantiles).tolist() == pytest.approx([0.04, 0.04, 0.04])


def test_fewer_than_two_quantiles_is_rejected(scores):
    with pytest.raises(ValueError, match='n_quantiles must be at least 2'):
        evaluation.quantile_buckets(scores, n_quantiles=1)


def test_quantile_buckets_span_zero_to_n_minus_one(scores):
    buckets = evaluation.quantile_buckets(scores, n_quantiles=5)
    assert buckets.iloc[0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


# --- decay, turnover, persistence ---

def test_ic_decay_reports_one_row_per_horizon(scores):
    closes = _frame([[100.0] * 5, [110.0] * 5, [121.0] * 5])
    decay = evaluation.ic_decay(scores, closes, horizons=(1, 2))
    assert decay['horizon'].tolist() == [1, 2]
    assert list(decay.columns) == ['horizon', 'mean_ic', 't_stat', 'n_periods']


def test_factor_turnover_is_zero_for_a_constant_score(scores):
    turnover = evaluation.factor_turnover(scores, n_quantiles=5)
    assert np.isnan(turnover.iloc[0])
    assert turnover.iloc[1:].tolist() == pytest.approx([0.0, 0.0])


def test_factor_turnover_is_one_when_the_top_bucket_is_replaced_entirely():
    flipping = _frame([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    assert evaluation.factor_turnover(flipping, n_quantiles=5).iloc[1:].tolist() == pytest.approx([1.0, 1.0])


def test_rank_autocorrelation_is_one_for_a_score_whose_ordering_never_changes(scores):
    autocorrelation = evaluation.rank_autocorrelation(scores)
    assert np.isnan(autocorrelation.iloc[0])
    assert autocorrelation.iloc[1:].tolist() == pytest.approx([1.0, 1.0])


def test_annualized_spread_scales_a_monthly_spread_to_a_year():
    spread = pd.Series([0.01, 0.01, 0.01])
    assert evaluation.annualized_spread(spread, horizon=21) == pytest.approx(0.01 * 252 / 21)


# --- widening a stored panel ---

def test_factor_scores_widens_a_long_panel():
    panel = pd.DataFrame({
        'Ticker': ['AAA', 'BBB', 'AAA', 'BBB'],
        'Date': pd.to_datetime(['2020-01-31', '2020-01-31', '2020-02-28', '2020-02-28']),
        'roe': [0.1, 0.2, 0.3, 0.4],
    })
    wide = evaluation.factor_scores(panel, 'roe')
    assert wide.shape == (2, 2)
    assert wide.loc[pd.Timestamp('2020-02-28'), 'BBB'] == pytest.approx(0.4)


def test_factor_scores_rejects_a_factor_the_panel_lacks():
    panel = pd.DataFrame({'Ticker': ['AAA'], 'Date': pd.to_datetime(['2020-01-31']), 'roe': [0.1]})
    with pytest.raises(ValueError, match='is not a column of this panel'):
        evaluation.factor_scores(panel, 'sharpe_of_the_moon')


# --- the array-shaped helpers shared with experiments/ ---

def test_spearman_by_group_averages_the_within_group_correlations():
    preds = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    targets = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    groups = np.array(['a', 'a', 'a', 'b', 'b', 'b'])
    assert evaluation.spearman_by_group(preds, targets, groups) == pytest.approx(0.0)


def test_spearman_by_group_skips_groups_too_small_to_correlate():
    preds = np.array([1.0, 2.0, 3.0, 9.0])
    targets = np.array([1.0, 2.0, 3.0, 9.0])
    groups = np.array(['a', 'a', 'a', 'b'])
    assert evaluation.spearman_by_group(preds, targets, groups) == pytest.approx(1.0)


def test_spearman_by_group_is_nan_when_no_group_is_large_enough():
    assert np.isnan(evaluation.spearman_by_group(np.array([1.0]), np.array([1.0]), np.array(['a'])))


def test_relevance_grades_rank_each_group_independently():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 1.0, 2.0])
    groups = np.array(['a'] * 5 + ['b'] * 2)
    grades = evaluation.relevance_grades(y, groups, n_grades=5)
    assert grades[:5].tolist() == sorted(grades[:5].tolist())
    assert grades[5] < grades[6]


def test_relevance_grades_put_the_top_two_percentiles_in_the_same_grade():
    """The percentile of the best name is exactly 1.0, so it lands on the grade below the ceiling.

    This is the grading the ranking experiments were trained against, kept as it was so their
    recorded runs stay comparable.
    """
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    groups = np.array(['a'] * 5)
    assert evaluation.relevance_grades(y, groups, n_grades=5).tolist() == [1, 2, 3, 4, 4]


def test_precision_at_k_is_one_when_the_predicted_top_matches_the_true_top():
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array(['a'] * 4)
    assert evaluation.precision_at_k(preds, targets, groups, k_frac=0.5) == pytest.approx(1.0)


def test_precision_at_k_is_zero_when_the_prediction_is_inverted():
    preds = np.array([4.0, 3.0, 2.0, 1.0])
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array(['a'] * 4)
    assert evaluation.precision_at_k(preds, targets, groups, k_frac=0.5) == pytest.approx(0.0)
