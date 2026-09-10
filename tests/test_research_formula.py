"""Tests for the idea formula language: what it accepts, what it refuses, and what it computes."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from investalyze.research import formula

JAN, FEB = date(2020, 1, 31), date(2020, 2, 29)


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two dates, three names, with a sector so grouped operations have something to group on."""
    rows = []
    values = {('AAA', JAN): (10.0, 2.0), ('BBB', JAN): (20.0, 4.0), ('CCC', JAN): (30.0, 0.0),
              ('AAA', FEB): (11.0, 2.0), ('BBB', FEB): (25.0, 5.0), ('CCC', FEB): (30.0, 1.0)}
    sectors = {'AAA': 'tech', 'BBB': 'tech', 'CCC': 'energy'}
    for (ticker, when), (revenue, income) in values.items():
        rows.append({'Ticker': ticker, 'Date': when, 'revenue': revenue, 'income': income,
                     'sector': sectors[ticker]})
    return pd.DataFrame(rows).sort_values(['Date', 'Ticker']).reset_index(drop=True)


# --- what the language accepts ---

def test_arithmetic_combines_columns(panel):
    result = formula.evaluate('income / revenue', panel)
    assert result.tolist() == pytest.approx([0.2, 0.2, 0.0, 2 / 11, 0.2, 1 / 30])


def test_precedence_follows_python(panel):
    assert formula.evaluate('revenue + income * 2', panel).tolist() == [14.0, 28.0, 30.0, 15.0, 35.0, 32.0]


def test_a_comparison_yields_a_boolean(panel):
    assert formula.evaluate('revenue > 20', panel).tolist() == [False, False, True, False, True, True]


def test_where_picks_between_two_branches(panel):
    result = formula.evaluate('where(revenue > 20, 1, 0)', panel)
    assert result.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def test_mask_drops_a_name_rather_than_scoring_it(panel):
    result = formula.evaluate('mask(revenue, income > 3)', panel)
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[2])
    assert result.iloc[1] == pytest.approx(20.0)


def test_a_masked_name_leaves_the_cross_section(panel):
    """The difference from where(cond, x, 0): a zero still competes in the ranking, a NaN does not."""
    masked = formula.evaluate('rank(mask(revenue, income > 3))', panel)
    zeroed = formula.evaluate('rank(where(income > 3, revenue, 0))', panel)
    assert np.isnan(masked.iloc[0])
    assert np.isfinite(zeroed.iloc[0])


def test_a_ternary_reads_as_a_conditional(panel):
    result = formula.evaluate('1 if revenue > 20 else 0', panel)
    assert result.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def test_boolean_operators_combine_conditions(panel):
    result = formula.evaluate('(revenue > 15) and (income > 3)', panel)
    assert result.tolist() == [False, True, False, False, True, False]


# --- what it refuses ---

def test_an_unknown_name_is_refused_by_name(panel):
    with pytest.raises(ValueError, match='unknown name.*margin'):
        formula.validate('margin * 2', known={'revenue', 'income'})


def test_an_unknown_function_is_refused(panel):
    with pytest.raises(ValueError, match='unknown function.*wibble'):
        formula.validate('wibble(revenue)', known={'revenue'})


def test_attribute_access_is_refused():
    with pytest.raises(ValueError, match='not allowed'):
        formula.validate('revenue.__class__', known={'revenue'})


def test_a_lambda_is_refused():
    with pytest.raises(ValueError, match='not allowed'):
        formula.validate('(lambda: 1)()', known=set())


def test_a_call_on_a_non_function_is_refused():
    with pytest.raises(ValueError, match='not allowed|unknown function'):
        formula.validate('revenue(1)', known={'revenue'})


def test_syntax_errors_say_so():
    with pytest.raises(ValueError, match='could not be parsed'):
        formula.validate('revenue +', known={'revenue'})


# --- dependencies ---

def test_dependencies_lists_referenced_names():
    assert formula.dependencies('rank(a) + b / c') == {'a', 'b', 'c'}


def test_function_names_are_not_dependencies():
    assert 'rank' not in formula.dependencies('rank(a)')


def test_a_keyword_argument_naming_a_column_is_a_dependency():
    assert formula.dependencies('neutralize(a, by=sector)') == {'a', 'sector'}


# --- cross-sectional functions group by date ---

def test_rank_is_a_percentile_within_each_date(panel):
    result = formula.evaluate('rank(revenue)', panel)
    assert result.tolist() == pytest.approx([0.0, 50.0, 100.0, 0.0, 50.0, 100.0])


def test_zscore_is_computed_within_each_date(panel):
    result = formula.evaluate('zscore(revenue)', panel)
    assert result.groupby(panel['Date']).mean().tolist() == pytest.approx([0.0, 0.0])


def test_neutralize_removes_the_group_mean(panel):
    result = formula.evaluate('neutralize(revenue, by=sector)', panel)
    # tech on JAN is 10 and 20, so a mean of 15 leaves -5 and +5; energy is alone, so zero
    assert result.tolist() == pytest.approx([-5.0, 5.0, 0.0, -7.0, 7.0, 0.0])


def test_winsorize_clips_the_tails_within_a_date(panel):
    result = formula.evaluate('winsorize(revenue, 0.34)', panel)
    assert result.max() < 30.0


# --- time-series functions group by ticker ---

def test_lag_reads_the_previous_date_for_the_same_name(panel):
    result = formula.evaluate('lag(revenue, 1)', panel)
    assert np.isnan(result.tolist()[:3]).all()
    assert result.tolist()[3:] == pytest.approx([10.0, 20.0, 30.0])


def test_delta_is_the_change_against_an_earlier_date(panel):
    result = formula.evaluate('delta(revenue, 1)', panel)
    assert result.tolist()[3:] == pytest.approx([1.0, 5.0, 0.0])


# --- safety ---

def test_safe_div_yields_nan_rather_than_infinity(panel):
    result = formula.evaluate('safe_div(revenue, income)', panel)
    assert np.isnan(result.iloc[2])
    assert result.iloc[0] == pytest.approx(5.0)


def test_plain_division_by_zero_also_yields_nan(panel):
    result = formula.evaluate('revenue / income', panel)
    assert np.isnan(result.iloc[2])


# --- ideas may reference other ideas, which are simply columns by then ---

def test_an_idea_referencing_another_idea_reads_its_column(panel):
    enriched = panel.assign(margin=panel['income'] / panel['revenue'])
    # AAA and BBB both margin 0.2 on the first date, so they share the averaged rank
    assert formula.evaluate('rank(margin)', enriched).tolist() == pytest.approx(
        [75.0, 75.0, 0.0, 50.0, 100.0, 0.0])
