"""Tests for user-defined factors: parsing, validation, and the two ways one gets computed.

Definitions are built explicitly here rather than read from the repo's factors.toml, so what these
assert does not move when someone adds a factor of their own.

The ranking case is the one worth reading. An expression that references `rank_x` compares a name to
its own cross-section, so the same raw value can rank high on one date and low on another; the test
plants two dates on different scales to prove they are ranked apart rather than pooled.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from investalyze.analysis import custom_factors
from investalyze.analysis.factors import BUILTIN_FACTORS, BUILTIN_HIGHER_IS_BETTER

ORIENTATIONS = BUILTIN_HIGHER_IS_BETTER


def _parse(text: str) -> dict[str, custom_factors.CustomFactor]:
    """Parse definitions against the built-in factor library."""
    return custom_factors.parse(textwrap.dedent(text), BUILTIN_FACTORS)


# --- parsing and validation ---

def test_an_expression_factor_is_parsed_with_its_orientation_and_description():
    defined = _parse("""
        [factors.blend]
        expression = '0.5 * roe + 0.5 * roa'
        higher_is_better = false
        description = 'a blend'
    """)
    factor = defined['blend']
    assert factor.kind == custom_factors.EXPRESSION
    assert factor.expression == '0.5 * roe + 0.5 * roa'
    assert factor.higher_is_better is False
    assert factor.description == 'a blend'


def test_a_ratio_factor_is_parsed_with_both_of_its_terms():
    defined = _parse("""
        [factors.turnover]
        numerator = 'Revenue'
        denominator = 'Total Assets'
    """)
    factor = defined['turnover']
    assert factor.kind == custom_factors.RATIO
    assert factor.terms == ('Revenue', 'Total Assets')
    assert factor.higher_is_better is True


def test_an_empty_file_defines_nothing():
    assert _parse('') == {}


def test_an_expression_referencing_an_unknown_column_is_rejected():
    with pytest.raises(ValueError, match='references unknown column'):
        _parse("[factors.bad]\nexpression = 'roe + not_a_factor'")


def test_a_function_call_in_an_expression_is_rejected():
    with pytest.raises(ValueError, match='references unknown column'):
        _parse("[factors.bad]\nexpression = 'sqrt(roe)'")


def test_an_expression_may_use_a_rank_column():
    assert 'ranked' in _parse("[factors.ranked]\nexpression = 'rank_roe - rank_vol_252'")


def test_an_expression_may_use_an_identity_column():
    assert 'sized' in _parse("[factors.sized]\nexpression = 'roe / mcap'")


def test_reusing_a_built_in_factor_name_is_rejected():
    with pytest.raises(ValueError, match='already a built-in factor'):
        _parse("[factors.roe]\nexpression = 'roa'")


def test_a_name_that_cannot_be_a_column_is_rejected():
    with pytest.raises(ValueError, match='usable as a column name'):
        _parse("[factors.'not a name']\nexpression = 'roe'")


def test_a_factor_with_neither_form_is_rejected():
    with pytest.raises(ValueError, match='needs either an expression'):
        _parse("[factors.empty]\ndescription = 'nothing here'")


def test_a_factor_with_both_forms_is_rejected():
    with pytest.raises(ValueError, match='both an expression and a numerator'):
        _parse("[factors.both]\nexpression = 'roe'\nnumerator = 'Revenue'\ndenominator = 'Revenue'")


def test_reading_an_absent_file_defines_nothing():
    assert custom_factors.read(Path('/nowhere/factors.toml'), BUILTIN_FACTORS) == {}
    assert custom_factors.read(None, BUILTIN_FACTORS) == {}


def test_orientation_merges_the_custom_factors_over_the_built_ins():
    defined = _parse("[factors.risky]\nexpression = 'vol_252'\nhigher_is_better = false")
    merged = custom_factors.orientation(defined, BUILTIN_HIGHER_IS_BETTER)
    assert merged['risky'] is False
    assert merged['roe'] is True


# --- ranking ---

def test_ranks_are_oriented_so_a_hundred_is_always_the_good_end():
    frame = pd.DataFrame({'roe': [0.1, 0.3], 'vol_252': [0.1, 0.3]})
    ranked = custom_factors.with_ranks(frame, BUILTIN_FACTORS, ORIENTATIONS)
    assert ranked['rank_roe'].tolist() == [50.0, 100.0]        # higher roe is better
    assert ranked['rank_vol_252'].tolist() == [100.0, 50.0]    # lower volatility is better


def test_ranks_are_taken_within_each_group_when_a_grouping_column_is_given():
    frame = pd.DataFrame({'Date': ['d1', 'd1', 'd2', 'd2'], 'roe': [0.01, 0.02, 100.0, 200.0]})
    ranked = custom_factors.with_ranks(frame, BUILTIN_FACTORS, ORIENTATIONS, by='Date')
    assert ranked['rank_roe'].tolist() == [50.0, 100.0, 50.0, 100.0]


def test_ranking_skips_a_factor_the_frame_does_not_carry():
    ranked = custom_factors.with_ranks(pd.DataFrame({'roe': [0.1]}), BUILTIN_FACTORS, ORIENTATIONS)
    assert 'rank_roe' in ranked.columns
    assert 'rank_vol_252' not in ranked.columns


# --- expression factors ---

def test_an_expression_factor_is_computed_from_the_frames_own_columns():
    defined = _parse("[factors.blend]\nexpression = '0.5 * roe + 0.5 * roa'")
    frame = pd.DataFrame({'roe': [0.2, 0.4], 'roa': [0.1, 0.2]})
    out = custom_factors.apply_expressions(frame, defined, BUILTIN_FACTORS, ORIENTATIONS)
    assert out['blend'].tolist() == pytest.approx([0.15, 0.30])


def test_applying_expressions_adds_only_the_custom_columns():
    defined = _parse("[factors.blend]\nexpression = 'roe + roa'")
    frame = pd.DataFrame({'roe': [0.2], 'roa': [0.1]})
    out = custom_factors.apply_expressions(frame, defined, BUILTIN_FACTORS, ORIENTATIONS)
    assert [column for column in out.columns if column not in frame.columns] == ['blend']


def test_an_expression_over_ranks_compares_each_date_separately():
    defined = _parse("[factors.momentum_rank]\nexpression = 'rank_ret_12_1'")
    frame = pd.DataFrame({'Date': ['d1', 'd1', 'd2', 'd2'], 'ret_12_1': [0.01, 0.02, 5.0, 9.0]})
    out = custom_factors.apply_expressions(frame, defined, BUILTIN_FACTORS, ORIENTATIONS, by='Date')
    assert out['momentum_rank'].tolist() == [50.0, 100.0, 50.0, 100.0]


def test_a_division_by_zero_in_an_expression_becomes_missing_rather_than_infinite():
    defined = _parse("[factors.ratio]\nexpression = 'roe / vol_252'")
    frame = pd.DataFrame({'roe': [0.2, 0.2], 'vol_252': [0.1, 0.0]})
    out = custom_factors.apply_expressions(frame, defined, BUILTIN_FACTORS, ORIENTATIONS)
    assert out['ratio'].iloc[0] == pytest.approx(2.0)
    assert np.isnan(out['ratio'].iloc[1])


def test_a_frame_missing_a_referenced_column_fails_with_the_factor_named():
    defined = _parse("[factors.blend]\nexpression = 'roe + roa'")
    with pytest.raises(ValueError, match="factor 'blend' failed to evaluate"):
        custom_factors.apply_expressions(pd.DataFrame({'roe': [0.1]}), defined,
                                         BUILTIN_FACTORS, ORIENTATIONS)


def test_applying_no_definitions_returns_the_frame_unchanged():
    frame = pd.DataFrame({'roe': [0.1]})
    assert custom_factors.apply_expressions(frame, {}, BUILTIN_FACTORS, ORIENTATIONS) is frame


# --- ratio factors ---

@pytest.fixture
def schema() -> dict[str, set[str]]:
    """A statement schema small enough to reason about, with one column in two statements."""
    return {
        'income': {'Ticker', 'Revenue', 'Research & Development', 'Net Income'},
        'cashflow': {'Ticker', 'Net Cash from Operating Activities', 'Net Income'},
        'balance': {'Ticker', 'Total Assets', 'Total Equity'},
    }


def test_a_ratio_term_is_resolved_to_the_statement_holding_it(schema):
    defined = _parse("[factors.turnover]\nnumerator = 'Revenue'\ndenominator = 'Total Assets'")
    resolved = custom_factors.resolve_terms(defined, schema)
    assert resolved == {'Revenue': 'income', 'Total Assets': 'balance'}


def test_a_term_no_statement_holds_is_rejected(schema):
    defined = _parse("[factors.bad]\nnumerator = 'Vibes'\ndenominator = 'Revenue'")
    with pytest.raises(ValueError, match="names 'Vibes', which is not a column"):
        custom_factors.resolve_terms(defined, schema)


def test_a_term_two_statements_hold_is_rejected_as_ambiguous(schema):
    defined = _parse("[factors.bad]\nnumerator = 'Net Income'\ndenominator = 'Revenue'")
    with pytest.raises(ValueError, match='the term is ambiguous'):
        custom_factors.resolve_terms(defined, schema)


def test_a_term_the_built_in_factors_already_pull_reuses_their_alias(schema):
    defined = _parse("[factors.turnover]\nnumerator = 'Revenue'\ndenominator = 'Total Assets'")
    resolved = custom_factors.resolve_terms(defined, schema)
    existing = {'income': {'Revenue': 'revenue'}, 'cashflow': {}, 'balance': {'Total Assets': 'assets'}}
    aliases = custom_factors.term_aliases(defined, resolved, existing)
    assert aliases == {'Revenue': 'revenue', 'Total Assets': 'assets'}
    assert custom_factors.extra_columns(defined, resolved, aliases, existing) == {
        'income': {}, 'cashflow': {}, 'balance': {}}


def test_a_term_the_built_ins_do_not_pull_gets_its_own_alias(schema):
    defined = _parse("[factors.intensity]\nnumerator = 'Research & Development'\ndenominator = 'Revenue'")
    resolved = custom_factors.resolve_terms(defined, schema)
    existing = {'income': {'Revenue': 'revenue'}, 'cashflow': {}, 'balance': {}}
    aliases = custom_factors.term_aliases(defined, resolved, existing)
    assert aliases['Research & Development'] == 'term_research_development'
    extras = custom_factors.extra_columns(defined, resolved, aliases, existing)
    assert extras['income'] == {'Research & Development': 'term_research_development'}


def test_a_ratio_factor_divides_the_merged_terms(schema):
    defined = _parse("[factors.turnover]\nnumerator = 'Revenue'\ndenominator = 'Total Assets'")
    resolved = custom_factors.resolve_terms(defined, schema)
    aliases = {'Revenue': 'revenue', 'Total Assets': 'assets'}
    frame = pd.DataFrame({'revenue_ttm': [100.0, 50.0], 'assets': [200.0, 25.0]})
    out = custom_factors.apply_ratios(frame, defined, resolved, aliases)
    assert out['turnover'].tolist() == pytest.approx([0.5, 2.0])


def test_a_ratio_with_a_non_positive_denominator_is_missing(schema):
    defined = _parse("[factors.turnover]\nnumerator = 'Revenue'\ndenominator = 'Total Assets'")
    resolved = custom_factors.resolve_terms(defined, schema)
    aliases = {'Revenue': 'revenue', 'Total Assets': 'assets'}
    frame = pd.DataFrame({'revenue_ttm': [100.0, 100.0], 'assets': [0.0, -50.0]})
    out = custom_factors.apply_ratios(frame, defined, resolved, aliases)
    assert out['turnover'].isna().all()


def test_a_leading_minus_reads_an_expense_as_a_positive_magnitude(schema):
    """SimFin stores research spend as a negative number, so the sign has to be requestable."""
    defined = _parse("[factors.intensity]\nnumerator = '-Research & Development'\ndenominator = 'Revenue'")
    resolved = custom_factors.resolve_terms(defined, schema)
    aliases = {'Research & Development': 'term_research_development', 'Revenue': 'revenue'}
    frame = pd.DataFrame({'term_research_development_ttm': [-30.0], 'revenue_ttm': [100.0]})
    out = custom_factors.apply_ratios(frame, defined, resolved, aliases)
    assert out['intensity'].iloc[0] == pytest.approx(0.3)


def test_stripping_a_sign_leaves_the_column_name():
    assert custom_factors.strip_sign('-Research & Development') == ('Research & Development', -1.0)
    assert custom_factors.strip_sign('Revenue') == ('Revenue', 1.0)


def test_a_column_alias_is_usable_as_an_identifier():
    assert custom_factors.column_alias('Research & Development').isidentifier()
    assert custom_factors.column_alias('Total Liabilities & Equity').isidentifier()


def test_the_statement_schema_is_read_from_the_database():
    con = duckdb.connect()
    con.execute('CREATE TABLE income (Ticker VARCHAR, Revenue BIGINT)')
    con.execute('CREATE TABLE cashflow (Ticker VARCHAR, "Net Change in Cash" BIGINT)')
    con.execute('CREATE TABLE balance (Ticker VARCHAR, "Total Assets" BIGINT)')
    schema = custom_factors.statement_schema(con)
    assert schema['income'] == {'Ticker', 'Revenue'}
    assert 'Total Assets' in schema['balance']


# --- the merge into the factor library ---

def test_a_factors_file_adds_its_definitions_to_the_library(tmp_path: Path):
    """Run in a subprocess, since the merge happens once when analysis.factors is imported."""
    definitions = tmp_path / 'factors.toml'
    definitions.write_text("[factors.my_blend]\nexpression = '0.5 * roe + 0.5 * roa'\n"
                           "higher_is_better = false\ndescription = 'mine'\n")
    script = (
        'from investalyze.analysis import factors\n'
        "print(factors.FACTORS[-1])\n"
        "print(factors.HIGHER_IS_BETTER['my_blend'])\n"
        "print(factors.FAMILIES['custom'])\n"
        "print(factors.FACTOR_DOCS['my_blend'].formula)\n"
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env={'INVESTALYZE_FACTORS_FILE': str(definitions), 'PATH': '/usr/bin'},
                            cwd=Path.cwd())
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == 'my_blend'
    assert lines[1] == 'False'
    assert lines[2] == "['my_blend']"
    assert lines[3] == '0.5 * roe + 0.5 * roa'


def test_switching_the_factors_file_off_leaves_the_built_in_library(tmp_path: Path):
    script = ('from investalyze.analysis import factors\n'
              'print(len(factors.FACTORS))\nprint(list(factors.FAMILIES))\n')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env={'INVESTALYZE_FACTORS_FILE': '', 'PATH': '/usr/bin'}, cwd=Path.cwd())
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == str(len(BUILTIN_FACTORS))
    assert 'custom' not in lines[1]

# --- noticing a definition that changed under a stored panel ---

def _factor(expression: str) -> custom_factors.CustomFactor:
    """One expression definition, parsed the way the file would give it."""
    text = custom_factors.serialize_factor('blend', {'expression': expression,
                                                     'higher_is_better': True, 'description': ''})
    return custom_factors.parse(text, BUILTIN_FACTORS)['blend']


def test_a_fingerprint_is_stable_for_the_same_definitions():
    customs = {'blend': _factor('rank_roe + rank_roa')}
    assert custom_factors.fingerprint(customs) == custom_factors.fingerprint(dict(customs))


def test_an_edited_formula_changes_the_fingerprint():
    before = custom_factors.fingerprint({'blend': _factor('rank_roe + rank_roa')})
    after = custom_factors.fingerprint({'blend': _factor('rank_roe - rank_roa')})
    assert before != after


def test_the_edited_factor_is_named():
    recorded = custom_factors.fingerprint({'blend': _factor('rank_roe + rank_roa')})
    now = {'blend': _factor('rank_roe - rank_roa')}
    assert custom_factors.changed_since(recorded, now) == ['blend']


def test_an_untouched_definition_is_not_named():
    customs = {'blend': _factor('rank_roe + rank_roa')}
    assert custom_factors.changed_since(custom_factors.fingerprint(customs), customs) == []


def test_nothing_recorded_means_nothing_can_be_said():
    assert custom_factors.changed_since(None, {'blend': _factor('rank_roe')}) == []
