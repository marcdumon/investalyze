"""Tests for the factor library: factor definitions, validation, and fingerprints."""

import pytest

from investalyze.research import factors

FIELDS = {'revenue', 'income', 'assets', 'sector'}


def _factor(name: str, expression: str, **overrides) -> factors.Factor:
    body = {'name': name, 'expression': expression, 'description': 'a test factor'} | overrides
    return factors.Factor(**body)


# --- factor definitions ---

def test_a_factor_round_trips_through_a_file(tmp_path):
    original = _factor('margin', 'income / revenue', description='profit per unit of revenue')
    factors.save(original, tmp_path)
    loaded = factors.load(tmp_path)
    assert loaded['margin'] == original


def test_saving_writes_one_file_per_factor(tmp_path):
    factors.save(_factor('margin', 'income / revenue'), tmp_path)
    factors.save(_factor('turnover', 'revenue / assets'), tmp_path)
    assert sorted(path.name for path in tmp_path.glob('*.toml')) == ['margin.toml', 'turnover.toml']


def test_a_factor_starts_as_a_candidate():
    assert _factor('margin', 'income / revenue').status == factors.Status.CANDIDATE


def test_an_unusable_name_is_refused():
    with pytest.raises(ValueError, match='usable as a column name'):
        _factor('not a name', 'revenue')


def test_an_unknown_status_is_refused():
    with pytest.raises(ValueError, match='status'):
        _factor('margin', 'income / revenue', status='pending')


# --- validation across the whole library ---

def test_an_unknown_field_is_refused_naming_it():
    library = {'margin': _factor('margin', 'income / turnover')}
    with pytest.raises(ValueError, match='turnover'):
        factors.validate_library(library, FIELDS)


def test_a_factor_may_reference_another_factor():
    library = {'margin': _factor('margin', 'income / revenue'),
               'ranked': _factor('ranked', 'rank(margin)')}
    factors.validate_library(library, FIELDS)


def test_a_cycle_is_refused():
    library = {'a': _factor('a', 'b + 1'), 'b': _factor('b', 'a + 1')}
    with pytest.raises(ValueError, match='cycle'):
        factors.validate_library(library, FIELDS)


def test_a_factor_referring_to_itself_is_a_cycle():
    library = {'a': _factor('a', 'a + 1')}
    with pytest.raises(ValueError, match='cycle'):
        factors.validate_library(library, FIELDS)


def test_evaluation_order_puts_dependencies_first():
    library = {'ranked': _factor('ranked', 'rank(margin)'),
               'margin': _factor('margin', 'income / revenue')}
    assert factors.evaluation_order(library, FIELDS) == ['margin', 'ranked']


# --- fingerprints drive cache invalidation ---

def test_the_fingerprint_moves_when_the_formula_changes():
    before = {'margin': _factor('margin', 'income / revenue')}
    after = {'margin': _factor('margin', 'income / assets')}
    assert factors.fingerprint('margin', before) != factors.fingerprint('margin', after)


def test_the_fingerprint_holds_when_only_the_description_changes():
    """The rationale does not change a value, so it must not invalidate cached results."""
    before = {'margin': _factor('margin', 'income / revenue', description='one')}
    after = {'margin': _factor('margin', 'income / revenue', description='two')}
    assert factors.fingerprint('margin', before) == factors.fingerprint('margin', after)


def test_the_fingerprint_holds_when_only_the_status_changes():
    before = {'margin': _factor('margin', 'income / revenue', status=factors.Status.CANDIDATE)}
    after = {'margin': _factor('margin', 'income / revenue', status=factors.Status.ACCEPTED)}
    assert factors.fingerprint('margin', before) == factors.fingerprint('margin', after)


def test_the_fingerprint_moves_when_a_dependency_changes():
    """Without this, a downstream factor would read results from the old definition."""
    before = {'margin': _factor('margin', 'income / revenue'), 'ranked': _factor('ranked', 'rank(margin)')}
    after = {'margin': _factor('margin', 'income / assets'), 'ranked': _factor('ranked', 'rank(margin)')}
    assert factors.fingerprint('ranked', before) != factors.fingerprint('ranked', after)


def test_the_fingerprint_ignores_an_unrelated_factor():
    before = {'margin': _factor('margin', 'income / revenue'), 'other': _factor('other', 'revenue')}
    after = {'margin': _factor('margin', 'income / revenue'), 'other': _factor('other', 'assets')}
    assert factors.fingerprint('margin', before) == factors.fingerprint('margin', after)


# --- direction ---

def test_a_factor_records_its_direction():
    assert _factor('margin', 'income / revenue').higher_is_better is True
    assert _factor('leverage', 'debt', higher_is_better=False).higher_is_better is False
