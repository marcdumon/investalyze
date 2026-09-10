"""Tests for the idea store: what an idea is, how it is validated, and when its fingerprint moves."""

import pytest

from investalyze.research import ideas

FIELDS = {'revenue', 'income', 'assets', 'sector'}


def _idea(name: str, expression: str, **overrides) -> ideas.Idea:
    body = {'name': name, 'expression': expression, 'description': 'a test idea'} | overrides
    return ideas.Idea(**body)


# --- what an idea is ---

def test_an_idea_round_trips_through_a_file(tmp_path):
    original = _idea('margin', 'income / revenue', description='profit per unit of revenue')
    ideas.save(original, tmp_path)
    loaded = ideas.load(tmp_path)
    assert loaded['margin'] == original


def test_saving_writes_one_file_per_idea(tmp_path):
    ideas.save(_idea('margin', 'income / revenue'), tmp_path)
    ideas.save(_idea('turnover', 'revenue / assets'), tmp_path)
    assert sorted(path.name for path in tmp_path.glob('*.toml')) == ['margin.toml', 'turnover.toml']


def test_an_idea_starts_as_a_draft():
    assert _idea('margin', 'income / revenue').status == ideas.DRAFT


def test_an_unusable_name_is_refused():
    with pytest.raises(ValueError, match='usable as a column name'):
        _idea('not a name', 'revenue')


def test_an_unknown_status_is_refused():
    with pytest.raises(ValueError, match='status'):
        _idea('margin', 'income / revenue', status='pondering')


# --- validation across the whole library ---

def test_an_unknown_field_is_refused_naming_it():
    library = {'margin': _idea('margin', 'income / turnover')}
    with pytest.raises(ValueError, match='turnover'):
        ideas.validate_library(library, FIELDS)


def test_an_idea_may_reference_another_idea():
    library = {'margin': _idea('margin', 'income / revenue'),
               'ranked': _idea('ranked', 'rank(margin)')}
    ideas.validate_library(library, FIELDS)


def test_a_cycle_is_refused():
    library = {'a': _idea('a', 'b + 1'), 'b': _idea('b', 'a + 1')}
    with pytest.raises(ValueError, match='cycle'):
        ideas.validate_library(library, FIELDS)


def test_an_idea_referring_to_itself_is_a_cycle():
    library = {'a': _idea('a', 'a + 1')}
    with pytest.raises(ValueError, match='cycle'):
        ideas.validate_library(library, FIELDS)


def test_evaluation_order_puts_dependencies_first():
    library = {'ranked': _idea('ranked', 'rank(margin)'),
               'margin': _idea('margin', 'income / revenue')}
    assert ideas.evaluation_order(library, FIELDS) == ['margin', 'ranked']


# --- fingerprints drive cache invalidation ---

def test_the_fingerprint_moves_when_the_formula_changes():
    before = {'margin': _idea('margin', 'income / revenue')}
    after = {'margin': _idea('margin', 'income / assets')}
    assert ideas.fingerprint('margin', before) != ideas.fingerprint('margin', after)


def test_the_fingerprint_holds_when_only_the_description_changes():
    """Prose does not change a value, so it must not throw away an expensive computation."""
    before = {'margin': _idea('margin', 'income / revenue', description='one')}
    after = {'margin': _idea('margin', 'income / revenue', description='two')}
    assert ideas.fingerprint('margin', before) == ideas.fingerprint('margin', after)


def test_the_fingerprint_holds_when_only_the_status_changes():
    before = {'margin': _idea('margin', 'income / revenue', status=ideas.DRAFT)}
    after = {'margin': _idea('margin', 'income / revenue', status=ideas.KEPT)}
    assert ideas.fingerprint('margin', before) == ideas.fingerprint('margin', after)


def test_the_fingerprint_moves_when_a_dependency_changes():
    """The reason this matters: a stale answer downstream is invisible without it."""
    before = {'margin': _idea('margin', 'income / revenue'), 'ranked': _idea('ranked', 'rank(margin)')}
    after = {'margin': _idea('margin', 'income / assets'), 'ranked': _idea('ranked', 'rank(margin)')}
    assert ideas.fingerprint('ranked', before) != ideas.fingerprint('ranked', after)


def test_the_fingerprint_ignores_an_unrelated_idea():
    before = {'margin': _idea('margin', 'income / revenue'), 'other': _idea('other', 'revenue')}
    after = {'margin': _idea('margin', 'income / revenue'), 'other': _idea('other', 'assets')}
    assert ideas.fingerprint('margin', before) == ideas.fingerprint('margin', after)


# --- the direction a factor is read in ---

def test_an_idea_says_which_end_is_good():
    assert _idea('margin', 'income / revenue').higher_is_better is True
    assert _idea('leverage', 'debt', higher_is_better=False).higher_is_better is False
