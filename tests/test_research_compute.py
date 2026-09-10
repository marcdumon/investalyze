"""Tests for resolving what an idea needs before any query runs."""

import pytest

from investalyze.research import compute, fields, ideas

CATALOG = {name: fields.Field(name, name.title(), 'income')
           for name in ('revenue', 'income_net_income', 'total_assets')}


def _idea(name: str, expression: str) -> ideas.Idea:
    return ideas.Idea(name=name, expression=expression)


def test_a_simple_idea_needs_only_its_own_fields():
    library = {'margin': _idea('margin', 'income_net_income / revenue')}
    assert compute.required_fields(['margin'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_fields_are_followed_through_a_referenced_idea():
    library = {'margin': _idea('margin', 'income_net_income / revenue'),
               'ranked': _idea('ranked', 'rank(margin)')}
    assert compute.required_fields(['ranked'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_an_unrelated_idea_contributes_nothing():
    library = {'margin': _idea('margin', 'income_net_income / revenue'),
               'other': _idea('other', 'total_assets')}
    assert compute.required_fields(['margin'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_an_identity_column_counts_as_a_field():
    library = {'neutral': _idea('neutral', 'neutralize(revenue, by=sector)')}
    assert compute.required_fields(['neutral'], library, CATALOG) == {'revenue', 'sector'}


def test_an_unknown_name_is_refused_naming_the_idea_that_reads_it():
    library = {'margin': _idea('margin', 'income_net_income / wibble')}
    with pytest.raises(ValueError, match="idea 'margin' reads unknown name 'wibble'"):
        compute.required_fields(['margin'], library, CATALOG)
