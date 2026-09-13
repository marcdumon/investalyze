"""Tests for resolving the data fields a factor needs before any query runs."""

import pytest

from investalyze.research import compute, factors, fields

CATALOG = {name: fields.Field(name, name.title(), 'income')
           for name in ('revenue', 'income_net_income', 'total_assets')}


def _factor(name: str, expression: str) -> factors.Factor:
    return factors.Factor(name=name, expression=expression)


def test_a_factor_needs_only_its_own_fields():
    library = {'margin': _factor('margin', 'income_net_income / revenue')}
    assert compute.required_fields(['margin'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_fields_are_followed_through_a_referenced_factor():
    library = {'margin': _factor('margin', 'income_net_income / revenue'),
               'ranked': _factor('ranked', 'rank(margin)')}
    assert compute.required_fields(['ranked'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_an_unrelated_factor_contributes_nothing():
    library = {'margin': _factor('margin', 'income_net_income / revenue'),
               'other': _factor('other', 'total_assets')}
    assert compute.required_fields(['margin'], library, CATALOG) == {'income_net_income', 'revenue'}


def test_an_identity_column_counts_as_a_field():
    library = {'neutral': _factor('neutral', 'neutralize(revenue, by=sector)')}
    assert compute.required_fields(['neutral'], library, CATALOG) == {'revenue', 'sector'}


def test_an_unknown_name_is_refused_naming_the_factor_that_reads_it():
    library = {'margin': _factor('margin', 'income_net_income / wibble')}
    with pytest.raises(ValueError, match="factor 'margin' reads unknown name 'wibble'"):
        compute.required_fields(['margin'], library, CATALOG)
