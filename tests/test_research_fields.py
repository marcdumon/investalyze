"""Tests for the data field catalog: how statement columns become expression names."""

import pytest

from investalyze.research import fields

SCHEMA = {
    'income': ['Ticker', 'Report Date', 'Revenue', 'Net Income', 'Research & Development'],
    'balance': ['Ticker', 'Report Date', 'Total Assets', 'Total Equity'],
    'cashflow': ['Ticker', 'Report Date', 'Net Income', 'Dividends Paid'],
}


def test_a_plain_column_becomes_a_snake_case_alias():
    catalog = fields.build_catalog(SCHEMA)
    assert catalog['revenue'].source == 'Revenue'
    assert catalog['revenue'].statement == 'income'


def test_punctuation_is_stripped_from_an_alias():
    catalog = fields.build_catalog(SCHEMA)
    assert catalog['research_development'].source == 'Research & Development'


def test_every_alias_is_usable_as_a_column_name():
    for alias in fields.build_catalog(SCHEMA):
        assert alias.isidentifier(), alias


def test_metadata_columns_are_not_offered_as_fields():
    catalog = fields.build_catalog(SCHEMA)
    assert 'ticker' not in catalog
    assert 'report_date' not in catalog


def test_a_name_in_two_statements_is_reachable_under_each():
    catalog = fields.build_catalog(SCHEMA)
    assert catalog['income_net_income'].statement == 'income'
    assert catalog['cashflow_net_income'].statement == 'cashflow'


def test_an_ambiguous_bare_alias_is_not_offered():
    """`net_income` exists on two statements, so the bare name would silently pick one."""
    assert 'net_income' not in fields.build_catalog(SCHEMA)


def test_an_unambiguous_name_keeps_its_bare_alias():
    assert 'revenue' in fields.build_catalog(SCHEMA)


def test_flow_items_are_ttm_and_balance_items_are_mrq():
    catalog = fields.build_catalog(SCHEMA)
    assert catalog['revenue'].window == fields.TTM
    assert catalog['total_assets'].window == fields.MRQ
    assert catalog['dividends_paid'].window == fields.TTM


def test_the_catalog_reports_which_statement_a_field_came_from():
    catalog = fields.build_catalog(SCHEMA)
    assert {field.statement for field in catalog.values()} == {'income', 'balance', 'cashflow'}
