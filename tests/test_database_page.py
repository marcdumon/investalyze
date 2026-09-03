"""Tests for the database page: the catalogue, the shaping, and the guards on the query box.

The catalogue test is the one that matters over time. A table added to the database without a line
saying what it is for is exactly the situation this page exists to fix, so it fails here rather than
appearing on screen with nothing beside it.
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from investalyze.apps.database import catalog, data, logic

_DB = Path('data') / 'investalyze.duckdb'


def _described(**overrides) -> dict:
    """A described table, shaped as `describe_table` returns it."""
    base = {'name': 'prices', 'rows': 1_000,
            'columns': [{'column_name': 'Ticker', 'data_type': 'VARCHAR'},
                        {'column_name': 'Date', 'data_type': 'DATE'},
                        {'column_name': 'C', 'data_type': 'DOUBLE'}],
            'nulls': {'Ticker': 0, 'Date': 0, 'C': 250}, 'nulls_counted': True,
            'span': (pd.Timestamp('2000-01-03'), pd.Timestamp('2026-09-02')), 'tickers': 12}
    return base | overrides


# --- the catalogue ---

def test_every_catalogued_table_says_what_it_is_for_and_what_writes_it():
    for name, table in catalog.TABLES.items():
        assert table.purpose.strip(), name
        assert table.written_by.strip(), name
        assert table.area in catalog.AREAS, name


def test_a_description_is_a_sentence_rather_than_a_restatement_of_the_name():
    """A column list already says a table has a Date; the point here is what a row means."""
    for name, table in catalog.TABLES.items():
        assert len(table.purpose) > 40, name


def test_areas_group_every_table():
    grouped = {name for members in catalog.by_area().values() for name in members}
    assert grouped == set(catalog.TABLES)


@pytest.mark.skipif(not _DB.exists(), reason='live database absent')
def test_every_table_in_the_database_has_a_catalogue_entry():
    """A table appearing on the page with nothing beside it is the problem this page solves."""
    try:
        con = duckdb.connect(str(_DB), read_only=True)
    except duckdb.IOException:
        pytest.skip('database write-locked')
    try:
        live = {row[0] for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
    finally:
        con.close()
    assert not live - set(catalog.TABLES), f'undescribed: {sorted(live - set(catalog.TABLES))}'


# --- shaping ---

def test_a_table_row_carries_the_area_from_its_catalogue_entry():
    frame = pd.DataFrame([{'name': 'factor_panel', 'rows': 10, 'columns': 38}])
    assert logic.table_rows(frame)[0]['area'] == catalog.TABLES['factor_panel'].area


def test_a_table_with_no_entry_is_marked_rather_than_dropped():
    frame = pd.DataFrame([{'name': 'something_new', 'rows': 1, 'columns': 2}])
    row = logic.table_rows(frame)[0]
    assert row['area'] == 'Uncatalogued'
    assert logic.purpose_of('something_new')[0] == logic.UNCATALOGUED


def test_the_facts_name_the_rows_columns_tickers_and_span():
    facts = logic.header_facts(_described())
    assert facts == ['1,000 rows', '3 columns', '12 tickers', '2000-01-03 to 2026-09-02']


def test_a_table_without_dates_or_tickers_says_nothing_about_them():
    facts = logic.header_facts(_described(span=None, tickers=None))
    assert facts == ['1,000 rows', '3 columns']


def test_an_empty_share_is_shown_only_where_there_is_one():
    rows = {row['column_name']: row['empty'] for row in logic.column_rows(_described())}
    assert rows['Ticker'] == ''
    assert rows['C'] == '25%'


def test_a_column_that_is_entirely_empty_says_so():
    rows = {row['column_name']: row['empty']
            for row in logic.column_rows(_described(nulls={'Ticker': 0, 'Date': 0, 'C': 1_000}))}
    assert rows['C'] == 'all'


def test_uncounted_columns_are_dashed_and_explained():
    described = _described(nulls_counted=False, nulls={})
    assert all(row['empty'] == '-' for row in logic.column_rows(described))
    assert 'skipped above' in logic.skipped_note(described, 5_000_000)


def test_a_counted_table_needs_no_explanation():
    assert logic.skipped_note(_described(), 5_000_000) == ''


# --- the query box ---

def test_an_empty_query_is_refused():
    assert logic.check_query('   ') == 'Type a query first.'


def test_two_statements_are_refused():
    assert logic.check_query('SELECT 1; SELECT 2') == 'One statement at a time.'


def test_a_trailing_semicolon_is_not_two_statements():
    assert logic.check_query('SELECT 1;') is None


def test_writes_are_left_to_the_database_to_refuse():
    """A rule here could only recognise the forms someone thought of; read-only catches them all."""
    assert logic.check_query('DELETE FROM prices') is None


@pytest.mark.skipif(not _DB.exists(), reason='live database absent')
def test_the_connection_itself_rejects_every_kind_of_write():
    try:
        _rows, _more, failure = data.run_query('DELETE FROM prices')
    except duckdb.IOException:
        pytest.skip('database write-locked')
    assert failure is not None
    assert 'read-only' in failure.lower() or 'cannot execute' in failure.lower()


@pytest.mark.skipif(not _DB.exists(), reason='live database absent')
def test_a_query_returning_more_than_the_cap_says_it_was_cut_short():
    try:
        rows, truncated, failure = data.run_query('SELECT * FROM prices', limit=10)
    except duckdb.IOException:
        pytest.skip('database write-locked')
    assert failure is None
    assert len(rows) == 10
    assert truncated is True


@pytest.mark.skipif(not _DB.exists(), reason='live database absent')
def test_a_statement_that_is_not_a_select_still_runs():
    """DESCRIBE and SHOW are how someone finds their way around, and neither writes anything."""
    try:
        rows, _more, failure = data.run_query('DESCRIBE prices')
    except duckdb.IOException:
        pytest.skip('database write-locked')
    assert failure is None
    assert len(rows) > 0


def test_the_page_layout_builds():
    from investalyze.apps.database import page

    assert page.layout() is not None
