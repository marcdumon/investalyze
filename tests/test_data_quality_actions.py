"""Tests for the data-quality pure helpers: row normalization and log-entry field assembly."""

from datetime import date

import pytest

from investalyze.apps.data_quality import actions, quality_log, toml_io

PRICE_ROW = {'CheckName': 'nonpositive_price', 'Severity': 'error', 'SrcTable': 'prices',
             'Ticker': 'ATLX', 'Date': '2015-07-10', 'Key': None, 'Details': 'C=0'}
FUND_ROW = {'CheckName': 'fundamentals_sanity', 'Severity': 'error', 'SrcTable': 'balance',
            'Ticker': 'LZB', 'Date': None, 'Key': 'us|A|2005|Q4|false', 'Details': 'Total Assets=-5'}


def test_parse_key_splits_fundamentals_composite():
    assert actions.parse_key('us|A|2005|Q4|false') == {
        'market': 'us', 'period': 'A', 'fiscal_year': 2005, 'fiscal_period': 'Q4', 'is_restated': 'false'}


def test_parse_key_returns_none_for_non_fundamentals_key():
    assert actions.parse_key(None) is None
    assert actions.parse_key('') is None
    assert actions.parse_key('just-a-date') is None


def test_log_fields_price_row_uses_date():
    fields = actions.log_fields(PRICE_ROW, 'real-problem', 'wrong close')
    assert fields['date'] == date(2015, 7, 10)
    assert fields['key'] is None
    assert (fields['tag'], fields['table'], fields['comment']) == ('real-problem', 'prices', 'wrong close')


def test_log_fields_fundamentals_row_keeps_key_and_drops_empty_comment():
    fields = actions.log_fields(FUND_ROW, 'false-alarm', '')
    assert fields['key'] == 'us|A|2005|Q4|false'
    assert fields['date'] is None
    assert fields['comment'] is None


def test_log_fields_round_trip_through_log_parser():
    block = toml_io.serialize_block('log', actions.log_fields(PRICE_ROW, 'investigate', "it's odd"))
    entry = quality_log.parse_log(block)[0]
    assert entry.check == 'nonpositive_price'
    assert entry.tag == 'investigate'
    assert entry.comment == "it's odd"
    assert entry.date == date(2015, 7, 10)

# ---------- Details number formatting ----------


def test_format_details_numbers_adds_separators():
    text = 'gp+opex: lhs=-1414514 rhs=-1141514 diff=23.92%'
    assert actions.format_details_numbers(text) == 'gp+opex: lhs=-1,414,514 rhs=-1,141,514 diff=23.92%'


def test_format_details_numbers_keeps_short_numbers_dates_and_decimals():
    assert actions.format_details_numbers('Revenue: Q1..Q4=15844000000 FY=14198000000 diff=11.59%') \
        == 'Revenue: Q1..Q4=15,844,000,000 FY=14,198,000,000 diff=11.59%'
    assert actions.format_details_numbers('gap 12 days (2003-11-10 to 2003-11-22)') \
        == 'gap 12 days (2003-11-10 to 2003-11-22)'
    assert actions.format_details_numbers('C=12345.67') == 'C=12,345.67'


# ---------- involved line items per check ----------


def test_involved_items_identity_checks_use_details_prefix():
    assert actions.involved_items('balance_identity', 'liab+equity: lhs=1 rhs=2 diff=50%') \
        == {'Total Liabilities', 'Total Equity', 'Total Assets'}
    assert actions.involved_items('income_chain', 'gp+opex: lhs=1 rhs=2') \
        == {'Gross Profit', 'Operating Expenses', 'Other Operating Income', 'Operating Income (Loss)'}


def test_involved_items_qsum_names_its_column():
    assert actions.involved_items('quarters_vs_fy', 'Net Income: Q1..Q4=1 FY=2 diff=50%') == {'Net Income'}


def test_involved_items_sanity_and_unknown_checks():
    assert actions.involved_items('hard_invariants', 'Shares (Basic)=0 Shares (Diluted)=null') \
        == {'Shares (Basic)', 'Shares (Diluted)'}
    assert actions.involved_items('hard_invariants', 'Total Assets=-5') == {'Total Assets'}
    assert actions.involved_items('negative_revenue', 'Revenue=-3') == {'Revenue'}
    assert actions.involved_items('some_future_check', 'whatever') == set()


# ---------- fix_entries ----------


def price_row(ticker: str, day: str) -> dict:
    return {'CheckName': 'nonpositive_price', 'Severity': 'error', 'SrcTable': 'prices',
            'Ticker': ticker, 'Date': day, 'Key': None, 'Details': 'L=0'}


def test_fix_entries_delete_rows_merges_tickers_per_date():
    rows = [price_row('ATLX', '2015-07-22'), price_row('RCAT', '2015-07-22'), price_row('ATLX', '2015-08-05')]
    entries = actions.fix_entries('delete_rows', rows, None, None, 'bad bars')
    assert entries == [
        ('delete_date_range', {'table': 'prices', 'tickers': ['ATLX', 'RCAT'], 'start': date(2015, 7, 22),
                               'end': date(2015, 7, 22), 'reason': 'bad bars'}),
        ('delete_date_range', {'table': 'prices', 'tickers': ['ATLX'], 'start': date(2015, 8, 5),
                               'end': date(2015, 8, 5), 'reason': 'bad bars'}),
    ]


def test_fix_entries_delete_span_covers_min_to_max_per_ticker():
    rows = [price_row('ATLX', '2015-08-05'), price_row('ATLX', '2015-07-22'), price_row('RCAT', '2008-08-27')]
    entries = actions.fix_entries('delete_span', rows, None, None, 'garbage run')
    assert entries == [
        ('delete_date_range', {'table': 'prices', 'tickers': ['ATLX'], 'start': date(2015, 7, 22),
                               'end': date(2015, 8, 5), 'reason': 'garbage run'}),
        ('delete_date_range', {'table': 'prices', 'tickers': ['RCAT'], 'start': date(2008, 8, 27),
                               'end': date(2008, 8, 27), 'reason': 'garbage run'}),
    ]


def test_fix_entries_delete_ticker_merges_per_table():
    rows = [price_row('ATLX', '2015-07-22'), price_row('RCAT', '2008-08-27')]
    entries = actions.fix_entries('delete_ticker', rows, None, None, 'dead listing')
    assert entries == [('delete_date_range', {'table': 'prices', 'tickers': ['ATLX', 'RCAT'], 'reason': 'dead listing'})]


def test_fix_entries_set_value_single_row_only():
    entries = actions.fix_entries('set_value', [price_row('CXE', '2000-01-03')], 'AC', 5.1, 'sign flip')
    assert entries == [('set_value', {'table': 'prices', 'tickers': ['CXE'], 'start': date(2000, 1, 3),
                                      'end': date(2000, 1, 3), 'column': 'AC', 'value': 5.1, 'reason': 'sign flip'})]
    with pytest.raises(ValueError, match='exactly one'):
        actions.fix_entries('set_value', [price_row('A', '2000-01-03'), price_row('B', '2000-01-03')], 'AC', 5.1, 'r')
    with pytest.raises(ValueError, match='corrected value'):
        actions.fix_entries('set_value', [price_row('CXE', '2000-01-03')], 'AC', None, 'r')


def test_fix_entries_clear_value_needs_column_and_omits_value():
    entries = actions.fix_entries('clear_value', [price_row('CXE', '2000-01-03')], 'AC', None, 'unusable')
    assert entries == [('set_value', {'table': 'prices', 'tickers': ['CXE'], 'start': date(2000, 1, 3),
                                      'end': date(2000, 1, 3), 'column': 'AC', 'reason': 'unusable'})]
    with pytest.raises(ValueError, match='column'):
        actions.fix_entries('clear_value', [price_row('CXE', '2000-01-03')], None, None, 'r')


def test_fix_entries_rebuild_adjusted_close_requires_prices():
    entries = actions.fix_entries('rebuild_adjusted_close', [price_row('VHI', '2000-01-03')], None, None, 'poison div')
    assert entries == [('rebuild_adjusted_close', {'table': 'prices', 'tickers': ['VHI'], 'reason': 'poison div'})]
    with pytest.raises(ValueError, match='prices'):
        actions.fix_entries('rebuild_adjusted_close', [dict(price_row('LZB', '2000-01-03'), SrcTable='balance')],
                            None, None, 'r')


def test_fix_entries_rejects_bad_input():
    with pytest.raises(ValueError, match='at least one'):
        actions.fix_entries('delete_rows', [], None, None, 'r')
    with pytest.raises(ValueError, match='reason'):
        actions.fix_entries('delete_rows', [price_row('A', '2000-01-03')], None, None, '   ')
    with pytest.raises(ValueError, match='date-keyed'):
        actions.fix_entries('delete_rows', [FUND_ROW], None, None, 'r')
    with pytest.raises(ValueError, match='action'):
        actions.fix_entries('frobnicate', [price_row('A', '2000-01-03')], None, None, 'r')
