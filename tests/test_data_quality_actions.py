"""Tests for the data-quality pure helpers: row normalization and log-entry field assembly."""

from datetime import date

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
