"""Tests for the quality triage log: parsing, anti-join key frame, and validate-then-append."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from investalyze.apps.data_quality import actions, quality_log, toml_io

PRICE_ROW = {'CheckName': 'nonpositive_price', 'Severity': 'error', 'SrcTable': 'prices',
             'Ticker': 'ATLX', 'Date': '2015-07-10', 'Key': None, 'Details': 'C=0'}


def test_parse_log_round_trips():
    text = """\
[[log]]
check = 'extreme_return'
ticker = 'GME'
date = 2021-01-27
tag = 'false-alarm'
comment = 'genuine squeeze'

[[log]]
check = 'negative_revenue'
ticker = 'JPM'
key = 'us|Q|2020|Q1|false'
tag = 'known'
"""
    entries = quality_log.parse_log(text)
    assert entries[0] == quality_log.LogEntry('extreme_return', 'GME', date(2021, 1, 27), None,
                                              'false-alarm', 'genuine squeeze', '', '', '')
    assert entries[1].key == 'us|Q|2020|Q1|false' and entries[1].date is None and entries[1].comment == ''


def test_parse_log_rejects_unknown_section():
    with pytest.raises(ValueError, match='unknown log section'):
        quality_log.parse_log("[[note]]\ncheck = 'x'\nticker = 'Y'\ntag = 'known'\n")


def test_parse_log_rejects_missing_required():
    with pytest.raises(ValueError, match='tag'):
        quality_log.parse_log("[[log]]\ncheck = 'x'\nticker = 'Y'\n")


def test_read_log_missing_file_is_empty(tmp_path: Path):
    assert quality_log.read_log(tmp_path / 'nope.toml') == []


def test_log_keys_frame_columns_and_null_date():
    entries = [
        quality_log.LogEntry('extreme_return', 'GME', date(2021, 1, 27), None, 'known', '', '', '', ''),
        quality_log.LogEntry('negative_revenue', 'JPM', None, 'us|Q|2020|Q1|false', 'known', '', '', '', ''),
    ]
    frame = quality_log.log_keys_frame(entries)
    assert list(frame.columns) == ['check_name', 'ticker', 'log_date', 'log_key']
    assert frame.iloc[0]['log_date'] == '2021-01-27'
    assert pd.isna(frame.iloc[1]['log_date'])  # missing date is SQL NULL in the anti-join
    assert frame.iloc[1]['log_key'] == 'us|Q|2020|Q1|false'


def test_append_log_validates_and_grows(tmp_path: Path):
    path = tmp_path / 'quality_log.toml'
    block = toml_io.serialize_block('log', actions.log_fields(PRICE_ROW, 'real-problem', 'zero close'))
    quality_log.append_log(path, block)
    quality_log.append_log(path, toml_io.serialize_block('log', actions.log_fields(
        {**PRICE_ROW, 'Date': '2015-07-11'}, 'investigate', '')))
    entries = quality_log.read_log(path)
    assert [e.date for e in entries] == [date(2015, 7, 10), date(2015, 7, 11)]
    assert entries[0].tag == 'real-problem'


def test_append_log_atomic_on_invalid_entry(tmp_path: Path):
    path = tmp_path / 'quality_log.toml'
    original = "[[log]]\ncheck = 'a'\nticker = 'B'\ntag = 'known'\n"
    path.write_text(original)
    with pytest.raises(ValueError, match='missing required'):
        quality_log.append_log(path, "[[log]]\ncheck = 'x'\n")
    assert path.read_text() == original
