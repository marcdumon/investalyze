"""Tests for the control-panel blacklist-review pure helper: reading manual_blacklist.csv."""

from pathlib import Path

import pandas as pd

from investalyze.apps.control_panel import blacklist_review


def test_read_blacklist_returns_empty_list_when_none_blacklisted(tmp_path: Path):
    assert blacklist_review.read_blacklist(tmp_path / 'missing.csv') == []


def test_read_blacklist_reads_the_saved_entries(tmp_path: Path):
    path = tmp_path / 'manual_blacklist.csv'
    pd.DataFrame([{'ticker': 'SPCX', 'market': 'nasdaq', 'company_name': 'SPAC Corp',
                   'reason': 'Unknown sector', 'blacklisted_on': '2026-08-01'}]).to_csv(path, index=False)
    rows = blacklist_review.read_blacklist(path)
    assert rows[0] == {'ticker': 'SPCX', 'market': 'nasdaq', 'company_name': 'SPAC Corp',
                       'reason': 'Unknown sector', 'blacklisted_on': '2026-08-01'}


def test_read_blacklist_fills_in_columns_a_file_predating_them_lacks(tmp_path: Path):
    path = tmp_path / 'manual_blacklist.csv'
    pd.DataFrame([{'ticker': 'SPCX', 'reason': 'Unknown sector', 'blacklisted_on': '2026-08-01'}]).to_csv(path, index=False)
    rows = blacklist_review.read_blacklist(path)
    assert rows[0]['market'] == '' and rows[0]['company_name'] == ''


def test_filter_blacklist_with_no_search_returns_every_row():
    rows = [{'ticker': 'SPCX', 'reason': 'x', 'blacklisted_on': '2026-08-01'}]
    assert blacklist_review.filter_blacklist(rows, None) == rows
    assert blacklist_review.filter_blacklist(rows, '') == rows


def test_filter_blacklist_matches_a_ticker_substring_case_insensitively():
    rows = [{'ticker': 'SPCX', 'reason': 'x', 'blacklisted_on': '2026-08-01'},
            {'ticker': 'AAPL', 'reason': 'y', 'blacklisted_on': '2026-08-01'}]
    assert blacklist_review.filter_blacklist(rows, 'spc') == [rows[0]]
    assert blacklist_review.filter_blacklist(rows, ' PCX ') == [rows[0]]


def test_filter_blacklist_with_no_match_returns_an_empty_list():
    rows = [{'ticker': 'SPCX', 'reason': 'x', 'blacklisted_on': '2026-08-01'}]
    assert blacklist_review.filter_blacklist(rows, 'ZZZ') == []
