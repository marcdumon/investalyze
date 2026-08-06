"""Tests for the control-panel blacklist-review pure helper: reading manual_blacklist.csv."""

from pathlib import Path

import pandas as pd

from investalyze.apps.control_panel import blacklist_review


def test_read_blacklist_returns_empty_list_when_none_blacklisted(tmp_path: Path):
    assert blacklist_review.read_blacklist(tmp_path / 'missing.csv') == []


def test_read_blacklist_reads_the_saved_entries(tmp_path: Path):
    path = tmp_path / 'manual_blacklist.csv'
    pd.DataFrame([{'ticker': 'SPCX', 'reason': 'Unknown sector', 'blacklisted_on': '2026-08-01'}]).to_csv(path, index=False)
    rows = blacklist_review.read_blacklist(path)
    assert rows[0] == {'ticker': 'SPCX', 'reason': 'Unknown sector', 'blacklisted_on': '2026-08-01'}
