"""Tests for the control-panel listings-review pure helpers: reading candidates, adding tickers."""

from pathlib import Path

import pandas as pd

from investalyze.apps.control_panel import listings_review


def test_read_candidates_returns_empty_list_when_no_scan_has_run(tmp_path: Path):
    assert listings_review.read_candidates(tmp_path / 'missing.csv') == []


def test_read_candidates_reads_the_saved_scan(tmp_path: Path):
    path = tmp_path / 'candidates.csv'
    pd.DataFrame([{'ticker': 'SPCX', 'market': 'nasdaq', 'company_name': 'SpaceX Corp',
                  'classification': 'new', 'suggested_new_ticker': ''}]).to_csv(path, index=False)
    rows = listings_review.read_candidates(path)
    assert rows[0]['ticker'] == 'SPCX'


def test_read_candidates_returns_empty_list_for_headerless_zero_candidate_scan(tmp_path: Path):
    path = tmp_path / 'candidates.csv'
    pd.DataFrame([]).to_csv(path, index=False)
    assert listings_review.read_candidates(path) == []


def test_add_tickers_appends_new_and_reactivated_safe_rows(tmp_path: Path):
    ticker_csv = tmp_path / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAPL'], 'market': ['nasdaq']}).to_csv(ticker_csv, index=False)
    rows = [
        {'ticker': 'SPCX', 'market': 'nasdaq', 'classification': 'new'},
        {'ticker': 'RELS', 'market': 'nyse', 'classification': 'reactivated_safe'},
        {'ticker': 'ZOMB', 'market': 'nyse', 'classification': 'reactivated_collision'},
    ]
    added = listings_review.add_tickers(rows, ticker_csv)
    assert added == 2
    tickers = set(pd.read_csv(ticker_csv)['ticker'])
    assert tickers == {'AAPL', 'SPCX', 'RELS'}
    assert 'ZOMB' not in tickers   # collision rows need a rename staged first, never auto-added


def test_add_tickers_skips_duplicates(tmp_path: Path):
    ticker_csv = tmp_path / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAPL'], 'market': ['nasdaq']}).to_csv(ticker_csv, index=False)
    added = listings_review.add_tickers([{'ticker': 'AAPL', 'market': 'nasdaq', 'classification': 'new'}], ticker_csv)
    assert added == 1   # counted as processed...
    assert len(pd.read_csv(ticker_csv)) == 1   # ...but not duplicated
