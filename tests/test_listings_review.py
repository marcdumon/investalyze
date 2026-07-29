"""Tests for the control-panel listings-review pure helper: reading saved scan candidates."""

from pathlib import Path

import pandas as pd

from investalyze.apps.control_panel import listings_review


def test_read_candidates_returns_empty_list_when_no_scan_has_run(tmp_path: Path):
    assert listings_review.read_candidates(tmp_path / 'missing.csv') == []


def test_read_candidates_reads_the_saved_scan(tmp_path: Path):
    path = tmp_path / 'candidates.csv'
    pd.DataFrame([{'ticker': 'SPCX', 'market': 'nasdaq', 'company_name': 'SpaceX Corp',
                  'classification': 'new', 'suggested_new_ticker': '', 'previous_company_name': ''}]
                ).to_csv(path, index=False)
    rows = listings_review.read_candidates(path)
    assert rows[0]['ticker'] == 'SPCX'


def test_read_candidates_adds_a_human_readable_classification_label(tmp_path: Path):
    path = tmp_path / 'candidates.csv'
    pd.DataFrame([
        {'ticker': 'RELS', 'market': 'nyse', 'company_name': 'Relisted Co', 'classification': 'reactivated_safe',
        'suggested_new_ticker': '', 'previous_company_name': 'Relisted Co'},
        {'ticker': 'BID', 'market': 'nasdaq', 'company_name': 'New Co', 'classification': 'reactivated_collision',
        'suggested_new_ticker': 'BID.1', 'previous_company_name': 'Old Co'},
    ]).to_csv(path, index=False)
    rows = listings_review.read_candidates(path)
    labels = {r['ticker']: r['classification_label'] for r in rows}
    assert labels == {'RELS': 'Reactivated (safe)', 'BID': 'Reactivated (collision)'}
    assert rows[0]['classification'] == 'reactivated_safe'   # raw value kept, request_run matches on it


def test_read_candidates_title_cases_company_names(tmp_path: Path):
    path = tmp_path / 'candidates.csv'
    pd.DataFrame([
        {'ticker': 'BID', 'market': 'nasdaq', 'company_name': 'Tribeca Strategic Acquisition Corp.',
        'classification': 'reactivated_collision', 'suggested_new_ticker': 'BID.1', 'previous_company_name': 'SOTHEBYS'},
        {'ticker': 'VIDA', 'market': 'nysemkt', 'company_name': 'VIDA Global Inc.',
        'classification': 'reactivated_collision', 'suggested_new_ticker': 'VIDA.1',
        'previous_company_name': 'VIDAROO Corp'},
    ]).to_csv(path, index=False)
    rows = listings_review.read_candidates(path)
    names = {r['ticker']: r['previous_company_name'] for r in rows}
    assert names['BID'] == 'Sothebys'
    assert names['VIDA'] == 'Vidaroo Corp'   # every name gets title-cased, no exceptions


def test_read_candidates_returns_empty_list_for_headerless_zero_candidate_scan(tmp_path: Path):
    path = tmp_path / 'candidates.csv'
    pd.DataFrame([]).to_csv(path, index=False)
    assert listings_review.read_candidates(path) == []
