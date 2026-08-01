"""Tests for the control-panel page's pure argv-building helpers."""

import json

from investalyze.apps.control_panel import page


def test_build_listings_scan_argv():
    title, argv = page.build_listings_scan_argv()
    assert title == 'scan listings'
    assert argv == ['-m', 'investalyze.ingest', 'scan-listings']


def test_build_blacklist_argv_for_one_ticker():
    title, argv = page.build_blacklist_argv(['BID'], None, 'reused by a different company')
    assert title == 'blacklist BID'
    assert argv == ['-m', 'investalyze.ingest', 'blacklist-ticker', '--ticker', 'BID',
                    '--reason', 'reused by a different company']


def test_build_blacklist_argv_for_multiple_tickers():
    title, argv = page.build_blacklist_argv(['AAPL', 'MSFT'], None, 'batch cleanup')
    assert title == 'blacklist 2 tickers'
    assert argv == ['-m', 'investalyze.ingest', 'blacklist-ticker', '--ticker', 'AAPL', '--ticker', 'MSFT',
                    '--reason', 'batch cleanup']


def test_build_blacklist_argv_for_a_universe():
    title, argv = page.build_blacklist_argv(None, 'penny_stocks', 'not interesting')
    assert title == 'blacklist universe penny_stocks'
    assert argv == ['-m', 'investalyze.ingest', 'blacklist-ticker', '--universe', 'penny_stocks',
                    '--reason', 'not interesting']


def test_build_add_listings_argv():
    rows = [
        {'ticker': 'RELS', 'market': 'nyse', 'classification': 'reactivated_safe', 'suggested_new_ticker': ''},
        {'ticker': 'BID', 'market': 'nasdaq', 'classification': 'reactivated_collision', 'suggested_new_ticker': 'BID.1'},
    ]
    title, argv = page.build_add_listings_argv(rows)
    assert title == 'add 2 listing(s)'
    assert argv[:3] == ['-m', 'investalyze.ingest', 'add-listings']
    assert argv[3] == '--rows'
    payload = json.loads(argv[4])
    assert payload == [
        {'ticker': 'RELS', 'market': 'nyse', 'classification': 'reactivated_safe', 'new_ticker': ''},
        {'ticker': 'BID', 'market': 'nasdaq', 'classification': 'reactivated_collision', 'new_ticker': 'BID.1'},
    ]
