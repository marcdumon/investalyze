"""Tests for investalyze.ingest.listings: NASDAQ/SEC source parsing and classification."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from investalyze.ingest import listings

NASDAQLISTED_SAMPLE = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AACG|ATA Creativity Global - American Depositary Shares|G|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
ZTEST|Test Issue Common Stock|G|Y|N|100|N|N
ACONW|Acon S2 Acquisition Corp - Warrant|G|N|N|100|N|N
File Creation Time: 07/27/2026 05:00:01
"""

OTHERLISTED_SAMPLE = """\
ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
A|Agilent Technologies Inc. Common Stock|N|A|N|100|N|A
SPCX|SpaceX Inc Common Stock|N|SPCX|N|100|N|SPCX
IEXV|Some IEX Thing|V|IEXV|N|100|N|IEXV
AMKT|American Market Co Common Stock|A|AMKT|N|100|N|AMKT
File Creation Time: 07/27/2026 05:00:01
"""

SEC_TICKERS_SAMPLE = {
    '0': {'cik_str': 320193, 'ticker': 'aapl', 'title': 'Apple Inc.'},
    '1': {'cik_str': 1318605, 'ticker': 'TSLA', 'title': 'Tesla Inc'},
}


def test_parse_nasdaqlisted_drops_etf_test_issue_and_warrants():
    frame = listings._parse_nasdaqlisted(NASDAQLISTED_SAMPLE)
    assert list(frame['ticker']) == ['AACG']
    assert list(frame['market']) == ['nasdaq']


NASDAQLISTED_KEYWORD_SAMPLE = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
BFAM|Bright Horizons Family Solutions Inc - Common Stock|G|N|N|100|N|N
RGHT|XYZ Corp Rights|G|N|N|100|N|N
PFD|ABC Preferred Stock|G|N|N|100|N|N
UNIT|DEF Acquisition Corp Units|G|N|N|100|N|N
File Creation Time: 07/27/2026 05:00:01
"""


def test_parse_nasdaqlisted_keeps_substring_match_but_drops_whole_word_keyword_matches():
    frame = listings._parse_nasdaqlisted(NASDAQLISTED_KEYWORD_SAMPLE)
    # 'Bright Horizons' merely contains 'right' as a substring and must be kept; 'Rights',
    # 'Preferred', and 'Units' are whole-word matches and must be dropped
    assert list(frame['ticker']) == ['BFAM']


def test_parse_otherlisted_keeps_only_nyse_and_nysemkt():
    frame = listings._parse_otherlisted(OTHERLISTED_SAMPLE)
    assert sorted(frame['ticker']) == ['A', 'AMKT', 'SPCX']
    by_ticker = dict(zip(frame['ticker'], frame['market']))
    assert by_ticker['AMKT'] == 'nysemkt'
    assert set(frame['market']) == {'nyse', 'nysemkt'}


def test_parse_sec_cik_map_upper_cases_tickers():
    cik_map = listings._parse_sec_cik_map(SEC_TICKERS_SAMPLE)
    assert cik_map == {'AAPL': 320193, 'TSLA': 1318605}


def sample_listings() -> pd.DataFrame:
    return pd.DataFrame([
        {'ticker': 'SPCX', 'market': 'nasdaq', 'company_name': 'SpaceX Corp'},
        {'ticker': 'AAPL', 'market': 'nasdaq', 'company_name': 'Apple Inc.'},
        {'ticker': 'ZOMB', 'market': 'nyse', 'company_name': 'Zombie Reincorp'},
        {'ticker': 'RELS', 'market': 'nyse', 'company_name': 'Relisted Co'},
    ])


def test_classify_never_tracked_ticker_as_new():
    candidates = listings.classify(sample_listings(), tracked={'AAPL'}, dead_tickers=set(), live_cik={}, last_cik={})
    by_ticker = {c.ticker: c for c in candidates}
    assert by_ticker['SPCX'].classification == 'new'
    assert 'AAPL' not in by_ticker   # already tracked, not a candidate


def test_classify_reactivated_ticker_with_matching_cik_as_safe():
    candidates = listings.classify(sample_listings(), tracked=set(), dead_tickers={'RELS'},
                                   live_cik={'RELS': 42}, last_cik={'RELS': 42})
    assert next(c for c in candidates if c.ticker == 'RELS').classification == 'reactivated_safe'


def test_classify_reactivated_ticker_with_mismatched_cik_as_collision():
    candidates = listings.classify(sample_listings(), tracked=set(), dead_tickers={'ZOMB'},
                                   live_cik={'ZOMB': 99}, last_cik={'ZOMB': 1})
    assert next(c for c in candidates if c.ticker == 'ZOMB').classification == 'reactivated_collision'


def test_classify_reactivated_ticker_with_unknown_cik_defaults_to_collision():
    candidates = listings.classify(sample_listings(), tracked=set(), dead_tickers={'ZOMB'}, live_cik={}, last_cik={})
    assert next(c for c in candidates if c.ticker == 'ZOMB').classification == 'reactivated_collision'


def test_next_free_suffix_finds_the_first_unused_key():
    assert listings.next_free_suffix('SPCX', existing=set()) == 'SPCX.1'
    assert listings.next_free_suffix('SPCX', existing={'SPCX.1'}) == 'SPCX.2'
    assert listings.next_free_suffix('SPCX', existing={'SPCX.1', 'SPCX.2'}) == 'SPCX.3'


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A minimal data_root: ticker.csv with one tracked ticker, empty state dir."""
    raw_dir = tmp_path / 'yahoo' / 'raw'
    state_dir = tmp_path / 'yahoo' / 'state'
    raw_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    pd.DataFrame({'ticker': ['AAPL'], 'market': ['nasdaq']}).to_csv(raw_dir / 'ticker.csv', index=False)
    return tmp_path


@pytest.fixture
def scan_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(tmp_path / 'scan.duckdb'))
    con.execute('CREATE TABLE companies (Ticker VARCHAR, CIK BIGINT)')
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE)')
    return con


def test_scan_classifies_and_writes_candidates_csv(monkeypatch, scan_con, data_root):
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {})
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    assert result.saved_to == data_root / 'yahoo' / 'state' / listings.CANDIDATES_FILE
    saved = pd.read_csv(result.saved_to)
    assert set(saved['ticker']) == {'SPCX', 'ZOMB', 'RELS'}   # AAPL already tracked, excluded


def test_scan_suggests_a_free_key_for_collision_candidates(monkeypatch, scan_con, data_root):
    state_dir = data_root / 'yahoo' / 'state'
    pd.DataFrame([{'ticker': 'ZOMB', 'attempts': 3, 'first_blacklisted': '2020-01-01', 'died_on': '2020-06-01'}]
                ).to_csv(state_dir / 'price_dead.csv', index=False)
    scan_con.execute("INSERT INTO companies VALUES ('ZOMB', 1)")   # old company's CIK on file
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {'ZOMB': 99})   # different CIK now trading
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    zomb = next(c for c in result.candidates if c.ticker == 'ZOMB')
    assert zomb.classification == 'reactivated_collision'
    assert zomb.suggested_new_ticker == 'ZOMB.1'


def test_scan_clears_dead_list_once_a_staged_rename_has_moved_the_old_rows(monkeypatch, scan_con, data_root):
    state_dir = data_root / 'yahoo' / 'state'
    pd.DataFrame([{'ticker': 'ZOMB', 'attempts': 3, 'first_blacklisted': '2020-01-01', 'died_on': '2020-06-01'}]
                ).to_csv(state_dir / 'price_dead.csv', index=False)
    cleaning_path = data_root / 'cleaning.toml'
    cleaning_path.write_text("[[rename_ticker]]\ntickers = ['ZOMB']\nnew_ticker = 'ZOMB.1'\nreason = 'r'\n")
    # the rename has already been applied: no ZOMB rows remain anywhere
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {})
    result = listings.scan(scan_con, data_root, cleaning_config=cleaning_path)
    zomb = next(c for c in result.candidates if c.ticker == 'ZOMB')
    assert zomb.classification == 'new'   # resolved: no longer treated as a dead/collision ticker
    assert pd.read_csv(state_dir / 'price_dead.csv').empty


def test_scan_checks_identity_for_a_ticker_stuck_in_blacklist_limbo(monkeypatch, scan_con, data_root):
    """A ticker pruned out of ticker.csv while blacklisted (but not yet aged out to fully dead)
    still holds price rows, so a live listing under it must go through the CIK identity check
    rather than being classified new, even though it appears in neither dead-list CSV.
    """
    scan_con.execute("INSERT INTO prices VALUES ('LIMBO', '2020-01-01', 10.0)")
    scan_con.execute("INSERT INTO companies VALUES ('LIMBO', 7)")   # CIK on file from before blacklisting
    limbo_listing = pd.DataFrame([{'ticker': 'LIMBO', 'market': 'nasdaq', 'company_name': 'Limbo Corp'}])
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: limbo_listing)
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {'LIMBO': 7})   # same company, CIK matches
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    limbo = next(c for c in result.candidates if c.ticker == 'LIMBO')
    assert limbo.classification == 'reactivated_safe'
