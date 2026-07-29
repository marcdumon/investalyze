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


NASDAQLISTED_BLANK_SYMBOL_SAMPLE = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AACG|ATA Creativity Global - American Depositary Shares|G|N|N|100|N|N
|Some Blank Symbol Co - Common Stock|G|N|N|100|N|N
File Creation Time: 07/27/2026 05:00:01
"""


def test_parse_nasdaqlisted_drops_a_blank_symbol_row():
    frame = listings._parse_nasdaqlisted(NASDAQLISTED_BLANK_SYMBOL_SAMPLE)
    assert list(frame['ticker']) == ['AACG']   # the blank-symbol row never reaches ticker.csv


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


def test_classify_fills_in_the_previous_company_name_for_a_dead_ticker():
    candidates = listings.classify(sample_listings(), tracked=set(), dead_tickers={'ZOMB'},
                                   live_cik={'ZOMB': 99}, last_cik={'ZOMB': 1},
                                   last_company_name={'ZOMB': 'Old Zombie Inc'})
    assert next(c for c in candidates if c.ticker == 'ZOMB').previous_company_name == 'Old Zombie Inc'


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
    con.execute('CREATE TABLE companies (Ticker VARCHAR, CIK BIGINT, CompanyName VARCHAR)')
    con.execute('CREATE TABLE prices (Ticker VARCHAR, Date DATE, C DOUBLE)')
    return con


def test_scan_classifies_and_writes_candidates_csv(monkeypatch, scan_con, data_root):
    state_dir = data_root / 'yahoo' / 'state'
    pd.DataFrame([{'ticker': 'RELS', 'attempts': 3, 'first_blacklisted': '2020-01-01', 'died_on': '2020-06-01'}]
                ).to_csv(state_dir / 'price_dead.csv', index=False)
    scan_con.execute("INSERT INTO companies VALUES ('RELS', 1, 'Relisted Co')")
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {'RELS': 1})   # same CIK: reactivated_safe
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    assert result.saved_to == data_root / 'yahoo' / 'state' / listings.CANDIDATES_FILE
    # SPCX/ZOMB are 'new' (added straight to ticker.csv, not saved for review); RELS needs review
    assert set(c.ticker for c in result.candidates) == {'SPCX', 'ZOMB', 'RELS'}
    saved = pd.read_csv(result.saved_to)
    assert set(saved['ticker']) == {'RELS'}


def test_scan_adds_new_candidates_to_ticker_csv_automatically(monkeypatch, scan_con, data_root):
    # no dead-list/companies setup here, so SPCX/ZOMB/RELS all classify as 'new'
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {})
    listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    tracked = set(pd.read_csv(data_root / 'yahoo' / 'raw' / 'ticker.csv')['ticker'])
    assert tracked == {'AAPL', 'SPCX', 'ZOMB', 'RELS'}


def test_scan_does_not_add_reactivated_candidates_to_ticker_csv(monkeypatch, scan_con, data_root):
    state_dir = data_root / 'yahoo' / 'state'
    pd.DataFrame([{'ticker': 'ZOMB', 'attempts': 3, 'first_blacklisted': '2020-01-01', 'died_on': '2020-06-01'}]
                ).to_csv(state_dir / 'price_dead.csv', index=False)
    scan_con.execute("INSERT INTO companies VALUES ('ZOMB', 1, 'Zombie Corp Original')")
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {'ZOMB': 99})   # collision: different CIK
    listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    tracked = set(pd.read_csv(data_root / 'yahoo' / 'raw' / 'ticker.csv')['ticker'])
    assert 'ZOMB' not in tracked   # a collision must be reviewed, never auto-added


def test_scan_skips_manually_blacklisted_tickers(monkeypatch, scan_con, data_root):
    state_dir = data_root / 'yahoo' / 'state'
    pd.DataFrame([{'ticker': 'SPCX', 'reason': 'not interesting', 'blacklisted_on': '2026-01-01'}]
                ).to_csv(state_dir / listings.MANUAL_BLACKLIST_FILE, index=False)
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {})
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    assert 'SPCX' not in {c.ticker for c in result.candidates}
    assert 'SPCX' not in set(pd.read_csv(data_root / 'yahoo' / 'raw' / 'ticker.csv')['ticker'])


def test_scan_suggests_a_free_key_for_collision_candidates(monkeypatch, scan_con, data_root):
    state_dir = data_root / 'yahoo' / 'state'
    pd.DataFrame([{'ticker': 'ZOMB', 'attempts': 3, 'first_blacklisted': '2020-01-01', 'died_on': '2020-06-01'}]
                ).to_csv(state_dir / 'price_dead.csv', index=False)
    scan_con.execute("INSERT INTO companies VALUES ('ZOMB', 1, 'Zombie Corp Original')")   # old company on file
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: sample_listings())
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {'ZOMB': 99})   # different CIK now trading
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    zomb = next(c for c in result.candidates if c.ticker == 'ZOMB')
    assert zomb.classification == 'reactivated_collision'
    assert zomb.suggested_new_ticker == 'ZOMB.1'
    assert zomb.previous_company_name == 'Zombie Corp Original'   # so the UI can explain the collision


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
    scan_con.execute("INSERT INTO companies VALUES ('LIMBO', 7, 'Limbo Corp')")   # on file from before blacklisting
    limbo_listing = pd.DataFrame([{'ticker': 'LIMBO', 'market': 'nasdaq', 'company_name': 'Limbo Corp'}])
    monkeypatch.setattr(listings, 'fetch_nasdaq_listings', lambda: limbo_listing)
    monkeypatch.setattr(listings, 'fetch_sec_cik_map', lambda: {'LIMBO': 7})   # same company, CIK matches
    result = listings.scan(scan_con, data_root, cleaning_config=data_root / 'cleaning.toml')
    limbo = next(c for c in result.candidates if c.ticker == 'LIMBO')
    assert limbo.classification == 'reactivated_safe'


# --- blacklist_tickers --------------------------------------------------------------


def test_blacklist_tickers_deletes_rows_drops_from_ticker_csv_and_records_it(scan_con, data_root):
    scan_con.execute("INSERT INTO prices VALUES ('JUNK', '2020-01-01', 1.0), ('OTHER', '2020-01-01', 2.0)")
    scan_con.execute("INSERT INTO companies VALUES ('JUNK', 5, 'Junk Corp')")
    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAPL', 'JUNK'], 'market': ['nasdaq', 'nasdaq']}).to_csv(ticker_csv, index=False)

    deleted = listings.blacklist_tickers(scan_con, data_root, ['JUNK'], 'not a real business')

    assert deleted == 1   # only the JUNK price row; companies isn't a ticker-keyed table
    remaining = scan_con.execute("SELECT Ticker FROM prices").fetchall()
    assert remaining == [('OTHER',)]
    assert set(pd.read_csv(ticker_csv)['ticker']) == {'AAPL'}
    manual = pd.read_csv(data_root / 'yahoo' / 'state' / listings.MANUAL_BLACKLIST_FILE)
    assert manual.iloc[0]['ticker'] == 'JUNK'
    assert manual.iloc[0]['reason'] == 'not a real business'


def test_blacklist_tickers_applies_one_reason_to_every_ticker(scan_con, data_root):
    scan_con.execute("INSERT INTO prices VALUES ('AAA', '2020-01-01', 1.0), ('BBB', '2020-01-01', 2.0)")
    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAA', 'BBB', 'CCC'], 'market': ['nasdaq'] * 3}).to_csv(ticker_csv, index=False)

    listings.blacklist_tickers(scan_con, data_root, ['AAA', 'BBB'], 'blacklisted as a universe')

    assert set(pd.read_csv(ticker_csv)['ticker']) == {'CCC'}
    manual = pd.read_csv(data_root / 'yahoo' / 'state' / listings.MANUAL_BLACKLIST_FILE)
    assert set(manual['ticker']) == {'AAA', 'BBB'}
    assert set(manual['reason']) == {'blacklisted as a universe'}   # same reason recorded for both


def test_blacklist_tickers_with_empty_list_is_a_no_op(scan_con, data_root):
    assert listings.blacklist_tickers(scan_con, data_root, [], 'reason') == 0


# --- add_listings ------------------------------------------------------------------


def test_add_listings_adds_safe_rows_directly(scan_con, data_root):
    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAPL'], 'market': ['nasdaq']}).to_csv(ticker_csv, index=False)
    rows = [{'ticker': 'RELS', 'market': 'nyse', 'classification': 'reactivated_safe', 'new_ticker': None}]

    added, renamed, skipped = listings.add_listings(scan_con, data_root, rows)

    assert (added, renamed, skipped) == (1, 0, 0)
    assert set(pd.read_csv(ticker_csv, keep_default_na=False)['ticker']) == {'AAPL', 'RELS'}


def test_add_listings_does_not_corrupt_a_ticker_that_looks_like_a_na_sentinel(scan_con, data_root):
    # pandas treats the bare string 'NA' as a missing value by default; ticker.csv reads/writes
    # in this module must survive a real ticker symbol that happens to collide with that
    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    pd.DataFrame({'ticker': ['NA'], 'market': ['nyse']}).to_csv(ticker_csv, index=False)
    rows = [{'ticker': 'RELS', 'market': 'nyse', 'classification': 'reactivated_safe', 'new_ticker': None}]

    listings.add_listings(scan_con, data_root, rows)

    tickers = pd.read_csv(ticker_csv, keep_default_na=False)['ticker'].tolist()
    assert 'NA' in tickers
    assert not any(pd.isna(t) for t in tickers)


def test_add_listings_renames_collision_before_adding(scan_con, data_root):
    scan_con.execute("INSERT INTO prices VALUES ('BID', '2019-01-01', 10.0), ('OTHER', '2019-01-01', 5.0)")
    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAPL'], 'market': ['nasdaq']}).to_csv(ticker_csv, index=False)
    rows = [{'ticker': 'BID', 'market': 'nasdaq', 'classification': 'reactivated_collision', 'new_ticker': 'BID.1'}]

    added, renamed, skipped = listings.add_listings(scan_con, data_root, rows)

    assert (added, renamed, skipped) == (1, 1, 0)
    old_rows = scan_con.execute("SELECT Ticker FROM prices WHERE Ticker = 'BID'").fetchall()
    assert old_rows == []
    moved_rows = scan_con.execute("SELECT Ticker FROM prices WHERE Ticker = 'BID.1'").fetchall()
    assert moved_rows == [('BID.1',)]
    assert set(pd.read_csv(ticker_csv)['ticker']) == {'AAPL', 'BID'}


def test_add_listings_skips_collision_row_with_blank_new_ticker(scan_con, data_root):
    scan_con.execute("INSERT INTO prices VALUES ('BID', '2019-01-01', 10.0)")
    ticker_csv = data_root / 'yahoo' / 'raw' / 'ticker.csv'
    pd.DataFrame({'ticker': ['AAPL'], 'market': ['nasdaq']}).to_csv(ticker_csv, index=False)
    rows = [{'ticker': 'BID', 'market': 'nasdaq', 'classification': 'reactivated_collision', 'new_ticker': ''}]

    added, renamed, skipped = listings.add_listings(scan_con, data_root, rows)

    assert (added, renamed, skipped) == (0, 0, 1)
    assert 'BID' not in set(pd.read_csv(ticker_csv)['ticker'])   # never added unrenamed
    still_there = scan_con.execute("SELECT Ticker FROM prices WHERE Ticker = 'BID'").fetchall()
    assert still_there == [('BID',)]   # untouched, not merged
