"""Tests for investalyze.ingest.listings: NASDAQ/SEC source parsing and classification."""

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


def test_parse_otherlisted_keeps_only_nyse_and_nysemkt():
    frame = listings._parse_otherlisted(OTHERLISTED_SAMPLE)
    assert sorted(frame['ticker']) == ['A', 'SPCX']
    assert set(frame['market']) == {'nyse'}


def test_parse_sec_cik_map_upper_cases_tickers():
    cik_map = listings._parse_sec_cik_map(SEC_TICKERS_SAMPLE)
    assert cik_map == {'AAPL': 320193, 'TSLA': 1318605}
