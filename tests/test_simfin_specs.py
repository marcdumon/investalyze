"""Tests for the SimFin download matrix and freshness check."""
import datetime
import os

from investalyze.ingest.providers.simfin.fundamental_data import _needs_download, _Spec, _specs


def test_spec_filename_and_url():
    s = _Spec('income', 'annual-full', 'us', 90)
    assert s.filename == 'us-income-annual-full.zip'
    assert s.url == ('https://prod.simfin.com/api/bulk-download/s3'
                     '?dataset=income&variant=annual-full&market=us')


def test_industries_spec_has_no_market_or_variant():
    s = _Spec('industries', None, None, 90)
    assert s.filename == 'industries.zip'
    assert s.url.endswith('?dataset=industries')


def test_specs_cover_the_locked_matrix():
    specs = _specs(90, 30)
    names = {s.filename for s in specs}
    # 3 statements x 4 variants = 12 fundamentals + companies + industries
    assert len([s for s in specs if s.dataset in {'income', 'balance', 'cashflow'}]) == 12
    assert 'us-income-annual-full-asreported.zip' in names
    assert 'us-companies.zip' in names
    assert 'industries.zip' in names
    assert 'us-income-annual.zip' not in names           # 'derived' variant — excluded
    assert not any(s.market == 'de' for s in specs)       # us only
    assert {s.refresh_days for s in specs if s.dataset == 'industries'} == {30}


def test_needs_download(tmp_path):
    missing = tmp_path / 'x.zip'
    assert _needs_download(missing, 90) is True
    fresh = tmp_path / 'fresh.zip'
    fresh.write_bytes(b'z')
    assert _needs_download(fresh, 90) is False
    stale = tmp_path / 'stale.zip'
    stale.write_bytes(b'z')
    old = datetime.datetime.now().timestamp() - 100 * 86400
    os.utime(stale, (old, old))
    assert _needs_download(stale, 90) is True
