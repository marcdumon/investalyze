"""Tests for SimFin acquire: redirect handling + refresh-by-age orchestration."""
from pathlib import Path

from investalyze.ingest.providers.simfin import fundamental_data as provider


class _Resp:
    """Minimal stand-in for a requests.Response (context-manager + streaming)."""
    def __init__(self, status=200, headers=None, content=b''):
        self.status_code = status
        self.headers = headers or {}
        self._content = content
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size=1):
        yield self._content
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_download_follows_s3_redirect_without_auth(tmp_path: Path, monkeypatch):
    calls = []
    def fake_get(url, headers=None, allow_redirects=True, stream=False, timeout=None):
        calls.append((url, headers))
        if 'simfin.com' in url:
            return _Resp(302, headers={'Location': 'https://s3.example/file'})
        return _Resp(200, content=b'zipbytes')
    monkeypatch.setattr(provider.requests, 'get', fake_get)
    dest = tmp_path / 'us-income-annual-full.zip'
    provider._download_file(_spec_url(), {'Authorization': 'api-key K'}, dest)
    assert dest.read_bytes() == b'zipbytes'
    s3_headers = [h for (u, h) in calls if 's3.example' in u][0]
    assert s3_headers == {}                       # auth NOT forwarded to S3


def _spec_url():
    return ('https://prod.simfin.com/api/bulk-download/s3'
            '?dataset=income&variant=annual-full&market=us')


def test_acquire_downloads_stale_skips_fresh(tmp_path: Path, monkeypatch):
    downloaded = []
    monkeypatch.setattr(provider, '_download_file',
                        lambda url, headers, dest: downloaded.append(dest.name))
    # everything reports stale -> every spec downloaded
    monkeypatch.setattr(provider, '_needs_download', lambda dest, days: True)
    provider._acquire(tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 30}, 'K')
    assert 'us-income-annual-full.zip' in downloaded
    assert 'industries.zip' in downloaded
    assert len(downloaded) == 14                  # 12 fundamentals + companies + industries

    downloaded.clear()
    monkeypatch.setattr(provider, '_needs_download', lambda dest, days: False)
    provider._acquire(tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 30}, 'K')
    assert downloaded == []                        # all fresh -> nothing downloaded


def test_acquire_continues_after_one_download_error(tmp_path: Path, monkeypatch):
    def flaky(url, headers, dest):
        if 'balance' in dest.name:
            raise RuntimeError('boom')
        dest.write_bytes(b'z')
    monkeypatch.setattr(provider, '_download_file', flaky)
    monkeypatch.setattr(provider, '_needs_download', lambda dest, days: True)
    provider._acquire(tmp_path, {'refresh_days_fundamentals': 90, 'refresh_days_meta': 30}, 'K')
    assert (tmp_path / 'us-income-annual-full.zip').exists()      # others still written
