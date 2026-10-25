"""The SEC contact address: SEC_CONTACT_EMAIL in .env in the app folder (not in git), in the User-Agent of every SEC
request; SEC refuses a request without a real contact address."""

import pytest

from investalyze import paths, sec_contact


def test_the_user_agent_has_the_contact_address_of_env(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('SEC_CONTACT_EMAIL=jan.peters@example.com\n')
    monkeypatch.setattr(paths, 'APP_DIR', tmp_path)
    monkeypatch.delenv('SEC_CONTACT_EMAIL', raising=False)
    assert sec_contact.user_agent('ingest') == 'investalyze ingest (jan.peters@example.com)'


def test_without_a_contact_address_a_request_stops_with_the_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'APP_DIR', tmp_path)
    monkeypatch.delenv('SEC_CONTACT_EMAIL', raising=False)
    with pytest.raises(RuntimeError, match='SEC_CONTACT_EMAIL'):
        sec_contact.user_agent('ingest')
