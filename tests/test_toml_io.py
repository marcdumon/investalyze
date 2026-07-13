"""Tests for the data-quality TOML serializer and validate-then-write append."""

from datetime import date
from pathlib import Path

import pytest

from investalyze.apps.data_quality import toml_io


def test_serialize_block_scalars_dates_and_lists():
    block = toml_io.serialize_block('log', {'check': 'x', 'ticker': 'FOO', 'date': date(2019, 3, 11),
                                            'value': 42.5, 'tickers': ['A', 'B']})
    assert "check = 'x'" in block
    assert 'date = 2019-03-11' in block
    assert 'value = 42.5' in block
    assert "tickers = ['A', 'B']" in block


def test_serialize_block_omits_none_fields():
    block = toml_io.serialize_block('log', {'check': 'x', 'ticker': 'FOO', 'key': None})
    assert 'key' not in block


def test_serialize_block_double_quotes_apostrophe():
    block = toml_io.serialize_block('log', {'comment': "it's fine"})
    assert '"it\'s fine"' in block


def test_append_block_grows_and_validates(tmp_path: Path):
    path = tmp_path / 'x.toml'
    seen = []
    toml_io.append_block(path, "[[log]]\ncheck = 'a'\n", lambda text: seen.append(text) or [])
    toml_io.append_block(path, "[[log]]\ncheck = 'b'\n", lambda text: seen.append(text) or [])
    assert path.read_text().count('[[log]]') == 2
    assert "check = 'b'" in seen[-1]  # validator saw the whole combined file


def test_append_block_atomic_on_invalid(tmp_path: Path):
    path = tmp_path / 'x.toml'
    original = "[[log]]\ncheck = 'a'\n"
    path.write_text(original)

    def reject(_text):
        raise ValueError('bad entry')

    with pytest.raises(ValueError, match='bad entry'):
        toml_io.append_block(path, "[[log]]\ncheck = 'b'\n", reject)
    assert path.read_text() == original
