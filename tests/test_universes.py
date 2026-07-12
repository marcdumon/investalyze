"""Tests for the shared named-universe file helpers."""
import pytest

from investalyze.apps.universes import list_universes, load_universe, save_universe


def test_save_load_roundtrip(tmp_path):
    clean = save_universe(' My Universe! ', ['BBB', 'AAA'], universe_dir=tmp_path)
    assert clean == 'My_Universe'
    assert load_universe('My_Universe', universe_dir=tmp_path) == ['AAA', 'BBB']  # saved sorted
    assert list_universes(universe_dir=tmp_path) == ['My_Universe']


def test_empty_name_raises(tmp_path):
    with pytest.raises(ValueError):
        save_universe('***', ['AAA'], universe_dir=tmp_path)


def test_list_missing_dir(tmp_path):
    assert list_universes(universe_dir=tmp_path / 'nope') == []
