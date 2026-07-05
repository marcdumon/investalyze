"""Tests for config.read — TOML settings with built-in defaults."""
from pathlib import Path

from investalyze.ingest import config


def test_read_uses_defaults_when_file_missing(tmp_path: Path):
    cfg = config.read(tmp_path / 'absent.toml')
    assert cfg.data_root == Path('data')
    assert cfg.db == 'investalyze.duckdb'
    assert cfg.provider('stooq') == {}


def test_read_none_uses_defaults():
    cfg = config.read(None)
    assert cfg.data_root == Path('data')
    assert cfg.db == 'investalyze.duckdb'


def test_read_reads_toml_values(tmp_path: Path):
    p = tmp_path / 'ingest.toml'
    p.write_text(
        'data_root = "/srv/data"\n'
        'db = "custom.duckdb"\n'
        '\n'
        '[stooq]\n'
        'update_file = "u.txt"\n'
    )
    cfg = config.read(p)
    assert cfg.data_root == Path('/srv/data')
    assert cfg.db == 'custom.duckdb'
    assert cfg.provider('stooq') == {'update_file': 'u.txt'}
