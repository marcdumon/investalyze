"""Tests for the orchestrator: connect, run selected providers, summarize."""
from pathlib import Path

from investalyze.ingest import orchestrator, storage
from investalyze.ingest.config import Config

_HEADER = '<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>'


def _stooq_tree(data_root: Path, ticker: str) -> None:
    f = data_root / 'stooq' / 'raw' / 'data' / 'world' / 'bonds' / f'{ticker.lower()}.txt'
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f'{_HEADER}\n{ticker},D,20240321,000000,1.0,2.0,0.5,1.5,0,0\n')


def _config(data_root: Path) -> Config:
    return Config(data_root=data_root, db='investalyze.duckdb', providers={})


def test_run_executes_stooq_and_writes_db(tmp_path: Path):
    _stooq_tree(tmp_path, '10YUSY.B')
    summary = orchestrator.run(_config(tmp_path), providers=['stooq'])
    assert summary == {'stooq': 1}
    con = storage.connect(tmp_path)
    assert con.execute('SELECT COUNT(*) FROM market_data').fetchone()[0] == 1


def test_run_honors_provider_selection(tmp_path: Path):
    _stooq_tree(tmp_path, '10YUSY.B')
    summary = orchestrator.run(_config(tmp_path), providers=['stooq'])
    assert list(summary) == ['stooq']


def test_yahoo_is_registered():
    assert 'yahoo' in orchestrator.PROVIDERS
