"""End-to-end + fetch tests for the Yahoo provider (yfinance mocked)."""
from pathlib import Path

import pandas as pd
import pytest

from investalyze.ingest.providers.yahoo import price_data as provider


def _yahoo_multi(symbols):
    """Fake yf.download multi-ticker frame: MultiIndex columns (ticker, field)."""
    idx = pd.DatetimeIndex([pd.Timestamp('2024-03-21')], name='Date')
    fields = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    cols = pd.MultiIndex.from_product([symbols, fields])
    data = {(s, f): [1.0 if f != 'Volume' else 100] for s in symbols for f in fields}
    return pd.DataFrame(data, index=idx).reindex(columns=cols)


def test_fetch_splits_batch_into_per_ticker_frames(monkeypatch):
    monkeypatch.setattr(provider.yf, 'download', lambda *a, **k: _yahoo_multi(['A', 'AACB']))
    monkeypatch.setattr(provider.time, 'sleep', lambda *_: None)
    out = provider._fetch(['A', 'AACB'], start=None)
    assert set(out) == {'A', 'AACB'}
    assert list(out['A'].columns) == ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
    assert out['A'].index.name == 'Date'


def _single(symbol, close, adj, div=0.0, split=0.0):
    idx = pd.DatetimeIndex([pd.Timestamp('2024-03-21')], name='Date')
    return pd.DataFrame({'Open': [close], 'High': [close], 'Low': [close], 'Close': [close],
                         'Adj Close': [adj], 'Volume': [100], 'Dividends': [div],
                         'Stock Splits': [split]}, index=idx)


def _ticker_csv(tmp_path: Path, *symbols):
    raw = tmp_path / 'yahoo' / 'raw'
    raw.mkdir(parents=True)
    (raw / 'ticker.csv').write_text('ticker,market\n' + '\n'.join(f'{s},nyse' for s in symbols) + '\n')


def test_run_loads_prices_dividends_splits(tmp_path, monkeypatch):
    import duckdb
    _ticker_csv(tmp_path, 'AAA')
    monkeypatch.setattr(provider, '_fetch',
                        lambda syms, **k: {'AAA': _single('AAA', 10.0, 10.0, div=0.5, split=2.0)})
    con = duckdb.connect()
    n = provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})
    assert n == 1
    assert con.execute('SELECT Ticker, O, H, L, C, V, AC FROM prices').fetchall() == [('AAA', 10.0, 10.0, 10.0, 10.0, 100, 10.0)]
    assert con.execute('SELECT Ticker, Dividend FROM dividends').fetchall() == [('AAA', 0.5)]
    assert con.execute('SELECT Ticker, Ratio FROM splits').fetchall() == [('AAA', 2.0)]


def test_run_skips_empty_ticker_and_blacklists_it(tmp_path, monkeypatch):
    import duckdb
    _ticker_csv(tmp_path, 'DEAD')
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'DEAD': pd.DataFrame()})
    con = duckdb.connect()
    n = provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})
    assert n == 0
    blacklist = pd.read_csv(tmp_path / 'yahoo' / 'state' / 'price_blacklist.csv')
    assert blacklist.loc[0, 'ticker'] == 'DEAD'
    assert blacklist.loc[0, 'market'] == 'nyse'
    assert blacklist.loc[0, 'attempts'] == 1
    assert blacklist.loc[0, 'first_blacklisted'] == blacklist.loc[0, 'last_checked']


def test_run_skips_tickers_already_dead(tmp_path, monkeypatch):
    import duckdb
    _ticker_csv(tmp_path, 'GONE', 'AAA')
    state_dir = tmp_path / 'yahoo' / 'state'
    state_dir.mkdir(parents=True)
    pd.DataFrame([{'ticker': 'GONE', 'attempts': 5, 'first_blacklisted': '2024-01-01', 'died_on': '2024-02-01'}]
                 ).to_csv(state_dir / 'price_dead.csv', index=False)
    fetched: list[str] = []
    def fake_fetch(syms, **k):
        fetched.extend(syms)
        return {'AAA': _single('AAA', 10.0, 10.0)}
    monkeypatch.setattr(provider, '_fetch', fake_fetch)
    con = duckdb.connect()
    provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})
    assert 'GONE' not in fetched
    assert 'AAA' in fetched


def test_update_recomputes_ac_after_new_dividend(tmp_path, monkeypatch):
    import duckdb
    _ticker_csv(tmp_path, 'AAA')
    con = duckdb.connect()
    # initial full load: two days, no events -> AC == close
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'AAA': pd.DataFrame(
        {'Open': [10.0, 11.0], 'High': [10.0, 11.0], 'Low': [10.0, 11.0], 'Close': [10.0, 11.0],
         'Adj Close': [10.0, 11.0], 'Volume': [100, 100], 'Dividends': [0.0, 0.0], 'Stock Splits': [0.0, 0.0]},
        index=pd.DatetimeIndex([pd.Timestamp('2024-03-20'), pd.Timestamp('2024-03-21')], name='Date'))})
    provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})
    assert con.execute("SELECT AC FROM prices WHERE Date = '2024-03-20'").fetchone() == (10.0,)

    # update: new day 3 with a dividend -> day 1 & 2 AC must shift
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {'AAA': pd.DataFrame(
        {'Open': [12.0], 'High': [12.0], 'Low': [12.0], 'Close': [12.0], 'Adj Close': [12.0],
         'Volume': [100], 'Dividends': [0.6], 'Stock Splits': [0.0]},
        index=pd.DatetimeIndex([pd.Timestamp('2024-03-22')], name='Date'))})
    provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001}, update=True)
    ac_day1 = con.execute("SELECT AC FROM prices WHERE Date = '2024-03-20'").fetchone()
    assert ac_day1 is not None and ac_day1[0] < 10.0   # back-adjusted by the new dividend


def test_update_batches_from_earliest_start(tmp_path, monkeypatch):
    """Two tickers with different last dates update in ONE call from the earliest start."""
    import duckdb
    _ticker_csv(tmp_path, 'AAA', 'BBB')
    con = duckdb.connect()
    # seed: AAA last = 03-20, BBB last = 03-22
    monkeypatch.setattr(provider, '_fetch', lambda syms, **k: {
        'AAA': _single('AAA', 10.0, 10.0).set_axis(pd.DatetimeIndex([pd.Timestamp('2024-03-20')], name='Date')),
        'BBB': pd.DataFrame({'Open': [9.0], 'High': [9.0], 'Low': [9.0], 'Close': [9.0], 'Adj Close': [9.0],
                             'Volume': [100], 'Dividends': [0.0], 'Stock Splits': [0.0]},
                            index=pd.DatetimeIndex([pd.Timestamp('2024-03-22')], name='Date'))})
    provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001})

    calls = []
    monkeypatch.setattr(provider, '_fetch',
                        lambda syms, **k: (calls.append((tuple(syms), k.get('start'))) or {}))
    provider.run(con, tmp_path, {'ticker_file': 'ticker.csv', 'sleep': 0, 'batch_size': 10, 'ac_tolerance': 0.001}, update=True)
    assert len(calls) == 1                       # one batched call, not one per ticker
    syms, start = calls[0]
    assert set(syms) == {'AAA', 'BBB'}
    assert start == '2024-03-21'                 # earliest (AAA's 03-20) + 1 day
