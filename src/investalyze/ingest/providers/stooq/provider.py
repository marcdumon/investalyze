"""Stooq provider — bonds, currencies, indices (OHLCV). Manual bulk/update download.

Owns its whole flow: fetch -> transform -> save (through storage.write). Data dirs
are scaffolded separately by orchestrator.create_data_dirs (run once up front).
Split into more files inside this folder if it grows.
"""
import logging
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest import storage

log = logging.getLogger('investalyze.ingest.stooq')

_TABLE = 'market_data'
_KEY = ['Ticker', 'Date']


# Stooq source categories we keep (bonds/currencies/indices); everything else
# — equities (Yahoo owns them), money market, crypto — is out of scope.
_IN_SCOPE: frozenset[str] = frozenset({'bonds', 'currencies', 'indices'})


def _asset_class_from_category(category: str) -> str | None:
    """Map a Stooq source category dir to its AssetClass, or None if out of scope."""
    cat = category.lower()
    return cat if cat in _IN_SCOPE else None


def _asset_class_from_ticker(ticker: str) -> str | None:
    """Classify a flat-file ticker by pattern, or None if out of scope.

    The flat update file carries no category, so we infer it: `.B` suffix = bond,
    `^` prefix = index, a bare 6-letter code = currency pair. Everything else
    (`.US` equities, `.V` crypto, money-market, commodities) is dropped.
    """
    if ticker.endswith('.B'):
        return 'bonds'
    if ticker.startswith('^'):
        return 'indices'
    if len(ticker) == 6 and ticker.isalpha():
        return 'currencies'
    return None


_BULK_ZIP = 'd_world_txt.zip'   # world only — d_us is equities, which Yahoo owns
_UPDATE_FILE = 'data_d.txt'


def _extract_bulk(raw: Path, zip_name: str = _BULK_ZIP) -> None:
    """Unzip the world bulk into `raw/` (producing `raw/data/...`) when needed.

    No-op if no zip is present (rely on an already-extracted tree) or if the
    extracted tree is already newer than the zip. Otherwise extract — overwriting
    a stale tree.
    """
    zip_path = raw / zip_name
    if not zip_path.exists():
        log.debug(f'no bulk zip at {zip_path} — using existing tree')
        return
    tree = raw / 'data'
    if tree.exists() and tree.stat().st_mtime >= zip_path.stat().st_mtime:
        log.debug(f'extracted tree newer than {zip_name} — skipping unzip')
        return
    log.info(f'extracting {zip_name}')
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw)


def run(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict | None = None, *, update: bool = False) -> int:
    """Acquire Stooq data and load it into `market_data`. Returns the table row count.

    Full run reads the unzipped bulk tree under the provider's `raw/data/`;
    update reads the flat update file. Both upsert into the same table, so the
    daily update converges with the full history. `settings` (from config) can
    override the bulk-zip / update-file names.
    """
    settings = settings or {}
    raw = data_root / 'stooq' / 'raw'
    if update:
        log.info('update from flat file')
        df = _read_update_file(raw / settings.get('update_file', _UPDATE_FILE))
    else:
        log.info('full load from bulk tree')
        _extract_bulk(raw, settings.get('bulk_zip', _BULK_ZIP))
        df = _read_tree(raw / 'data')
    rows = storage.write(con, _TABLE, df, key=_KEY)
    log.info(f'done — {rows} rows in {_TABLE}')
    return rows


_CANONICAL_COLS = ['Ticker', 'Date', 'O', 'H', 'L', 'C', 'AssetClass']


def _read_update_file(path: Path) -> pd.DataFrame:
    """Read the flat Stooq update file (`data_d.txt`) into canonical market_data.

    The file mixes all instruments with no category, so each row is classified by
    ticker pattern; out-of-scope rows are dropped and the rest transformed with
    their per-row AssetClass.
    """
    raw = pd.read_csv(path)
    asset_class = raw['<TICKER>'].map(_asset_class_from_ticker)
    keep = asset_class.notna()
    log.info(f'{int(keep.sum())}/{len(raw)} update rows in scope')
    if not keep.any():
        return pd.DataFrame(columns=_CANONICAL_COLS)
    return _transform(raw[keep], asset_class[keep]).reset_index(drop=True)


def _read_tree(root: Path) -> pd.DataFrame:
    """Read every in-scope Stooq ticker file under `root` into canonical `market_data`.

    Stooq nests its files as `.../daily/<region>/<category>/[sub]/<ticker>.txt`,
    but only the category folder matters here. We find category folders by name
    anywhere under `root` (bonds / currencies / indices — see `_asset_class_from_category`),
    ignoring the frequency and region levels, and transform every `.txt` beneath
    each one, tagged with that folder's AssetClass.
    """
    frames: list[pd.DataFrame] = []
    for category_dir in sorted(p for p in root.rglob('*') if p.is_dir()):
        asset_class = _asset_class_from_category(category_dir.name)
        if asset_class is None:
            continue
        # zero-byte files exist in the real tree — skip them
        ticker_files = [f for f in sorted(category_dir.rglob('*.txt')) if f.stat().st_size > 0]
        before = len(frames)
        for ticker_file in ticker_files:
            frames.append(_transform(pd.read_csv(ticker_file), asset_class))
            log.debug(f'{ticker_file.name} read')
        rows = sum(len(f) for f in frames[before:])
        log.info(f'{asset_class} ({category_dir.parent.name}) — {len(ticker_files)} files, {rows} rows')
    if not frames:
        log.warning(f'no in-scope ticker files under {root}')
        return pd.DataFrame(columns=_CANONICAL_COLS)
    return pd.concat(frames, ignore_index=True)


def _transform(raw: pd.DataFrame, asset_class: str | pd.Series) -> pd.DataFrame:
    """Normalize raw Stooq OHLCV rows into canonical `market_data` rows.

    Strips the exchange suffix from the ticker (`10YUSY.B` -> `10YUSY`), parses
    the `YYYYMMDD` date, keeps O/H/L/C (volume is meaningless for these
    instruments), and tags the asset class (bonds / currencies / indices).
    `asset_class` may be a single label (folder path) or a per-row Series (flat
    file, classified by ticker).
    """
    return pd.DataFrame({
        'Ticker': raw['<TICKER>'].str.split('.').str[0].str.upper(),
        'Date': pd.to_datetime(raw['<DATE>'].astype(str), format='%Y%m%d').dt.date,
        'O': raw['<OPEN>'].astype(float),
        'H': raw['<HIGH>'].astype(float),
        'L': raw['<LOW>'].astype(float),
        'C': raw['<CLOSE>'].astype(float),
        'AssetClass': asset_class,
    })
