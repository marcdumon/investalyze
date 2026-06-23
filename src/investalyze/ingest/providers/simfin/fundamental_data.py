"""SimFin provider — fundamentals (income/balance/cashflow) + company metadata.

US market only. Both vintages (as-reported + restated) fold into an `IsRestated`
column, annual + quarterly into a `Period` column. Bulk REST download (api-key ->
S3 redirect), refreshed by file age; no incremental feed. Owns its whole flow:
acquire zips into raw/ -> extract + union via DuckDB -> merge-upsert via storage.
Split into more files in this folder if it grows.
"""
import datetime
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import requests

from investalyze.ingest import storage
from investalyze.ingest.providers.simfin import columns

log = logging.getLogger('investalyze.ingest.simfin')

_BASE_URL = 'https://prod.simfin.com/api/bulk-download/s3'
_MARKET = 'us'
_STATEMENTS = ['income', 'balance', 'cashflow']
# (variant suffix, Period, IsRestated)
_VARIANTS: list[tuple[str, str, bool]] = [
    ('annual-full-asreported', 'A', False),
    ('quarterly-full-asreported', 'Q', False),
    ('annual-full', 'A', True),
    ('quarterly-full', 'Q', True),
]
_KEY_FUND = ['Ticker', 'Market', 'Period', 'IsRestated', 'Fiscal Year', 'Fiscal Period']
_KEY_COMPANIES = ['Ticker', 'Market']


@dataclass(frozen=True)
class _Spec:
    """One bulk file to download."""
    dataset: str
    variant: str | None
    market: str | None
    refresh_days: int

    @property
    def filename(self) -> str:
        """`market-dataset-variant.zip`, omitting absent parts."""
        parts = [p for p in (self.market, self.dataset, self.variant) if p]
        return '-'.join(parts) + '.zip'

    @property
    def url(self) -> str:
        """The SimFin bulk-download URL for this file."""
        params = f'dataset={self.dataset}'
        if self.variant:
            params += f'&variant={self.variant}'
        if self.market:
            params += f'&market={self.market}'
        return f'{_BASE_URL}?{params}'


def _specs(refresh_fundamentals: int, refresh_meta: int) -> list[_Spec]:
    """The full download matrix: fundamentals (3 statements x 4 variants) + metadata."""
    fundamentals = [_Spec(s, v, _MARKET, refresh_fundamentals)
                    for s in _STATEMENTS for v, _period, _restated in _VARIANTS]
    meta = [_Spec('companies', None, _MARKET, refresh_meta),
            _Spec('industries', None, None, refresh_meta)]
    return fundamentals + meta


def _needs_download(dest: Path, refresh_days: int) -> bool:
    """True if `dest` is missing or at least `refresh_days` old."""
    if not dest.exists():
        return True
    age = (datetime.date.today() - datetime.date.fromtimestamp(dest.stat().st_mtime)).days
    return age >= refresh_days


def _download_file(url: str, headers: dict, dest: Path) -> None:
    """Download `url` to `dest` atomically.

    SimFin replies with a 30x redirect to a presigned S3 URL; auth goes only to
    SimFin, never forwarded to S3 (S3 rejects extra Authorization headers).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    try:
        r = requests.get(url, headers=headers, allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308):
            download_url, extra = r.headers['Location'], {}
        else:
            r.raise_for_status()
            download_url, extra = url, headers
        with requests.get(download_url, headers=extra, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        tmp.rename(dest)
        log.info(f'downloaded {dest.name}')
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _acquire(raw_dir: Path, settings: dict, api_key: str) -> None:
    """Download every missing/stale bulk zip into `raw_dir`; a failure is logged, not fatal."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {'Authorization': f'api-key {api_key}'}
    specs = _specs(settings['refresh_days_fundamentals'], settings['refresh_days_meta'])
    for spec in specs:
        dest = raw_dir / spec.filename
        if not _needs_download(dest, spec.refresh_days):
            log.debug(f'{spec.filename} fresh — skip')
            continue
        log.info(f'downloading {spec.filename}')
        try:
            _download_file(spec.url, headers, dest)
        except Exception as e:   # one bad file must not abort the rest
            log.error(f'failed {spec.filename}: {e}')


def _extract(zip_path: Path, dest_dir: Path) -> Path:
    """Extract the single CSV member of `zip_path` into `dest_dir`; return its path."""
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        z.extract(name, dest_dir)
        return dest_dir / name


def _read_statement(con: duckdb.DuckDBPyConnection, raw_dir: Path, tmp: Path, statement: str) -> pd.DataFrame:
    """Union the present variant zips for one statement, tagged + SimFinId->SrcId.

    Missing variants are skipped (partial coverage OK); no zip at all -> empty frame.
    """
    parts: list[str] = []
    for variant, period, is_restated in _VARIANTS:
        zp = raw_dir / f'{_MARKET}-{statement}-{variant}.zip'
        if not zp.exists():
            log.warning(f'missing {zp.name} — skip')
            continue
        csv = _extract(zp, tmp)
        parts.append(
            f"SELECT Ticker, SimFinId AS SrcId, 'simfin' AS Src, '{_MARKET}' AS Market, "
            f"'{period}' AS Period, {str(is_restated).lower()} AS IsRestated, "
            f"* EXCLUDE (Ticker, SimFinId) "
            f"FROM read_csv('{csv}', delim=';', union_by_name=true, null_padding=true)"
        )
    if not parts:
        return pd.DataFrame()
    return con.execute(' UNION ALL BY NAME '.join(parts)).df()


def _read_companies(con: duckdb.DuckDBPyConnection, raw_dir: Path, tmp: Path) -> pd.DataFrame:
    """Companies left-joined to industries (Industry/Sector), SimFinId->SrcId.

    `parallel=false` on the companies read: Business Summary holds free text that
    can break parallel CSV splitting.
    """
    czip = raw_dir / f'{_MARKET}-companies.zip'
    izip = raw_dir / 'industries.zip'
    if not czip.exists() or not izip.exists():
        log.warning('missing companies/industries zip — skip companies')
        return pd.DataFrame()
    ccsv = _extract(czip, tmp)
    icsv = _extract(izip, tmp)
    return con.execute(
        f"""SELECT c.Ticker, c.SimFinId AS SrcId, 'simfin' AS Src, c.Market,
                   i.Industry, i.Sector, c."Company Name", c.IndustryId,
                   c.* EXCLUDE (Ticker, SimFinId, Market, "Company Name", IndustryId)
            FROM read_csv('{ccsv}', delim=';', union_by_name=true, null_padding=true, parallel=false) c
            LEFT JOIN read_csv('{icsv}', delim=';') i ON c.IndustryId = i.IndustryId"""
    ).df().rename(columns=columns.COMPANIES)


def run(con: duckdb.DuckDBPyConnection, data_root: Path, settings: dict, *, update: bool = False) -> int:
    """Download SimFin bulk fundamentals + companies and merge-upsert into the DB.

    No incremental feed: `update` is ignored (refresh-by-age governs downloads).
    `settings` is the `[simfin]` config (missing key raises). Returns the total
    row count across the simfin tables.
    """
    api_key = os.environ.get('SIMFIN_API_KEY')
    if not api_key:
        raise RuntimeError('SIMFIN_API_KEY not set (put it in .env or the environment)')
    raw_dir = data_root / 'simfin' / 'raw'
    _acquire(raw_dir, settings, api_key)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for statement in _STATEMENTS:
            df = _read_statement(con, raw_dir, tmp, statement)
            if df.empty:
                log.warning(f'no data for {statement} — skip')
                continue
            storage.write(con, statement, df, key=_KEY_FUND)
            log.info(f'{statement}: {len(df)} rows')
        companies = _read_companies(con, raw_dir, tmp)
        if not companies.empty:
            storage.write(con, 'companies', companies, key=_KEY_COMPANIES)
            log.info(f'companies: {len(companies)} rows')
    counts = {t: con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
              for t in _STATEMENTS + ['companies']
              if t in {r[0] for r in con.execute('SHOW TABLES').fetchall()}}
    total = sum(counts.values())
    log.info(f'done — {total} rows across simfin tables ({counts})')
    return total
