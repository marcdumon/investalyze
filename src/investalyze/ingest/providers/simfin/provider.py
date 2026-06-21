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
