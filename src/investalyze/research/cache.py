"""Cached results, stored as parquet files named by factor, parameters and fingerprint.

A result is a pure function of the factor definition, the parameters it was computed with, and the
data underneath, so its file name is `<factor>__<params_hash>__<fingerprint>`. Writing a result
deletes every other file for the same factor and parameters. A redefined factor therefore leaves no
stale result behind, and there is only ever one result per factor and parameter set.

Files rather than database tables: invalidating is deleting a file, a new parameter needs no schema
migration, and computing needs no write lock on a single-writer database. DuckDB reads and writes the
files, since it is already a dependency and pandas has no parquet engine installed here.
"""

import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd

VALUES, IC, BACKTEST, SUMMARY = 'values', 'ic', 'backtest', 'summary'


def _quoted(path: Path) -> str:
    """A path safe to place inside a SQL string literal."""
    return str(path).replace("'", "''")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame to a parquet file through an in-memory DuckDB."""
    with duckdb.connect() as con:
        con.register('result', frame)
        con.execute(f"COPY result TO '{_quoted(path)}' (FORMAT PARQUET)")


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a frame through an in-memory DuckDB."""
    with duckdb.connect() as con:
        return con.execute(f"SELECT * FROM read_parquet('{_quoted(path)}')").df()


def params_hash(**params: object) -> str:
    """A short stable hash of the parameters a result was computed with."""
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class Cache:
    """Results on disk, addressed by kind, factor, parameters and fingerprint."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _folder(self, kind: str) -> Path:
        return self.root / kind

    def _stem(self, factor: str, params: str, fingerprint: str) -> str:
        return f'{factor}__{params}__{fingerprint}'

    def path(self, kind: str, factor: str, params: str, fingerprint: str) -> Path:
        """Where one result is stored."""
        return self._folder(kind) / f'{self._stem(factor, params, fingerprint)}.parquet'

    def get(self, kind: str, factor: str, params: str, fingerprint: str) -> pd.DataFrame | None:
        """The stored result, or None when it is missing or was computed from another definition."""
        path = self.path(kind, factor, params, fingerprint)
        if not path.is_file():
            return None
        return _read_parquet(path)

    def meta(self, kind: str, factor: str, params: str, fingerprint: str) -> dict | None:
        """The metadata recorded with a result."""
        path = self.path(kind, factor, params, fingerprint).with_suffix('.json')
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def put(self, kind: str, factor: str, params: str, fingerprint: str, frame: pd.DataFrame,
            meta: dict | None = None) -> Path:
        """Store a result and delete any earlier result for the same factor and parameters."""
        folder = self._folder(kind)
        folder.mkdir(parents=True, exist_ok=True)
        superseded = self.stale(kind, factor, params, fingerprint)
        path = self.path(kind, factor, params, fingerprint)
        _write_parquet(frame, path)
        record = {'factor': factor, 'params': params, 'fingerprint': fingerprint,
                  'rows': int(len(frame)), 'computed_at': pd.Timestamp.now().isoformat()}
        path.with_suffix('.json').write_text(json.dumps(record | (meta or {}), indent=2, default=str))
        for old in superseded:
            old.unlink(missing_ok=True)
            old.with_suffix('.json').unlink(missing_ok=True)
        return path

    def stale(self, kind: str, factor: str, params: str, fingerprint: str) -> list[Path]:
        """Results for this factor and parameters computed from a different definition."""
        folder = self._folder(kind)
        if not folder.is_dir():
            return []
        current = self._stem(factor, params, fingerprint)
        found = []
        for path in folder.glob(f'{factor}__{params}__*.parquet'):
            if path.stem != current:
                found.append(path)
        return sorted(found)

    def _summary_path(self, factor: str, fingerprint: str) -> Path:
        return self._folder(SUMMARY) / f'{factor}__{fingerprint}.json'

    def get_summary(self, factor: str, fingerprint: str) -> dict:
        """Headline results recorded for the current definition of a factor, empty when none."""
        path = self._summary_path(factor, fingerprint)
        return json.loads(path.read_text()) if path.is_file() else {}

    def put_summary(self, factor: str, fingerprint: str, updates: dict) -> dict:
        """Merge headline results into a factor's summary and delete summaries of earlier definitions."""
        folder = self._folder(SUMMARY)
        folder.mkdir(parents=True, exist_ok=True)
        path = self._summary_path(factor, fingerprint)
        merged = self.get_summary(factor, fingerprint) | updates
        path.write_text(json.dumps(merged, indent=2, default=str))
        for old in folder.glob(f'{factor}__*.json'):
            if old != path:
                old.unlink(missing_ok=True)
        return merged

    def forget(self, factor: str) -> int:
        """Delete every result for one factor."""
        removed = 0
        for kind_folder in self.root.glob('*'):
            for path in kind_folder.glob(f'{factor}__*'):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def clear(self) -> None:
        """Delete the whole cache. Always safe, since every result can be recomputed."""
        shutil.rmtree(self.root, ignore_errors=True)
