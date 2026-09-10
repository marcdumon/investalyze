"""Where computed answers live, and how they stop being wrong.

An answer is a pure function of the idea that produced it, the question asked, and the data
underneath. So it is stored under a name carrying all three: `<idea>__<question>__<fingerprint>`.
Nothing else is needed to know whether a stored answer is still the current one.

That naming is what makes staleness impossible rather than merely detectable. Writing an answer
sweeps every other file for the same idea and question, so a redefined idea's old answers are gone
the moment a new one lands. There is never a second answer to choose between, and so no picker, no
versions, and nothing for a user to delete by hand.

Parquet files on disk rather than database tables: invalidating is unlinking a file, no schema has
to be migrated when a question gains a parameter, and computing an answer needs no write lock on a
database that only allows one writer at a time. DuckDB reads and writes the files, since it is
already a dependency and pandas has no parquet engine installed here.
"""

import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd

VALUES, PREDICT, BACKTEST, VALIDATE = 'values', 'predict', 'backtest', 'validate'


def _quoted(path: Path) -> str:
    """A path safe to place inside a SQL string literal."""
    return str(path).replace("'", "''")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    try:
        con.register('answer', frame)
        con.execute(f"COPY answer TO '{_quoted(path)}' (FORMAT PARQUET)")
    finally:
        con.close()


def _read_parquet(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(f"SELECT * FROM read_parquet('{_quoted(path)}')").df()
    finally:
        con.close()


def signature(**parts) -> str:
    """A short stable hash of the parameters that make a question the question it is."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class Cache:
    """Answers on disk, addressed by idea, question and fingerprint."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _folder(self, kind: str) -> Path:
        return self.root / kind

    def _stem(self, idea: str, question: str, fingerprint: str) -> str:
        return f'{idea}__{question}__{fingerprint}'

    def path(self, kind: str, idea: str, question: str, fingerprint: str) -> Path:
        """Where one answer is stored."""
        return self._folder(kind) / f'{self._stem(idea, question, fingerprint)}.parquet'

    def get(self, kind: str, idea: str, question: str, fingerprint: str) -> pd.DataFrame | None:
        """The stored answer, or None when it was never computed or is no longer current."""
        path = self.path(kind, idea, question, fingerprint)
        if not path.is_file():
            return None
        return _read_parquet(path)

    def meta(self, kind: str, idea: str, question: str, fingerprint: str) -> dict | None:
        """What was recorded alongside an answer: when it ran, and what it covered."""
        path = self.path(kind, idea, question, fingerprint).with_suffix('.json')
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def put(self, kind: str, idea: str, question: str, fingerprint: str, frame: pd.DataFrame,
            meta: dict | None = None) -> Path:
        """Store an answer, discarding any earlier answer to the same question about the same idea."""
        folder = self._folder(kind)
        folder.mkdir(parents=True, exist_ok=True)
        superseded = self.stale(kind, idea, question, fingerprint)
        path = self.path(kind, idea, question, fingerprint)
        _write_parquet(frame, path)
        record = {'idea': idea, 'question': question, 'fingerprint': fingerprint,
                  'rows': int(len(frame)), 'computed_at': pd.Timestamp.now().isoformat()}
        path.with_suffix('.json').write_text(json.dumps(record | (meta or {}), indent=2, default=str))
        for old in superseded:
            old.unlink(missing_ok=True)
            old.with_suffix('.json').unlink(missing_ok=True)
        return path

    def stale(self, kind: str, idea: str, question: str, fingerprint: str) -> list[Path]:
        """Answers to this question about this idea that a different definition produced."""
        folder = self._folder(kind)
        if not folder.is_dir():
            return []
        current = self._stem(idea, question, fingerprint)
        prefix = f'{idea}__{question}__'
        found = []
        for path in folder.glob(f'{prefix}*.parquet'):
            if path.stem != current:
                found.append(path)
        return sorted(found)

    def forget(self, idea: str) -> int:
        """Drop every answer about one idea, for when the idea itself is deleted."""
        removed = 0
        for kind_folder in self.root.glob('*'):
            for path in kind_folder.glob(f'{idea}__*'):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def clear(self) -> None:
        """Throw the whole cache away. Always safe, since every answer can be recomputed."""
        shutil.rmtree(self.root, ignore_errors=True)
