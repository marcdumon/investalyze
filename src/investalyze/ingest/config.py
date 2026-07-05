"""Configuration reading.

Reads `ingest.toml` (paths + per-provider settings) into a `Config`, with
built-in defaults so the package runs with no config file at all. Secrets (e.g.
the SimFin API key) are NOT here — they come from the environment.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_DATA_ROOT = 'data'
_DEFAULT_DB = 'investalyze.duckdb'
_DEFAULT_LOG_LEVEL = 'INFO'


@dataclass(frozen=True)
class Config:
    """Resolved ingest settings."""

    data_root: Path
    db: str
    log_level: str
    providers: dict[str, dict]

    def provider(self, name: str) -> dict:
        """Settings for one provider (empty dict if unconfigured)."""
        return self.providers.get(name, {})


def read(path: Path | None = None) -> Config:
    """Read `Config` from a TOML file, falling back to defaults.

    Top-level keys set `data_root` / `db`; every TOML table (e.g. `[stooq]`) is a
    per-provider settings section. A missing or unspecified file yields all
    defaults.
    """
    raw = tomllib.loads(path.read_text()) if path is not None and path.exists() else {}
    providers = {name: section for name, section in raw.items() if isinstance(section, dict)}
    return Config(
        data_root=Path(raw.get('data_root', _DEFAULT_DATA_ROOT)),
        db=raw.get('db', _DEFAULT_DB),
        log_level=raw.get('log_level', _DEFAULT_LOG_LEVEL),
        providers=providers,
    )
