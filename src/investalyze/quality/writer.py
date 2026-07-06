"""The one path quality-check findings reach the DB.

Each run replaces a check's rows wholesale (delete-then-insert), so `anomalies`
always reflects the latest run per check, including a check going clean.
"""

import duckdb
import pandas as pd

FINDING_COLS = ['SrcTable', 'Ticker', 'Date', 'Key', 'Details']

_DDL = """
    CREATE TABLE IF NOT EXISTS anomalies (
        CheckName  VARCHAR NOT NULL,
        Severity   VARCHAR NOT NULL,
        SrcTable   VARCHAR NOT NULL,
        Ticker     VARCHAR NOT NULL,
        Date       DATE,
        Key        VARCHAR,
        Details    VARCHAR NOT NULL,
        DetectedAt TIMESTAMP NOT NULL
    )
"""


def ensure_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create the anomalies table if it does not exist."""
    con.execute(_DDL)


def replace_findings(con: duckdb.DuckDBPyConnection, check_name: str, severity: str, findings: pd.DataFrame) -> int:
    """Replace all of `check_name`'s anomalies rows with `findings`, returning len(findings).

    Delete-then-insert keeps re-runs idempotent; an empty frame clears the check's
    previous findings. Raises ValueError if `findings` deviates from FINDING_COLS.
    """
    if list(findings.columns) != FINDING_COLS:
        raise ValueError(f'findings columns must be {FINDING_COLS}, got {list(findings.columns)}')
    con.execute('DELETE FROM anomalies WHERE CheckName = ?', [check_name])
    con.register('_findings', findings)
    try:
        con.execute(
            'INSERT INTO anomalies SELECT ?, ?, SrcTable, Ticker, Date, Key, Details, now() FROM _findings',
            [check_name, severity],
        )
    finally:
        con.unregister('_findings')
    return len(findings)
