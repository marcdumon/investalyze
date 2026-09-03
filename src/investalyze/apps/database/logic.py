"""Pure shaping for the Database page: grid columns, the header a table gets, and query checks.

Kept free of Dash and DuckDB imports so it is testable standalone; the page module wires it up.
"""

import pandas as pd

from investalyze.apps.database import catalog

TABLE_COLUMNS = [
    {'field': 'name', 'headerName': 'Table', 'flex': 1},
    {'field': 'area', 'headerName': 'Area', 'width': 130},
    {'field': 'rows', 'headerName': 'Rows', 'width': 120,
     'valueFormatter': {'function': 'd3.format(",")(params.value)'}},
    {'field': 'columns', 'headerName': 'Cols', 'width': 90},
]

COLUMN_COLUMNS = [
    {'field': 'column', 'headerName': 'Column', 'flex': 1},
    {'field': 'type', 'headerName': 'Type', 'width': 130},
    {'field': 'empty', 'headerName': 'Empty', 'width': 110},
]

WRITES_NOTHING = 'Nothing writes this table yet.'
NO_DESCRIPTION = 'No description written for this table yet.'
STATS_SKIPPED = ('Too many rows to count empty values quickly, so that column is left out. '
                 'Everything else here is read from metadata.')


def table_rows(tables: pd.DataFrame) -> list[dict]:
    """The table list as grid rows, each carrying the area its catalogue entry puts it in."""
    if tables.empty:
        return []
    rows = []
    for table in tables.itertuples():
        entry = catalog.TABLES.get(table.name)
        rows.append({'name': table.name, 'rows': int(table.rows), 'columns': int(table.columns),
                     'area': entry.area if entry else '', 'internal': bool(entry and entry.internal)})
    return rows


def empty_share(nulls: int | None, rows: int) -> str:
    """How much of a column is empty, as a share rather than a raw count nobody can scale."""
    if nulls is None or rows <= 0:
        return ''
    if nulls == 0:
        return '-'
    return f'{nulls / rows:.1%}'


def column_rows(described: dict) -> list[dict]:
    """A table's columns as grid rows, with the empty share where it was cheap enough to count."""
    columns = described.get('columns')
    if columns is None or columns.empty:
        return []
    rows = []
    for column in columns.itertuples():
        nulls = getattr(column, 'nulls', None)
        rows.append({'column': column.column, 'type': column.type,
                     'empty': empty_share(nulls, described.get('rows', 0))})
    return rows


def header_lines(described: dict) -> list[str]:
    """What a selected table is, in the order a reader wants it: purpose, writer, key, span."""
    name = described.get('name', '')
    entry = catalog.TABLES.get(name)
    lines = [entry.purpose if entry else NO_DESCRIPTION]
    if entry:
        lines.append(f'Written by: {entry.written_by}' if entry.written_by else WRITES_NOTHING)
        if entry.key:
            lines.append(f'One row per: {", ".join(entry.key)}')

    facts = [f'{described.get("rows", 0):,} rows']
    if described.get('first'):
        facts.append(f'{described["first"]} to {described["last"]}')
    if described.get('tickers'):
        facts.append(f'{described["tickers"]:,} tickers')
    lines.append(', '.join(facts))
    if not described.get('stats', True):
        lines.append(STATS_SKIPPED)
    return lines


def check_query(sql: str | None) -> str | None:
    """The reason a query will not be run, or None when it is worth sending to the database.

    Only the cheap, unambiguous refusals belong here. Whether a statement is allowed at all is the
    database's own answer, given by the read-only connection, and repeating that judgement in a list
    of forbidden words would only be a second opinion that can be wrong.
    """
    text = (sql or '').strip().rstrip(';').strip()
    if not text:
        return 'Write a query first.'
    if ';' in text:
        return 'One statement at a time.'
    return None
