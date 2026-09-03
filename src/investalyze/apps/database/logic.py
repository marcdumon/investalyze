"""Pure shaping for the database page: grid columns, the header of a selected table, query guards.

Free of Dash and DuckDB, so every sentence the page shows can be asserted without a database.
"""

import pandas as pd

from investalyze.apps.database import catalog

TABLE_COLUMNS = [
    {'field': 'name', 'headerName': 'Table'},
    {'field': 'area', 'headerName': 'Area'},
    {'field': 'rows', 'headerName': 'Rows',
     'valueFormatter': {'function': 'd3.format(",")(params.value)'}},
    {'field': 'columns', 'headerName': 'Cols'},
]

COLUMN_COLUMNS = [
    {'field': 'column_name', 'headerName': 'Column'},
    {'field': 'data_type', 'headerName': 'Type'},
    {'field': 'empty', 'headerName': 'Empty'},
]

UNCATALOGUED = 'No description written for this table yet.'


def table_rows(tables: pd.DataFrame) -> list[dict]:
    """The table list as grid rows, each carrying the area its catalogue entry puts it in."""
    rows = []
    for table in tables.itertuples():
        entry = catalog.described(table.name)
        rows.append({'name': table.name, 'rows': int(table.rows), 'columns': int(table.columns),
                     'area': entry.area if entry else 'Uncatalogued',
                     'internal': bool(entry.internal) if entry else False})
    return rows


def purpose_of(name: str) -> tuple[str, str, str]:
    """A table's description, what writes it, and what makes a row unique."""
    entry = catalog.described(name)
    if entry is None:
        return UNCATALOGUED, '', ''
    return entry.purpose, entry.written_by, entry.key


def header_facts(described: dict) -> list[str]:
    """The short facts about a selected table, each left out when it does not apply.

    A table with no Date has no span to report and a table with no Ticker covers no names, so
    saying nothing beats saying none.
    """
    facts = [f'{described["rows"]:,} rows', f'{len(described["columns"])} columns']
    if described.get('tickers') is not None:
        facts.append(f'{described["tickers"]:,} tickers')
    span = described.get('span')
    if span:
        facts.append(f'{_as_date(span[0])} to {_as_date(span[1])}')
    return facts


def _as_date(value) -> str:
    """A date or timestamp as a plain ISO day."""
    return pd.Timestamp(value).date().isoformat()


def column_rows(described: dict) -> list[dict]:
    """The columns of a selected table, with how empty each one is."""
    counted, rows = described.get('nulls_counted', False), described.get('rows', 0)
    output = []
    for column in described['columns']:
        name = column['column_name']
        output.append({'column_name': name, 'data_type': column['data_type'],
                       'empty': _empty_share(described.get('nulls', {}).get(name), rows, counted)})
    return output


def _empty_share(nulls: int | None, rows: int, counted: bool) -> str:
    """How much of a column is empty, as a share, or a dash when it was not counted."""
    if not counted or nulls is None or not rows:
        return '-'
    if nulls == 0:
        return ''
    return f'{nulls / rows:.0%}' if nulls < rows else 'all'


def skipped_note(described: dict, limit: int) -> str:
    """Why the empty column is blank on a large table, or nothing when it was filled in."""
    if described.get('nulls_counted', True):
        return ''
    return (f'Empty counts are skipped above {limit:,} rows: they need a full pass, which costs '
            'seconds on a table this size.')


def check_query(sql: str) -> str | None:
    """The reason a query cannot be run, or None when it can.

    Writes are not checked for here. The connection is read-only, so the database refuses them
    itself, and a rule written in this file could only ever recognise the forms someone thought of.
    """
    text = (sql or '').strip().rstrip(';').strip()
    if not text:
        return 'Type a query first.'
    if ';' in text:
        return 'One statement at a time.'
    return None
