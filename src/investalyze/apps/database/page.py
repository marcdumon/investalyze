"""Database page: what every table holds, what wrote it, and a query box for the rest.

The app names `factor_panel`, `factor_ic` and `backtest_runs` on screen without ever saying what
they are. This page answers that, from a hand-written catalogue rather than a schema, because no
amount of column metadata explains what a row means.

The query box runs on a read-only connection, so the database itself refuses anything that writes.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
from dash import Input, Output, State, callback, clientside_callback, dcc, html
from dash.exceptions import PreventUpdate

from investalyze.apps.database import data, logic

BUSY = 'database busy, a job is currently running, try again once it finishes'
PICK = 'Select a table to see what it holds.'


def _grid(grid_id: str, columns: list[dict], height: int, selectable: bool = False) -> dag.AgGrid:
    """One AG Grid with the page's shared styling."""
    options = {'theme': 'legacy'}
    if selectable:
        options['rowSelection'] = {'mode': 'singleRow', 'checkboxes': False,
                                   'enableClickSelection': True}
    return dag.AgGrid(
        id=grid_id, columnDefs=columns, rowData=[],
        defaultColDef={'sortable': True, 'resizable': True},
        dashGridOptions=options, columnSize='autoSize', columnSizeOptions={'skipHeader': False},
        style={'height': height}, className='ag-theme-alpine-dark',
    )


def _note(text: str, colour: str = 'gray') -> dmc.Alert:
    """A short inline note for an empty, unavailable or refused panel."""
    return dmc.Alert(text, color=colour, variant='light')


def layout() -> html.Div:
    """The page body: tables on the left, what one holds on the right, a query box beneath."""
    header = dmc.Group([
        dmc.Title('Database', order=3),
        dmc.Text('what every table holds, and who writes it', size='sm', c='dimmed'),
    ], gap='md', mb=8)

    tables = dmc.Paper([
        dmc.Text('Tables', fw=600, size='sm', mb=4),
        html.Div(id='db-tables-notice'),
        _grid('db-tables', logic.TABLE_COLUMNS, 560, selectable=True),
    ], p='sm', withBorder=True)

    detail = dmc.Paper([
        html.Div(id='db-detail', children=_note(PICK)),
        dmc.Text('Columns', fw=600, size='sm', mt=10, mb=4),
        html.Div(id='db-columns-note'),
        _grid('db-columns', logic.COLUMN_COLUMNS, 260),
        dmc.Text('First rows', fw=600, size='sm', mt=10, mb=4),
        _grid('db-sample', [], 260),
    ], p='sm', withBorder=True)

    query = dmc.Paper([
        dmc.Group([dmc.Text('Query', fw=600, size='sm'),
                   dmc.Text('read-only: the database refuses anything that writes', size='xs',
                            c='dimmed')], gap=8, mb=6),
        dmc.Textarea(id='db-sql', placeholder='SELECT Ticker, count(*) FROM prices GROUP BY 1',
                     autosize=True, minRows=3, maxRows=10, size='xs', mb=6),
        dmc.Group([
            dmc.Button('Run query', id='db-btn-run', size='xs', color='orange'),
            dmc.Text(id='db-query-note', size='xs', c='dimmed'),
        ], gap='xs', mb=6),
        _grid('db-results', [], 320),
    ], p='sm', withBorder=True)

    return html.Div([
        header,
        dmc.Grid([
            dmc.GridCol(tables, span={'base': 12, 'lg': 4}),
            dmc.GridCol(detail, span={'base': 12, 'lg': 8}),
        ], gutter=12),
        html.Div(style={'height': '10px'}),
        query,
    ], style={'padding': '4px'})


@callback(Output('db-tables', 'rowData'), Output('db-tables-notice', 'children'),
          Input('db-tables', 'id'))
def load_tables(_):
    """Fill the table list once the page mounts."""
    try:
        return logic.table_rows(data.list_tables()), None
    except duckdb.Error:
        return [], _note(BUSY)


@callback(Output('db-detail', 'children'), Output('db-columns', 'rowData'),
          Output('db-columns-note', 'children'), Output('db-sample', 'columnDefs'),
          Output('db-sample', 'rowData'),
          Input('db-tables', 'selectedRows'))
def show_table(selected):
    """Describe the selected table: what it is for, its columns, and what its rows look like."""
    if not selected:
        return _note(PICK), [], None, [], []
    name = selected[0]['name']
    try:
        described = data.describe_table(name)
        sample = data.sample_rows(name)
    except duckdb.Error:
        return _note(BUSY), [], None, [], []
    if not described:
        return _note(f'{name} is no longer in the database.'), [], None, [], []

    purpose, written_by, key = logic.purpose_of(name)
    detail = [
        dmc.Group([dmc.Text(name, fw=700, size='sm'),
                   dmc.Text(' | '.join(logic.header_facts(described)), size='xs', c='dimmed')], gap=8),
        dmc.Text(purpose, size='sm', mt=6),
    ]
    if key:
        detail.append(dmc.Text(f'One row per: {key}', size='xs', c='dimmed', mt=6))
    if written_by:
        detail.append(dmc.Code(written_by, block=True))

    skipped = logic.skipped_note(described, data.NULL_COUNT_LIMIT)
    columns = [{'field': column} for column in sample.columns]
    return (html.Div(detail), logic.column_rows(described),
            _note(skipped) if skipped else None, columns, sample.to_dict('records'))


@callback(Output('db-results', 'columnDefs'), Output('db-results', 'rowData'),
          Output('db-query-note', 'children'),
          Input('db-btn-run', 'n_clicks'), State('db-sql', 'value'),
          prevent_initial_call=True)
def run_query(n_clicks, sql):
    """Run the typed query and show what came back, or why nothing did."""
    if not n_clicks:
        raise PreventUpdate
    refusal = logic.check_query(sql)
    if refusal:
        return [], [], refusal

    rows, truncated, failure = data.run_query(sql)
    if failure:
        return [], [], failure.splitlines()[0]
    note = f'{len(rows):,} rows' + (f', stopped at the first {data.QUERY_ROWS:,}' if truncated else '')
    return [{'field': column} for column in rows.columns], rows.to_dict('records'), note


dash.register_page(__name__, path='/database', name='Database', layout=layout)

for _grid_id in ('db-tables', 'db-columns', 'db-sample', 'db-results'):
    clientside_callback(
        "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
        Output(_grid_id, 'className'),
        Input('theme-switch', 'checked'),
    )
