"""Data Quality page: review open anomalies with evidence and log each one to build understanding.

The grid shows anomalies that have not yet been logged (an anti-join against quality_log.toml through
a short-lived read-only connection), so you always work the unseen findings. Clicking a row shows
evidence (price candlestick or transposed fundamentals). Logging a finding appends a `[[log]]` entry
(tag + comment) to quality_log.toml and the row drops off the list. Nothing here modifies the raw
tables or the anomalies table; the log is a non-destructive triage overlay.
"""

from pathlib import Path

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from investalyze.apps.data_quality import actions, evidence, quality_log, toml_io
from investalyze.apps.screener.data import get_pool
from investalyze.ingest import storage

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / 'data'
LOG_PATH = REPO_ROOT / 'quality_log.toml'
ROW_LIMIT = 1000

BUSY = 'database busy, a job is currently running, try again once it finishes'
PREVIEW_STYLE = {'background': 'var(--mantine-color-default-hover)', 'padding': '8px', 'borderRadius': '6px',
                 'fontSize': '12px', 'whiteSpace': 'pre-wrap', 'marginTop': '6px', 'minHeight': '48px'}
PICK_PROMPT = dmc.Text('select an anomaly to inspect', size='sm', c='dimmed')

ANOMALY_COLUMNS = [
    {'field': 'Severity', 'width': 90},
    {'field': 'CheckName', 'headerName': 'Check', 'width': 175},
    # filter: False: the sidebar's dq-filter-table already filters this field server-side, before the
    # grid's row cap; a second, page-local column filter here would look equivalent but silently
    # search a different, arbitrary subset (the exact bug this replaced).
    {'field': 'SrcTable', 'headerName': 'Table', 'width': 105, 'filter': False},
    {'field': 'Ticker', 'width': 85, 'cellRenderer': 'TickerLinkIfStock'},
    {'field': 'Date', 'width': 105},
    {'field': 'Key', 'width': 150},
    {'field': 'Details', 'flex': 1, 'minWidth': 200},
]


def read_open_anomalies(con: duckdb.DuckDBPyConnection, log_entries: list, checks: list[str] | None,
                        severity: str, ticker: str | None, tables: list[str] | None,
                        limit: int) -> tuple[list[dict], int, list[str], list[str]]:
    """Return (capped open rows, total open, distinct checks, distinct tables): anomalies not yet in the log, filtered.

    `tables` filters server-side, before the limit is applied; the grid's own Table column filter
    only searches the already-loaded page, so without this a table with few anomalies relative to
    others (which sort ahead of it) could be entirely capped out before that column filter ever sees it.
    """
    clauses: list[str] = []
    params: list = []
    if checks:
        clauses.append(f"CheckName IN ({', '.join('?' for _ in checks)})")
        params += checks
    if severity in ('error', 'warn'):
        clauses.append('Severity = ?')
        params.append(severity)
    if ticker:
        clauses.append('Ticker ILIKE ?')
        params.append(f'%{ticker.strip()}%')
    if tables:
        clauses.append(f"SrcTable IN ({', '.join('?' for _ in tables)})")
        params += tables
    if log_entries:
        clauses.append(
            'NOT EXISTS (SELECT 1 FROM logged_raw l WHERE l.check_name = anomalies.CheckName '
            'AND l.ticker = anomalies.Ticker AND l.log_date::DATE IS NOT DISTINCT FROM anomalies.Date '
            'AND l.log_key IS NOT DISTINCT FROM anomalies.Key)'
        )
    where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''

    if log_entries:
        con.register('logged_raw', quality_log.log_keys_frame(log_entries))
    try:
        total = int(con.execute(f'SELECT count(*) FROM anomalies{where}', params).fetchone()[0])  # type: ignore[index]
        frame = con.execute(
            f'SELECT CheckName, Severity, SrcTable, Ticker, Date, Key, Details FROM anomalies{where} '
            f'ORDER BY Severity, CheckName, Ticker LIMIT {limit}', params,
        ).df()
        all_checks = [row[0] for row in con.execute('SELECT DISTINCT CheckName FROM anomalies ORDER BY CheckName').fetchall()]
        all_tables = [row[0] for row in con.execute('SELECT DISTINCT SrcTable FROM anomalies ORDER BY SrcTable').fetchall()]
    finally:
        if log_entries:
            con.unregister('logged_raw')
    frame['Date'] = frame['Date'].apply(lambda value: None if value is None or str(value) == 'NaT' else str(value)[:10])
    frame['Key'] = frame['Key'].where(frame['Key'].notna(), None)
    frame['Details'] = frame['Details'].apply(
        lambda value: actions.format_details_numbers(value) if isinstance(value, str) else value)
    frame['IsStock'] = frame['Ticker'].isin(set(get_pool()['Ticker']))
    return frame.to_dict('records'), total, all_checks, all_tables


def _labeled(label: str, component) -> html.Div:
    """Small label above its control."""
    return html.Div([dmc.Text(label, size='xs', c='dimmed', mb=2), component], style={'marginBottom': '8px'})


def _log_panel() -> dmc.Card:
    """Evidence panel plus the log form for the selected finding."""
    return dmc.Card([
        html.Div(id='dq-evidence', children=PICK_PROMPT),
        dmc.Divider(my=10, label='Log this finding', labelPosition='center'),
        _labeled('tag', dmc.Select(id='dq-log-tag', data=quality_log.STANDARD_TAGS, value=quality_log.STANDARD_TAGS[0],
                                   size='xs')),
        _labeled('comment', dmc.Textarea(id='dq-log-comment', autosize=True, minRows=2, size='xs',
                                         placeholder='what you observed / what to check later')),
        html.Pre(id='dq-log-preview', style=PREVIEW_STYLE),
        dmc.Group([
            dmc.Button('Log finding', id='dq-btn-log', size='xs', color='blue',
                       leftSection=DashIconify(icon='tabler:notebook')),
            dmc.Text(id='dq-log-status', size='xs', c='dimmed'),
        ], mt=8, gap=10),
    ], withBorder=True, radius='md', padding='sm')


def layout() -> html.Div:
    """Build the data-quality page: filter bar + open-anomaly grid on the left, evidence + log on the right."""
    header = dmc.Group([
        dmc.Text('Data Quality', fw=700, size='lg'),
        dmc.Text('review and log problems; logged findings drop off the list', size='xs', c='dimmed'),
        dmc.Button('Refresh', id='dq-btn-refresh', size='xs', variant='subtle',
                   leftSection=DashIconify(icon='tabler:reload')),
    ], mb=10, gap=12)

    filters = dmc.Group([
        dmc.MultiSelect(id='dq-filter-check', data=[], placeholder='checks', clearable=True, size='xs',
                        style={'flex': 1, 'minWidth': 0}),
        dmc.MultiSelect(id='dq-filter-table', data=[], placeholder='tables', clearable=True, size='xs',
                        style={'width': '160px'}),
        dmc.SegmentedControl(id='dq-filter-severity', value='all', size='xs',
                             data=[{'label': 'All', 'value': 'all'}, {'label': 'Error', 'value': 'error'},
                                   {'label': 'Warn', 'value': 'warn'}]),
        dmc.TextInput(id='dq-filter-ticker', placeholder='ticker', size='xs', style={'width': '130px'}),
    ], mb=8, gap=10, wrap='nowrap')

    grid = dag.AgGrid(
        id='dq-grid', columnDefs=ANOMALY_COLUMNS, rowData=[],
        defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
        dashGridOptions={'animateRows': False, 'theme': 'legacy'},
        className='ag-theme-alpine-dark', style={'height': '70vh', 'width': '100%'},
    )

    return html.Div([
        header,
        dmc.Grid([
            dmc.GridCol([filters, html.Div(id='dq-count', style={'fontSize': '12px', 'marginBottom': '4px'}),
                         html.Div(id='dq-notice'), grid], span={'base': 12, 'lg': 7}),
            dmc.GridCol(_log_panel(), span={'base': 12, 'lg': 5}),
        ], gutter=12),
        dcc.Store(id='dq-selected'),
        dcc.Store(id='dq-log-tick', data=0),
    ], style={'padding': '4px'})


dash.register_page(__name__, path='/quality', name='Data Quality', layout=layout)

clientside_callback(
    "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
    Output('dq-grid', 'className'), Input('theme-switch', 'checked'),
)


@callback(
    Output('dq-grid', 'rowData'), Output('dq-count', 'children'), Output('dq-filter-check', 'data'),
    Output('dq-filter-table', 'data'), Output('dq-notice', 'children'),
    Input('dq-btn-refresh', 'n_clicks'), Input('dq-filter-check', 'value'), Input('dq-filter-severity', 'value'),
    Input('dq-filter-ticker', 'value'), Input('dq-filter-table', 'value'), Input('dq-log-tick', 'data'),
)
def refresh_grid(_refresh, checks, severity, ticker, tables, _tick):
    """Reload the open-anomaly grid for the current filters and the latest log state."""
    log_entries = quality_log.read_log(LOG_PATH)
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        rows, total, all_checks, all_tables = read_open_anomalies(
            con, log_entries, checks, severity, ticker, tables, ROW_LIMIT)
    except duckdb.Error:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dmc.Text(BUSY, size='xs', c='dimmed')
    finally:
        con.close()
    shown = (f'showing {len(rows)} of {total:,} open' + (' (filter to see more)' if total > len(rows) else '')
             + f' | {len(log_entries)} logged')
    return rows, shown, all_checks, all_tables, None


@callback(
    Output('dq-selected', 'data'), Output('dq-evidence', 'children'),
    Input('dq-grid', 'cellClicked'), State('dq-grid', 'rowData'), State('theme-switch', 'checked'),
    prevent_initial_call=True,
)
def select_row(cell, rows, dark):
    """Store the clicked row and render its evidence."""
    row_id = (cell or {}).get('rowId')
    if row_id is None or not str(row_id).isdigit() or not rows or int(row_id) >= len(rows):
        raise PreventUpdate
    row = rows[int(row_id)]
    return row, evidence.safe_evidence_children(row, bool(dark))


@callback(
    Output('dq-log-preview', 'children'),
    Input('dq-selected', 'data'), Input('dq-log-tag', 'value'), Input('dq-log-comment', 'value'),
)
def log_preview(row, tag, comment):
    """Render the quality_log.toml block the Log button would append."""
    if not row:
        return 'select an anomaly first'
    return toml_io.serialize_block('log', actions.log_fields(row, tag or quality_log.STANDARD_TAGS[0], comment or ''))


@callback(
    Output('dq-log-tick', 'data'), Output('dq-log-status', 'children'),
    Output('dq-selected', 'data', allow_duplicate=True), Output('dq-evidence', 'children', allow_duplicate=True),
    Input('dq-btn-log', 'n_clicks'),
    State('dq-selected', 'data'), State('dq-log-tag', 'value'), State('dq-log-comment', 'value'),
    State('dq-log-tick', 'data'),
    prevent_initial_call=True,
)
def do_log(_n, row, tag, comment, tick):
    """Append the selected finding to quality_log.toml, then drop it from the grid and clear the panel."""
    if not row:
        return dash.no_update, 'select a finding first', dash.no_update, dash.no_update
    block = toml_io.serialize_block('log', actions.log_fields(row, tag or quality_log.STANDARD_TAGS[0], comment or ''))
    quality_log.append_log(LOG_PATH, block)
    return (tick or 0) + 1, f"logged {row['CheckName']} for {row['Ticker']}", None, PICK_PROMPT
