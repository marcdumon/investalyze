"""Data Quality page: review open anomalies with evidence, log them, or stage cleaning fixes.

The grid shows anomalies that have not yet been logged (an anti-join against quality_log.toml through
a short-lived read-only connection), so you always work the unseen findings. Clicking a row shows
evidence (price candlestick or transposed fundamentals); checkboxes select multiple rows for bulk
action. Logging appends `[[log]]` entries (tag + comment) to quality_log.toml and the rows drop off
the list. Staging a fix appends entries to cleaning.toml for the selected rows and logs those rows
as fix-staged, so they drop off the list too; the fixes wait there until run from Control Panel ->
Cleaning (Preview / Apply). Everything on this page writes only the two TOML files; the database
changes when apply is run from the control panel.
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
from investalyze.cleaning import registry
from investalyze.ingest import storage

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / 'data'
LOG_PATH = REPO_ROOT / 'quality_log.toml'
CLEANING_PATH = REPO_ROOT / 'cleaning.toml'
ROW_LIMIT = 1000

BUSY = 'database busy, a job is currently running, try again once it finishes'
PREVIEW_STYLE = {'background': 'var(--mantine-color-default-hover)', 'padding': '8px', 'borderRadius': '6px',
                 'fontSize': '12px', 'whiteSpace': 'pre-wrap', 'marginTop': '6px', 'minHeight': '48px'}
PICK_PROMPT = dmc.Text('select an anomaly to inspect', size='sm', c='dimmed')

ANOMALY_COLUMNS = [
    {'field': 'Severity', 'width': 120, 'checkboxSelection': True,
     'headerCheckboxSelection': True, 'headerCheckboxSelectionFilteredOnly': True},
    {'field': 'CheckName', 'headerName': 'Check', 'width': 175},
    {'field': 'SrcTable', 'headerName': 'Table', 'width': 105},
    {'field': 'Ticker', 'width': 85, 'cellRenderer': 'TickerLinkIfStock'},
    {'field': 'Date', 'width': 105},
    {'field': 'Key', 'width': 150},
    {'field': 'Details', 'flex': 1, 'minWidth': 200},
]


def read_open_anomalies(con: duckdb.DuckDBPyConnection, log_entries: list, checks: list[str] | None,
                        severity: str, ticker: str | None, tables: list[str] | None,
                        limit: int) -> tuple[list[dict], int, list[str], list[str]]:
    """Return (capped open rows, total open, distinct checks, distinct tables): anomalies not yet in the log, filtered.

    Every filter applies before the limit, so rows matching a filter are found and counted even when
    the unfiltered sort order would have capped them out.
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


def _target_rows(selected: list | None, clicked: dict | None) -> list[dict]:
    """The rows an action applies to: the checked rows, or the clicked row when none are checked."""
    if selected:
        return selected
    return [clicked] if clicked else []


def _number(value: object) -> float | None:
    """A NumberInput's value as a number; a cleared input reports '' and becomes None."""
    return value if isinstance(value, (int, float)) else None


def _log_panel() -> dmc.Card:
    """Evidence panel plus the log form and the fix-staging form.

    Both forms act on the checked rows, falling back to the clicked row when none are checked.
    """
    return dmc.Card([
        html.Div(id='dq-evidence', children=PICK_PROMPT),
        dmc.Divider(my=10, label='Log finding(s)', labelPosition='center'),
        _labeled('tag', dmc.Select(id='dq-log-tag', data=quality_log.STANDARD_TAGS, value=quality_log.STANDARD_TAGS[0],
                                   size='xs')),
        _labeled('comment', dmc.Textarea(id='dq-log-comment', autosize=True, minRows=2, size='xs',
                                         placeholder='what you observed / what to check later')),
        html.Pre(id='dq-log-preview', style=PREVIEW_STYLE),
        dmc.Group([
            dmc.Button('Log finding(s)', id='dq-btn-log', size='xs', color='blue',
                       leftSection=DashIconify(icon='tabler:notebook')),
            dmc.Text(id='dq-log-status', size='xs', c='dimmed'),
        ], mt=8, gap=10),
        dmc.Divider(my=10, label='Stage cleaning fix', labelPosition='center'),
        _labeled('action', dmc.Select(id='dq-fix-action', size='xs', clearable=True,
                                      placeholder='what apply should do with the selected rows',
                                      data=[{'value': key, 'label': label} for key, label in actions.FIX_ACTIONS.items()])),
        dmc.Group([
            html.Div(_labeled('column', dmc.Select(id='dq-fix-column', data=[], size='xs', searchable=True,
                                                   clearable=True, placeholder='column')), style={'flex': 1}),
            html.Div(_labeled('new value', dmc.NumberInput(id='dq-fix-value', size='xs', hideControls=True)),
                     style={'flex': 1}),
        ], gap=10),
        _labeled('reason', dmc.Textarea(id='dq-fix-reason', autosize=True, minRows=2, size='xs',
                                        placeholder='why this correction is right (stored in cleaning.toml)')),
        html.Pre(id='dq-fix-preview', style=PREVIEW_STYLE),
        dmc.Group([
            dmc.Button('Stage fix', id='dq-btn-stage', size='xs', color='blue',
                       leftSection=DashIconify(icon='tabler:playlist-add')),
            dmc.Text(id='dq-fix-status', size='xs', c='dimmed'),
        ], mt=8, gap=10),
        dmc.Text('staging logs the rows as fix-staged (they drop off the list) and queues the fix in cleaning.toml '
                 'until you run it from Control Panel -> Cleaning', size='xs', c='dimmed', mt=4),
    ], withBorder=True, radius='md', padding='sm')


def layout() -> html.Div:
    """Build the data-quality page: filter bar + open-anomaly grid on the left, evidence + log on the right."""
    header = dmc.Group([
        dmc.Text('Data Quality', fw=700, size='lg'),
        dmc.Text('log problems or stage cleaning fixes; logged findings drop off the list', size='xs', c='dimmed'),
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

    # column filters stay off: they would search only the ROW_LIMIT-capped page, silently missing rows
    # the cap dropped; all filtering goes through the filter bar, which filters server-side before the cap.
    grid = dag.AgGrid(
        id='dq-grid', columnDefs=ANOMALY_COLUMNS, rowData=[],
        defaultColDef={'sortable': True, 'resizable': True},
        dashGridOptions={'animateRows': False, 'theme': 'legacy', 'rowSelection': 'multiple',
                         'suppressRowClickSelection': True},
        className='ag-theme-alpine-dark', style={'height': '70vh', 'width': '100%', '--ag-cell-horizontal-padding': '2px'},
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
    Input('dq-selected', 'data'), Input('dq-grid', 'selectedRows'),
    Input('dq-log-tag', 'value'), Input('dq-log-comment', 'value'),
)
def log_preview(clicked, selected, tag, comment):
    """Render the quality_log.toml blocks the Log button would append (first three, plus a count)."""
    rows = _target_rows(selected, clicked)
    if not rows:
        return 'check or click an anomaly first'
    blocks = [toml_io.serialize_block('log', actions.log_fields(row, tag or quality_log.STANDARD_TAGS[0], comment or ''))
              for row in rows[:3]]
    more = f'... and {len(rows) - 3} more entries\n' if len(rows) > 3 else ''
    return '\n'.join(blocks) + more


@callback(
    Output('dq-log-tick', 'data'), Output('dq-log-status', 'children'),
    Output('dq-selected', 'data', allow_duplicate=True), Output('dq-evidence', 'children', allow_duplicate=True),
    Output('dq-grid', 'selectedRows', allow_duplicate=True),
    Input('dq-btn-log', 'n_clicks'),
    State('dq-selected', 'data'), State('dq-grid', 'selectedRows'),
    State('dq-log-tag', 'value'), State('dq-log-comment', 'value'),
    State('dq-log-tick', 'data'),
    prevent_initial_call=True,
)
def do_log(_n, clicked, selected, tag, comment, tick):
    """Append the targeted findings to quality_log.toml, then drop them from the grid and clear the panel."""
    rows = _target_rows(selected, clicked)
    if not rows:
        return dash.no_update, 'check or click a finding first', dash.no_update, dash.no_update, dash.no_update
    blocks = [toml_io.serialize_block('log', actions.log_fields(row, tag or quality_log.STANDARD_TAGS[0], comment or ''))
              for row in rows]
    quality_log.append_log(LOG_PATH, '\n'.join(blocks))
    return (tick or 0) + 1, f'logged {len(rows)} finding(s)', None, PICK_PROMPT, []


@callback(
    Output('dq-fix-column', 'data'),
    Input('dq-grid', 'selectedRows'), Input('dq-selected', 'data'),
)
def column_options(selected, clicked):
    """Offer the value columns of the one table the targeted rows come from."""
    rows = _target_rows(selected, clicked)
    tables = sorted({row.get('SrcTable') for row in rows if row.get('SrcTable')})
    if len(tables) != 1:
        return []
    try:
        con = storage.connect(DATA_ROOT, read_only=True)
        try:
            described = con.execute(f'SELECT * FROM {tables[0]} LIMIT 0').description
        finally:
            con.close()
    except duckdb.Error:
        return []
    return [name for name, *_ in described if name not in ('Ticker', 'Date')]


@callback(
    Output('dq-fix-reason', 'value'),
    Input('dq-grid', 'selectedRows'), Input('dq-selected', 'data'),
)
def default_reason(selected, clicked):
    """Prefill the fix reason from the targeted issue's check and details; editable afterwards."""
    rows = _target_rows(selected, clicked)
    if not rows:
        return ''
    check = str(rows[0].get('CheckName') or '').strip()
    details = str(rows[0].get('Details') or '').strip()
    base = f'{check}: {details}' if details else check
    return base + (f' (+{len(rows) - 1} more rows)' if len(rows) > 1 else '')


@callback(
    Output('dq-fix-preview', 'children'),
    Input('dq-grid', 'selectedRows'), Input('dq-selected', 'data'), Input('dq-fix-action', 'value'),
    Input('dq-fix-column', 'value'), Input('dq-fix-value', 'value'), Input('dq-fix-reason', 'value'),
)
def fix_preview(selected, clicked, action, column, value, reason):
    """Render the cleaning.toml blocks the Stage button would append, or the reason it can't."""
    if not action:
        return 'choose an action'
    try:
        entries = actions.fix_entries(action, _target_rows(selected, clicked), column, _number(value), reason or '')
    except ValueError as err:
        return str(err)
    return '\n'.join(toml_io.serialize_block(section, fields) for section, fields in entries)


@callback(
    Output('dq-fix-status', 'children'), Output('dq-log-tick', 'data', allow_duplicate=True),
    Output('dq-selected', 'data', allow_duplicate=True), Output('dq-evidence', 'children', allow_duplicate=True),
    Output('dq-grid', 'selectedRows', allow_duplicate=True),
    Input('dq-btn-stage', 'n_clicks'),
    State('dq-grid', 'selectedRows'), State('dq-selected', 'data'), State('dq-fix-action', 'value'),
    State('dq-fix-column', 'value'), State('dq-fix-value', 'value'), State('dq-fix-reason', 'value'),
    State('dq-log-tick', 'data'),
    prevent_initial_call=True,
)
def stage_fix(_n, selected, clicked, action, column, value, reason, tick):
    """Append the previewed fix entries to cleaning.toml and log the rows as fix-staged, dropping them."""
    keep = (dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    if not action:
        return 'choose an action first', *keep
    rows = _target_rows(selected, clicked)
    try:
        entries = actions.fix_entries(action, rows, column, _number(value), reason or '')
    except ValueError as err:
        return str(err), *keep
    block = '\n'.join(toml_io.serialize_block(section, fields) for section, fields in entries)
    toml_io.append_block(CLEANING_PATH, block, registry.parse_fixes)
    log_blocks = [toml_io.serialize_block('log', actions.log_fields(row, 'fix-staged', (reason or '').strip()))
                  for row in rows]
    quality_log.append_log(LOG_PATH, '\n'.join(log_blocks))
    status = f'staged {len(entries)} entr{"y" if len(entries) == 1 else "ies"}; apply from Control Panel -> Cleaning'
    return status, (tick or 0) + 1, None, PICK_PROMPT, []
