"""Screener page: metadata + factor-percentile filters over one pool, two-grid curation, saved universes.

The pool (metrics metadata + factor columns) is built once on first visit through a short-lived
read-only connection and cached at module level; Refresh rebuilds it. Factor ranks are recomputed
over the metadata-filtered candidates, so a percentile always means 'among my current candidates'.
While a job holds the DB's write lock the page shows a busy notice instead of a traceback.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
from dash import ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from investalyze.analysis import factors
from investalyze.apps.screener import metrics
from investalyze.apps.screener.data import clear_cache, get_pool
from investalyze.apps.screener.detail import safe_detail_children
from investalyze.apps.screener.logic import apply_filters, apply_metadata_filters, composite_score, compute_ranks
from investalyze.apps.universes import list_universes, load_universe, save_universe

IDENTITY_FIELDS = ['Ticker', 'name', 'sector', 'industry', 'mcap_bn', 'mcap_bucket']
IDENTITY_COLUMNS = [
    {'field': 'sel', 'headerName': '', 'editable': True, 'cellDataType': 'boolean', 'width': 46, 'pinned': 'left'},
    {'field': 'Ticker', 'pinned': 'left', 'width': 82, 'cellRenderer': 'TickerLink'},
    {'field': 'name', 'headerName': 'Company', 'width': 170},
    {'field': 'sector', 'headerName': 'Sector', 'width': 120},
    {'field': 'industry', 'headerName': 'Industry', 'width': 145},
    {'field': 'mcap_bn', 'headerName': 'MCap $bn', 'width': 88},
    {'field': 'mcap_bucket', 'headerName': 'Bucket', 'width': 78},
]
SEL_FIELDS = ['Ticker', 'name']
SEL_COLUMNS = [
    {'field': 'sel', 'headerName': '', 'editable': True, 'cellDataType': 'boolean', 'width': 50},
    {'field': 'Ticker', 'width': 90, 'cellRenderer': 'TickerLink'},
    {'field': 'name', 'headerName': 'Company', 'flex': 1},
]

BUSY = 'database busy, a job is currently running, try again once it finishes'


def grid_title(text: str, button_text: str, button_id: str) -> dmc.Group:
    """Slim bar above a grid: a bold caption on the left, an action button on the right."""
    return dmc.Group([
        dmc.Text(text, size='xs', fw=700),
        dmc.Button(button_text, id=button_id, size='compact-xs', variant='light'),
    ], justify='space-between', mb=4, wrap='nowrap')


def factor_row(factor: str) -> dmc.Group:
    """One sidebar row: factor name with min/max percentile inputs, kept on a single line."""
    return dmc.Group([
        dmc.Text(factor, size='xs', style={'width': '106px'}),
        dmc.NumberInput(id={'type': 'fmin', 'factor': factor}, placeholder='min', min=0, max=100,
                        size='xs', style={'width': '72px'}),
        dmc.NumberInput(id={'type': 'fmax', 'factor': factor}, placeholder='max', min=0, max=100,
                        size='xs', style={'width': '72px'}),
    ], gap=6, wrap='nowrap')


def apply_action(
    action: str, selection: list[str], checkbox_events: list[dict],
    filtered: list[str], universe_name: str | None
) -> tuple[list[str], str]:
    """Return the new (selection, status message) for one user action.

    Every ticker is either undecided (the candidates table) or selected (the condensed table).
    """
    sel = set(selection)
    if action == 'grid':
        for event in checkbox_events:
            if event.get('colId') != 'sel':
                continue
            ticker = event.get('data', {}).get('Ticker') or event.get('rowId')
            if event.get('value'):
                sel.add(ticker)
            else:
                sel.discard(ticker)
        message = f'{len(sel)} selected'
    elif action == 'sel-grid':
        for event in checkbox_events:
            if event.get('colId') == 'sel' and not event.get('value'):
                sel.discard(event.get('data', {}).get('Ticker') or event.get('rowId'))
        message = f'{len(sel)} selected'
    elif action == 'btn-select-all':
        sel |= set(filtered)
        message = f'added {len(filtered)} tickers, {len(sel)} selected'
    elif action == 'btn-deselect-all':
        sel = set()
        message = 'selection cleared'
    elif action == 'btn-load':
        if not universe_name:
            return sorted(sel), 'pick a universe to load first'
        sel = set(load_universe(universe_name))
        message = f"loaded '{universe_name}' ({len(sel)} tickers)"
    else:
        message = ''
    return sorted(sel), message


def layout() -> html.Div:
    """Build the screener page. The pool is loaded (and cached) on first visit."""
    try:
        df = get_pool()
    except duckdb.Error:
        return html.Div(BUSY, style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '14px', 'padding': '24px'})

    filter_row = dmc.Group([
        dmc.TextInput(id='f-search', label='Search', placeholder='ticker or company', debounce=500, size='xs',
                      leftSection=DashIconify(icon='tabler:search', width=14), style={'width': '190px'}),
        dmc.MultiSelect(id='f-sector', label='Sector', data=sorted(df['sector'].unique()), placeholder='all',
                        clearable=True, searchable=True, size='xs', style={'width': '190px'}),
        dmc.MultiSelect(id='f-industry', label='Industry', data=[], placeholder='all',
                        clearable=True, searchable=True, size='xs', style={'width': '210px'}),
        dmc.NumberInput(id='f-mindvol', label='Min $vol (mn/d)', min=0, size='xs', style={'width': '110px'}),
        dmc.NumberInput(id='f-minyears', label='Min years', min=0, size='xs', style={'width': '90px'}),
        dmc.NumberInput(id='f-maxanom', label='Max anomalies', min=0, size='xs', style={'width': '110px'}),
        html.Div([dmc.Text('Listing', size='xs', fw=500, mb=2),
                  dmc.SegmentedControl(id='f-active', value='all', size='xs', data=['all', 'active', 'delisted'])]),
        html.Div([dmc.Text('Market-cap bucket', size='xs', fw=500, mb=2),
                  html.Div(dmc.ChipGroup(dmc.Group([dmc.Chip(label, value=label, size='xs')
                                                    for label in metrics.MCAP_LABELS], gap=4),
                                         id='f-bucket', multiple=True, value=[]),
                           style={'height': '30px', 'display': 'flex', 'alignItems': 'center'})]),
        dmc.Text(id='filter-count', size='sm', fw=700, style={'marginLeft': 'auto', 'alignSelf': 'flex-end'}),
    ], gap=10, align='flex-end', px=12, pt=8, pb=6, style={'flexWrap': 'wrap'})

    factor_panel = dmc.Accordion([
        dmc.AccordionItem([
            dmc.AccordionControl('Factor percentile filters & composite score', py=2),
            dmc.AccordionPanel([
                dmc.MultiSelect(id='score-factors', data=factors.FACTORS, label='Composite score factors',
                                size='xs', clearable=True, mb=10, style={'maxWidth': '520px'}),
                dmc.Group([
                    dmc.Stack([dmc.Text(family, size='xs', fw=700)] + [factor_row(f) for f in members], gap=4)
                    for family, members in factors.FAMILIES.items()
                ], gap=24, align='flex-start', style={'flexWrap': 'wrap'}),
            ]),
        ], value='factors'),
    ], multiple=True, styles={'control': {'paddingLeft': '12px'}, 'content': {'padding': '4px 12px 10px'}},
       style={'borderBottom': '1px solid var(--mantine-color-default-border)'})

    header = dmc.Group([
        dmc.Text('Screener', fw=700, size='lg'),
        dmc.Text(id='sel-count', size='xs', c='dimmed'),
        dmc.SegmentedControl(id='mode', value='raw', size='xs',
                             data=[{'label': 'Raw', 'value': 'raw'}, {'label': 'Rank', 'value': 'rank'}]),
        dmc.TextInput(id='universe-name', placeholder='universe name', size='xs', style={'width': '150px'}),
        dmc.Button('Save', id='btn-save', size='xs', variant='default',
                   leftSection=DashIconify(icon='tabler:device-floppy', width=14)),
        dmc.Select(id='universe-dd', data=list_universes(), placeholder='saved universes', clearable=True,
                   size='xs', style={'width': '190px'}),
        dmc.Button('Load', id='btn-load', size='xs', variant='default',
                   leftSection=DashIconify(icon='tabler:folder-open', width=14)),
        dmc.Button('Refresh', id='btn-refresh', size='xs', variant='subtle',
                   leftSection=DashIconify(icon='tabler:reload', width=14)),
        dmc.Text(id='save-status', size='xs', c='teal'),
    ], gap=8, px=12, py=8,
       style={'borderBottom': '1px solid var(--mantine-color-default-border)', 'flexWrap': 'wrap'})

    grids = html.Div([
        html.Div([
            grid_title('Candidates', 'select all', 'btn-select-all'),
            dag.AgGrid(
                id='grid', columnDefs=[], rowData=[],
                defaultColDef={'sortable': True, 'filter': True, 'resizable': True, 'width': 85},
                dashGridOptions={'singleClickEdit': True, 'animateRows': False, 'theme': 'legacy'},
                className='ag-theme-alpine-dark',
                style={'flex': 1, 'width': '100%', 'minHeight': 0, '--ag-cell-horizontal-padding': '3px'},
            ),
        ], style={'flex': 1, 'minWidth': 0, 'display': 'flex', 'flexDirection': 'column'}),
        html.Div([
            grid_title('Selected (uncheck to remove)', 'deselect all', 'btn-deselect-all'),
            dag.AgGrid(
                id='sel-grid', columnDefs=SEL_COLUMNS, rowData=[],
                defaultColDef={'sortable': True, 'resizable': True},
                dashGridOptions={'singleClickEdit': True, 'animateRows': False, 'theme': 'legacy'},
                className='ag-theme-alpine-dark',
                style={'flex': 1, 'width': '100%', 'minHeight': 0, '--ag-cell-horizontal-padding': '3px'},
            ),
        ], style={'width': '340px', 'flexShrink': 0, 'display': 'flex', 'flexDirection': 'column'}),
    ], style={'display': 'flex', 'gap': '10px', 'flex': 1, 'minHeight': 0, 'padding': '8px 12px 0'})

    return html.Div([
        dcc.Store(id='selection', data=[]),
        header,
        filter_row,
        factor_panel,
        html.Div(id='notice'),
        grids,
        dmc.Group([
            dmc.Text('click a row to inspect it below; click a Ticker cell to open its full analysis in a new tab',
                     size='xs', c='dimmed'),
            dmc.Text(id='sel-status', size='xs', c='teal'),
        ], gap=16, px=12, py=6),
        html.Div(id='detail-content', style={'overflowY': 'auto', 'padding': '0 12px 12px', 'maxHeight': '45vh',
                                             'flexShrink': 0}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': 'calc(100vh - 32px)'})


dash.register_page(__name__, path='/screener', name='Screener', layout=layout)

clientside_callback(
    "(dark) => dark ? ['ag-theme-alpine-dark', 'ag-theme-alpine-dark'] : ['ag-theme-alpine', 'ag-theme-alpine']",
    Output('grid', 'className'), Output('sel-grid', 'className'),
    Input('theme-switch', 'checked'),
)


@callback(Output('f-industry', 'data'), Input('f-sector', 'value'))
def industry_options(sectors: list[str] | None) -> list[str]:
    """Industry choices narrowed to the picked sectors."""
    try:
        df = get_pool()
    except duckdb.Error:
        return []
    subset = df if not sectors else df[df['sector'].isin(sectors)]
    return sorted(subset['industry'].unique())


@callback(
    Output('grid', 'rowData'), Output('grid', 'columnDefs'), Output('sel-grid', 'rowData'),
    Output('filter-count', 'children'), Output('sel-count', 'children'), Output('notice', 'children'),
    Input('f-search', 'value'), Input('f-sector', 'value'), Input('f-industry', 'value'), Input('f-bucket', 'value'),
    Input('f-mindvol', 'value'), Input('f-minyears', 'value'), Input('f-active', 'value'), Input('f-maxanom', 'value'),
    Input({'type': 'fmin', 'factor': ALL}, 'value'), Input({'type': 'fmax', 'factor': ALL}, 'value'),
    Input('score-factors', 'value'), Input('mode', 'value'), Input('btn-refresh', 'n_clicks'),
    Input('selection', 'data'),
)
def update_grid(search, sectors, industries, buckets, min_dvol, min_years, active, max_anom,
                fmin_values, fmax_values, score_factors, mode, refresh_clicks, selection):
    """Rebuild both grids for the current filters, score selection, display mode and curation state."""
    # dmc.NumberInput reports '' when cleared; the filters expect None for "no bound"
    min_dvol = min_dvol if min_dvol not in (None, '') else None
    min_years = min_years if min_years not in (None, '') else None
    max_anom = max_anom if max_anom not in (None, '') else None
    if ctx.triggered_id == 'btn-refresh':
        clear_cache()
    try:
        pool = get_pool()
    except duckdb.Error:
        notice = html.Div(BUSY, style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '13px', 'padding': '8px 12px'})
        return [], [], [], '', '', notice

    undecided = pool[~pool['Ticker'].isin(selection)]
    candidates = apply_metadata_filters(undecided, search, sectors, industries, buckets,
                                        min_dvol, min_years, active, max_anom)
    ranked = compute_ranks(candidates)

    # pattern-matching inputs arrive in layout order; recover each one's factor from its id
    filters: dict[str, tuple[float | None, float | None]] = {}
    for spec, value in zip(ctx.inputs_list[8], fmin_values):
        filters[spec['id']['factor']] = (value if value not in (None, '') else None, None)
    for spec, value in zip(ctx.inputs_list[9], fmax_values):
        lo, _ = filters.get(spec['id']['factor'], (None, None))
        filters[spec['id']['factor']] = (lo, value if value not in (None, '') else None)
    passed = apply_filters(ranked, filters)

    chosen = score_factors or []
    if chosen:
        passed = passed.copy()
        passed['score'] = composite_score(passed, chosen)
        passed = passed.sort_values('score', ascending=False)

    display = passed.copy()
    if chosen:
        display['score'] = display['score'].round(1)
    if mode == 'rank':
        for factor in factors.FACTORS:
            display[factor] = display[f'rank_{factor}'].round(0)
    else:
        for factor in factors.FACTORS:
            display[factor] = display[factor].round(4)

    fields = IDENTITY_FIELDS + (['score'] if chosen else []) + factors.FACTORS
    rows = display[fields].copy()
    rows.insert(0, 'sel', False)
    columns = list(IDENTITY_COLUMNS)
    if chosen:
        columns.append({'field': 'score', 'headerName': 'Score', 'width': 90, 'pinned': 'left'})
    columns += [{'field': factor, 'headerName': factor} for factor in factors.FACTORS]

    sel_rows = pool.loc[pool['Ticker'].isin(selection), SEL_FIELDS].copy()
    sel_rows.insert(0, 'sel', True)
    count = f'{len(passed)} of {len(undecided)} candidates match'
    return rows.to_dict('records'), columns, sel_rows.to_dict('records'), count, f'selected: {len(selection)}', None


@callback(
    Output('selection', 'data'), Output('sel-status', 'children'),
    Input('grid', 'cellValueChanged'), Input('sel-grid', 'cellValueChanged'),
    Input('btn-select-all', 'n_clicks'), Input('btn-deselect-all', 'n_clicks'), Input('btn-load', 'n_clicks'),
    State('selection', 'data'), State('grid', 'rowData'), State('universe-dd', 'value'),
    prevent_initial_call=True,
)
def update_state(grid_events, sel_events, select_all, deselect_all, load, selection, row_data, universe):
    """Route one user action (checkbox edit, button click) through apply_action."""
    action = str(ctx.triggered_id)
    events = grid_events if action == 'grid' else sel_events if action == 'sel-grid' else None
    events = events if isinstance(events, list) else [events] if events else []
    filtered = [row['Ticker'] for row in (row_data or [])]
    return apply_action(action, selection, events, filtered, universe)


@callback(
    Output('universe-dd', 'data'), Output('save-status', 'children'),
    Input('btn-save', 'n_clicks'), State('selection', 'data'), State('universe-name', 'value'),
    prevent_initial_call=True,
)
def save_current(n_clicks, selection, name):
    """Save the selected tickers as a named universe."""
    if not name or not name.strip():
        return list_universes(), 'enter a universe name first'
    if not selection:
        return list_universes(), 'selection is empty, nothing saved'
    clean = save_universe(name, selection)
    return list_universes(), f"saved '{clean}' ({len(selection)} tickers)"


@callback(
    Output('detail-content', 'children'),
    Input('grid', 'cellClicked'), Input('sel-grid', 'cellClicked'),
    State('grid', 'rowData'), State('sel-grid', 'rowData'),
    prevent_initial_call=True,
)
def show_detail(cell, sel_cell, rows, sel_rows):
    """Show the detail panel for the clicked row of either grid.

    rowId is AG Grid's auto-assigned id (== original rowData array index) and stays stable under a
    client-side column sort; rowIndex is the sorted DISPLAY position and would resolve wrongly.
    """
    cell, data = (sel_cell, sel_rows) if ctx.triggered_id == 'sel-grid' else (cell, rows)
    if not cell or cell.get('colId') in ('sel', 'Ticker'):
        raise PreventUpdate
    row_id = cell.get('rowId')
    if row_id is None or not str(row_id).isdigit() or not data or int(row_id) >= len(data):
        raise PreventUpdate
    ticker = data[int(row_id)].get('Ticker')
    try:
        pool = get_pool()
        row = pool.loc[pool['Ticker'] == ticker].iloc[0]
    except duckdb.Error:
        return [html.Div(BUSY, style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '13px', 'padding': '12px'})]
    return safe_detail_children(ticker, row)
