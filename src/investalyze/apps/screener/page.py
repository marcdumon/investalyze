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

from investalyze.analysis import factors
from investalyze.apps.screener import metrics
from investalyze.apps.screener.data import clear_cache, get_pool
from investalyze.apps.screener.detail import safe_detail_children
from investalyze.apps.screener.logic import apply_filters, apply_metadata_filters, composite_score, compute_ranks
from investalyze.apps.universes import list_universes, load_universe, save_universe

IDENTITY_FIELDS = ['Ticker', 'name', 'sector', 'industry', 'mcap_bn', 'mcap_bucket']
IDENTITY_COLUMNS = [
    {'field': 'sel', 'headerName': '', 'editable': True, 'cellDataType': 'boolean', 'width': 50, 'pinned': 'left'},
    {'field': 'Ticker', 'pinned': 'left', 'width': 95},
    {'field': 'name', 'headerName': 'Company', 'width': 200},
    {'field': 'sector', 'headerName': 'Sector', 'width': 140},
    {'field': 'industry', 'headerName': 'Industry', 'width': 170},
    {'field': 'mcap_bn', 'headerName': 'MCap $bn', 'width': 100},
    {'field': 'mcap_bucket', 'headerName': 'Bucket', 'width': 85},
]
SEL_FIELDS = ['Ticker', 'name', 'sector', 'mcap_bucket', 'dvol_mn']
SEL_COLUMNS = [
    {'field': 'sel', 'headerName': '', 'editable': True, 'cellDataType': 'boolean', 'width': 50},
    {'field': 'Ticker', 'width': 95},
    {'field': 'name', 'headerName': 'Company', 'flex': 1},
    {'field': 'sector', 'headerName': 'Sector', 'width': 140},
    {'field': 'mcap_bucket', 'headerName': 'Bucket', 'width': 85},
    {'field': 'dvol_mn', 'headerName': '$Vol mn/d', 'width': 100},
]

BTN = {'marginRight': '6px', 'marginBottom': '4px'}
BUSY = 'database busy, a job is currently running, try again once it finishes'


def labeled(text: str, component) -> html.Div:
    """Sidebar row: a small label above its control."""
    return html.Div([html.Label(text, style={'fontSize': '12px', 'color': 'var(--mantine-color-dimmed)'}), component],
                    style={'marginBottom': '10px'})


def grid_title(text: str, button_text: str, button_id: str) -> html.Div:
    """Slim bar above a grid: a bold caption on the left, an action button on the right."""
    return html.Div([
        html.Div(text, style={'fontSize': '12px', 'fontWeight': 'bold'}),
        html.Button(button_text, id=button_id, style=BTN | {'whiteSpace': 'nowrap', 'flexShrink': 0}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '0 0 4px'})


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

    sidebar = html.Div([
        labeled('Search ticker / company', dcc.Input(id='f-search', type='text', debounce=True, style={'width': '100%'})),
        labeled('Sector', dcc.Dropdown(id='f-sector', options=sorted(df['sector'].unique()), multi=True)),
        labeled('Industry', dcc.Dropdown(id='f-industry', multi=True)),
        labeled('Market-cap bucket', dcc.Checklist(id='f-bucket', options=metrics.MCAP_LABELS, inline=True,
                                                   style={'fontSize': '13px'})),
        labeled('Min median $ volume (mn/day)', dcc.Input(id='f-mindvol', type='number', min=0, style={'width': '100%'})),
        labeled('Min history (years)', dcc.Input(id='f-minyears', type='number', min=0, style={'width': '100%'})),
        labeled('Listing', dcc.RadioItems(id='f-active', options=['all', 'active', 'delisted'], value='all', inline=True,
                                          style={'fontSize': '13px'})),
        labeled('Max anomalies', dcc.Input(id='f-maxanom', type='number', min=0, style={'width': '100%'})),
        dmc.MultiSelect(id='score-factors', data=factors.FACTORS, label='Composite score factors',
                        size='xs', clearable=True, mb=8),
        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl(family, py=4),
                dmc.AccordionPanel(dmc.Stack([factor_row(f) for f in members], gap=4)),
            ], value=family)
            for family, members in factors.FAMILIES.items()
        ], multiple=True),
        html.Div(id='filter-count', style={'fontSize': '13px', 'fontWeight': 'bold', 'margin': '8px 0'}),
        html.Div(id='sel-status', style={'fontSize': '12px', 'color': '#0a7', 'marginTop': '6px'}),
    ], style={'width': '340px', 'flexShrink': 0, 'padding': '12px', 'overflowY': 'auto',
              'borderRight': '1px solid var(--mantine-color-default-border)'})

    header = html.Div([
        html.B('Screener', style={'marginRight': '20px'}),
        html.Span(id='sel-count', style={'marginRight': '20px', 'color': 'var(--mantine-color-anchor)'}),
        dmc.SegmentedControl(id='mode', value='raw', size='xs',
                             data=[{'label': 'Raw', 'value': 'raw'}, {'label': 'Rank', 'value': 'rank'}]),
        dcc.Input(id='universe-name', type='text', placeholder='universe name',
                  style={'width': '160px', 'marginLeft': '12px', 'marginRight': '6px'}),
        html.Button('Save', id='btn-save', style=BTN),
        dcc.Dropdown(id='universe-dd', options=list_universes(), placeholder='saved universes',
                     style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '6px'}),
        html.Button('Load', id='btn-load', style=BTN),
        html.Button('Refresh', id='btn-refresh', style=BTN),
        html.Span(id='save-status', style={'fontSize': '12px', 'color': '#0a7', 'marginLeft': '10px'}),
    ], style={'padding': '8px 12px', 'borderBottom': '1px solid var(--mantine-color-default-border)',
              'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'})

    main = html.Div([
        html.Div(id='notice'),
        html.Div([
            html.Div([
                grid_title('Candidates', 'select all', 'btn-select-all'),
                dag.AgGrid(
                    id='grid', columnDefs=[], rowData=[],
                    defaultColDef={'sortable': True, 'filter': True, 'resizable': True, 'width': 110},
                    dashGridOptions={'singleClickEdit': True, 'animateRows': False},
                    className='ag-theme-alpine-dark',
                    style={'flex': 1, 'width': '100%', 'minHeight': 0},
                ),
            ], style={'flex': 2, 'minWidth': 0, 'display': 'flex', 'flexDirection': 'column'}),
            html.Div([
                grid_title('Selected (uncheck to send back to candidates)', 'deselect all', 'btn-deselect-all'),
                dag.AgGrid(
                    id='sel-grid', columnDefs=SEL_COLUMNS, rowData=[],
                    defaultColDef={'sortable': True, 'resizable': True},
                    dashGridOptions={'singleClickEdit': True, 'animateRows': False},
                    className='ag-theme-alpine-dark',
                    style={'flex': 1, 'width': '100%', 'minHeight': 0},
                ),
            ], style={'flex': 1, 'minWidth': 0, 'display': 'flex', 'flexDirection': 'column'}),
        ], style={'display': 'flex', 'gap': '10px', 'height': '52%'}),
        html.Div(
            html.Span('click a grid row to inspect a ticker', style={'fontSize': '12px', 'color': 'var(--mantine-color-dimmed)'}),
            style={'padding': '6px 12px'},
        ),
        html.Div(id='detail-content', style={'flex': 1, 'overflowY': 'auto', 'padding': '0 12px 12px'}),
    ], style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'minWidth': 0})

    return html.Div([
        dcc.Store(id='selection', data=[]),
        header,
        html.Div([sidebar, main], style={'display': 'flex', 'flex': 1, 'minHeight': 0}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': 'calc(100vh - 32px)', 'fontFamily': 'sans-serif'})


dash.register_page(__name__, path='/screener', name='Screener', layout=layout)

clientside_callback(
    "(dark) => dark ? ['ag-theme-alpine-dark', 'ag-theme-alpine-dark'] : ['ag-theme-alpine', 'ag-theme-alpine']",
    Output('grid', 'className'), Output('sel-grid', 'className'),
    Input('theme-switch', 'checked'),
)


@callback(Output('f-industry', 'options'), Input('f-sector', 'value'))
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
    Output('universe-dd', 'options'), Output('save-status', 'children'),
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
    State('grid', 'rowData'), State('sel-grid', 'rowData'), State('theme-switch', 'checked'),
    prevent_initial_call=True,
)
def show_detail(cell, sel_cell, rows, sel_rows, dark):
    """Show the detail panel for the clicked row of either grid.

    rowId is AG Grid's auto-assigned id (== original rowData array index) and stays stable under a
    client-side column sort; rowIndex is the sorted DISPLAY position and would resolve wrongly.
    """
    cell, data = (sel_cell, sel_rows) if ctx.triggered_id == 'sel-grid' else (cell, rows)
    if not cell or cell.get('colId') == 'sel':
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
    return safe_detail_children(ticker, row, bool(dark))
