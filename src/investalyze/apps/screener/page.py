"""Screener page: metadata + factor-value filters over one pool, two-grid curation, saved universes.

The pool (metrics metadata + factor columns) is built once on first visit through a short-lived
read-only connection and cached at module level; Refresh rebuilds it. Factor filters bound each
factor's raw value. Ranks, which drive the composite score and the Rank display mode, are recomputed
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
from investalyze.apps.control_panel.jobs import MANAGER
from investalyze.apps.control_panel.page import build_blacklist_argv
from investalyze.apps.screener import metrics
from investalyze.apps.screener.data import clear_cache, get_pool, get_scales
from investalyze.apps.screener.detail import safe_detail_children
from investalyze.apps.screener.logic import (apply_filters, apply_metadata_filters, collect_bounds, composite_score,
                                             compute_ranks)
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


def grid_title(text: str, button_text: str, button_id: str, count_id: str | None = None) -> dmc.Group:
    """Slim bar above a grid: a bold caption on the left, an action button on the right.

    Args:
        count_id: when given, an id'd Text sits beside the caption to carry a 'matched/total' count.
    """
    caption = [dmc.Text(text, size='xs', fw=700)]
    if count_id:
        caption.append(dmc.Text(id=count_id, size='xs', fw=600))
    return dmc.Group([
        dmc.Group(caption, gap=6, wrap='nowrap'),
        dmc.Button(button_text, id=button_id, size='compact-xs', variant='light'),
    ], justify='space-between', mb=4, wrap='nowrap')


def factor_row(factor: str, step: float) -> dmc.Group:
    """One sidebar row: factor name with min/max value inputs, kept on a single line.

    The inputs carry no min/max attribute of their own: factor values are unbounded and often
    negative, so any clamp here would silently refuse legitimate bounds.

    Args:
        step: how much one spinner click moves the bound, sized to this factor's own spread.
    """
    return dmc.Group([
        factor_help(factor),
        dmc.NumberInput(id={'type': 'fmin', 'factor': factor}, placeholder='min', step=step,
                        size='xs', style={'width': '72px'}),
        dmc.NumberInput(id={'type': 'fmax', 'factor': factor}, placeholder='max', step=step,
                        size='xs', style={'width': '72px'}),
    ], gap=6, wrap='nowrap')


def factor_help(factor: str) -> dmc.HoverCard:
    """The factor's name, explaining itself on hover.

    The name doubles as the hover target rather than adding a help icon, so the definition costs
    no width in a panel that already wraps.
    """
    doc = factors.FACTOR_DOCS[factor]
    best = 'higher is better' if factors.HIGHER_IS_BETTER[factor] else 'lower is better'
    return dmc.HoverCard([
        dmc.HoverCardTarget(
            dmc.Text(factor, size='xs',
                     style={'width': '106px', 'cursor': 'help', 'textDecoration': 'underline dotted'}),
        ),
        dmc.HoverCardDropdown([
            dmc.Group([dmc.Text(factor, fw=700, size='sm'), dmc.Text(best, size='xs', c='dimmed')], gap=8, mb=6),
            dmc.Text(doc.what, size='xs'),
            dmc.Code(doc.formula, block=True, mt=8, style={'fontSize': '11px', 'whiteSpace': 'pre-wrap'}),
        ]),
    ], width=380, shadow='md', openDelay=200, position='right', withArrow=True)


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
        scales = get_scales()
    except duckdb.Error:
        return html.Div(BUSY, style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '14px', 'padding': '24px'})

    filter_row = dmc.Group([
        dmc.TextInput(id='f-search', label='Search', placeholder='Ticker or company', debounce=500, size='xs',
                      leftSection=DashIconify(icon='tabler:search', width=14), style={'width': '160px'}),
        dmc.MultiSelect(id='f-sector', label='Sector', data=sorted(df['sector'].unique()), placeholder='All',
                        clearable=True, searchable=True, size='xs', style={'width': '155px'}),
        dmc.MultiSelect(id='f-industry', label='Industry', data=[], placeholder='All',
                        clearable=True, searchable=True, size='xs', style={'width': '170px'}),
        dmc.NumberInput(id='f-mindvol', label='Min $vol (mn/d)', min=0, size='xs', style={'width': '100px'}),
        dmc.NumberInput(id='f-minyears', label='Min years', min=0, size='xs', style={'width': '85px'}),
        dmc.NumberInput(id='f-maxanom', label='Max anomalies', min=0, size='xs', style={'width': '100px'}),
        html.Div([dmc.Text('Listing', size='xs', fw=500, mb=2),
                  dmc.SegmentedControl(id='f-active', value='all', size='xs', data=['all', 'active', 'delisted'])]),
        html.Div([dmc.Text('Market-cap bucket', size='xs', fw=500, mb=2),
                  html.Div(dmc.ChipGroup(dmc.Group([dmc.Chip(label, value=label, size='xs')
                                                    for label in metrics.MCAP_LABELS], gap=4),
                                         id='f-bucket', multiple=True, value=[]),
                           style={'height': '30px', 'display': 'flex', 'alignItems': 'center'})]),
    ], gap=8, align='flex-end', px=12, pt=4, pb=4, style={'flexWrap': 'wrap'})

    factor_panel = dmc.Accordion([
        dmc.AccordionItem([
            dmc.AccordionControl('Factor value filters & composite score', py=2),
            dmc.AccordionPanel([
                dmc.Group([
                    dmc.MultiSelect(id='score-factors', data=factors.FACTORS, label='Composite score factors',
                                    size='xs', clearable=True, style={'width': '520px'}),
                    dmc.Button('Clear bounds', id='btn-clear-bounds', size='compact-xs', variant='default'),
                ], gap=8, align='flex-end', mb=6),
                dmc.Text(f'Hover a factor name for its definition and formula. {factors.TTM_NOTE}',
                         size='xs', c='dimmed', mb=10),
                dmc.Group([
                    dmc.Stack([dmc.Text(family, size='xs', fw=700)]
                              + [factor_row(f, scales[f].step) for f in members], gap=4)
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
        dmc.TextInput(id='universe-name', placeholder='Universe name', size='xs', style={'width': '120px'}),
        # Save/Load are icon-only: as labelled buttons the four universe controls cost 496px
        # of the 862px row on their own, which forced the whole header onto a second line.
        dmc.Tooltip(dmc.ActionIcon(DashIconify(icon='tabler:device-floppy', width=16), id='btn-save',
                                   size='input-xs', variant='default'), label='Save universe'),
        dmc.Select(id='universe-dd', data=list_universes(), placeholder='Saved universes', clearable=True,
                   size='xs', style={'width': '150px'}),
        dmc.Tooltip(dmc.ActionIcon(DashIconify(icon='tabler:folder-open', width=16), id='btn-load',
                                   size='input-xs', variant='default'), label='Load universe'),
        dmc.Button('Blacklist', id='scr-btn-blacklist', size='xs', color='red', variant='outline',
                   leftSection=DashIconify(icon='tabler:trash', width=14)),
        dmc.Text(id='scr-blacklist-status', size='xs'),
        dmc.Button('Refresh', id='btn-refresh', size='xs', variant='subtle',
                   leftSection=DashIconify(icon='tabler:reload', width=14)),
        dmc.Text(id='save-status', size='xs', c='teal'),
    ], gap=6, px=12, py=6, align='center',
       style={'borderBottom': '1px solid var(--mantine-color-default-border)', 'flexWrap': 'wrap'})

    grids = html.Div([
        html.Div([
            grid_title('Candidates', 'select all', 'btn-select-all', count_id='filter-count'),
            dag.AgGrid(
                id='grid', columnDefs=[], rowData=[],
                defaultColDef={'sortable': True, 'filter': True, 'resizable': True, 'width': 85},
                dashGridOptions={'singleClickEdit': True, 'animateRows': False, 'theme': 'legacy',
                                 'rowHeight': 26, 'headerHeight': 28},
                className='ag-theme-alpine-dark',
                style={'flex': 1, 'width': '100%', 'minHeight': 0, '--ag-cell-horizontal-padding': '3px'},
            ),
        ], style={'flex': 1, 'minWidth': 0, 'display': 'flex', 'flexDirection': 'column'}),
        html.Div([
            grid_title('Selected (uncheck to remove)', 'deselect all', 'btn-deselect-all'),
            dag.AgGrid(
                id='sel-grid', columnDefs=SEL_COLUMNS, rowData=[],
                defaultColDef={'sortable': True, 'resizable': True},
                dashGridOptions={'singleClickEdit': True, 'animateRows': False, 'theme': 'legacy',
                                 'rowHeight': 26, 'headerHeight': 28},
                className='ag-theme-alpine-dark',
                style={'flex': 1, 'width': '100%', 'minHeight': 0, '--ag-cell-horizontal-padding': '3px'},
            ),
        ], style={'width': '340px', 'flexShrink': 0, 'display': 'flex', 'flexDirection': 'column'}),
    # flex:1 fills the window when there is room; minHeight keeps the table a usable size
    # once the blocks above it grow, and the page scrolls to reach whatever no longer fits.
    ], style={'display': 'flex', 'gap': '10px', 'flex': 1, 'minHeight': '60vh', 'padding': '8px 12px 0'})

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
        html.Div(id='detail-content', style={'overflowY': 'auto', 'padding': '0 12px 12px', 'maxHeight': '30vh',
                                             'flexShrink': 0}),
        dmc.Modal(
            id='scr-confirm-modal', title='Confirm', centered=True,
            children=[
                dmc.Text(id='scr-confirm-text'),
                dmc.Textarea(id='scr-blacklist-reason', autosize=True, minRows=2, size='xs', mt=8,
                             placeholder='Reason (stored against every ticker blacklisted)'),
                dmc.Group([
                    dmc.Button('Cancel', id='scr-confirm-no', variant='default', size='xs'),
                    dmc.Button('Confirm', id='scr-confirm-yes', color='red', size='xs'),
                ], justify='flex-end', mt=12),
            ],
        ),
        dcc.Store(id='scr-blacklist-pending'),
        dcc.Interval(id='scr-blacklist-poll', interval=1000, disabled=True),
    ], style={'display': 'flex', 'flexDirection': 'column', 'minHeight': 'calc(100vh - 32px)'})


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
                _fmin_values, _fmax_values, score_factors, mode, refresh_clicks, selection):
    """Rebuild both grids for the current filters, score selection, display mode and curation state.

    The factor bounds are read back off ctx.inputs_list by id, so the two pattern-matching arguments
    serve only to trigger this callback.
    """
    # dmc.NumberInput reports '' when cleared; the filters expect None for "no bound"
    min_dvol = min_dvol if min_dvol not in (None, '') else None
    min_years = min_years if min_years not in (None, '') else None
    max_anom = max_anom if max_anom not in (None, '') else None
    if ctx.triggered_id == 'btn-refresh':
        clear_cache()
    try:
        pool = get_pool()
        scales = get_scales()
    except duckdb.Error:
        notice = html.Div(BUSY, style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '13px', 'padding': '8px 12px'})
        return [], [], [], '', '', notice

    undecided = pool[~pool['Ticker'].isin(selection)]
    candidates = apply_metadata_filters(undecided, search, sectors, industries, buckets,
                                        min_dvol, min_years, active, max_anom)
    ranked = compute_ranks(candidates)
    passed = apply_filters(ranked, collect_bounds(ctx.inputs_list))

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
            display[factor] = display[factor].round(scales[factor].decimals)

    fields = IDENTITY_FIELDS + (['score'] if chosen else []) + factors.FACTORS
    rows = display[fields].copy()
    rows.insert(0, 'sel', False)
    columns = list(IDENTITY_COLUMNS)
    if chosen:
        columns.append({'field': 'score', 'headerName': 'Score', 'width': 90, 'pinned': 'left'})
    columns += [{'field': factor, 'headerName': factor} for factor in factors.FACTORS]

    sel_rows = pool.loc[pool['Ticker'].isin(selection), SEL_FIELDS].copy()
    sel_rows.insert(0, 'sel', True)
    count = f'{len(passed):,}/{len(undecided):,}'
    return rows.to_dict('records'), columns, sel_rows.to_dict('records'), count, f'selected: {len(selection)}', None


@callback(
    Output({'type': 'fmin', 'factor': ALL}, 'value'), Output({'type': 'fmax', 'factor': ALL}, 'value'),
    Input('btn-clear-bounds', 'n_clicks'),
    prevent_initial_call=True,
)
def clear_bounds(_n_clicks):
    """Blank every factor bound at once, since the panel holds two inputs for each of 24 factors.

    The inputs are blanked with '' rather than None: None leaves dmc.NumberInput showing its old
    text even though the callback has already dropped the bound, so the panel would contradict the
    grid. collect_bounds reads '' as no bound.
    """
    blank = [''] * len(factors.FACTORS)
    return blank, blank


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


@callback(
    Output('scr-confirm-modal', 'opened'), Output('scr-confirm-text', 'children'), Output('scr-blacklist-reason', 'value'),
    Input('scr-btn-blacklist', 'n_clicks'),
    State('selection', 'data'),
    prevent_initial_call=True,
)
def open_blacklist_confirm(_n_clicks, selection):
    """Open the confirm modal for the current selection; no-op if nothing is selected."""
    if not selection:
        raise PreventUpdate
    target = selection[0] if len(selection) == 1 else f'the {len(selection)} selected tickers'
    verb = 'it' if len(selection) == 1 else 'them'
    return True, f'Permanently delete all stored data for {target} and block {verb} from future scans?', ''


@callback(
    Output('scr-confirm-modal', 'opened', allow_duplicate=True), Output('scr-blacklist-pending', 'data'),
    Output('scr-blacklist-status', 'children'), Output('scr-btn-blacklist', 'disabled'),
    Output('scr-blacklist-poll', 'disabled', allow_duplicate=True),
    Input('scr-confirm-yes', 'n_clicks'), Input('scr-confirm-no', 'n_clicks'),
    State('selection', 'data'), State('scr-blacklist-reason', 'value'),
    prevent_initial_call=True,
)
def resolve_blacklist_confirm(_yes_n, _no_n, selection, reason):
    """Start the blacklist job for the current selection on Confirm; just close the modal on Cancel."""
    if ctx.triggered_id != 'scr-confirm-yes':
        return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    reason = (reason or '').strip()
    if not selection or not reason:
        raise PreventUpdate
    title, argv = build_blacklist_argv(selection, None, reason)
    if not MANAGER.start(title, argv):
        return (False, False, dmc.Text('a job is already running elsewhere, try again shortly', size='xs', c='red'),
                False, True)
    return False, True, dmc.Text('blacklisting...', size='xs', c='yellow'), True, False


@callback(
    Output('scr-btn-blacklist', 'disabled', allow_duplicate=True), Output('scr-blacklist-status', 'children', allow_duplicate=True),
    Output('scr-blacklist-pending', 'data', allow_duplicate=True), Output('selection', 'data', allow_duplicate=True),
    Output('scr-blacklist-poll', 'disabled', allow_duplicate=True),
    Input('scr-blacklist-poll', 'n_intervals'),
    State('scr-blacklist-pending', 'data'), State('selection', 'data'),
    prevent_initial_call=True,
)
def poll_blacklist_job(_n_intervals, pending, selection):
    """Once our own blacklist job finishes, clear the pool cache, drop blacklisted tickers from selection,
    and stop polling."""
    running = MANAGER.is_running()
    if not pending or running:
        return running, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    job = MANAGER.history[0] if MANAGER.history else None
    if job is None or job.returncode != 0:
        return (False, dmc.Text('blacklist failed, check the control panel job log', size='xs', c='red'),
                False, dash.no_update, True)
    clear_cache()
    try:
        remaining = set(get_pool()['Ticker'])
    except duckdb.Error:
        return False, dmc.Text('blacklisted (refresh to update the grid)', size='xs', c='teal'), False, dash.no_update, True
    return False, dmc.Text('blacklisted', size='xs', c='teal'), False, [t for t in selection if t in remaining], True
