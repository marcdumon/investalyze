"""Screener page: rank the universe on factors, filter by percentile, save the result as a universe.

Factors are built once on first visit through a short-lived read-only connection and cached at
module level; Refresh rebuilds. While a job holds the DB's write lock the page shows a busy
notice instead of a traceback (same pattern as the other pages).
"""

from pathlib import Path

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
import pandas as pd
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from investalyze.analysis import factors
from investalyze.apps.screener.logic import apply_filters, composite_score, compute_ranks
from investalyze.apps.ticker_selector.page import safe_detail_children
from investalyze.apps.universes import list_universes, load_universe, save_universe
from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'

IDENTITY_COLUMNS = [
    {'field': 'Ticker', 'pinned': 'left', 'width': 95},
    {'field': 'name', 'headerName': 'Company', 'width': 200},
    {'field': 'sector', 'headerName': 'Sector', 'width': 140},
    {'field': 'mcap_bn', 'headerName': 'MCap $bn', 'width': 100},
]

_FACTORS_CACHE: pd.DataFrame | None = None


def get_factors() -> pd.DataFrame:
    """Return the per-ticker factor table, building and caching it on first use."""
    global _FACTORS_CACHE
    if _FACTORS_CACHE is None:
        con = storage.connect(DATA_ROOT, read_only=True)
        try:
            _FACTORS_CACHE = factors.build_factors(con)
        finally:
            con.close()
    return _FACTORS_CACHE


def clear_cache() -> None:
    """Drop the cached factor table so the next access rebuilds it."""
    global _FACTORS_CACHE
    _FACTORS_CACHE = None


def filter_row(factor: str) -> dmc.Group:
    """One sidebar row: factor name with min/max percentile inputs."""
    return dmc.Group([
        dmc.Text(factor, size='xs', style={'width': '150px'}),
        dmc.NumberInput(id={'type': 'sc-fmin', 'factor': factor}, placeholder='min', min=0, max=100,
                        size='xs', style={'width': '80px'}),
        dmc.NumberInput(id={'type': 'sc-fmax', 'factor': factor}, placeholder='max', min=0, max=100,
                        size='xs', style={'width': '80px'}),
    ], gap=6)


def layout() -> html.Div:
    """Screener page: top bar, factor filters, composite score picker, results grid, detail."""
    top = dmc.Group([
        dmc.Select(id='sc-universe', data=['all'] + list_universes(), value='all', size='xs',
                   label='Universe', style={'width': '220px'}),
        dmc.Text(id='sc-count', size='xs', c='dimmed'),
        dmc.SegmentedControl(id='sc-mode', value='raw', size='xs',
                             data=[{'label': 'Raw', 'value': 'raw'}, {'label': 'Rank', 'value': 'rank'}]),
        dmc.TextInput(id='sc-save-name', placeholder='universe name', size='xs', style={'width': '170px'}),
        dmc.Button('Save as universe', id='sc-save', size='xs', variant='light'),
        dmc.Text(id='sc-save-status', size='xs', c='green'),
        dmc.Button('Refresh factors', id='sc-refresh', size='xs', variant='subtle',
                   leftSection=DashIconify(icon='tabler:refresh')),
    ], gap=10, align='flex-end', mb=8)

    sidebar = dmc.Stack([
        dmc.MultiSelect(id='sc-score-factors', data=factors.FACTORS, label='Composite score factors',
                        size='xs', clearable=True),
        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl(family, py=4),
                dmc.AccordionPanel(dmc.Stack([filter_row(f) for f in members], gap=4)),
            ], value=family)
            for family, members in factors.FAMILIES.items()
        ], multiple=True),
    ], gap=8, style={'width': '360px', 'flexShrink': 0, 'overflowY': 'auto'})

    grid = dag.AgGrid(
        id='sc-grid', columnDefs=[], rowData=[],
        defaultColDef={'sortable': True, 'resizable': True, 'width': 110},
        className='ag-theme-alpine-dark',
        dashGridOptions={'animateRows': False},
        style={'height': '55vh', 'width': '100%'},
    )

    return html.Div([
        html.Div(id='sc-notice'),
        top,
        dmc.Group([sidebar, html.Div([grid, html.Div(id='sc-detail')], style={'flex': 1, 'minWidth': 0})],
                  gap=12, align='flex-start'),
    ])


dash.register_page(__name__, path='/screener', name='Screener', layout=layout)


def _busy_notice() -> dmc.Alert:
    """Standard notice shown when a writer holds the DB lock."""
    return dmc.Alert('database busy, a job is currently running, try again once it finishes',
                     color='yellow', variant='light')


@callback(
    Output('sc-grid', 'rowData'), Output('sc-grid', 'columnDefs'),
    Output('sc-count', 'children'), Output('sc-notice', 'children'),
    Input('sc-universe', 'value'), Input({'type': 'sc-fmin', 'factor': ALL}, 'value'),
    Input({'type': 'sc-fmax', 'factor': ALL}, 'value'),
    Input('sc-score-factors', 'value'), Input('sc-mode', 'value'), Input('sc-refresh', 'n_clicks'),
)
def update_grid(universe, fmin_values, fmax_values, score_factors, mode, refresh_clicks):
    """Rebuild the grid for the current universe, filters, score selection and display mode."""
    if ctx.triggered_id == 'sc-refresh':
        clear_cache()
    try:
        pool = get_factors()
    except duckdb.Error:
        return [], [], '', _busy_notice()

    if universe and universe != 'all':
        tickers = set(load_universe(universe))
        pool = pool[pool['Ticker'].isin(tickers)]

    # pattern-matching inputs arrive in layout order; recover each one's factor from its id
    filters: dict[str, tuple[float | None, float | None]] = {}
    for spec, value in zip(ctx.inputs_list[1], fmin_values):
        filters[spec['id']['factor']] = (value if value not in (None, '') else None, None)
    for spec, value in zip(ctx.inputs_list[2], fmax_values):
        lo, _ = filters.get(spec['id']['factor'], (None, None))
        filters[spec['id']['factor']] = (lo, value if value not in (None, '') else None)

    ranked = compute_ranks(pool)
    filtered = apply_filters(ranked, filters)

    selected = score_factors or []
    if selected:
        filtered = filtered.copy()
        filtered['score'] = composite_score(filtered, selected).round(1)
        filtered = filtered.sort_values('score', ascending=False)

    display = filtered.copy()
    if mode == 'rank':
        for factor in factors.FACTORS:
            display[factor] = display[f'rank_{factor}'].round(0)
    else:
        for factor in factors.FACTORS:
            display[factor] = display[factor].round(4)

    columns = list(IDENTITY_COLUMNS)
    if selected:
        columns.append({'field': 'score', 'headerName': 'Score', 'width': 90, 'pinned': 'left'})
    columns += [{'field': factor, 'headerName': factor} for factor in factors.FACTORS]
    fields = [c['field'] for c in columns]
    count = f'{len(filtered)} of {len(pool)} tickers pass'
    return display[fields].to_dict('records'), columns, count, None


@callback(
    Output('sc-universe', 'data'), Output('sc-save-status', 'children'),
    Input('sc-save', 'n_clicks'), State('sc-save-name', 'value'), State('sc-grid', 'rowData'),
    prevent_initial_call=True,
)
def save_current(n_clicks, name, row_data):
    """Save the currently displayed tickers as a named universe."""
    options = ['all'] + list_universes()
    if not name or not name.strip():
        return options, 'enter a universe name first'
    tickers = [row['Ticker'] for row in (row_data or [])]
    if not tickers:
        return options, 'nothing to save'
    clean = save_universe(name, tickers)
    return ['all'] + list_universes(), f"saved '{clean}' ({len(tickers)} tickers)"


@callback(
    Output('sc-detail', 'children'),
    Input('sc-grid', 'cellClicked'), State('sc-grid', 'rowData'),
    prevent_initial_call=True,
)
def show_detail(cell, row_data):
    """Show the read-only ticker detail for the clicked row."""
    # rowId is AG Grid's auto-assigned id (== original rowData array index) and stays stable
    # under a client-side column sort; rowIndex is the sorted DISPLAY position and would not.
    row_id = cell.get('rowId') if cell else None
    if row_id is None or not row_id.isdigit() or not row_data:
        raise PreventUpdate
    row_index = int(row_id)
    if row_index >= len(row_data):
        raise PreventUpdate
    return safe_detail_children(row_data[row_index].get('Ticker'))
