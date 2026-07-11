"""Dash app for assembling named ticker universes from the investalyze database.

Run with:  .venv/bin/python -m investalyze.apps.ticker_selector.app  then open http://127.0.0.1:8050
Universes are saved as data/universes/<name>.csv and loaded in experiments via dataset.load_universe(name).
"""

import re
from pathlib import Path

import dash_ag_grid as dag
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from investalyze.apps.ticker_selector import metrics
from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'
UNIVERSE_DIR = DATA_ROOT / 'universes'

CON = storage.connect(DATA_ROOT, read_only=True)
METRICS = metrics.build_metrics(CON)

GRID_FIELDS = ['Ticker', 'name', 'sector', 'industry', 'country', 'mcap_bn', 'mcap_bucket', 'last_close',
               'dvol_mn', 'years', 'last_date', 'active', 'n_periods', 'n_anomalies']
GRID_COLUMNS = [
    {'field': 'sel', 'headerName': '', 'editable': True, 'cellDataType': 'boolean', 'width': 50, 'pinned': 'left'},
    {'field': 'Ticker', 'pinned': 'left', 'width': 95},
    {'field': 'name', 'headerName': 'Company', 'width': 230},
    {'field': 'sector', 'headerName': 'Sector', 'width': 150},
    {'field': 'industry', 'headerName': 'Industry', 'width': 190},
    {'field': 'country', 'headerName': 'Country', 'width': 110},
    {'field': 'mcap_bn', 'headerName': 'MCap $bn', 'width': 105},
    {'field': 'mcap_bucket', 'headerName': 'Bucket', 'width': 90},
    {'field': 'last_close', 'headerName': 'Close', 'width': 85},
    {'field': 'dvol_mn', 'headerName': '$Vol mn/d', 'width': 105},
    {'field': 'years', 'headerName': 'Years', 'width': 80},
    {'field': 'last_date', 'headerName': 'Last date', 'width': 110},
    {'field': 'active', 'headerName': 'Active', 'width': 85, 'cellDataType': 'boolean'},
    {'field': 'n_periods', 'headerName': 'Fund. periods', 'width': 115},
    {'field': 'n_anomalies', 'headerName': 'Anomalies', 'width': 100},
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


# ---------- pure helpers (also exercised headless by the verification script) ----------

def filter_metrics(
    df: pd.DataFrame, search: str | None, sectors: list[str] | None, industries: list[str] | None,
    buckets: list[str] | None, min_dvol_mn: float | None, min_years: float | None,
    active: str, max_anomalies: int | None
) -> pd.DataFrame:
    """Apply every sidebar filter to the metrics table; empty/None controls leave their angle unfiltered."""
    mask = pd.Series(True, index=df.index)
    if search:
        needle = search.strip().upper()
        mask &= df['Ticker'].str.upper().str.contains(needle, regex=False) | df['name'].str.upper().str.contains(needle, regex=False)
    if sectors:
        mask &= df['sector'].isin(sectors)
    if industries:
        mask &= df['industry'].isin(industries)
    if buckets:
        mask &= df['mcap_bucket'].isin(buckets)
    if min_dvol_mn is not None:
        mask &= df['dollar_vol'] >= min_dvol_mn * 1e6
    if min_years is not None:
        mask &= df['years'] >= min_years
    if active == 'active':
        mask &= df['active']
    elif active == 'delisted':
        mask &= ~df['active']
    if max_anomalies is not None:
        mask &= df['n_anomalies'] <= max_anomalies
    return df[mask]


def apply_action(
    action: str, selection: list[str], discarded: list[str], checkbox_events: list[dict],
    filtered: list[str], universe_name: str | None, detail_ticker: str | None
) -> tuple[list[str], list[str], str]:
    """Return the new (selection, discarded, status message) for one user action.

    Every ticker is in exactly one of three groups: undecided (the candidates table),
    selected (the condensed table) or discarded (hidden until Clear).
    """
    sel, out = set(selection), set(discarded)
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
    elif action == 'btn-add':
        sel |= set(filtered)
        message = f'added {len(filtered)} tickers, {len(sel)} selected'
    elif action == 'btn-remove':
        out |= set(filtered)
        message = f'removed {len(filtered)} tickers from the candidates'
    elif action == 'btn-clear':
        sel, out = set(), set()
        message = 'selection and removals cleared'
    elif action == 'btn-load':
        if not universe_name:
            return sorted(sel), sorted(out), 'pick a universe to load first'
        sel, out = set(load_universe(universe_name)), set()
        message = f"loaded '{universe_name}' ({len(sel)} tickers)"
    elif action == 'btn-detail-toggle' and detail_ticker:
        if detail_ticker in sel:
            sel.discard(detail_ticker)
            message = f'{detail_ticker} back in the candidates'
        else:
            sel.add(detail_ticker)
            out.discard(detail_ticker)
            message = f'{detail_ticker} selected'
    else:
        message = ''
    return sorted(sel), sorted(out), message


def list_universes() -> list[str]:
    """Names of every saved universe file."""
    if not UNIVERSE_DIR.exists():
        return []
    return sorted(p.stem for p in UNIVERSE_DIR.glob('*.csv'))


def save_universe(name: str, tickers: list[str]) -> str:
    """Write the selection to data/universes/<name>.csv and return the cleaned name."""
    clean = re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip()).strip('_')
    if not clean:
        raise ValueError('universe name is empty')
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Ticker': sorted(tickers)}).to_csv(UNIVERSE_DIR / f'{clean}.csv', index=False)
    return clean


def load_universe(name: str) -> list[str]:
    """Read a saved universe back as a ticker list."""
    return pd.read_csv(UNIVERSE_DIR / f'{name}.csv')['Ticker'].tolist()


# ---------- detail panel ----------

def price_figure(ticker: str) -> go.Figure:
    """Adjusted close (log) and volume for the full history of one ticker."""
    prices = CON.cursor().execute(
        "SELECT Date, AC, V FROM prices WHERE Ticker = ? ORDER BY Date", [ticker]
    ).df()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=prices['Date'], y=prices['AC'], name='AC', line={'width': 1}), row=1, col=1)
    fig.add_trace(go.Bar(x=prices['Date'], y=prices['V'], name='Volume', marker={'line': {'width': 0}}), row=2, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(height=420, margin={'l': 40, 'r': 10, 't': 30, 'b': 20}, showlegend=False,
                      title={'text': f'{ticker} adjusted close (log) and volume', 'font': {'size': 14}})
    return fig


def fmt_money(value: object) -> str:
    """Millions with thousands separators, or '-' when the field is missing."""
    if value is None or pd.isna(value):
        return '-'
    return f'{value / 1e6:,.0f}m'


def two_col_table(rows: list[tuple[str, object]]) -> html.Table:
    """Small label/value table used for the stats and fundamentals blocks."""
    return html.Table(
        [html.Tr([html.Td(label, style={'color': '#666', 'paddingRight': '12px'}), html.Td(str(value))]) for label, value in rows],
        style={'fontSize': '13px', 'borderSpacing': '0 2px'},
    )


def detail_children(ticker: str) -> list:
    """Stats, latest fundamentals, anomalies and the price chart for one ticker."""
    row = METRICS.loc[METRICS['Ticker'] == ticker].iloc[0]
    stats = two_col_table([
        ('Company', row['name']), ('Sector', row['sector']), ('Industry', row['industry']),
        ('Country', row['country']), ('Employees', row['employees'] if pd.notna(row['employees']) else '-'),
        ('History', f"{row['first_date']} to {row['last_date']} ({row['years']}y)"),
        ('Active', bool(row['active'])), ('Last close', row['last_close']),
        ('Median $ volume', f"{row['dvol_mn']}mn/day"),
        ('Market cap', f"{row['mcap_bn']}bn" if pd.notna(row['mcap_bn']) else '- (no share count)'),
        ('Fundamental periods', row['n_periods']), ('Anomalies', row['n_anomalies']),
    ])

    fundamentals = CON.cursor().execute("""
        SELECT "Fiscal Year", "Fiscal Period", Revenue, "Gross Profit", "Operating Income (Loss)",
               "Net Income", "Shares (Basic)"
        FROM income WHERE Ticker = ? ORDER BY "Report Date" DESC LIMIT 1
    """, [ticker]).df()
    if len(fundamentals):
        f = fundamentals.iloc[0]
        fundamentals_block = two_col_table([
            ('Latest report', f"{f['Fiscal Year']} {f['Fiscal Period']}"),
            ('Revenue', fmt_money(f['Revenue'])), ('Gross profit', fmt_money(f['Gross Profit'])),
            ('Operating income', fmt_money(f['Operating Income (Loss)'])), ('Net income', fmt_money(f['Net Income'])),
            ('Shares (basic)', fmt_money(f['Shares (Basic)'])),
        ])
    else:
        fundamentals_block = html.Div('no fundamentals in DB', style={'color': '#888', 'fontSize': '13px'})

    anomalies = CON.cursor().execute(
        "SELECT CheckName, Severity, Date, Details FROM anomalies WHERE Ticker = ? ORDER BY Date", [ticker]
    ).df()
    if len(anomalies):
        shown = anomalies.head(15)
        anomaly_rows = [html.Tr([html.Td(str(v), style={'paddingRight': '10px'}) for v in row]) for row in shown.itertuples(index=False)]
        extra = f' (+{len(anomalies) - len(shown)} more)' if len(anomalies) > len(shown) else ''
        anomalies_block = html.Div([
            html.Div(f'{len(anomalies)} anomalies{extra}', style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Table(anomaly_rows, style={'fontSize': '12px'}),
        ])
    else:
        anomalies_block = html.Div('no anomalies recorded', style={'color': '#888', 'fontSize': '13px'})

    left = html.Div(
        [html.H4(ticker, style={'margin': '0 0 6px'}), stats,
         html.H4('Fundamentals', style={'margin': '10px 0 4px', 'fontSize': '14px'}), fundamentals_block,
         html.H4('Anomalies', style={'margin': '10px 0 4px', 'fontSize': '14px'}), anomalies_block],
        style={'width': '440px', 'flexShrink': 0, 'overflowY': 'auto'},
    )
    chart = html.Div(dcc.Graph(figure=price_figure(ticker)), style={'flex': 1, 'minWidth': 0})
    return [html.Div([left, chart], style={'display': 'flex', 'gap': '16px'})]


# ---------- layout ----------

def labeled(text: str, component) -> html.Div:
    """Sidebar row: a small label above its control."""
    return html.Div([html.Label(text, style={'fontSize': '12px', 'color': '#555'}), component], style={'marginBottom': '10px'})


BTN = {'marginRight': '6px', 'marginBottom': '4px'}

sidebar = html.Div([
    labeled('Search ticker / company', dcc.Input(id='f-search', type='text', debounce=True, style={'width': '100%'})),
    labeled('Sector', dcc.Dropdown(id='f-sector', options=sorted(METRICS['sector'].unique()), multi=True)),
    labeled('Industry', dcc.Dropdown(id='f-industry', multi=True)),
    labeled('Market-cap bucket', dcc.Checklist(id='f-bucket', options=metrics.MCAP_LABELS, inline=True,
                                               style={'fontSize': '13px'})),
    labeled('Min median $ volume (mn/day)', dcc.Input(id='f-mindvol', type='number', min=0, style={'width': '100%'})),
    labeled('Min history (years)', dcc.Input(id='f-minyears', type='number', min=0, style={'width': '100%'})),
    labeled('Listing', dcc.RadioItems(id='f-active', options=['all', 'active', 'delisted'], value='all', inline=True,
                                      style={'fontSize': '13px'})),
    labeled('Max anomalies', dcc.Input(id='f-maxanom', type='number', min=0, style={'width': '100%'})),
    html.Div(id='filter-count', style={'fontSize': '13px', 'fontWeight': 'bold', 'margin': '8px 0'}),
    html.Div([
        html.Button('Add filtered', id='btn-add', style=BTN),
        html.Button('Remove filtered', id='btn-remove', style=BTN),
    ]),
    html.Div(id='sel-status', style={'fontSize': '12px', 'color': '#0a7', 'marginTop': '6px'}),
], style={'width': '300px', 'flexShrink': 0, 'padding': '12px', 'overflowY': 'auto', 'borderRight': '1px solid #ddd'})

header = html.Div([
    html.B('Ticker Selector', style={'marginRight': '20px'}),
    html.Span(id='sel-count', style={'marginRight': '20px', 'color': '#06c'}),
    dcc.Input(id='universe-name', type='text', placeholder='universe name', style={'width': '160px', 'marginRight': '6px'}),
    html.Button('Save', id='btn-save', style=BTN),
    dcc.Dropdown(id='universe-dd', options=list_universes(), placeholder='saved universes',
                 style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '6px'}),
    html.Button('Load', id='btn-load', style=BTN),
    html.Button('Clear', id='btn-clear', style=BTN),
    html.Span(id='save-status', style={'fontSize': '12px', 'color': '#0a7', 'marginLeft': '10px'}),
], style={'padding': '8px 12px', 'borderBottom': '1px solid #ddd', 'display': 'flex', 'alignItems': 'center',
          'flexWrap': 'wrap'})

main = html.Div([
    html.Div([
        dag.AgGrid(
            id='grid', columnDefs=GRID_COLUMNS, rowData=[],
            defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
            getRowId='params.data.Ticker',
            dashGridOptions={'singleClickEdit': True, 'animateRows': False},
            style={'height': '100%', 'flex': 2, 'minWidth': 0},
        ),
        html.Div([
            html.Div('Selected (uncheck to send back to candidates)', style={'fontSize': '12px', 'fontWeight': 'bold',
                                                                              'padding': '0 0 4px'}),
            dag.AgGrid(
                id='sel-grid', columnDefs=SEL_COLUMNS, rowData=[],
                defaultColDef={'sortable': True, 'resizable': True},
                getRowId='params.data.Ticker',
                dashGridOptions={'singleClickEdit': True, 'animateRows': False},
                style={'flex': 1, 'width': '100%', 'minHeight': 0},
            ),
        ], style={'flex': 1, 'minWidth': 0, 'display': 'flex', 'flexDirection': 'column'}),
    ], style={'display': 'flex', 'gap': '10px', 'height': '52%'}),
    html.Div([
        html.Button('select / deselect', id='btn-detail-toggle', disabled=True, style=BTN),
        html.Span('click a grid row to inspect a ticker', style={'fontSize': '12px', 'color': '#888'}),
    ], style={'padding': '6px 12px'}),
    html.Div(id='detail-content', style={'flex': 1, 'overflowY': 'auto', 'padding': '0 12px 12px'}),
], style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'minWidth': 0})

app = Dash(__name__, title='Ticker Selector')
app.layout = html.Div([
    dcc.Store(id='selection', data=[]),
    dcc.Store(id='discarded', data=[]),
    dcc.Store(id='detail-ticker', data=None),
    header,
    html.Div([sidebar, main], style={'display': 'flex', 'flex': 1, 'minHeight': 0}),
], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'fontFamily': 'sans-serif'})


# ---------- callbacks ----------

@app.callback(Output('f-industry', 'options'), Input('f-sector', 'value'))
def industry_options(sectors: list[str] | None) -> list[str]:
    subset = METRICS if not sectors else METRICS[METRICS['sector'].isin(sectors)]
    return sorted(subset['industry'].unique())


@app.callback(
    Output('grid', 'rowData'), Output('sel-grid', 'rowData'), Output('filter-count', 'children'), Output('sel-count', 'children'),
    Input('f-search', 'value'), Input('f-sector', 'value'), Input('f-industry', 'value'), Input('f-bucket', 'value'),
    Input('f-mindvol', 'value'), Input('f-minyears', 'value'), Input('f-active', 'value'),
    Input('f-maxanom', 'value'), Input('selection', 'data'), Input('discarded', 'data'),
)
def update_grid(search, sectors, industries, buckets, min_dvol, min_years, active, max_anom, selection, discarded):
    pool = METRICS[~METRICS['Ticker'].isin(set(selection) | set(discarded))]  # undecided candidates only
    filtered = filter_metrics(pool, search, sectors, industries, buckets, min_dvol, min_years, active, max_anom)
    rows = filtered[GRID_FIELDS].copy()
    rows.insert(0, 'sel', False)
    sel_rows = METRICS.loc[METRICS['Ticker'].isin(selection), SEL_FIELDS].copy()
    sel_rows.insert(0, 'sel', True)
    count = f'{len(filtered)} of {len(pool)} candidates match ({len(discarded)} removed)'
    return rows.to_dict('records'), sel_rows.to_dict('records'), count, f'selected: {len(selection)}'


@app.callback(
    Output('selection', 'data'), Output('discarded', 'data'), Output('sel-status', 'children'),
    Input('grid', 'cellValueChanged'), Input('sel-grid', 'cellValueChanged'),
    Input('btn-add', 'n_clicks'), Input('btn-remove', 'n_clicks'), Input('btn-clear', 'n_clicks'),
    Input('btn-load', 'n_clicks'), Input('btn-detail-toggle', 'n_clicks'),
    State('selection', 'data'), State('discarded', 'data'), State('grid', 'rowData'),
    State('universe-dd', 'value'), State('detail-ticker', 'data'),
    prevent_initial_call=True,
)
def update_state(grid_events, sel_events, add, remove, clear, load, toggle, selection, discarded, row_data, universe, detail):
    action = str(ctx.triggered_id)
    events = grid_events if action == 'grid' else sel_events if action == 'sel-grid' else None
    events = events if isinstance(events, list) else [events] if events else []
    filtered = [row['Ticker'] for row in (row_data or [])]
    return apply_action(action, selection, discarded, events, filtered, universe, detail)


@app.callback(
    Output('universe-dd', 'options'), Output('save-status', 'children'),
    Input('btn-save', 'n_clicks'), State('selection', 'data'), State('universe-name', 'value'),
    prevent_initial_call=True,
)
def save_current(n_clicks, selection, name):
    if not name or not name.strip():
        return list_universes(), 'enter a universe name first'
    if not selection:
        return list_universes(), 'selection is empty, nothing saved'
    clean = save_universe(name, selection)
    return list_universes(), f"saved '{clean}' ({len(selection)} tickers)"


@app.callback(
    Output('detail-content', 'children'), Output('detail-ticker', 'data'),
    Input('grid', 'cellClicked'), Input('sel-grid', 'cellClicked'), prevent_initial_call=True,
)
def show_detail(cell, sel_cell):
    cell = sel_cell if ctx.triggered_id == 'sel-grid' else cell
    if not cell or cell.get('colId') == 'sel':
        raise PreventUpdate
    ticker = cell.get('rowId')
    return detail_children(ticker), ticker


@app.callback(
    Output('btn-detail-toggle', 'children'), Output('btn-detail-toggle', 'disabled'),
    Input('detail-ticker', 'data'), Input('selection', 'data'),
)
def toggle_label(detail: str | None, selection: list[str]) -> tuple[str, bool]:
    if not detail:
        return 'select / deselect', True
    return (f'deselect {detail}' if detail in selection else f'select {detail}'), False


if __name__ == '__main__':
    print(f'{len(METRICS)} tickers loaded, universes dir: {UNIVERSE_DIR}')
    app.run(debug=False)
