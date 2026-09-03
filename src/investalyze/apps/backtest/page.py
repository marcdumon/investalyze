"""Backtest page: stored runs, their equity curves, and the factor evaluation behind them.

Read only. Runs are produced by `python -m investalyze.backtest run` and evaluations by the
`evaluate` command, both reachable as buttons on the Control Panel; this page shows what they left in
the database. Selecting a run draws its equity against the benchmark with a drawdown band beneath,
and lists the fills.

The survivorship note travels with the run rather than being written into the page, so a run made
against a universe that does contain delisted names will say so on its own.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from datetime import date

from investalyze.apps import universes
from investalyze.apps.backtest import data, logic
from investalyze.apps.control_panel.jobs import MANAGER
from investalyze.backtest import registry

BUSY = 'database busy, a job is currently running, try again once it finishes'
EMPTY_FACTORS = 'no factor evaluations yet; press Evaluate factors above'
EMPTY_RUNS = 'no stored runs yet; configure one above and press Run signal'

STRATEGY_COLOR = '#4C72B0'
BENCHMARK_COLOR = '#8C8C8C'
DRAWDOWN_COLOR = '#C44E52'

TRADE_COLUMNS = [
    {'field': 'Date', 'width': 110},
    {'field': 'Ticker', 'width': 100},
    {'field': 'Side', 'width': 80},
    {'field': 'Shares', 'width': 110, 'valueFormatter': {'function': 'd3.format(",.1f")(params.value)'}},
    {'field': 'Price', 'width': 100, 'valueFormatter': {'function': 'd3.format(",.2f")(params.value)'}},
    {'field': 'Notional', 'width': 120, 'valueFormatter': {'function': 'd3.format(",.0f")(params.value)'}},
    {'field': 'Cost', 'width': 90, 'valueFormatter': {'function': 'd3.format(",.2f")(params.value)'}},
    {'field': 'Kind', 'width': 90},
]


def _grid(grid_id: str, columns: list[dict], height: int, selectable: bool = False) -> dag.AgGrid:
    """One AG Grid with the page's shared styling."""
    return dag.AgGrid(
        id=grid_id, columnDefs=columns, rowData=[],
        dashGridOptions={'rowSelection': 'single'} if selectable else {},
        columnSize='sizeToFit', style={'height': height},
        className='ag-theme-alpine-dark',
    )


def _note(text: str) -> dmc.Alert:
    """A dimmed inline note for an empty or unavailable panel."""
    return dmc.Alert(text, color='gray', variant='light')





def _run_panel() -> dmc.Paper:
    """The controls a run needs, universe included, and the buttons that launch one."""
    today = date.today().isoformat()
    return dmc.Paper([
        dmc.Group([dmc.Text('Run a backtest', fw=600, size='sm'),
                   dmc.Badge('writes to the database', color='orange', variant='light', size='xs')],
                  gap=8, mb=6),
        dmc.Group([
            dmc.Select(id='bt-run-signal', label='Signal', value='factor_composite', size='xs', w=200,
                       data=[{'value': name, 'label': name} for name in sorted(registry.SIGNALS)]),
            dmc.TextInput(id='bt-run-start', label='From', value='2015-01-01', size='xs', w=120),
            dmc.TextInput(id='bt-run-end', label='To', value=today, size='xs', w=120),
            dmc.Select(id='bt-run-freq', label='Rebalance', value='M', size='xs', w=100,
                       data=[{'value': f, 'label': f} for f in ('D', 'W', 'M', 'Q')]),
        ], gap='xs', mb=6),
        dmc.Group([
            dmc.Select(id='bt-run-universe', label='Universe', placeholder='whole database', data=[],
                       clearable=True, searchable=True, size='xs', w=260),
            dmc.NumberInput(id='bt-run-ntop', label='Top N', value=500, min=1, size='xs', w=100),
            dmc.Select(id='bt-run-rankby', label='Ranked by', value='liquidity', size='xs', w=130,
                       data=[{'value': r, 'label': r} for r in ('liquidity', 'mcap', 'ticker')]),
            dmc.Select(id='bt-run-vintage', label='Vintage', value='as_reported', size='xs', w=140,
                       data=[{'value': 'as_reported', 'label': 'as reported'},
                             {'value': 'restated', 'label': 'restated'}]),
            dmc.NumberInput(id='bt-run-horizon', label='IC horizon', value=21, min=1, size='xs', w=110),
        ], gap='xs', mb=6),
        dmc.Text('Universes are saved from the Screener. Leaving it blank ranks the whole database; '
                 'Top N then cuts it to the names priced across the whole range.',
                 size='xs', c='dimmed', mb=8),
        dmc.Group([
            dmc.Button('1. Build factor panel', id='bt-btn-factors', size='xs', color='orange'),
            dmc.Button('2. Evaluate factors', id='bt-btn-evaluate', size='xs', color='orange'),
            dmc.Button('3. Run signal', id='bt-btn-run', size='xs', color='orange'),
        ], gap='xs'),
        html.Div(id='bt-run-status', style={'marginTop': '8px'}),
        dcc.Interval(id='bt-poll', interval=2000),
    ], p='sm', withBorder=True)


def layout() -> html.Div:
    """The page body, built per request so a run stored since the last view shows up."""
    header = dmc.Group([
        dmc.Title('Backtest', order=3),
        dmc.Text('configure a run, launch it, and read what it stored', size='sm', c='dimmed'),
    ], gap='md', mb=8)

    runs_panel = html.Div([
        dmc.Text('Stored runs', fw=600, size='sm', mb=4),
        html.Div(id='bt-runs-notice'),
        _grid('bt-runs', logic.RUN_COLUMNS, 220, selectable=True),
    ])

    detail_panel = html.Div([
        html.Div(id='bt-caveats', style={'marginBottom': '8px'}),
        dcc.Graph(id='bt-curve', config={'displayModeBar': False}, style={'height': '380px'}),
        dmc.Text('Fills', fw=600, size='sm', mt=8, mb=4),
        _grid('bt-trades', TRADE_COLUMNS, 240),
    ])

    factors_panel = html.Div([
        dmc.Text('Factor evaluation', fw=600, size='sm', mb=4),
        dmc.Text('IC is measured on raw values; the oriented column flips the factors whose lower '
                 'values are the better ones, so the column ranks the library on one scale.',
                 size='xs', c='dimmed', mb=6),
        html.Div(id='bt-factors-notice'),
        _grid('bt-factors', logic.FACTOR_COLUMNS, 420),
    ])

    return html.Div([
        header,
        _run_panel(),
        html.Div(style={'height': '10px'}),
        dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab('Runs', value='runs'),
                dmc.TabsTab('Factors', value='factors'),
            ]),
            dmc.TabsPanel(html.Div([runs_panel, html.Div(detail_panel, style={'marginTop': '10px'})]),
                          value='runs', pt=10),
            dmc.TabsPanel(factors_panel, value='factors', pt=10),
        ], value='runs'),
        dcc.Store(id='bt-selected-run'),
        dcc.Store(id='bt-runs-tick', data=0),
    ], style={'padding': '4px'})


@callback(Output('bt-runs', 'rowData'), Output('bt-runs-notice', 'children'),
          Input('bt-runs-tick', 'data'))
def load_runs(_tick):
    """Fill the run table once the page mounts."""
    try:
        runs = data.get_runs()
    except duckdb.IOException:
        return [], _note(BUSY)
    if runs.empty:
        return [], _note(EMPTY_RUNS)
    return runs.to_dict('records'), None


@callback(Output('bt-factors', 'rowData'), Output('bt-factors-notice', 'children'),
          Input('bt-runs-tick', 'data'))
def load_factors(_tick):
    """Fill the factor evaluation table once the page mounts."""
    try:
        table = logic.orient_ic(data.get_factor_ic())
    except duckdb.IOException:
        return [], _note(BUSY)
    if table.empty:
        return [], _note(EMPTY_FACTORS)
    return table.to_dict('records'), None


@callback(Output('bt-curve', 'figure'), Output('bt-trades', 'rowData'), Output('bt-caveats', 'children'),
          Input('bt-runs', 'selectedRows'))
def show_run(selected):
    """Draw the selected run's equity curve, drawdown band, caveats and fills."""
    if not selected:
        return _empty_figure('select a run'), [], None
    run = selected[0]
    try:
        curve = data.get_equity(run['RunId'])
        trades = data.get_trades(run['RunId'])
    except duckdb.IOException:
        return _empty_figure(BUSY), [], _note(BUSY)
    if curve.empty:
        return _empty_figure('this run stored no equity curve'), [], None

    rebased = logic.rebase(curve, ['Equity', 'BenchmarkEquity'])
    drawdown = logic.drawdown_from_equity(curve['Equity'])
    caveats = dmc.Alert(run['SurvivorshipNote'], title='Read this first', color='yellow', variant='light')
    return _curve_figure(rebased, drawdown, run), trades.to_dict('records'), caveats


def _empty_figure(message: str) -> go.Figure:
    """A blank chart carrying one centred message."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
    figure.update_layout(template='plotly_dark', margin={'l': 40, 'r': 20, 't': 20, 'b': 30},
                         xaxis={'visible': False}, yaxis={'visible': False})
    return figure


def _curve_figure(rebased, drawdown, run: dict) -> go.Figure:
    """Equity against benchmark on top, drawdown beneath, sharing one date axis."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    figure.add_trace(go.Scatter(x=rebased['Date'], y=rebased['Equity'], name=run['Signal'],
                                line={'color': STRATEGY_COLOR, 'width': 2}), row=1, col=1)
    if rebased['BenchmarkEquity'].notna().any():
        figure.add_trace(go.Scatter(x=rebased['Date'], y=rebased['BenchmarkEquity'], name=run['Benchmark'],
                                    line={'color': BENCHMARK_COLOR, 'width': 1.5, 'dash': 'dot'}), row=1, col=1)
    figure.add_trace(go.Scatter(x=rebased['Date'], y=drawdown, name='drawdown', fill='tozeroy',
                                line={'color': DRAWDOWN_COLOR, 'width': 1},
                                hovertemplate='%{y:.1%}<extra>drawdown</extra>'), row=2, col=1)
    figure.update_yaxes(title_text='rebased to 100', row=1, col=1)
    figure.update_yaxes(title_text='drawdown', tickformat='.0%', row=2, col=1)
    figure.update_layout(template='plotly_dark', margin={'l': 50, 'r': 20, 't': 20, 'b': 30},
                         legend={'orientation': 'h', 'y': 1.08}, hovermode='x unified')
    return figure





@callback(Output('bt-run-universe', 'data'), Input('bt-run-universe', 'id'))
def load_universe_options(_):
    """Offer every universe saved from the Screener."""
    return universes.list_universes()


@callback(Output('bt-run-status', 'children'), Input('bt-poll', 'n_intervals'))
def show_job_status(_n):
    """Show what the job runner is doing, so a launched run is visibly running."""
    job = MANAGER.current or (MANAGER.history[0] if MANAGER.history else None)
    if job is None:
        return None
    if MANAGER.current is not None:
        return dmc.Group([dmc.Loader(size='xs'), dmc.Text(job.title, size='xs')], gap=6)
    tail = '\n'.join(job.lines[-12:]) if getattr(job, 'lines', None) else ''
    colour = 'green' if getattr(job, 'returncode', 1) == 0 else 'red'
    return dmc.Alert(dmc.Code(tail, block=True) if tail else 'finished',
                     title=f'{job.title} finished', color=colour, variant='light')


@callback(Output('bt-runs-tick', 'data'),
          Input('bt-btn-factors', 'n_clicks'), Input('bt-btn-evaluate', 'n_clicks'),
          Input('bt-btn-run', 'n_clicks'),
          State('bt-run-signal', 'value'), State('bt-run-start', 'value'), State('bt-run-end', 'value'),
          State('bt-run-freq', 'value'), State('bt-run-universe', 'value'), State('bt-run-ntop', 'value'),
          State('bt-run-rankby', 'value'), State('bt-run-vintage', 'value'),
          State('bt-run-horizon', 'value'), State('bt-runs-tick', 'data'),
          prevent_initial_call=True)
def launch(_factors, _evaluate, _run, signal, start, end, freq, universe, n_top, rank_by, vintage,
           horizon, tick):
    """Start the chosen backtest command as a job, with the universe the form selected."""
    if MANAGER.current is not None:
        raise PreventUpdate
    trigger = ctx.triggered_id
    if trigger == 'bt-btn-factors':
        title, argv = logic.factors_argv(start, end, freq, vintage, universe, n_top, rank_by)
    elif trigger == 'bt-btn-evaluate':
        title, argv = logic.evaluate_argv(int(horizon or 21), vintage)
    else:
        title, argv = logic.run_argv(signal, start, end, freq, universe, n_top, rank_by)
    MANAGER.start(title, argv)
    return (tick or 0) + 1

dash.register_page(__name__, path='/backtest', name='Backtest', layout=layout)

clientside_callback(
    "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
    Output('bt-runs', 'className'),
    Input('theme-switch', 'checked'),
)
