"""Backtest page: backtest factor portfolios and timing rules, both built from the factor library.

A factor portfolio holds the top quantile of one factor, or of the average rank of several, from the
same library and point-in-time universe the Factors page analyses. A timing rule enters and exits each
name on its own entry and exit factors, decided at every close. Every run goes through the job runner,
since writing a run needs the database's single write lock and the page itself only reads. Selecting a
stored run draws its equity against the benchmark with a drawdown band, splits it in-sample and
out-of-sample, and lists its settings and fills.
"""

from urllib.parse import parse_qs

import dash
import dash_mantine_components as dmc
import duckdb
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from investalyze.apps import components, theming, universes
from investalyze.apps.backtest import data, logic
from investalyze.apps.control_panel.jobs import MANAGER
from investalyze.backtest import config as backtest_config

BUSY = 'Database busy: a job is running. Try again once it finishes.'
EMPTY_RUNS = 'No stored runs yet. Set one up above and press Run backtest.'
JOB_RUNNING = 'A job is already running. Wait for it to finish, or cancel it.'

STRATEGY_COLOR = '#4C72B0'
BENCHMARK_COLOR = '#8C8C8C'
DRAWDOWN_COLOR = '#C44E52'

TRADE_COLUMNS = [
    {'field': 'Date', 'width': 96},
    {'field': 'Ticker', 'width': 64},
    {'field': 'Side', 'width': 60},
    {'field': 'Shares', 'width': 100, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.1f")(params.value)'}},
    {'field': 'Price', 'width': 80, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.2f")(params.value)'}},
    {'field': 'Notional', 'width': 100, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.0f")(params.value)'}},
    {'field': 'Cost', 'width': 80, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.2f")(params.value)'}},
    {'field': 'Kind', 'width': 84},
]

PERIOD_COLUMNS = [
    {'field': 'period', 'headerName': 'Period', 'width': 120},
    {'field': 'start', 'headerName': 'Start', 'width': 104},
    {'field': 'end', 'headerName': 'End', 'width': 104},
    {'field': 'cagr', 'headerName': 'CAGR', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
    {'field': 'benchmark_cagr', 'headerName': 'Benchmark CAGR', 'width': 140, 'type': 'rightAligned',
     'valueFormatter': {'function': 'params.value == null ? "" : d3.format(".1%")(params.value)'}},
    {'field': 'vol', 'headerName': 'Volatility', 'width': 100, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
    {'field': 'sharpe', 'headerName': 'Sharpe', 'width': 84, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(".2f")(params.value)'}},
    {'field': 'max_drawdown', 'headerName': 'Max drawdown', 'width': 130, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
]

_T_FORMAT = {'function': 'params.value == null ? "" : d3.format("+.2f")(params.value)'}

FACTOR_LIBRARY_COLUMNS = [
    {'field': 'name', 'headerName': 'Factor', 'width': 230},
    {'field': 'status', 'headerName': 'Status', 'width': 100},
    {'field': 'direction', 'headerName': 'Direction', 'width': 96},
    {'field': 't_stat', 'headerName': 't-stat', 'width': 80, 'type': 'rightAligned', 'valueFormatter': _T_FORMAT},
    {'field': 'oos_t_stat', 'headerName': 'OOS t', 'width': 80, 'type': 'rightAligned', 'valueFormatter': _T_FORMAT},
    {'field': 'sharpe', 'headerName': 'Sharpe', 'width': 84, 'type': 'rightAligned',
     'valueFormatter': {'function': 'params.value == null ? "" : d3.format(".2f")(params.value)'}},
    {'field': 'description', 'headerName': 'Rationale', 'width': 360},
    {'field': 'expression', 'headerName': 'Expression', 'width': 360},
]

SETTING_COLUMNS = [
    {'field': 'setting', 'headerName': 'Setting', 'width': 180},
    {'field': 'value', 'headerName': 'Value', 'flex': 1},
]

def _date_fields(prefix: str, start: str, end: str) -> list:
    """The From and To inputs a card carries."""
    return [
        dmc.TextInput(id=f'{prefix}-start', label='From', value=start, size='xs', w=120),
        dmc.TextInput(id=f'{prefix}-end', label='To', value=end, size='xs', w=120),
    ]


def _factor_panel() -> dmc.Paper:
    """A factor portfolio: the top quantile of factors from the factor library."""
    cfg = backtest_config.read()
    costs = cfg.costs
    return dmc.Paper([
        dmc.Group([dmc.Text('Factor portfolio', fw=600, size='sm'),
                   dmc.Text('Top quantile of one factor, or of the average rank of several.', size='xs', c='dimmed')],
                  gap=8, mb=6),
        dmc.Text('Factor library: tick the factors to backtest. t-stat and OOS t come from IC analysis on the '
                 'Factors page, Sharpe from the latest single-factor backtest.', size='xs', c='dimmed', mb=4),
        components.grid('bt-factor-library', FACTOR_LIBRARY_COLUMNS, 280, selection=components.MULTI_ROW,
                        row_id='params.data.name', column_size=None, row_height=28, min_width=None, filterable=True),
        html.Div(id='bt-selected-factors', style={'margin': '6px 0'}),
        dmc.Group([
            *_date_fields('bt-fp', '2016-01-31', '2024-12-31'),
            dmc.Select(id='bt-fp-rebalance', label='Rebalance', value='M', size='xs', w=120,
                       data=[{'value': key, 'label': label.title()} for key, label in
                             (('M', 'monthly'), ('Q', 'quarterly'))]),
            dmc.Select(id='bt-fp-quantiles', label='Quantiles', value='5', size='xs', w=100,
                       data=[{'value': str(n), 'label': str(n)} for n in logic.QUANTILE_CHOICES]),
            dmc.Select(id='bt-fp-size', label='Universe size', value='1000', size='xs', w=120,
                       data=[{'value': str(n), 'label': f'{n:,}'} for n in logic.UNIVERSE_SIZES]),
            dmc.TextInput(id='bt-fp-holdout', label='Out-of-sample from', value='2022-01-01', size='xs', w=140),
        ], gap='xs', mb=6),
        dmc.Text(f'Universe: the most liquid names by trailing dollar volume on each rebalance date. Long the top '
                 f'quantile, equal weight capped at {cfg.limits.max_weight:.0%} per name, filled at the next open. '
                 f'Costs from backtest.toml: {costs.slippage_bps:g} bps slippage, {costs.commission_bps:g} bps '
                 f'commission, impact coefficient {costs.impact_coef:g}. Benchmark {cfg.benchmark}.',
                 size='xs', c='dimmed', mb=8),
        dmc.Button('Run backtest', id='bt-btn-factor-run', size='xs', color='orange'),
    ], p='sm', withBorder=True)


def _timing_panel() -> dmc.Paper:
    """A timing rule: entry and exit factors from the factor library, decided daily per name."""
    return dmc.Paper([
        dmc.Group([dmc.Text('Timing rule', fw=600, size='sm'),
                   dmc.Text('Each name is entered and exited on its own factors, decided at every close.',
                            size='xs', c='dimmed')], gap=8, mb=6),
        dmc.Group([
            dmc.Select(id='bt-tr-entry', label='Entry factor (true = enter)', data=[], searchable=True, size='xs', w=240),
            dmc.Select(id='bt-tr-exit', label='Exit factor (true = exit)', data=[], searchable=True, clearable=True,
                       size='xs', w=240, placeholder='None'),
            dmc.NumberInput(id='bt-tr-stop', label='Trailing stop %', min=0, max=99, size='xs', w=120,
                            placeholder='None'),
            dmc.Checkbox(id='bt-tr-exit-on-false', label='Exit when the entry factor is no longer true', checked=True,
                         size='xs', mt=18),
        ], gap='xs', mb=6, align='flex-start'),
        dmc.Group([
            dmc.NumberInput(id='bt-tr-max-positions', label='Max positions', value=20, min=1, size='xs', w=110),
            dmc.Select(id='bt-tr-priority', label='Priority factor (highest first)', data=[], searchable=True,
                       clearable=True, size='xs', w=240, placeholder='Most liquid first'),
            dmc.Select(id='bt-tr-universe', label='Universe', data=[], clearable=True, searchable=True, size='xs', w=220,
                       placeholder='Most liquid names'),
            dmc.Select(id='bt-tr-size', label='Universe size', value='500', size='xs', w=120,
                       data=[{'value': str(n), 'label': f'{n:,}'} for n in logic.UNIVERSE_SIZES]),
            *_date_fields('bt-tr', '2016-01-01', '2024-12-31'),
            dmc.TextInput(id='bt-tr-holdout', label='Out-of-sample from', value='2022-01-01', size='xs', w=140),
        ], gap='xs', mb=6),
        dmc.Text('Positions are equal weight, 1 / max positions each. Universe size applies when no saved universe is '
                 'picked: the most liquid names by trailing dollar volume, rechosen each month-end.',
                 size='xs', c='dimmed', mb=8),
        dmc.Button('Run backtest', id='bt-btn-timing-run', size='xs', color='orange'),
    ], p='sm', withBorder=True)


def layout(**_query: str) -> html.Div:
    """The page body, built per request so a run stored since the last view shows up.

    Dash passes URL query parameters as keywords; `?factor=` is read by `load_factor_options` instead.
    """
    header = dmc.Group([
        dmc.Title('Backtest', order=3),
        dmc.Text('Backtest factor portfolios and timing rules built from the factor library.', size='sm', c='dimmed'),
    ], gap='md', mb=8)

    runs_panel = html.Div([
        dmc.Group([
            dmc.Text('Stored runs', fw=600, size='sm'),
            dmc.Button('Delete selected', id='bt-btn-delete-run', size='xs', color='red', variant='light',
                       disabled=True),
        ], justify='space-between', mb=4),
        html.Div(id='bt-runs-notice'),
        components.grid('bt-runs', logic.RUN_COLUMNS, 240, selection=components.MULTI_ROW, column_size='autoSize'),
    ])

    detail_panel = html.Div([
        html.Div(id='bt-caveats', style={'marginBottom': '8px'}),
        dcc.Graph(id='bt-curve', config={'displayModeBar': False}, style={'height': '380px'}),
        dmc.Text('In-sample and out-of-sample', fw=600, size='sm', mt=8, mb=4),
        components.grid('bt-periods', PERIOD_COLUMNS, 150, column_size=None, row_height=26),
        dmc.Text('Settings', fw=600, size='sm', mt=8, mb=4),
        components.grid('bt-settings', SETTING_COLUMNS, 220, row_height=26),
        dmc.Text('Fills', fw=600, size='sm', mt=8, mb=4),
        components.grid('bt-trades', TRADE_COLUMNS, 360, column_size=None, row_height=26),
    ])

    return html.Div([
        header,
        _factor_panel(),
        html.Div(style={'height': '10px'}),
        _timing_panel(),
        html.Div(style={'height': '10px'}),
        html.Div(id='bt-launch-note'),
        html.Div(id='bt-job-status'),
        dcc.Interval(id='bt-poll', interval=2000),
        html.Div(style={'height': '10px'}),
        runs_panel,
        html.Div(detail_panel, style={'marginTop': '10px'}),
        dcc.Store(id='bt-cancel-tick'),
        dmc.Modal(
            id='bt-delete-modal', title='Delete runs', centered=True, size='lg',
            children=[
                dmc.Text(id='bt-delete-text', size='sm', style={'whiteSpace': 'pre-line'}),
                dmc.Group([
                    dmc.Button('Cancel', id='bt-delete-no', variant='default', size='xs'),
                    dmc.Button('Delete', id='bt-delete-yes', color='red', size='xs'),
                ], justify='flex-end', mt=12),
            ],
        ),
        dcc.Store(id='bt-pending-delete'),
        dcc.Store(id='bt-runs-tick', data=0),
    ], style={'padding': '4px'})


def _ticked_names(selected: list[dict] | dict | None) -> list[str]:
    """Factor names from the grid's selection, which holds row objects or, right after a load, {'ids': [...]}."""
    if isinstance(selected, dict):
        return list(selected.get('ids', []))
    return [row['name'] for row in (selected or [])]


@callback(Output('bt-factor-library', 'rowData'), Output('bt-factor-library', 'selectedRows'),
          Input('bt-runs-tick', 'data'), Input('url', 'search'), State('bt-factor-library', 'selectedRows'))
def load_factor_library(_tick: str | int | None, search: str | None, current: list[dict] | dict | None) -> tuple:
    """Fill the factor library table, ticking a factor passed from the Factors page as ?factor=name."""
    rows = data.factor_rows()
    names = {row['name'] for row in rows}
    requested = parse_qs((search or '').lstrip('?')).get('factor', [])
    # the first load reports no trigger, and that is when a link from the Factors page arrives
    if requested and ctx.triggered_id in (None, 'url'):
        return rows, {'ids': [name for name in requested if name in names]}
    return rows, {'ids': [name for name in _ticked_names(current) if name in names]}


@callback(Output('bt-selected-factors', 'children'), Input('bt-factor-library', 'selectedRows'))
def show_selected_factors(selected: list[dict] | dict | None) -> components.Children:
    """Name the factors a press of Run backtest will trade."""
    names = _ticked_names(selected)
    if not names:
        return dmc.Text('No factor ticked yet.', size='xs', c='dimmed')
    kind = 'Composite of the average rank of' if len(names) > 1 else 'Single factor:'
    return dmc.Text(f'{kind} {" + ".join(names)}', size='sm', fw=500)


@callback(Output('bt-tr-entry', 'data'), Output('bt-tr-exit', 'data'), Output('bt-tr-priority', 'data'),
          Output('bt-tr-universe', 'data'), Input('bt-runs-tick', 'data'))
def load_timing_options(_tick: str | int | None) -> tuple:
    """Offer the factor library for entry, exit and priority, and every universe saved from the Screener."""
    names = [row['name'] for row in data.factor_rows()]
    return names, names, names, universes.list_universes()


@callback(Output('bt-runs', 'rowData'), Output('bt-runs-notice', 'children'), Input('bt-runs-tick', 'data'))
def load_runs(_tick: str | int | None) -> tuple:
    """Fill the run table once the page mounts, and again after every job."""
    try:
        runs = data.get_runs()
    except duckdb.IOException:
        return [], components.note(BUSY)
    if runs.empty:
        return [], components.note(EMPTY_RUNS)
    return runs.to_dict('records'), None


@callback(Output('bt-curve', 'figure'), Output('bt-trades', 'rowData'), Output('bt-caveats', 'children'),
          Output('bt-periods', 'rowData'), Output('bt-settings', 'rowData'),
          Input('bt-runs', 'selectedRows'), Input('theme-switch', 'checked'))
def show_run(selected: list[dict] | dict | None, dark: bool) -> tuple:
    """Draw the selected run's equity curve and drawdown, and list its periods, settings and fills."""
    dark = bool(dark)
    if not selected:
        return components.empty_figure(dark, 'Select a run'), [], None, [], []
    run = selected[0]
    try:
        curve = data.get_equity(run['RunId'])
        trades = data.get_trades(run['RunId'])
    except duckdb.IOException:
        return components.empty_figure(dark, BUSY), [], components.note(BUSY), [], []
    if curve.empty:
        return components.empty_figure(dark, 'This run stored no equity curve'), [], None, [], []

    holdout = logic.holdout_of(run.get('ConfigJson'))
    rebased = logic.rebase(curve, ['Equity', 'BenchmarkEquity'])
    drawdown = logic.drawdown_from_equity(curve['Equity'])
    caveats = None
    if len(selected) > 1:
        caveats = components.note(f'{len(selected)} runs selected; the chart below is run {run["RunId"]}.', 'yellow')
    periods = logic.run_periods(curve, holdout)
    return (_curve_figure(rebased, drawdown, run, holdout, dark), trades.to_dict('records'), caveats,
            periods.to_dict('records'), logic.run_settings_rows(run.get('ConfigJson')))


def _curve_figure(rebased: pd.DataFrame, drawdown: pd.Series, run: dict, holdout: pd.Timestamp | None,
                  dark: bool) -> go.Figure:
    """Equity against benchmark on top, drawdown beneath, sharing one date axis."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    figure.add_trace(go.Scatter(x=rebased['Date'], y=rebased['Equity'], name=run['Signal'],
                                line={'color': STRATEGY_COLOR, 'width': 2}), row=1, col=1)
    if rebased['BenchmarkEquity'].notna().any():
        figure.add_trace(go.Scatter(x=rebased['Date'], y=rebased['BenchmarkEquity'], name=run['Benchmark'],
                                    line={'color': BENCHMARK_COLOR, 'width': 1.5, 'dash': 'dot'}), row=1, col=1)
    figure.add_trace(go.Scatter(x=rebased['Date'], y=drawdown, name='Drawdown', fill='tozeroy',
                                line={'color': DRAWDOWN_COLOR, 'width': 1},
                                hovertemplate='%{y:.1%}<extra>Drawdown</extra>'), row=2, col=1)
    if holdout is not None:
        figure.add_vline(x=holdout, line={'dash': 'dot', 'width': 1, 'color': DRAWDOWN_COLOR})
        figure.add_annotation(x=holdout, y=1, yref='paper', text='Out-of-sample', showarrow=False,
                              xanchor='left', font={'size': 11, 'color': DRAWDOWN_COLOR})
    figure.update_yaxes(title_text='Rebased to 100', row=1, col=1)
    figure.update_yaxes(title_text='Drawdown', tickformat='.0%', row=2, col=1)
    figure.update_layout(**theming.figure_layout(dark), margin={'l': 50, 'r': 20, 't': 20, 'b': 30},
                         legend={'orientation': 'h', 'y': 1.08}, hovermode='x unified')
    return figure


@callback(Output('bt-job-status', 'children'), Input('bt-poll', 'n_intervals'))
def show_job_status(_n: int | None) -> components.Children:
    """Show what the job runner is doing, with a way out of it while it is still running."""
    job = MANAGER.current or (MANAGER.history[0] if MANAGER.history else None)
    if job is None:
        return None
    if MANAGER.current is not None:
        return dmc.Group([dmc.Loader(size='xs'), dmc.Text(job.title, size='xs'),
                          dmc.Button('Cancel', id='bt-btn-cancel', size='xs', color='red', variant='subtle')], gap=6)
    tail = '\n'.join(job.lines[-12:]) if getattr(job, 'lines', None) else ''
    colour = 'green' if getattr(job, 'returncode', 1) == 0 else 'red'
    return dmc.Alert(dmc.Code(tail, block=True) if tail else 'Finished', title=f'{job.title} finished',
                     color=colour, variant='light')


@callback(Output('bt-launch-note', 'children'),
          Input('bt-btn-factor-run', 'n_clicks'), Input('bt-btn-timing-run', 'n_clicks'),
          State('bt-factor-library', 'selectedRows'), State('bt-fp-start', 'value'), State('bt-fp-end', 'value'),
          State('bt-fp-rebalance', 'value'), State('bt-fp-quantiles', 'value'), State('bt-fp-size', 'value'),
          State('bt-fp-holdout', 'value'),
          State('bt-tr-entry', 'value'), State('bt-tr-exit', 'value'), State('bt-tr-stop', 'value'),
          State('bt-tr-exit-on-false', 'checked'), State('bt-tr-max-positions', 'value'), State('bt-tr-priority', 'value'),
          State('bt-tr-universe', 'value'), State('bt-tr-size', 'value'), State('bt-tr-start', 'value'),
          State('bt-tr-end', 'value'), State('bt-tr-holdout', 'value'),
          prevent_initial_call=True)
def launch(_factor_run: int | None, _timing_run: int | None, selected_factors: list[dict] | dict | None, fp_start: str | None,
           fp_end: str | None, rebalance: str | None, quantiles: str | None, size: str | None, fp_holdout: str | None,
           entry: str | None, exit_factor: str | None, stop: float | str | None, exit_on_false: bool,
           max_positions: float | str | None,
           priority: str | None, universe: str | None, tr_size: str | None, tr_start: str | None, tr_end: str | None,
           tr_holdout: str | None) -> components.Children:
    """Start the pressed card's backtest as a job, or say why it cannot start."""
    if MANAGER.current is not None:
        return components.note(JOB_RUNNING, 'yellow')
    if ctx.triggered_id == 'bt-btn-factor-run':
        factor_names = _ticked_names(selected_factors)
        if not factor_names:
            return components.note('Tick at least one factor in the factor library table.', 'orange')
        title, argv = logic.factor_portfolio_argv(factor_names, fp_start, fp_end, rebalance or 'M',
                                                  int(quantiles or 5), int(size or 1000), fp_holdout)
    else:
        if not entry:
            return components.note('Pick an entry factor.', 'orange')
        stop_percent = float(stop) if stop not in (None, '') else None
        problem = logic.timing_rule_problem(exit_factor=exit_factor, exit_on_entry_false=bool(exit_on_false),
                                            trailing_stop_percent=stop_percent)
        if problem:
            return components.note(problem, 'orange')
        title, argv = logic.timing_rule_argv(entry, exit_factor=exit_factor, exit_on_entry_false=bool(exit_on_false),
                                             trailing_stop_percent=stop_percent, max_positions=int(max_positions or 20),
                                             priority_factor=priority, universe=universe,
                                             universe_size=int(tr_size or 500), start=tr_start, end=tr_end,
                                             holdout=tr_holdout)
    if not MANAGER.start(title, argv):
        return components.note(JOB_RUNNING, 'yellow')
    return None


@callback(Output('bt-cancel-tick', 'data'), Input('bt-btn-cancel', 'n_clicks'), prevent_initial_call=True)
def cancel_job(n_clicks: int | None) -> int | None:
    """Stop the running job."""
    if n_clicks:
        MANAGER.cancel()
    return n_clicks


@callback(Output('bt-runs-tick', 'data'), Input('bt-poll', 'n_intervals'),
          State('bt-runs-tick', 'data'), prevent_initial_call=True)
def refresh_after_job(_n: int | None, tick: str | int | None) -> str:
    """Reload the tables when a job finishes, keyed on when it finished so one reload happens per job."""
    if MANAGER.current is not None or not MANAGER.history:
        raise PreventUpdate
    finished = MANAGER.history[0].finished
    stamp = finished.isoformat() if finished else None
    if stamp is None or stamp == tick:
        raise PreventUpdate
    return stamp


@callback(Output('bt-btn-delete-run', 'disabled'), Input('bt-runs', 'selectedRows'))
def toggle_delete(selected: list[dict] | dict | None) -> bool:
    """There is nothing to delete until at least one run is ticked."""
    return not selected


@callback(Output('bt-delete-modal', 'opened'), Output('bt-delete-text', 'children'),
          Output('bt-pending-delete', 'data'),
          Input('bt-btn-delete-run', 'n_clicks'), State('bt-runs', 'selectedRows'), prevent_initial_call=True)
def ask_delete(_n: int | None, selected: list[dict] | dict | None) -> tuple:
    """Name every selected run and what goes with it, since the page offers no way back."""
    if not selected:
        raise PreventUpdate
    return True, logic.delete_prompt(selected), {'ids': [run['RunId'] for run in selected]}


@callback(Output('bt-delete-modal', 'opened', allow_duplicate=True),
          Output('bt-pending-delete', 'data', allow_duplicate=True),
          Output('bt-launch-note', 'children', allow_duplicate=True),
          Input('bt-delete-yes', 'n_clicks'), Input('bt-delete-no', 'n_clicks'),
          State('bt-pending-delete', 'data'), prevent_initial_call=True)
def resolve_delete(yes_clicks: int | None, _no: int | None, pending: dict | None) -> tuple:
    """Start the pending delete on Delete; just close the modal on Cancel.

    A click is required, not merely a callback firing with Delete as its trigger: a remounted button
    reports no clicks, and nothing irreversible should start from one.
    """
    if ctx.triggered_id == 'bt-delete-yes' and yes_clicks and pending:
        if not MANAGER.start(*logic.delete_argv(pending['ids'])):
            return False, None, components.note('A job is already running, so nothing was deleted. Try again once it finishes.',
                                      'yellow')
    return False, None, dash.no_update


dash.register_page(__name__, path='/backtest', name='Backtest', layout=layout)

# every grid on the page follows the switch, not just the first one
for _grid_id in ('bt-factor-library', 'bt-runs', 'bt-trades', 'bt-periods', 'bt-settings'):
    clientside_callback(
        "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
        Output(_grid_id, 'className'),
        Input('theme-switch', 'checked'),
    )
