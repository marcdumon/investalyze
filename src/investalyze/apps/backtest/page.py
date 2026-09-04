"""Backtest page: launching runs, and reading back the runs and factor evaluations they stored.

The panel at the top builds a factor panel, evaluates factors and runs a signal, each as a
`python -m investalyze.backtest` job, since every CLI writes to the same single-writer DuckDB file
and the page itself only ever reads. Selecting a run draws its equity against the benchmark with a
drawdown band beneath, and lists the fills. Deleting one goes through the same job runner.
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

from investalyze.apps import theming, universes
from investalyze.apps.backtest import data, logic
from investalyze.apps.control_panel.jobs import MANAGER
from investalyze.backtest import registry

BUSY = 'database busy, a job is currently running, try again once it finishes'
EMPTY_FACTORS = 'no factor evaluations yet; press Evaluate factors above'
EMPTY_RUNS = 'no stored runs yet; set one up above and press Run backtest'

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


# the checkbox column comes free with this selection mode; clicking the row still selects it
_MULTI_SELECT = {'mode': 'multiRow', 'checkboxes': True, 'headerCheckbox': True, 'enableClickSelection': True}


def _grid(grid_id: str, columns: list[dict], height: int, selectable: bool = False) -> dag.AgGrid:
    """One AG Grid with the page's shared styling.

    A selectable grid sizes its columns to their contents rather than stretching them to the width
    available, so a long signal name or a six-figure fill count is never cut short.
    """
    options = {'theme': 'legacy'}   # without this the className theme below is ignored
    if selectable:
        options['rowSelection'] = _MULTI_SELECT
    return dag.AgGrid(
        id=grid_id, columnDefs=columns, rowData=[],
        # minWidth is the floor under autoSize/sizeToFit: seen to under-size a column when
        # browser zoom or DPI scaling changes rendered text width independently of the
        # canvas autoSize measures with, which otherwise reads as truncated data
        defaultColDef={'sortable': True, 'resizable': True, 'minWidth': 70},
        dashGridOptions=options,
        columnSize='autoSize' if selectable else 'sizeToFit',
        columnSizeOptions={'skipHeader': False},
        style={'height': height}, className='ag-theme-alpine-dark',
    )


def _note(text: str) -> dmc.Alert:
    """A dimmed inline note for an empty or unavailable panel."""
    return dmc.Alert(text, color='gray', variant='light')


def _date_fields(prefix: str, today: str) -> list:
    """The From and To inputs a card carries, debounced so a query does not run per keystroke."""
    return [
        dmc.TextInput(id=f'{prefix}-start', label='From', value='2015-01-01', size='xs', w=120,
                      debounce=400),
        dmc.TextInput(id=f'{prefix}-end', label='To', value=today, size='xs', w=120, debounce=400),
    ]


def _universe_select(prefix: str) -> dmc.Select:
    """The universe picker a card carries, filled from the universes saved on the Screener."""
    return dmc.Select(id=f'{prefix}-universe', label='Universe', placeholder='whole database',
                      data=[], clearable=True, searchable=True, size='xs', w=260)


def _cadence_select(prefix: str) -> dmc.Select:
    """How often the factor panel recomputes its values, on whichever card might cause a build."""
    return dmc.Select(id=f'{prefix}-panel-freq', label='Panel cadence', value='M', size='xs', w=130,
                      data=[{'value': f, 'label': label} for f, label in
                            (('D', 'daily'), ('W', 'weekly'), ('M', 'monthly'), ('Q', 'quarterly'))])


def _version_select(prefix: str) -> dmc.Select:
    """Which fundamentals version a card works from."""
    return dmc.Select(id=f'{prefix}-version', label='Version', value='as_reported', size='xs', w=140,
                      data=[{'value': 'as_reported', 'label': 'as reported'},
                            {'value': 'restated', 'label': 'restated'}])


def _research_panel() -> dmc.Paper:
    """Scoring the factor library against forward returns. Owns its own settings."""
    return dmc.Paper([
        dmc.Group([dmc.Text('Research factors', fw=600, size='sm'),
                   dmc.Text('which factors predicted returns, and how strongly', size='xs', c='dimmed')],
                  gap=8, mb=6),
        dmc.Group(_date_fields('bt-res', date.today().isoformat()) + [
            _universe_select('bt-res'),
            _version_select('bt-res'),
            _cadence_select('bt-res'),
            dmc.NumberInput(id='bt-res-horizon', label='IC horizon', value=21, min=1, size='xs', w=110),
        ], gap='xs', mb=8),
        dmc.Button('Evaluate factors', id='bt-btn-evaluate', size='xs', color='orange'),
        html.Div(id='bt-res-plan', style={'marginTop': '8px'}),
    ], p='sm', withBorder=True)


def _backtest_panel() -> dmc.Paper:
    """Walking one signal through the engine. Owns its own settings."""
    return dmc.Paper([
        dmc.Group([dmc.Text('Backtest a strategy', fw=600, size='sm'),
                   dmc.Text('what one signal would have done, after costs', size='xs', c='dimmed')],
                  gap=8, mb=6),
        dmc.Group([
            dmc.Select(id='bt-run-signal', label='Signal', value='factor_composite', size='xs', w=200,
                       data=[{'value': name, 'label': name} for name in sorted(registry.SIGNALS)]),
        ] + _date_fields('bt-run', date.today().isoformat()) + [
            dmc.Select(id='bt-run-freq', label='Rebalance', value='M', size='xs', w=100,
                       data=[{'value': f, 'label': f} for f in ('D', 'W', 'M', 'Q')]),
        ], gap='xs', mb=6),
        dmc.Group([
            _universe_select('bt-run'),
            dmc.NumberInput(id='bt-run-ntop', label='Top N', value=500, min=1, size='xs', w=100),
            dmc.Select(id='bt-run-rankby', label='Ranked by', value='liquidity', size='xs', w=130,
                       data=[{'value': r, 'label': r} for r in ('liquidity', 'mcap', 'ticker')]),
            _version_select('bt-run'),
            _cadence_select('bt-run'),
        ], gap='xs', mb=6),
        dmc.Text('Rebalance is how often the portfolio trades; Panel cadence is how often the factor '
                 'values behind it are recomputed, and only matters when this run would build them. '
                 'A run trades the names in the universe that were priced across its whole range, '
                 'capped at Top N. Leaving the universe blank uses the whole database.',
                 size='xs', c='dimmed', mb=8),
        dmc.Button('Run backtest', id='bt-btn-run', size='xs', color='orange'),
        html.Div(id='bt-run-plan', style={'marginTop': '8px'}),
    ], p='sm', withBorder=True)


def layout() -> html.Div:
    """The page body, built per request so a run stored since the last view shows up."""
    header = dmc.Group([
        dmc.Title('Backtest', order=3),
        dmc.Text('research the factors, backtest a strategy, read what they stored', size='sm', c='dimmed'),
    ], gap='md', mb=8)

    runs_panel = html.Div([
        dmc.Group([
            dmc.Text('Stored runs', fw=600, size='sm'),
            dmc.Button('Delete selected', id='bt-btn-delete-run', size='xs', color='red',
                       variant='light', disabled=True),
        ], justify='space-between', mb=4),
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
        dmc.Group([
            dmc.Text('Factor evaluation', fw=600, size='sm'),
            dmc.Group([
                dmc.Select(id='bt-factors-universe', label=None, placeholder='every universe', data=[],
                           clearable=True, searchable=True, size='xs', w=280),
                dmc.Button('Delete evaluation', id='bt-btn-delete-evaluation', size='xs', color='red',
                           variant='light', disabled=True),
            ], gap='xs'),
        ], justify='space-between', mb=4),
        dmc.Text('IC is measured on raw values; the oriented column flips the factors whose lower '
                 'values are the better ones, so the column ranks the library on one scale. Each '
                 'evaluation belongs to a version and a universe together, so this reads the '
                 "version chosen on the research card above and the universe picked here.",
                 size='xs', c='dimmed', mb=6),
        html.Div(id='bt-factors-notice'),
        _grid('bt-factors', logic.FACTOR_COLUMNS, 420),
    ])

    return html.Div([
        header,
        _research_panel(),
        html.Div(style={'height': '10px'}),
        _backtest_panel(),
        html.Div(style={'height': '10px'}),
        html.Div(id='bt-job-status'),
        dcc.Interval(id='bt-poll', interval=2000),
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
        dcc.Store(id='bt-selected-run'),
        dcc.Store(id='bt-pending-delete'),
        dcc.Store(id='bt-launched'),
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
          Input('bt-runs-tick', 'data'), Input('bt-factors-universe', 'value'),
          Input('bt-res-version', 'value'))
def load_factors(_tick, universe, version):
    """Fill the factor evaluation table for the chosen universe on the research card's version.

    A stored evaluation belongs to a version and a universe together, so reading one without the
    other pools two measurements of different things into one table.
    """
    try:
        table = logic.orient_ic(data.get_factor_ic(version or 'as_reported', universe))
    except duckdb.IOException:
        return [], _note(BUSY)
    if table.empty:
        return [], _note(EMPTY_FACTORS)
    return table.to_dict('records'), None


@callback(Output('bt-factors-universe', 'data'), Input('bt-runs-tick', 'data'),
          Input('bt-res-version', 'value'))
def load_evaluated_universes(_tick, version):
    """Offer the universes an evaluation has been stored for on this version."""
    try:
        return data.get_evaluated_universes(version or 'as_reported')
    except duckdb.IOException:
        return []


@callback(Output('bt-btn-delete-evaluation', 'disabled'), Input('bt-factors-universe', 'value'))
def toggle_delete_evaluation(universe):
    """One stored evaluation is deleted at a time, so one has to be picked first."""
    return not universe


@callback(Output('bt-delete-modal', 'opened', allow_duplicate=True),
          Output('bt-delete-text', 'children', allow_duplicate=True),
          Output('bt-pending-delete', 'data', allow_duplicate=True),
          Input('bt-btn-delete-evaluation', 'n_clicks'),
          State('bt-factors-universe', 'value'), State('bt-res-version', 'value'),
          prevent_initial_call=True)
def ask_delete_evaluation(_n, universe, version):
    """Name the evaluation and what it holds, since deleting it means scoring it all again."""
    if not universe:
        raise PreventUpdate
    version = version or 'as_reported'
    scopes = [scope for scope in data.get_evaluation_scopes(version) if scope['universe'] == universe]
    if not scopes:
        raise PreventUpdate
    return True, logic.delete_evaluation_prompt(scopes[0]), {'kind': 'evaluation', 'version': version,
                                                             'universe': universe}


@callback(Output('bt-curve', 'figure'), Output('bt-trades', 'rowData'), Output('bt-caveats', 'children'),
          Input('bt-runs', 'selectedRows'), Input('theme-switch', 'checked'))
def show_run(selected, dark):
    """Draw the selected run's equity curve, drawdown band, caveats and fills, in the current theme."""
    dark = bool(dark)
    if not selected:
        return _empty_figure('select a run', dark), [], None
    run = selected[0]
    try:
        curve = data.get_equity(run['RunId'])
        trades = data.get_trades(run['RunId'])
    except duckdb.IOException:
        return _empty_figure(BUSY, dark), [], _note(BUSY)
    if curve.empty:
        return _empty_figure('this run stored no equity curve', dark), [], None

    rebased = logic.rebase(curve, ['Equity', 'BenchmarkEquity'])
    drawdown = logic.drawdown_from_equity(curve['Equity'])
    caveats = None
    if len(selected) > 1:
        note = f'{len(selected)} runs selected; the chart below is run {run["RunId"]}.'
        caveats = dmc.Alert(dmc.Text(note, size='sm'), title='Note', color='yellow', variant='light')
    return _curve_figure(rebased, drawdown, run, dark), trades.to_dict('records'), caveats


def _empty_figure(message: str, dark: bool) -> go.Figure:
    """A blank chart carrying one centred message."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
    figure.update_layout(**theming.figure_layout(dark), margin={'l': 40, 'r': 20, 't': 20, 'b': 30},
                         xaxis={'visible': False}, yaxis={'visible': False})
    return figure


def _curve_figure(rebased, drawdown, run: dict, dark: bool) -> go.Figure:
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
    figure.update_layout(**theming.figure_layout(dark), margin={'l': 50, 'r': 20, 't': 20, 'b': 30},
                         legend={'orientation': 'h', 'y': 1.08}, hovermode='x unified')
    return figure


def _plan_alert(ready: bool, sentence: str) -> dmc.Alert:
    """The line under a button saying what pressing it will do."""
    return dmc.Alert(sentence, color='green' if ready else 'yellow', variant='light')


def _gaps_for(version, universe, start, end, panel_freq, rebalance=None):
    """What the stored panel is missing for one card's settings, or None while the DB is busy.

    `panel_freq` is the cadence the card would build at; `rebalance` is how often a run would trade,
    which the research card has no opinion on and leaves unset.
    """
    try:
        return data.panel_gaps(version or 'as_reported', universe, start, end, panel_freq, rebalance)
    except (duckdb.IOException, duckdb.Error, ValueError):
        return None


@callback(Output('bt-res-plan', 'children'),
          Input('bt-runs-tick', 'data'), Input('bt-res-start', 'value'), Input('bt-res-end', 'value'),
          Input('bt-res-universe', 'value'), Input('bt-res-version', 'value'),
          Input('bt-res-panel-freq', 'value'))
def show_research_plan(_tick, start, end, universe, version, panel_freq):
    """Say what Evaluate will do before it is pressed, including anything it would have to build."""
    gaps = _gaps_for(version, universe, start, end, panel_freq or 'M')
    if gaps is None:
        return _note(BUSY)
    return _plan_alert(*logic.research_plan(gaps, universe, version or 'as_reported'))


@callback(Output('bt-run-plan', 'children'),
          Input('bt-runs-tick', 'data'), Input('bt-run-signal', 'value'),
          Input('bt-run-start', 'value'), Input('bt-run-end', 'value'), Input('bt-run-freq', 'value'),
          Input('bt-run-universe', 'value'), Input('bt-run-version', 'value'),
          Input('bt-run-panel-freq', 'value'), Input('bt-run-ntop', 'value'),
          Input('bt-run-rankby', 'value'))
def show_backtest_plan(_tick, signal, start, end, freq, universe, version, panel_freq, n_top, rank_by):
    """Say what Run will do before it is pressed, what it would build, and how much it will trade."""
    kind = registry.SIGNALS[signal][0] if signal in registry.SIGNALS else registry.CROSS_SECTIONAL
    tradable = _tradable_for(universe, start, end, n_top, rank_by)
    if kind != registry.CROSS_SECTIONAL:
        return _plan_alert(*logic.backtest_plan({'n_tickers': 0}, kind, signal, universe,
                                                tradable=tradable))
    gaps = _gaps_for(version, universe, start, end, panel_freq or 'M', freq or 'M')
    if gaps is None:
        return _note(BUSY)
    return _plan_alert(*logic.backtest_plan(gaps, kind, signal, universe, version or 'as_reported',
                                            tradable))


def _tradable_for(universe, start, end, n_top, rank_by):
    """How much of a universe a run would trade, or None while the database is busy."""
    try:
        return data.tradable_names(universe, start, end, int(n_top or 500), rank_by or 'liquidity')
    except (duckdb.Error, ValueError):
        return None


@callback(Output('bt-res-universe', 'data'), Output('bt-run-universe', 'data'),
          Input('bt-runs-tick', 'data'))
def load_universe_options(_tick):
    """Offer every universe saved from the Screener, on both cards."""
    saved = universes.list_universes()
    return saved, saved


@callback(Output('bt-job-status', 'children'), Input('bt-poll', 'n_intervals'))
def show_job_status(_n):
    """Show what the job runner is doing, with a way out of it while it is still running."""
    job = MANAGER.current or (MANAGER.history[0] if MANAGER.history else None)
    if job is None:
        return None
    if MANAGER.current is not None:
        return dmc.Group([dmc.Loader(size='xs'), dmc.Text(job.title, size='xs'),
                          dmc.Button('Cancel', id='bt-btn-cancel', size='xs', color='red',
                                     variant='subtle')], gap=6)
    tail = '\n'.join(job.lines[-12:]) if getattr(job, 'lines', None) else ''
    colour = 'green' if getattr(job, 'returncode', 1) == 0 else 'red'
    return dmc.Alert(dmc.Code(tail, block=True) if tail else 'finished',
                     title=f'{job.title} finished', color=colour, variant='light')


@callback(Output('bt-launched', 'data'),
          Input('bt-btn-evaluate', 'n_clicks'), Input('bt-btn-run', 'n_clicks'),
          State('bt-res-start', 'value'), State('bt-res-end', 'value'),
          State('bt-res-universe', 'value'), State('bt-res-version', 'value'),
          State('bt-res-horizon', 'value'), State('bt-res-panel-freq', 'value'),
          State('bt-run-signal', 'value'), State('bt-run-start', 'value'), State('bt-run-end', 'value'),
          State('bt-run-freq', 'value'), State('bt-run-universe', 'value'), State('bt-run-ntop', 'value'),
          State('bt-run-rankby', 'value'), State('bt-run-version', 'value'),
          State('bt-run-panel-freq', 'value'),
          prevent_initial_call=True)
def launch(_evaluate, _run, res_start, res_end, res_universe, res_version, res_horizon,
           res_panel_freq, signal, run_start, run_end, freq, run_universe, n_top, rank_by,
           run_version, run_panel_freq):
    """Start a card's job, building the factor panel first when the card said it would have to.

    One press: the line under the button has already said what will happen, and asking again would
    only repeat it back. What a press cannot promise is how long a build takes, so the job is
    interruptible instead, through the Cancel that appears beside it while it runs.
    """
    if MANAGER.current is not None:
        raise PreventUpdate

    if ctx.triggered_id == 'bt-btn-evaluate':
        gaps = _gaps_for(res_version, res_universe, res_start, res_end, res_panel_freq or 'M')
        ready = gaps is not None and logic.research_plan(gaps, res_universe, res_version)[0]
        pending = {'kind': 'evaluate', 'horizon': int(res_horizon or 21), 'version': res_version,
                   'universe': res_universe, 'start': res_start, 'end': res_end,
                   'panel_freq': res_panel_freq or 'M'}
    else:
        kind = registry.SIGNALS[signal][0] if signal in registry.SIGNALS else registry.CROSS_SECTIONAL
        gaps = _gaps_for(run_version, run_universe, run_start, run_end, run_panel_freq or 'M',
                         freq or 'M') if kind == registry.CROSS_SECTIONAL else None
        ready = kind != registry.CROSS_SECTIONAL or (
            gaps is not None and logic.backtest_plan(gaps, kind, signal, run_universe, run_version)[0])
        pending = {'kind': 'run', 'signal': signal, 'start': run_start, 'end': run_end, 'freq': freq,
                   'universe': run_universe, 'n_top': n_top, 'rank_by': rank_by,
                   'version': run_version, 'panel_freq': run_panel_freq or 'M'}

    if not ready and gaps is None:
        raise PreventUpdate   # the database is busy, so what is missing is unknown
    title, argv = _job_argv(pending, build_panel=not ready)
    MANAGER.start(title, argv)
    return title


def _job_argv(pending: dict, *, build_panel: bool) -> tuple[str, list[str]]:
    """The command one card's settings turn into, with or without the panel build in front of it."""
    if pending['kind'] == 'evaluate':
        return logic.evaluate_argv(pending['horizon'], pending['version'], pending['universe'],
                                   pending['start'], pending['end'], build_panel,
                                   pending['panel_freq'])
    return logic.run_argv(pending['signal'], pending['start'], pending['end'], pending['freq'],
                          pending['universe'], pending['n_top'], pending['rank_by'],
                          pending['version'], build_panel, pending['panel_freq'])


@callback(Output('bt-cancel-tick', 'data'), Input('bt-btn-cancel', 'n_clicks'),
          prevent_initial_call=True)
def cancel_job(n_clicks):
    """Stop the running job, since a single press cannot promise how long a build will take."""
    if n_clicks:
        MANAGER.cancel()
    return n_clicks


@callback(Output('bt-runs-tick', 'data'), Input('bt-poll', 'n_intervals'),
          State('bt-runs-tick', 'data'), prevent_initial_call=True)
def refresh_after_job(_n, tick):
    """Reload the tables when a job finishes, keyed on when it finished so one reload happens per job.

    Initial calls are prevented so the tables load on their own: a PreventUpdate raised here before
    any job has run would cancel the callbacks downstream of the tick along with it.
    """
    if MANAGER.current is not None or not MANAGER.history:
        raise PreventUpdate
    finished = MANAGER.history[0].finished
    stamp = finished.isoformat() if finished else None
    if stamp is None or stamp == tick:
        raise PreventUpdate
    return stamp


@callback(Output('bt-btn-delete-run', 'disabled'), Input('bt-runs', 'selectedRows'))
def toggle_delete(selected):
    """There is nothing to delete until at least one run is ticked."""
    return not selected


@callback(Output('bt-delete-modal', 'opened'), Output('bt-delete-text', 'children'),
          Output('bt-pending-delete', 'data'),
          Input('bt-btn-delete-run', 'n_clicks'), State('bt-runs', 'selectedRows'),
          prevent_initial_call=True)
def ask_delete(_n, selected):
    """Name every selected run and what goes with it, since the page offers no way back."""
    if not selected:
        raise PreventUpdate
    return True, logic.delete_prompt(selected), {'kind': 'runs',
                                                'ids': [run['RunId'] for run in selected]}


@callback(Output('bt-delete-modal', 'opened', allow_duplicate=True),
          Output('bt-pending-delete', 'data', allow_duplicate=True),
          Input('bt-delete-yes', 'n_clicks'), Input('bt-delete-no', 'n_clicks'),
          State('bt-pending-delete', 'data'), prevent_initial_call=True)
def resolve_delete(yes_clicks, _no, pending):
    """Start the pending delete on Delete; just close the modal on Cancel.

    A click is required, not merely a callback firing with Delete as its trigger: a remounted button
    reports no clicks, and nothing irreversible should start from one. The payload is cleared either
    way, so one left over from a closed dialog can never be spent twice.
    """
    if ctx.triggered_id == 'bt-delete-yes' and yes_clicks and pending:
        if pending['kind'] == 'runs':
            MANAGER.start(*logic.delete_argv(pending['ids']))
        else:
            MANAGER.start(*logic.delete_evaluation_argv(pending['version'], pending['universe']))
    return False, None

dash.register_page(__name__, path='/backtest', name='Backtest', layout=layout)

# every grid on the page follows the switch, not just the first one
for _grid_id in ('bt-runs', 'bt-trades', 'bt-factors'):
    clientside_callback(
        "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
        Output(_grid_id, 'className'),
        Input('theme-switch', 'checked'),
    )
