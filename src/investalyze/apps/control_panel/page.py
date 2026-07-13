"""Control panel page: run the investalyze CLIs and monitor the database from the browser.

Every 'Run' button builds an argv for `python -m investalyze.<module>` and hands it to
`jobs.MANAGER`, which runs it as a subprocess (see jobs.py for why: one DB writer at a time).
The status cards at the top query the DB directly through short-lived read-only connections
and are paused while a job is running, since the running job may hold the write lock.
"""

from pathlib import Path

import dash
import dash_mantine_components as dmc
import duckdb
from dash import Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from investalyze.apps.control_panel import status
from investalyze.apps.control_panel.jobs import MANAGER
from investalyze.ingest import housekeeping, orchestrator, storage
from investalyze.quality import registry as quality_registry

DATA_ROOT = Path('data')
REPO_ROOT = Path(__file__).resolve().parents[4]

SEVERITY_COLOR = {'error': 'red', 'warn': 'yellow'}


# ---------- pure helpers (argv building; exercised standalone, no Dash needed) ----------

def build_ingest_argv(providers: list[str] | None, update: bool) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.ingest`, with optional provider filter and --update."""
    argv = ['-m', 'investalyze.ingest']
    for provider in providers or []:
        argv += ['-p', provider]
    if update:
        argv.append('--update')
    title = f"ingest {'update' if update else 'full'} ({', '.join(providers) if providers else 'all providers'})"
    return title, argv


def build_housekeeping_argv(tasks: list[str] | None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.ingest housekeeping`, with an optional task filter."""
    argv = ['-m', 'investalyze.ingest', 'housekeeping']
    for task in tasks or []:
        argv += ['-t', task]
    title = f"housekeeping ({', '.join(tasks) if tasks else 'all tasks'})"
    return title, argv


def build_quality_argv(checks: list[str] | None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.quality`, with an optional check filter."""
    argv = ['-m', 'investalyze.quality', *(checks or [])]
    title = f"quality ({', '.join(checks) if checks else 'all checks'})"
    return title, argv


# ---------- status cards ----------

def freshness_card() -> dmc.Card:
    """Card showing how many days old each key table's latest row is."""
    return dmc.Card(
        [dmc.Text('Freshness', fw=700, size='sm', mb=6), html.Div(id='cp-freshness-body')],
        withBorder=True, radius='md', padding='sm',
    )


def row_counts_card() -> dmc.Card:
    """Card listing every table's row count, largest first."""
    return dmc.Card(
        [dmc.Text('Row counts', fw=700, size='sm', mb=6),
         html.Div(id='cp-rowcounts-body', style={'maxHeight': '220px', 'overflowY': 'auto'})],
        withBorder=True, radius='md', padding='sm',
    )


def anomalies_card() -> dmc.Card:
    """Card summarizing anomaly counts by check and severity."""
    return dmc.Card(
        [dmc.Text('Anomalies', fw=700, size='sm', mb=6),
         html.Div(id='cp-anomalies-body', style={'maxHeight': '220px', 'overflowY': 'auto'})],
        withBorder=True, radius='md', padding='sm',
    )


def days_ago_badge(days_ago: int | None) -> dmc.Badge:
    """Green/yellow/red badge for how stale a table's latest row is."""
    if days_ago is None:
        return dmc.Badge('no data', color='gray', variant='light')
    color = 'green' if days_ago <= 3 else 'yellow' if days_ago <= 10 else 'red'
    return dmc.Badge(f'{days_ago}d ago', color=color, variant='light')


# ---------- command cards ----------

def ingest_card() -> dmc.Card:
    """Provider chips + full/update toggle + Run button."""
    return dmc.Card([
        dmc.Text('Ingest', fw=700, size='sm', mb=8),
        dmc.ChipGroup(
            dmc.Group([dmc.Chip(desc, value=name, size='xs') for name, desc in orchestrator.PROVIDER_DESCRIPTIONS.items()]),
            id='cp-ingest-providers', multiple=True, value=[],
        ),
        dmc.SegmentedControl(
            id='cp-ingest-mode', value='update',
            data=[{'label': 'Update', 'value': 'update'}, {'label': 'Full load', 'value': 'full'}],
            mt=8, size='xs',
        ),
        dmc.Button('Run ingest', id='cp-btn-ingest', size='xs', mt=8, leftSection=DashIconify(icon='tabler:player-play')),
        dmc.Group([
            dmc.Anchor('stooq bulk file', href='http://stooq.com/db/h/', target='_blank', size='xs'),
            dmc.Anchor('stooq daily update', href='http://stooq.com/db/', target='_blank', size='xs'),
        ], mt=8, gap=12),
        dmc.Text("stooq downloads are captcha-protected: fetch manually, drop into data/stooq/raw/",
                 size='xs', c='dimmed', mt=2),
    ], withBorder=True, radius='md', padding='sm')


def housekeeping_card() -> dmc.Card:
    """Task chips + Run button."""
    return dmc.Card([
        dmc.Text('Housekeeping', fw=700, size='sm', mb=8),
        dmc.ChipGroup(
            dmc.Group([dmc.Chip(desc, value=name, size='xs') for name, desc in housekeeping.TASK_DESCRIPTIONS.items()]),
            id='cp-housekeeping-tasks', multiple=True, value=[],
        ),
        dmc.Button('Run housekeeping', id='cp-btn-housekeeping', size='xs', mt=8,
                   leftSection=DashIconify(icon='tabler:player-play')),
    ], withBorder=True, radius='md', padding='sm')


def quality_card() -> dmc.Card:
    """Check chips grouped by severity + Run button."""
    errors = [n for n, (sev, _) in quality_registry.CHECKS.items() if sev == 'error']
    warns = [n for n, (sev, _) in quality_registry.CHECKS.items() if sev == 'warn']
    return dmc.Card([
        dmc.Text('Quality checks', fw=700, size='sm', mb=8),
        dmc.Text('error', size='xs', c='red', mb=4),
        dmc.ChipGroup(
            dmc.Group([dmc.Chip(quality_registry.CHECK_DESCRIPTIONS[n], value=n, size='xs') for n in sorted(errors)]),
            id='cp-quality-errors', multiple=True, value=[],
        ),
        dmc.Text('warn', size='xs', c='yellow', mt=8, mb=4),
        dmc.ChipGroup(
            dmc.Group([dmc.Chip(quality_registry.CHECK_DESCRIPTIONS[n], value=n, size='xs') for n in sorted(warns)]),
            id='cp-quality-warns', multiple=True, value=[],
        ),
        dmc.Button('Run quality', id='cp-btn-quality', size='xs', mt=8, leftSection=DashIconify(icon='tabler:player-play')),
    ], withBorder=True, radius='md', padding='sm')


# ---------- job console ----------

def job_console() -> dmc.Card:
    """Live status + scrolling log of the current/last job, plus a run-history accordion."""
    return dmc.Card([
        dmc.Group([
            dmc.Text('Job', fw=700, size='sm'),
            html.Div(id='cp-job-badge'),
            dmc.Button('Cancel', id='cp-btn-cancel', size='xs', color='red', variant='subtle',
                       leftSection=DashIconify(icon='tabler:player-stop'), style={'marginLeft': 'auto'}),
        ], justify='space-between'),
        html.Pre(id='cp-job-log', style={
            'height': '360px', 'overflowY': 'auto', 'background': '#0b0b0f', 'color': '#ddd',
            'padding': '8px', 'borderRadius': '6px', 'fontSize': '12px', 'marginTop': '8px', 'marginBottom': '8px',
        }),
        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl('History'),
                dmc.AccordionPanel(html.Div(id='cp-job-history')),
            ], value='history'),
        ]),
        dcc.Interval(id='cp-poll', interval=1000),
        html.Div(id='cp-log-scroll-dummy', style={'display': 'none'}),
    ], withBorder=True, radius='md', padding='sm')


# ---------- layout ----------

def layout() -> html.Div:
    """Build the control panel page: status row, command cards, job console."""
    return html.Div([
        dmc.SimpleGrid([freshness_card(), row_counts_card(), anomalies_card()],
                       cols={'base': 1, 'md': 3}, mb=12),
        dmc.Grid([
            dmc.GridCol(dmc.Stack([ingest_card(), housekeeping_card(), quality_card()], gap=10), span={'base': 12, 'lg': 5}),
            dmc.GridCol(job_console(), span={'base': 12, 'lg': 7}),
        ], gutter=12),
        dmc.Modal(
            id='cp-confirm-modal', title='Confirm', centered=True,
            children=[
                dmc.Text(id='cp-confirm-text'),
                dmc.Group([
                    dmc.Button('Cancel', id='cp-confirm-no', variant='default', size='xs'),
                    dmc.Button('Confirm', id='cp-confirm-yes', color='red', size='xs'),
                ], justify='flex-end', mt=12),
            ],
        ),
        dcc.Store(id='cp-pending-action'),
        dcc.Store(id='cp-cancel-tick'),
    ])


dash.register_page(__name__, path='/', name='Control Panel', layout=layout)


# ---------- status callbacks ----------

def _read_status():
    """Open a short-lived read-only connection and run the monitor queries."""
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        return status.freshness(con), status.row_counts(con), status.anomaly_summary(con)
    finally:
        con.close()


@callback(
    Output('cp-freshness-body', 'children'), Output('cp-rowcounts-body', 'children'),
    Output('cp-anomalies-body', 'children'),
    Input('cp-poll', 'n_intervals'),
)
def refresh_status(_n: int):
    """Refresh the status cards; skip the query while the DB is locked by a writer.

    MANAGER only knows about jobs the panel itself launched, not a CLI run started directly in a
    terminal, so the write lock is detected directly (duckdb.Error) rather than trusting is_running().
    """
    if MANAGER.is_running():
        paused = dmc.Text('paused, job running', size='xs', c='dimmed')
        return paused, paused, paused

    try:
        fresh, counts, anomalies = _read_status()
    except duckdb.Error:
        paused = dmc.Text('paused, database busy', size='xs', c='dimmed')
        return paused, paused, paused

    fresh_rows = [dmc.Group([dmc.Text(r.table, size='xs'), days_ago_badge(r.days_ago)], justify='space-between')
                  for r in fresh.itertuples()]

    count_rows = [dmc.Group([dmc.Text(r.table, size='xs'), dmc.Text(f'{r.rows:,}', size='xs', c='dimmed')], justify='space-between')
                  for r in counts.itertuples()]

    anomaly_rows = [dmc.Group([dmc.Text(f'{r.CheckName}', size='xs'),
                               dmc.Badge(str(r.n), color=SEVERITY_COLOR.get(r.Severity, 'gray'), variant='light', size='sm')],
                              justify='space-between')
                    for r in anomalies.itertuples()]

    return html.Div(fresh_rows), html.Div(count_rows), html.Div(anomaly_rows)


# ---------- job callbacks ----------

@callback(
    Output('cp-pending-action', 'data'), Output('cp-confirm-modal', 'opened'), Output('cp-confirm-text', 'children'),
    Input('cp-btn-ingest', 'n_clicks'), Input('cp-btn-housekeeping', 'n_clicks'), Input('cp-btn-quality', 'n_clicks'),
    State('cp-ingest-providers', 'value'), State('cp-ingest-mode', 'value'),
    State('cp-housekeeping-tasks', 'value'), State('cp-quality-errors', 'value'), State('cp-quality-warns', 'value'),
    prevent_initial_call=True,
)
def request_run(ingest_n, housekeeping_n, quality_n,
                 providers, mode, tasks, quality_errors, quality_warns):
    """Build the argv for the clicked Run button. A full ingest load goes through a confirm modal."""
    trigger = ctx.triggered_id
    if trigger == 'cp-btn-ingest':
        title, argv = build_ingest_argv(providers, update=mode == 'update')
        if mode == 'full':
            return {'title': title, 'argv': argv}, True, f"Run a full ingest load? This re-downloads history for {', '.join(providers) if providers else 'all providers'}."
        MANAGER.start(title, argv)
        return dash.no_update, False, dash.no_update
    if trigger == 'cp-btn-housekeeping':
        title, argv = build_housekeeping_argv(tasks)
        MANAGER.start(title, argv)
        return dash.no_update, False, dash.no_update
    if trigger == 'cp-btn-quality':
        title, argv = build_quality_argv((quality_errors or []) + (quality_warns or []))
        MANAGER.start(title, argv)
        return dash.no_update, False, dash.no_update
    raise PreventUpdate


@callback(
    Output('cp-confirm-modal', 'opened', allow_duplicate=True),
    Input('cp-confirm-yes', 'n_clicks'), Input('cp-confirm-no', 'n_clicks'),
    State('cp-pending-action', 'data'),
    prevent_initial_call=True,
)
def resolve_confirm(yes_n, no_n, pending):
    """Start the pending job on Confirm; just close the modal on Cancel."""
    if ctx.triggered_id == 'cp-confirm-yes' and pending:
        MANAGER.start(pending['title'], pending['argv'])
    return False


@callback(
    Output('cp-job-badge', 'children'), Output('cp-job-log', 'children'), Output('cp-job-history', 'children'),
    Output('cp-btn-ingest', 'disabled'), Output('cp-btn-housekeeping', 'disabled'), Output('cp-btn-quality', 'disabled'),
    Input('cp-poll', 'n_intervals'),
)
def refresh_job(_n: int):
    """Update the job badge, live log, and history; disable Run buttons while a job is active."""
    running = MANAGER.is_running()
    job = MANAGER.current if running else (MANAGER.history[0] if MANAGER.history else None)

    if job is None:
        badge = dmc.Badge('idle', color='gray', variant='light')
        log = ''
    else:
        if running:
            badge = dmc.Group([dmc.Loader(size='xs'), dmc.Badge(job.title, color='blue', variant='light')], gap=6)
        else:
            color = 'green' if job.returncode == 0 else 'red'
            badge = dmc.Group([dmc.Badge(job.title, color=color, variant='light'),
                               dmc.Badge(f'exit {job.returncode}', color=color, variant='outline')], gap=6)
        log = '\n'.join(job.lines)

    history_rows = [
        dmc.Group([
            dmc.Text(h.title, size='xs'),
            dmc.Badge(f'exit {h.returncode}', color='green' if h.returncode == 0 else 'red', size='xs', variant='light'),
            dmc.Text(h.started.strftime('%H:%M:%S'), size='xs', c='dimmed'),
        ], justify='space-between')
        for h in MANAGER.history
    ]

    return (badge, log, html.Div(history_rows) if history_rows else dmc.Text('no runs yet', size='xs', c='dimmed'),
            running, running, running)


@callback(Output('cp-cancel-tick', 'data'), Input('cp-btn-cancel', 'n_clicks'), prevent_initial_call=True)
def cancel_job(n: int):
    """Cancel the currently running job, if any."""
    MANAGER.cancel()
    return n


dash.clientside_callback(
    """
    function(children) {
        const el = document.getElementById('cp-job-log');
        if (el) { el.scrollTop = el.scrollHeight; }
        return '';
    }
    """,
    Output('cp-log-scroll-dummy', 'children'),
    Input('cp-job-log', 'children'),
)
