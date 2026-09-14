"""Control panel page: run the investalyze CLIs and monitor the database from the browser.

Every 'Run' button builds an argv for `python -m investalyze.<module>` and hands it to
`jobs.MANAGER`, which runs it as a subprocess (see jobs.py for why: one DB writer at a time).
The status cards at the top query the DB directly through short-lived read-only connections
and are paused while a job is running, since the running job may hold the write lock.
"""

import json

import dash
import dash_mantine_components as dmc
import duckdb
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from investalyze import paths
from investalyze.apps import components, universes
from investalyze.apps.control_panel import blacklist_review, listings_review, status
from investalyze.apps.control_panel.jobs import MANAGER
from investalyze.ingest import housekeeping, listings, orchestrator, storage
from investalyze.quality import registry as quality_registry


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


def build_cleaning_argv(command: str) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.cleaning check|apply`."""
    return f'cleaning {command}', ['-m', 'investalyze.cleaning', command]


def build_listings_scan_argv() -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.ingest scan-listings`."""
    return 'scan listings', ['-m', 'investalyze.ingest', 'scan-listings']


def build_blacklist_argv(tickers: list[str] | None, universe: str | None, reason: str) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.ingest blacklist-ticker`, targeting either `tickers` or
    every ticker in a saved universe (exactly one of `tickers`/`universe` must be set)."""
    if universe:
        return f'blacklist universe {universe}', ['-m', 'investalyze.ingest', 'blacklist-ticker',
                                                   '--universe', universe, '--reason', reason]
    argv = ['-m', 'investalyze.ingest', 'blacklist-ticker']
    for ticker in tickers:
        argv += ['--ticker', ticker]
    argv += ['--reason', reason]
    title = f'blacklist {tickers[0]}' if len(tickers) == 1 else f'blacklist {len(tickers)} tickers'
    return title, argv


def build_add_listings_argv(rows: list[dict]) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.ingest add-listings --rows <json>`.

    Each row's current grid state (ticker, market, classification, and its possibly user-edited
    suggested_new_ticker) is passed straight through as the collision-fix target.
    """
    payload = json.dumps([{'ticker': r['ticker'], 'market': r['market'], 'classification': r['classification'],
                           'new_ticker': r.get('suggested_new_ticker')} for r in rows])
    return f'add {len(rows)} listing(s)', ['-m', 'investalyze.ingest', 'add-listings', '--rows', payload]


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
#
# Visual rule: every button that modifies the database is orange and sits next to an orange
# 'writes ...' badge naming what it writes; read-only buttons are plain with a dimmed 'read-only' label.

def writes_badge(what: str) -> dmc.Badge:
    """Orange badge marking a data-modifying action, naming what it writes."""
    return dmc.Badge(f'writes {what}', color='orange', variant='light', size='xs')


def ingest_card() -> dmc.Card:
    """Provider chips + full/update toggle + Run button."""
    return dmc.Card([
        dmc.Group([dmc.Text('Ingest', fw=700, size='sm'), writes_badge('price + fundamentals tables')], gap=8, mb=8),
        dmc.ChipGroup(
            dmc.Group([dmc.Chip(desc, value=name, size='xs') for name, desc in orchestrator.PROVIDER_DESCRIPTIONS.items()]),
            id='cp-ingest-providers', multiple=True, value=[],
        ),
        dmc.SegmentedControl(
            id='cp-ingest-mode', value='update',
            data=[{'label': 'Update', 'value': 'update'}, {'label': 'Full load', 'value': 'full'}],
            mt=8, size='xs',
        ),
        dmc.Button('Run ingest', id='cp-btn-ingest', size='xs', mt=8, color='orange',
                   leftSection=DashIconify(icon='tabler:database-import')),
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
        dmc.Group([dmc.Text('Housekeeping', fw=700, size='sm'), writes_badge('derived tables')], gap=8, mb=8),
        dmc.ChipGroup(
            dmc.Group([dmc.Chip(desc, value=name, size='xs') for name, desc in housekeeping.TASK_DESCRIPTIONS.items()]),
            id='cp-housekeeping-tasks', multiple=True, value=[],
        ),
        dmc.Button('Run housekeeping', id='cp-btn-housekeeping', size='xs', mt=8, color='orange',
                   leftSection=DashIconify(icon='tabler:database-import')),
    ], withBorder=True, radius='md', padding='sm')


def quality_card() -> dmc.Card:
    """Check chips grouped by severity + Run button."""
    errors = [n for n, (sev, _) in quality_registry.CHECKS.items() if sev == 'error']
    warns = [n for n, (sev, _) in quality_registry.CHECKS.items() if sev == 'warn']
    return dmc.Card([
        dmc.Group([dmc.Text('Quality checks', fw=700, size='sm'), writes_badge('anomalies table only')], gap=8, mb=4),
        dmc.Text('scans the data and records findings in the anomalies table; correcting data happens in Cleaning',
                 size='xs', c='dimmed', mb=8),
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
        dmc.Button('Run quality scan', id='cp-btn-quality', size='xs', mt=8, color='orange',
                   leftSection=DashIconify(icon='tabler:zoom-check')),
    ], withBorder=True, radius='md', padding='sm')


def cleaning_card() -> dmc.Card:
    """Preview/apply buttons for the persistent manual corrections in cleaning.toml."""
    return dmc.Card([
        dmc.Text('Cleaning', fw=700, size='sm', mb=4),
        dmc.Text('corrects known-bad rows using the fix list in cleaning.toml', size='xs', c='dimmed', mb=8),
        dmc.Group([
            dmc.Button('Preview fixes', id='cp-btn-cleaning-check', size='xs', variant='default',
                       leftSection=DashIconify(icon='tabler:eye')),
            dmc.Text('read-only: shows what apply would change', size='xs', c='dimmed'),
        ], gap=8),
        dmc.Group([
            dmc.Button('Apply fixes', id='cp-btn-cleaning-apply', size='xs', color='orange',
                       leftSection=DashIconify(icon='tabler:database-edit')),
            writes_badge('price rows'),
        ], gap=8, mt=6),
    ], withBorder=True, radius='md', padding='sm')


def listings_scan_card() -> dmc.Card:
    """Scan button for new/reactivated tickers; auto-adds new ones, reactivated ones need review below."""
    return dmc.Card([
        dmc.Group([dmc.Text('New listings', fw=700, size='sm'), writes_badge('ticker.csv (new tickers)')],
                  gap=8, mb=4),
        dmc.Text('Checks NASDAQ/NYSE listings against ticker.csv; new tickers are added automatically, '
                 'reactivated ones need review below.', size='xs', c='dimmed', mb=8),
        dmc.Button('Scan for new listings', id='cp-btn-listings-scan', size='xs', color='orange',
                   leftSection=DashIconify(icon='tabler:zoom-check')),
    ], withBorder=True, radius='md', padding='sm')


def blacklist_card() -> dmc.Card:
    """Delete a ticker's (or a whole universe's) data everywhere and block it from ever being
    re-added by a scan. Fill in either the ticker field or the universe field, not both; one
    reason is stored against every ticker blacklisted, so it stays clear later why."""
    return dmc.Card([
        dmc.Group([dmc.Text('Blacklist ticker', fw=700, size='sm'),
                  dmc.Badge('deletes ticker data', color='red', variant='light', size='xs')], gap=8, mb=4),
        dmc.Text("Permanently deletes the data and blocks future scans from re-adding it. Fill in "
                 'either a ticker or a universe, not both.', size='xs', c='dimmed', mb=8),
        dmc.Group([
            dmc.TextInput(id='cp-blacklist-ticker', placeholder='TICKER', size='xs', style={'width': '140px'}),
            dmc.Select(id='cp-blacklist-universe', placeholder='Or a whole universe', data=[], size='xs',
                      clearable=True, style={'flex': 1}),
        ], gap=8),
        dmc.Textarea(id='cp-blacklist-reason', autosize=True, minRows=2, size='xs', mt=6,
                    placeholder='Reason (stored against every ticker blacklisted)'),
        dmc.Button('Blacklist', id='cp-btn-blacklist', size='xs', mt=8, color='red',
                   leftSection=DashIconify(icon='tabler:trash')),
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
            'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word',
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
        # type='container': breakpoints measure this grid's own width, so the cards stop
        # reflowing at the width they actually need (~220px each) instead of at a viewport
        # size that ignores the navbar and page padding.
        dmc.SimpleGrid([freshness_card(), row_counts_card(), anomalies_card()],
                       type='container', cols={'base': 1, '460px': 2, '700px': 3}, mb=12),
        dmc.Grid([
            dmc.GridCol(dmc.Stack([ingest_card(), housekeeping_card(), blacklist_card(), quality_card(),
                                   cleaning_card(), listings_scan_card()], gap=10),
                        span={'base': 12, 'lg': 5}),
            dmc.GridCol(job_console(), span={'base': 12, 'lg': 7}),
        ], gutter=12),
        dmc.Grid([dmc.GridCol(listings_review.listings_card(), span=12)], gutter=12, mt=12),
        dmc.Grid([dmc.GridCol(blacklist_review.blacklist_review_card(), span=12)], gutter=12, mt=12),
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
        dcc.Store(id='cp-listings-signature'),
        dcc.Store(id='cp-blacklist-signature'),
    ])


dash.register_page(__name__, path='/', name='Control Panel', layout=layout)


# ---------- status callbacks ----------

def _read_status() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Open a short-lived read-only connection and run the monitor queries."""
    with storage.connect(read_only=True) as con:
        return status.freshness(con), status.row_counts(con), status.anomaly_summary(con)


@callback(
    Output('cp-freshness-body', 'children'), Output('cp-rowcounts-body', 'children'),
    Output('cp-anomalies-body', 'children'),
    Input('cp-poll', 'n_intervals'),
)
def refresh_status(_n: int | None) -> tuple:
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
    Input('cp-btn-cleaning-check', 'n_clicks'), Input('cp-btn-cleaning-apply', 'n_clicks'),
    Input('cp-btn-listings-scan', 'n_clicks'), Input('cp-btn-blacklist', 'n_clicks'), Input('cp-btn-listings-add', 'n_clicks'),
    State('cp-ingest-providers', 'value'), State('cp-ingest-mode', 'value'),
    State('cp-housekeeping-tasks', 'value'), State('cp-quality-errors', 'value'), State('cp-quality-warns', 'value'),
    State('cp-blacklist-ticker', 'value'), State('cp-blacklist-universe', 'value'), State('cp-blacklist-reason', 'value'),
    State('cp-listings-grid', 'selectedRows'),
    prevent_initial_call=True,
)
def request_run(ingest_n: int | None, housekeeping_n: int | None, quality_n: int | None, cleaning_check_n: int | None,
                cleaning_apply_n: int | None, listings_scan_n: int | None, blacklist_n: int | None, listings_add_n: int | None,
                providers: list[str] | None, mode: str | None, tasks: list[str] | None, quality_errors: list[str] | None,
                quality_warns: list[str] | None, blacklist_ticker: str | None, blacklist_universe: str | None,
                blacklist_reason: str | None, listings_selected: list[dict] | dict | None) -> tuple:
    """Build the argv for the clicked Run button. Full ingest loads and cleaning applies go through a confirm modal."""
    trigger = ctx.triggered_id
    if trigger == 'cp-btn-cleaning-check':
        title, argv = build_cleaning_argv('check')
        MANAGER.start(title, argv)
        return dash.no_update, False, dash.no_update
    if trigger == 'cp-btn-cleaning-apply':
        title, argv = build_cleaning_argv('apply')
        return {'title': title, 'argv': argv}, True, 'Apply the fixes in cleaning.toml? This writes to the database.'
    if trigger == 'cp-btn-ingest':
        title, argv = build_ingest_argv(providers, update=mode == 'update')
        if mode == 'full':
            scope = ', '.join(providers) if providers else 'all providers'
            return {'title': title, 'argv': argv}, True, f'Run a full ingest load? This re-downloads history for {scope}.'
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
    if trigger == 'cp-btn-listings-scan':
        title, argv = build_listings_scan_argv()
        MANAGER.start(title, argv)
        return dash.no_update, False, dash.no_update
    if trigger == 'cp-btn-blacklist':
        ticker = (blacklist_ticker or '').strip().upper()
        universe = (blacklist_universe or '').strip()
        reason = (blacklist_reason or '').strip()
        if not reason or bool(ticker) == bool(universe):
            raise PreventUpdate   # need a reason, and exactly one of ticker/universe
        title, argv = build_blacklist_argv([ticker] if ticker else None, universe or None, reason)
        target = f'universe "{universe}" ({len(universes.load_universe(universe))} ticker(s))' if universe else ticker
        prompt = f'Permanently delete all stored data for {target} and block them from future scans?'
        return {'title': title, 'argv': argv}, True, prompt
    if trigger == 'cp-btn-listings-add':
        rows = listings_selected or []
        if not rows:
            raise PreventUpdate
        title, argv = build_add_listings_argv(rows)
        MANAGER.start(title, argv)
        return dash.no_update, False, dash.no_update
    raise PreventUpdate


@callback(
    Output('cp-confirm-modal', 'opened', allow_duplicate=True),
    Input('cp-confirm-yes', 'n_clicks'), Input('cp-confirm-no', 'n_clicks'),
    State('cp-pending-action', 'data'),
    prevent_initial_call=True,
)
def resolve_confirm(yes_n: int | None, no_n: int | None, pending: dict | None) -> bool:
    """Start the pending job on Confirm; just close the modal on Cancel."""
    if ctx.triggered_id == 'cp-confirm-yes' and pending:
        MANAGER.start(pending['title'], pending['argv'])
    return False


@callback(
    Output('cp-job-badge', 'children'), Output('cp-job-log', 'children'), Output('cp-job-history', 'children'),
    Output('cp-btn-ingest', 'disabled'), Output('cp-btn-housekeeping', 'disabled'), Output('cp-btn-quality', 'disabled'),
    Output('cp-btn-cleaning-check', 'disabled'), Output('cp-btn-cleaning-apply', 'disabled'),
    Output('cp-btn-listings-scan', 'disabled'), Output('cp-btn-blacklist', 'disabled'), Output('cp-btn-listings-add', 'disabled'),
    Input('cp-poll', 'n_intervals'),
)
def refresh_job(_n: int | None) -> tuple:
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
            running, running, running, running, running, running, running, running)


@callback(Output('cp-cancel-tick', 'data'), Input('cp-btn-cancel', 'n_clicks'), prevent_initial_call=True)
def cancel_job(n: int | None) -> int | None:
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


# ---------- listings-review callbacks ----------

@callback(
    Output('cp-listings-grid', 'rowData'), Output('cp-listings-signature', 'data'),
    Input('cp-poll', 'n_intervals'),
    State('cp-listings-signature', 'data'),
)
def refresh_listings_candidates(_n: int | None, last_signature: list | None) -> tuple:
    """Reload the candidates grid from the last scan's saved CSV, skipping the update when the
    content hasn't changed since the last poll.

    The 1-second poll would otherwise reset rowData every tick even when nothing changed, which
    resets the grid's scroll position and makes it impossible to scroll through the list.
    """
    rows = listings_review.read_candidates()
    # plain lists, not tuples: dcc.Store round-trips data through JSON, which has no tuple type,
    # so a tuple signature would never compare equal to the list it deserializes back into
    signature = sorted([r['ticker'], r['classification'], r.get('suggested_new_ticker', '')] for r in rows)
    if signature == last_signature:
        return dash.no_update, dash.no_update
    return rows, signature


@callback(Output('cp-blacklist-universe', 'data'), Input('cp-poll', 'n_intervals'))
def refresh_blacklist_universe_options(_n: int | None) -> list[dict]:
    """List saved universes as options for the blacklist card's universe picker."""
    return universes.list_universes()


# ---------- blacklist-review callbacks ----------

@callback(
    Output('cp-blacklist-grid', 'rowData'), Output('cp-blacklist-signature', 'data'),
    Input('cp-poll', 'n_intervals'), Input('cp-blacklist-search', 'value'),
    State('cp-blacklist-signature', 'data'),
)
def refresh_blacklist_grid(_n: int | None, search: str | None, last_signature: list | None) -> tuple:
    """Reload the blacklisted-tickers grid from manual_blacklist.csv, filtered by the ticker search
    box, skipping the update on an unchanged poll tick (same scroll-preserving reasoning as
    refresh_listings_candidates); a search-box edit always re-renders, since the visible rows change
    even though the underlying file didn't."""
    rows = blacklist_review.read_blacklist()
    signature = sorted([r['ticker'], r['market'], r['company_name'], r['reason'], r['blacklisted_on']] for r in rows)
    if signature == last_signature and ctx.triggered_id == 'cp-poll':
        return dash.no_update, dash.no_update
    return blacklist_review.filter_blacklist(rows, search), signature


@callback(
    Output('cp-unblacklist-status', 'children'),
    Input('cp-btn-unblacklist', 'n_clicks'),
    State('cp-blacklist-grid', 'selectedRows'),
    prevent_initial_call=True,
)
def unblacklist_selected(_n: int | None, selected: list[dict] | dict | None) -> components.Children:
    """Remove the selected tickers from manual_blacklist.csv so a future scan can offer them again."""
    rows = selected or []
    if not rows:
        raise PreventUpdate
    removed = listings.unblacklist_tickers(paths.DATA_ROOT, [r['ticker'] for r in rows])
    return dmc.Text(f'{removed} ticker(s) unblacklisted', size='xs', c='teal')
