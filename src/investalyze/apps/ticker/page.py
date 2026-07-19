"""Ticker analysis page: one stock's performance, factor profile, peer standing, risk and history.

The select's options and all snapshot numbers come from the screener's cached pool, so values
match the screener. Time series are queried per selection. `?symbol=X` deep-links a ticker.
A peer-basis select picks the comparison group: the ticker's whole industry (default), its
whole sector, or any saved universe; it persists across navigation. The active group's members
come first in the ticker select, and a stepper walks through universe members. Every section
degrades to a dimmed note when its inputs are missing; a locked DB shows the shared busy notice.
"""

import dash
import dash_mantine_components as dmc
import duckdb
import pandas as pd
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from investalyze.apps.screener.data import get_pool
from investalyze.apps.ticker import charts, data
from investalyze.apps.universes import list_universes, load_universe

BUSY = 'database busy, a job is currently running, try again once it finishes'
DEFAULT_RANGE = '5y'
INDUSTRY_BASIS = 'industry'
SECTOR_BASIS = 'sector'
UNIVERSE_PREFIX = 'u:'
ANCHOR_STYLE = {'scrollMarginTop': '70px'}
PROMPT = dmc.Stack([
    DashIconify(icon='tabler:chart-candle', width=44, color='var(--mantine-color-dimmed)'),
    dmc.Text('pick a ticker to analyse', size='sm', c='dimmed'),
    dmc.Text('or choose a peer universe and step through it with the arrows', size='xs', c='dimmed'),
], align='center', gap=6, mt=80)


def _money(value: object) -> str:
    """Compact dollar amount: $x.xbn above a billion, $xmn above a million, else plain."""
    if value is None or pd.isna(value):
        return '-'
    number = float(value)
    if abs(number) >= 1e12:
        return f'${number / 1e12:,.2f}tn'
    if abs(number) >= 1e9:
        return f'${number / 1e9:,.1f}bn'
    if abs(number) >= 1e6:
        return f'${number / 1e6:,.0f}mn'
    return f'${number:,.2f}'


def _pct(value: object) -> str:
    """Signed percent with one decimal, or '-' when missing."""
    if value is None or pd.isna(value):
        return '-'
    return f'{value:+.1%}'


def _tile(label: str, value: str, sub: str | None = None, color: str | None = None) -> dmc.Paper:
    """One stat tile: small label, bold value (optionally colored as a verdict), optional context line.

    An explicit c=None serializes to null, which Mantine's color parser rejects; omit the prop instead.
    """
    value_kwargs = {'c': color} if color is not None else {}
    children = [dmc.Text(label, size='xs', c='dimmed'), dmc.Text(value, fw=700, size='lg', **value_kwargs)]
    if sub is not None:
        children.append(dmc.Text(sub, size='xs', c='dimmed'))
    return dmc.Paper(children, withBorder=True, radius='md', p='xs', style={'minWidth': '128px'})


def _section(title: str, children: list, caption: str | None = None, anchor: str | None = None) -> dmc.Card:
    """One page section: bold title, optional dimmed caption, then its content; `anchor` makes it a jump target."""
    head: list = [dmc.Text(title, fw=700, size='sm')]
    if caption is not None:
        head.append(dmc.Text(caption, size='xs', c='dimmed'))
    extra = {'id': anchor, 'style': ANCHOR_STYLE} if anchor else {}
    return dmc.Card(head + children, withBorder=True, radius='md', padding='sm', mb=12, **extra)


def _no_data(message: str) -> dmc.Text:
    """Dimmed in-section note for missing inputs."""
    return dmc.Text(message, size='sm', c='dimmed', my=8)


def _hero(ticker: str, row: pd.Series, ttm: pd.DataFrame, returns: pd.DataFrame | None,
          history: pd.DataFrame) -> html.Div:
    """Identity line plus the KPI tile row."""
    badges = [dmc.Badge(row['mcap_bucket'] or 'no mcap', variant='light', color='blue'),
              dmc.Badge('active' if row['active'] else 'delisted', variant='light',
                        color='green' if row['active'] else 'gray')]
    identity = dmc.Group([
        dmc.Text(ticker, fw=700, size='xl'),
        dmc.Text(row['name'], size='lg'),
        dmc.Text(f"{row['sector']} / {row['industry']}", size='sm', c='dimmed'),
        *badges,
    ], gap=10)

    latest = ttm.iloc[-1] if len(ttm) else None
    ret_1y = returns[returns['window'] == '1y'].iloc[0] if returns is not None else None
    margin = latest['net_margin'] if latest is not None else None
    margin_color = 'red' if margin is not None and pd.notna(margin) and margin < 0 else None
    ret_color = None
    if ret_1y is not None and pd.notna(ret_1y['ticker']):
        ret_color = 'teal' if ret_1y['ticker'] >= 0 else 'red'
    tiles = [
        _tile('last close', f"${row['last_close']:,.2f}"),
        _tile('market cap', _money(row['mcap'])),
        _tile('revenue TTM', _money(latest['revenue']) if latest is not None else '-'),
        _tile('net margin TTM', f'{margin:.1%}' if margin is not None and pd.notna(margin) else '-', color=margin_color),
        _tile('1y return', _pct(ret_1y['ticker']) if ret_1y is not None else '-',
              f"S&P 500 {_pct(ret_1y['market'])}" if ret_1y is not None else None, color=ret_color),
        _tile('volatility 1y', f"{row['vol_252']:.0%}" if pd.notna(row['vol_252']) else '-'),
    ]
    if not history.empty:
        stats = data.drawdown(history['AC'])
        max_dd, current_dd = float(stats.min()), float(stats.iloc[-1])
        tiles.append(_tile('max drawdown', f'{max_dd:.0%}', 'full history', color='red' if max_dd <= -0.2 else None))
        tiles.append(_tile('current drawdown', f'{current_dd:.0%}', 'from all-time high',
                           color='red' if current_dd <= -0.2 else None))
    return html.Div([identity, dmc.Group(tiles, gap=8, mt=10, align='stretch')], style={'marginBottom': '12px'})


def _performance_section(history: pd.DataFrame, returns: pd.DataFrame | None, ticker: str, dark: bool) -> dmc.Card:
    """Indexed line vs S&P 500 with range control, plus the trailing-returns bars."""
    if history.empty:
        return _section('Performance', [_no_data(f'no price history for {ticker}')], anchor='sec-performance')
    window = data.rebased(history, data.RANGE_SESSIONS[DEFAULT_RANGE])
    controls = dmc.SegmentedControl(id='tk-range', value=DEFAULT_RANGE, size='xs',
                                    data=list(data.RANGE_SESSIONS))
    charts_row = dmc.Grid([
        dmc.GridCol(dcc.Graph(id='tk-perf', figure=charts.performance_figure(window, ticker, dark)),
                    span={'base': 12, 'lg': 8}),
        dmc.GridCol(dcc.Graph(figure=charts.trailing_returns_figure(returns, ticker, dark)),
                    span={'base': 12, 'lg': 4}),
    ], gutter=8)
    return _section('Performance', [controls, charts_row],
                    caption='total return, both series rebased to 100 at the window start; drawdown beneath',
                    anchor='sec-performance')


def _universe_members(basis: str | None) -> tuple[str, list[str]] | None:
    """(name, tickers) when the basis points at a loadable saved universe."""
    if not basis or not basis.startswith(UNIVERSE_PREFIX):
        return None
    name = basis[len(UNIVERSE_PREFIX):]
    try:
        return name, load_universe(name)
    except FileNotFoundError:
        return None


def _resolve_peers(pool: pd.DataFrame, row: pd.Series, basis: str | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(capped peer frame, full frame for the violins, caption phrase) for the chosen peer basis.

    A missing universe file falls back to the industry scope. The capped frame drives the factor
    percentiles; the violins get every member so distributions stay complete.
    """
    loaded = _universe_members(basis)
    if loaded is not None:
        name, members = loaded
        peers = data.universe_peer_group(pool, row['Ticker'], members)
        full = data.universe_peer_group(pool, row['Ticker'], members, cap=None)
        return peers, full, f"peers from universe '{name}'"
    scope = 'sector' if basis == SECTOR_BASIS else 'industry'
    peers = data.scope_peer_group(pool, row['Ticker'], scope)
    full = data.scope_peer_group(pool, row['Ticker'], scope, cap=None)
    return peers, full, f"peers from {scope} '{row[scope]}'"


def _risk_section(peers: pd.DataFrame, violin_peers: pd.DataFrame, ticker: str, dark: bool) -> dmc.Card:
    """Market and balance-sheet risk against the peer distribution."""
    if len(peers) >= 3:
        children = [dcc.Graph(figure=charts.peer_violin_figure(violin_peers, ticker, charts.RISK_SPEC, dark))]
    else:
        children = [_no_data('not enough peers for a risk comparison')]
    return _section('Risk', children,
                    caption='the needle marks this ticker, P its peer percentile (100 = best), green strong / red weak',
                    anchor='sec-risk')


def build_sections(ticker: str, dark: bool, basis: str | None) -> list:
    """Assemble every section for one ticker; pool row missing yields a single note."""
    pool = get_pool()
    match = pool[pool['Ticker'] == ticker]
    if match.empty:
        return [_no_data(f'{ticker} is not in the pool')]
    row = match.iloc[0]

    peers, violin_peers, peer_source = _resolve_peers(pool, row, basis)
    history = data.price_history(ticker)
    quarters = data.fundamentals_history(ticker)
    ttm = data.ttm_history(quarters) if len(quarters) else quarters
    returns = data.trailing_returns(history) if len(history) else None

    peer_caption = (f'{len(violin_peers) - 1} {peer_source}; the needle marks this ticker, '
                    'P its peer percentile (100 = best), green strong / red weak')
    profile_caption = 'percentile within the peer group, 100 = best'
    if len(peers) < len(violin_peers):
        profile_caption = f'percentile within the {len(peers) - 1} closest-by-size peers, 100 = best'

    sections = [_hero(ticker, row, ttm, returns, history),
                _performance_section(history, returns, ticker, dark)]
    if len(ttm):
        sections.append(_section('Fundamentals history', [dcc.Graph(figure=charts.fundamentals_figure(ttm, dark))],
                                 caption='trailing-twelve-month flows from restated quarters; balance items point-in-time',
                                 anchor='sec-fundamentals'))
    elif len(quarters):
        sections.append(_section('Fundamentals history',
                                 [_no_data(f'only {len(quarters)} of the 4 quarterly reports a TTM view needs')],
                                 anchor='sec-fundamentals'))
    else:
        sections.append(_section('Fundamentals history', [_no_data(f'no quarterly fundamentals for {ticker}')],
                                 anchor='sec-fundamentals'))
    if len(peers) >= 3:
        families, ranks = data.peer_percentiles(peers, ticker)
        profile = dmc.Grid([
            dmc.GridCol(_section('Factor profile', [dcc.Graph(figure=charts.factor_profile_figure(families, ranks, dark))],
                                 caption=profile_caption),
                        span={'base': 12, 'lg': 6}),
            dmc.GridCol(_section('Peer comparison',
                                 [dcc.Graph(figure=charts.peer_violin_figure(violin_peers, ticker, charts.PEER_SPEC, dark))],
                                 caption=peer_caption),
                        span={'base': 12, 'lg': 6}),
        ], gutter=12)
        sections.append(html.Div(profile, id='sec-peers', style=ANCHOR_STYLE))
    else:
        sections.append(_section('Peer comparison', [_no_data('not enough peers for a comparison')], anchor='sec-peers'))
    sections.append(_risk_section(peers, violin_peers, ticker, dark))
    return sections


def _select_options(pool: pd.DataFrame, universe: tuple[str, list[str]] | None = None) -> list[dict]:
    """'TICKER - Company' select options; with a universe, its members form the first option group."""
    labels: dict[str, str] = {}
    for ticker, name in zip(pool['Ticker'], pool['name']):
        labels[ticker] = f'{ticker} - {name}' if name else ticker
    if universe is None:
        return [{'value': ticker, 'label': label} for ticker, label in labels.items()]
    universe_name, members = universe
    member_set = set(members)
    first = [{'value': ticker, 'label': labels[ticker]} for ticker in members if ticker in labels]
    rest = [{'value': ticker, 'label': label} for ticker, label in labels.items() if ticker not in member_set]
    return [{'group': universe_name, 'items': first}, {'group': 'all tickers', 'items': rest}]


def _basis_options() -> list[dict]:
    """Peer-basis select options: the ticker's whole industry or sector, plus every saved universe."""
    options = [{'value': INDUSTRY_BASIS, 'label': 'industry peers'},
               {'value': SECTOR_BASIS, 'label': 'sector peers'}]
    for name in list_universes():
        options.append({'value': UNIVERSE_PREFIX + name, 'label': name})
    return options


def layout(symbol: str | None = None, **_query) -> html.Div:
    """Page shell: searchable ticker select (options built here so ?symbol= preselects), peer basis and content."""
    try:
        options = _select_options(get_pool())
    except duckdb.Error:
        options = []
    stepper = dmc.Group([
        dmc.ActionIcon(DashIconify(icon='tabler:chevron-left', width=16), id='tk-prev', variant='default', size='md'),
        dmc.Text(id='tk-pos', size='xs', c='dimmed'),
        dmc.ActionIcon(DashIconify(icon='tabler:chevron-right', width=16), id='tk-next', variant='default', size='md'),
    ], id='tk-stepper', gap=6, style={'display': 'none'})

    nav_link_style = {'fontSize': '12px', 'color': 'var(--mantine-color-dimmed)', 'cursor': 'pointer'}
    section_nav = dmc.Group([
        html.Span(label, id=f'tk-nav-{anchor}', style=nav_link_style)
        for label, anchor in (('performance', 'sec-performance'), ('fundamentals', 'sec-fundamentals'),
                              ('peers', 'sec-peers'), ('risk', 'sec-risk'))
    ], gap=14, style={'marginLeft': 'auto'})

    control_bar = html.Div(
        dmc.Group([
            dmc.Select(id='tk-select', data=options, value=symbol, searchable=True, clearable=True,
                       placeholder='search ticker or company', limit=100, size='sm',
                       nothingFoundMessage='no match', style={'width': '360px'}),
            dmc.Select(id='tk-peer-basis', data=_basis_options(), value=INDUSTRY_BASIS, size='sm',
                       allowDeselect=False, persistence=True, leftSection=DashIconify(icon='tabler:users', width=14),
                       style={'width': '280px'}),
            stepper,
            section_nav,
        ], gap=10),
        style={'position': 'sticky', 'top': 0, 'zIndex': 1001, 'background': 'var(--mantine-color-body)',
               'margin': '-4px -20px 12px', 'padding': '10px 20px',
               'borderBottom': '1px solid var(--mantine-color-default-border)'},
    )

    return html.Div([
        control_bar,
        dcc.Loading(html.Div(id='tk-content', children=PROMPT), type='circle', color='#3987e5',
                    delay_show=250, overlay_style={'visibility': 'visible', 'opacity': 0.4}),
        dcc.Store(id='tk-nav-scroll'),
    ], style={'padding': '4px'})


dash.register_page(__name__, path='/ticker', name='Ticker', layout=layout)

# Scrolls to the clicked section clientside; href="#..." anchors would make the Dash router
# rebuild the whole page on the hash change.
clientside_callback(
    """
    (p, f, e, r) => {
        const trigger = window.dash_clientside.callback_context.triggered[0];
        if (!trigger) return window.dash_clientside.no_update;
        const el = document.getElementById(trigger.prop_id.split('.')[0].replace('tk-nav-', ''));
        if (el) el.scrollIntoView({behavior: 'smooth'});
        return window.dash_clientside.no_update;
    }
    """,
    Output('tk-nav-scroll', 'data'),
    Input('tk-nav-sec-performance', 'n_clicks'), Input('tk-nav-sec-fundamentals', 'n_clicks'),
    Input('tk-nav-sec-peers', 'n_clicks'), Input('tk-nav-sec-risk', 'n_clicks'),
    prevent_initial_call=True,
)


@callback(Output('tk-select', 'data'), Input('tk-peer-basis', 'value'), Input('tk-select', 'value'))
def update_ticker_options(basis, ticker):
    """Regroup the ticker select so the active peer group's members come first, largest first."""
    try:
        pool = get_pool()
    except duckdb.Error:
        raise PreventUpdate
    grouped = _universe_members(basis)
    if grouped is None and ticker:
        scope = 'sector' if basis == SECTOR_BASIS else 'industry'
        me = pool[pool['Ticker'] == ticker]
        if len(me) and me.iloc[0][scope] != 'unknown':
            value = me.iloc[0][scope]
            members = pool[pool[scope] == value].sort_values('mcap', ascending=False, na_position='last')
            grouped = (value, members['Ticker'].tolist())
    return _select_options(pool, grouped)


@callback(Output('tk-stepper', 'style'), Output('tk-pos', 'children'),
          Input('tk-peer-basis', 'value'), Input('tk-select', 'value'))
def update_stepper(basis, ticker):
    """Show the universe stepper with the current position; auto basis hides it."""
    loaded = _universe_members(basis)
    if loaded is None:
        return {'display': 'none'}, ''
    _name, members = loaded
    position = str(members.index(ticker) + 1) if ticker in members else '-'
    return {'display': 'flex'}, f'{position} / {len(members)}'


@callback(Output('tk-select', 'value'), Input('tk-prev', 'n_clicks'), Input('tk-next', 'n_clicks'),
          State('tk-select', 'value'), State('tk-peer-basis', 'value'), prevent_initial_call=True)
def step_ticker(_prev, _next, ticker, basis):
    """Step to the previous or next universe member, wrapping around at the ends."""
    loaded = _universe_members(basis)
    if loaded is None:
        raise PreventUpdate
    _name, members = loaded
    step = -1 if ctx.triggered_id == 'tk-prev' else 1
    if ticker in members:
        index = (members.index(ticker) + step) % len(members)
    else:
        index = 0 if step == 1 else len(members) - 1
    return members[index]


@callback(Output('tk-content', 'children'), Input('tk-select', 'value'), Input('tk-peer-basis', 'value'),
          Input('theme-switch', 'checked'))
def render(ticker, basis, dark):
    """Rebuild the whole page for the selected ticker, peer basis and theme."""
    if not ticker:
        return PROMPT
    try:
        return build_sections(ticker, bool(dark), basis)
    except duckdb.Error:
        return dmc.Text(BUSY, size='sm', c='yellow')


@callback(Output('tk-perf', 'figure'), Input('tk-range', 'value'),
          State('tk-select', 'value'), State('theme-switch', 'checked'), prevent_initial_call=True)
def update_range(range_key, ticker, dark):
    """Re-render only the performance chart for a new time range."""
    if not ticker:
        raise PreventUpdate
    try:
        history = data.price_history(ticker)
    except duckdb.Error:
        raise PreventUpdate
    window = data.rebased(history, data.RANGE_SESSIONS.get(range_key))
    return charts.performance_figure(window, ticker, bool(dark))
