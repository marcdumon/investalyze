"""Ticker analysis page: one stock's performance, factor profile, peer standing, risk and history.

The select's options and all snapshot numbers come from the screener's cached pool, so values
match the screener. Time series are queried per selection. `?symbol=X` deep-links a ticker.
Every section degrades to a dimmed note when its inputs are missing; a locked DB shows the
shared busy notice.
"""

import dash
import dash_mantine_components as dmc
import duckdb
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from investalyze.apps.screener.data import get_pool
from investalyze.apps.ticker import charts, data

BUSY = 'database busy, a job is currently running, try again once it finishes'
DEFAULT_RANGE = '5y'
PROMPT = dmc.Text('select a ticker to analyse', size='sm', c='dimmed')


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


def _tile(label: str, value: str, sub: str | None = None) -> dmc.Paper:
    """One stat tile: small label, bold value, optional context line."""
    children = [dmc.Text(label, size='xs', c='dimmed'), dmc.Text(value, fw=700, size='lg')]
    if sub is not None:
        children.append(dmc.Text(sub, size='xs', c='dimmed'))
    return dmc.Paper(children, withBorder=True, radius='md', p='xs', style={'minWidth': '128px'})


def _section(title: str, children: list, caption: str | None = None) -> dmc.Card:
    """One page section: bold title, optional dimmed caption, then its content."""
    head: list = [dmc.Text(title, fw=700, size='sm')]
    if caption is not None:
        head.append(dmc.Text(caption, size='xs', c='dimmed'))
    return dmc.Card(head + children, withBorder=True, radius='md', padding='sm', mb=12)


def _no_data(message: str) -> dmc.Text:
    """Dimmed in-section note for missing inputs."""
    return dmc.Text(message, size='sm', c='dimmed', my=8)


def _hero(ticker: str, row: pd.Series, ttm: pd.DataFrame, returns: pd.DataFrame | None) -> html.Div:
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
    tiles = dmc.Group([
        _tile('last close', f"${row['last_close']:,.2f}"),
        _tile('market cap', _money(row['mcap'])),
        _tile('revenue TTM', _money(latest['revenue']) if latest is not None else '-'),
        _tile('net margin TTM', f"{latest['net_margin']:.1%}" if latest is not None and pd.notna(latest['net_margin']) else '-'),
        _tile('1y return', _pct(ret_1y['ticker']) if ret_1y is not None else '-',
              f"S&P 500 {_pct(ret_1y['market'])}" if ret_1y is not None else None),
        _tile('volatility 1y', f"{row['vol_252']:.0%}" if pd.notna(row['vol_252']) else '-'),
    ], gap=8, mt=10)
    return html.Div([identity, tiles], style={'marginBottom': '12px'})


def _performance_section(history: pd.DataFrame, returns: pd.DataFrame | None, ticker: str, dark: bool) -> dmc.Card:
    """Indexed line vs S&P 500 with range control, plus the trailing-returns bars."""
    if history.empty:
        return _section('Performance', [_no_data(f'no price history for {ticker}')])
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
                    caption='total return, both series rebased to 100 at the window start; drawdown beneath')


def _risk_section(peers: pd.DataFrame, history: pd.DataFrame, ticker: str, dark: bool) -> dmc.Card:
    """Market and balance-sheet risk against the peer range, plus drawdown depth tiles."""
    children: list = []
    if not history.empty:
        stats = data.drawdown(history['AC'])
        children.append(dmc.Group([
            _tile('max drawdown', f'{stats.min():.0%}', 'full history'),
            _tile('current drawdown', f'{stats.iloc[-1]:.0%}', 'from all-time high'),
        ], gap=8, mb=4))
    if len(peers) >= 3:
        children.append(dcc.Graph(figure=charts.peer_strip_figure(peers, ticker, charts.RISK_SPEC, dark)))
    else:
        children.append(_no_data('not enough peers for a risk comparison'))
    return _section('Risk', children, caption='gray dots are peers, the diamond is this ticker')


def build_sections(ticker: str, dark: bool) -> list:
    """Assemble every section for one ticker; pool row missing yields a single note."""
    pool = get_pool()
    match = pool[pool['Ticker'] == ticker]
    if match.empty:
        return [_no_data(f'{ticker} is not in the pool')]
    row = match.iloc[0]

    peers = data.peer_group(pool, ticker)
    history = data.price_history(ticker)
    quarters = data.fundamentals_history(ticker)
    ttm = data.ttm_history(quarters) if len(quarters) else quarters
    returns = data.trailing_returns(history) if len(history) else None

    mates = peers.iloc[1:]
    if len(mates) and (mates['industry'] == row['industry']).all():
        basis = f"industry '{row['industry']}'"
    else:
        basis = f"sector '{row['sector']}'"
    peer_caption = (f'{len(peers) - 1} size-comparable peers from {basis}; '
                    'gray dots are peers, the diamond is this ticker')

    sections = [_hero(ticker, row, ttm, returns),
                _performance_section(history, returns, ticker, dark)]
    if len(peers) >= 3:
        families, ranks = data.peer_percentiles(peers, ticker)
        profile = dmc.Grid([
            dmc.GridCol(_section('Factor profile', [dcc.Graph(figure=charts.factor_profile_figure(families, ranks, dark))],
                                 caption='percentile within the peer group, 100 = best'),
                        span={'base': 12, 'lg': 6}),
            dmc.GridCol(_section('Peer comparison',
                                 [dcc.Graph(figure=charts.peer_strip_figure(peers, ticker, charts.PEER_SPEC, dark))],
                                 caption=peer_caption),
                        span={'base': 12, 'lg': 6}),
        ], gutter=12)
        sections.append(profile)
    else:
        sections.append(_section('Peer comparison', [_no_data('not enough peers in this industry/sector')]))
    sections.append(_risk_section(peers, history, ticker, dark))
    if len(ttm):
        sections.append(_section('Fundamentals history', [dcc.Graph(figure=charts.fundamentals_figure(ttm, dark))],
                                 caption='trailing-twelve-month flows from restated quarters; balance items point-in-time'))
    else:
        sections.append(_section('Fundamentals history', [_no_data(f'no quarterly fundamentals for {ticker}')]))
    return sections


def _select_options(pool: pd.DataFrame) -> list[dict]:
    """'TICKER - Company' select options for every pool row."""
    options = []
    for ticker, name in zip(pool['Ticker'], pool['name']):
        options.append({'value': ticker, 'label': f'{ticker} - {name}' if name else ticker})
    return options


def layout(symbol: str | None = None, **_query) -> html.Div:
    """Page shell: searchable ticker select (options built here so ?symbol= preselects) and content."""
    try:
        options = _select_options(get_pool())
    except duckdb.Error:
        options = []
    return html.Div([
        dmc.Select(id='tk-select', data=options, value=symbol, searchable=True, clearable=True,
                   placeholder='search ticker or company', limit=100, size='sm',
                   nothingFoundMessage='no match', style={'width': '360px', 'marginBottom': '10px'}),
        html.Div(id='tk-content', children=PROMPT),
    ], style={'padding': '4px'})


dash.register_page(__name__, path='/ticker', name='Ticker', layout=layout)


@callback(Output('tk-content', 'children'), Input('tk-select', 'value'), Input('theme-switch', 'checked'))
def render(ticker, dark):
    """Rebuild the whole page for the selected ticker and theme."""
    if not ticker:
        return PROMPT
    try:
        return build_sections(ticker, bool(dark))
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
