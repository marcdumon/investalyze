"""Per-ticker detail panel: stats, latest fundamentals, anomalies and the price chart."""

from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'


def price_figure(ticker: str, dark: bool) -> go.Figure:
    """Adjusted close (log) and volume for the full history of one ticker."""
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        prices = con.execute("SELECT Date, AC, V FROM prices WHERE Ticker = ? ORDER BY Date", [ticker]).df()
    finally:
        con.close()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=prices['Date'], y=prices['AC'], name='AC', line={'width': 1}), row=1, col=1)
    fig.add_trace(go.Bar(x=prices['Date'], y=prices['V'], name='Volume', marker={'line': {'width': 0}}), row=2, col=1)
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white',
                      height=420, margin={'l': 40, 'r': 10, 't': 30, 'b': 20}, showlegend=False,
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
        [html.Tr([html.Td(label, style={'color': 'var(--mantine-color-dimmed)', 'paddingRight': '12px'}),
                  html.Td(str(value))]) for label, value in rows],
        style={'fontSize': '13px', 'borderSpacing': '0 2px'},
    )


def detail_children(ticker: str, row: pd.Series, dark: bool) -> list:
    """Stats, latest fundamentals, anomalies and the price chart for one ticker (row = its pool row)."""
    stats = two_col_table([
        ('Company', row['name']), ('Sector', row['sector']), ('Industry', row['industry']),
        ('Country', row['country']), ('Employees', row['employees'] if pd.notna(row['employees']) else '-'),
        ('History', f"{row['first_date']} to {row['last_date']} ({row['years']}y)"),
        ('Active', bool(row['active'])), ('Last close', row['last_close']),
        ('Median $ volume', f"{row['dvol_mn']}mn/day"),
        ('Market cap', f"{row['mcap_bn']}bn" if pd.notna(row['mcap_bn']) else '- (no share count)'),
        ('Fundamental periods', row['n_periods']), ('Anomalies', row['n_anomalies']),
    ])

    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        fundamentals = con.execute("""
            SELECT "Fiscal Year", "Fiscal Period", Revenue, "Gross Profit", "Operating Income (Loss)",
                   "Net Income", "Shares (Basic)"
            FROM income WHERE Ticker = ? ORDER BY "Report Date" DESC LIMIT 1
        """, [ticker]).df()
        anomalies = con.execute(
            "SELECT CheckName, Severity, Date, Details FROM anomalies WHERE Ticker = ? ORDER BY Date", [ticker]
        ).df()
    finally:
        con.close()

    if len(fundamentals):
        f = fundamentals.iloc[0]
        fundamentals_block = two_col_table([
            ('Latest report', f"{f['Fiscal Year']} {f['Fiscal Period']}"),
            ('Revenue', fmt_money(f['Revenue'])), ('Gross profit', fmt_money(f['Gross Profit'])),
            ('Operating income', fmt_money(f['Operating Income (Loss)'])), ('Net income', fmt_money(f['Net Income'])),
            ('Shares (basic)', fmt_money(f['Shares (Basic)'])),
        ])
    else:
        fundamentals_block = html.Div('no fundamentals in DB', style={'color': 'var(--mantine-color-dimmed)', 'fontSize': '13px'})

    if len(anomalies):
        shown = anomalies.head(15)
        anomaly_rows = [
            html.Tr([html.Td(str(v), style={'paddingRight': '10px'}) for v in row])
            for row in shown.itertuples(index=False)
        ]
        extra = f' (+{len(anomalies) - len(shown)} more)' if len(anomalies) > len(shown) else ''
        anomalies_block = html.Div([
            html.Div(f'{len(anomalies)} anomalies{extra}', style={'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Table(anomaly_rows, style={'fontSize': '12px'}),
        ])
    else:
        anomalies_block = html.Div('no anomalies recorded', style={'color': 'var(--mantine-color-dimmed)', 'fontSize': '13px'})

    left = html.Div(
        [html.H4(ticker, style={'margin': '0 0 6px'}), stats,
         html.H4('Fundamentals', style={'margin': '10px 0 4px', 'fontSize': '14px'}), fundamentals_block,
         html.H4('Anomalies', style={'margin': '10px 0 4px', 'fontSize': '14px'}), anomalies_block],
        style={'width': '440px', 'flexShrink': 0, 'overflowY': 'auto'},
    )
    chart = html.Div(dcc.Graph(figure=price_figure(ticker, dark)), style={'flex': 1, 'minWidth': 0})
    return [html.Div([left, chart], style={'display': 'flex', 'gap': '16px'})]


def safe_detail_children(ticker: str, row: pd.Series, dark: bool) -> list:
    """detail_children, but a locked DB (a control-panel job writing) shows a notice instead of a traceback."""
    try:
        return detail_children(ticker, row, dark)
    except duckdb.Error:
        return [html.Div('database busy, a job is currently running, try again once it finishes',
                         style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '13px', 'padding': '12px'})]
