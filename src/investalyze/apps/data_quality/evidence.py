"""Evidence panel for a selected anomaly: a windowed price candlestick or a transposed fundamentals view.

Read-only, short-lived connections; a locked DB (a job writing) shows a busy notice instead of a
traceback. Price findings (prices / market_data) get an OHLC candlestick around the flagged date;
fundamentals findings (income / balance / cashflow) get the flagged fiscal period and its neighbours
transposed, line items on the rows, periods on the columns, the flagged period highlighted.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import dcc, html
from plotly.subplots import make_subplots

from investalyze.apps.data_quality import actions, edgar
from investalyze.ingest import storage

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / 'data'
BUSY = 'database busy, a job is currently running, try again once it finishes'

_PRICE_TABLES = ('prices', 'market_data')
_META_COLUMNS = {
    'Ticker', 'SrcId', 'Src', 'Market', 'Period', 'IsRestated', 'Currency', 'Fiscal Year',
    'Fiscal Period', 'Report Date', 'Publish Date', 'Restated Date',
}


def _flag_date(row: dict) -> date | None:
    """The finding's date as a date, or None."""
    value = row.get('Date')
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date() if not pd.isna(value) else None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return date.fromisoformat(text[:10]) if text and text.lower() not in ('nat', 'none') else None


def _yahoo_finance_link(ticker: str) -> html.A:
    """Small link to the ticker's Yahoo Finance quote page."""
    return html.A(ticker, href=f'https://finance.yahoo.com/quote/{ticker}', target='_blank',
                  style={'color': 'var(--mantine-color-blue-5)'})


def _edgar_filing_url(cik: int | None, fiscal_period: str, report_date) -> str | None:
    """Direct EDGAR document URL for the filing covering this period, or None without a CIK or match."""
    if cik is None or pd.isna(report_date):
        return None
    try:
        return edgar.filing_url(cik, pd.Timestamp(report_date).strftime('%Y-%m-%d'), fiscal_period)
    except requests.RequestException:
        return None


def _edgar_restated_url(cik: int | None, restated_date) -> str | None:
    """Direct EDGAR document URL for the filing that published these restated figures, or None."""
    if cik is None or pd.isna(restated_date):
        return None
    try:
        return edgar.restated_filing_url(cik, pd.Timestamp(restated_date).strftime('%Y-%m-%d'))
    except requests.RequestException:
        return None


def price_evidence(row: dict, dark: bool) -> list:
    """Candlestick (plus volume for prices) of the window around the flagged date for a price finding."""
    ticker, table = row['Ticker'], row['SrcTable']
    on_date = _flag_date(row)
    has_volume = table == 'prices'
    columns = 'Date, O, H, L, C' + (', V' if has_volume else '')
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        if on_date is not None:
            window = con.execute(
                f'SELECT {columns} FROM {table} WHERE Ticker = ? AND Date BETWEEN ? AND ? ORDER BY Date',
                [ticker, on_date - timedelta(days=120), on_date + timedelta(days=120)],
            ).df()
        else:
            window = con.execute(f'SELECT {columns} FROM {table} WHERE Ticker = ? ORDER BY Date', [ticker]).df().tail(500)
    finally:
        con.close()

    if window.empty:
        return [html.Div(f'no {table} rows for {ticker}', style={'color': 'var(--mantine-color-dimmed)', 'fontSize': '13px'})]

    rows_layout = 2 if has_volume else 1
    heights = [0.75, 0.25] if has_volume else [1.0]
    fig = make_subplots(rows=rows_layout, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=window['Date'], open=window['O'], high=window['H'], low=window['L'],
                                 close=window['C'], name='OHLC'), row=1, col=1)
    if has_volume:
        fig.add_trace(go.Bar(x=window['Date'], y=window['V'], name='Volume', marker={'line': {'width': 0}}), row=2, col=1)
    if on_date is not None:
        fig.add_vline(x=on_date, line_dash='dot', line_color='orange')
    fig.update_layout(template='plotly_dark' if dark else 'plotly_white', height=420, showlegend=False,
                      margin={'l': 40, 'r': 10, 't': 30, 'b': 20}, xaxis_rangeslider_visible=False,
                      title={'text': f'{ticker} {table} around {on_date or "full history"}', 'font': {'size': 14}})
    return [html.Div(_yahoo_finance_link(ticker), style={'fontSize': '13px', 'marginBottom': '4px'}),
            dcc.Graph(figure=fig)]


def fundamentals_evidence(row: dict, dark: bool) -> list:
    """Flagged fiscal period and its neighbours transposed: line items on rows, periods on columns."""
    ticker, table = row['Ticker'], row['SrcTable']
    parsed = actions.parse_key(row.get('Key'))
    con = storage.connect(DATA_ROOT, read_only=True)
    try:
        statement = con.execute(f'SELECT * FROM {table} WHERE Ticker = ? ORDER BY "Report Date"', [ticker]).df()
        cik_row = con.execute('SELECT CIK FROM companies WHERE Ticker = ?', [ticker]).fetchone()
    finally:
        con.close()
    cik = int(cik_row[0]) if cik_row and cik_row[0] is not None else None

    if statement.empty:
        return [html.Div(f'no {table} rows for {ticker}', style={'color': 'var(--mantine-color-dimmed)', 'fontSize': '13px'})]

    labels = statement.apply(lambda r: f"{r['Fiscal Year']} {r['Fiscal Period']}{' R' if r['IsRestated'] else ''}", axis=1)
    flagged_pos = _flagged_position(statement, parsed)
    lo = max(0, flagged_pos - 2)
    window = statement.iloc[lo:flagged_pos + 3]
    window_labels = labels.iloc[lo:flagged_pos + 3].tolist()
    flagged_label = labels.iloc[flagged_pos] if flagged_pos is not None else None

    line_items = [col for col in statement.columns if col not in _META_COLUMNS and window[col].notna().any()]
    header = html.Tr([html.Th('line item', style={'textAlign': 'left', 'paddingRight': '12px'})]
                     + [html.Th(label, style={'padding': '0 8px', 'textAlign': 'right',
                                              'color': 'var(--mantine-color-orange-6)' if label == flagged_label else None})
                        for label in window_labels])
    filing_cells = [html.Td('SEC filing', style={'paddingRight': '12px', 'color': 'var(--mantine-color-dimmed)'})]
    for _, srow in window.iterrows():
        url = (_edgar_restated_url(cik, srow['Restated Date']) if srow['IsRestated']
              else _edgar_filing_url(cik, srow['Fiscal Period'], srow['Report Date']))
        cell = html.A('EDGAR', href=url, target='_blank') if url else '-'
        filing_cells.append(html.Td(cell, style={'padding': '0 8px', 'textAlign': 'right'}))
    filing_row = html.Tr(filing_cells)
    check = str(row.get('CheckName', ''))
    involved = actions.involved_items(check, str(row.get('Details', '')))
    used_columns = []
    for label, (_, srow) in zip(window_labels, window.iterrows()):
        if label == flagged_label:
            used_columns.append(True)
        elif check == 'quarters_vs_fy' and parsed is not None:
            same_vintage = srow['Period'] == 'Q' and str(srow['IsRestated']).lower() == parsed['is_restated']
            used_columns.append(same_vintage and srow['Fiscal Year'] == parsed['fiscal_year'])
        else:
            used_columns.append(False)

    body = []
    for item in line_items:
        cells = [html.Td(item, style={'paddingRight': '12px', 'color': 'var(--mantine-color-dimmed)'})]
        for used, (_, srow) in zip(used_columns, window.iterrows()):
            value = srow[item]
            text = '' if pd.isna(value) else (f'{value:,.0f}' if isinstance(value, (int, float)) else str(value))
            cells.append(html.Td(text, style={'padding': '0 8px', 'textAlign': 'right',
                                              'fontWeight': 'bold' if used and item in involved else None}))
        body.append(html.Tr(cells))
    for item in ('Report Date', 'Publish Date', 'Restated Date'):
        cells = [html.Td(item, style={'paddingRight': '12px', 'color': 'var(--mantine-color-dimmed)'})]
        for _, srow in window.iterrows():
            value = srow[item]
            text = '' if pd.isna(value) else pd.Timestamp(value).strftime('%Y-%m-%d')
            cells.append(html.Td(text, style={'padding': '0 8px', 'textAlign': 'right'}))
        body.append(html.Tr(cells))
    caption = [_yahoo_finance_link(ticker), html.Span(f' {table}: {row.get("Details", "")}')]
    return [
        html.Div(caption, style={'fontSize': '13px', 'marginBottom': '6px'}),
        html.Div(html.Table([header, filing_row] + body, style={'fontSize': '12px', 'borderCollapse': 'collapse'}),
                 style={'overflowX': 'auto'}),
    ]


def _flagged_position(statement: pd.DataFrame, parsed: dict | None) -> int:
    """Row index of the flagged period in `statement`, or the last row when it cannot be matched."""
    if parsed is None:
        return len(statement) - 1
    match = statement[(statement['Fiscal Year'].astype('Int64') == parsed['fiscal_year'])
                      & (statement['Fiscal Period'] == parsed['fiscal_period'])
                      & (statement['Period'] == parsed['period'])
                      & (statement['IsRestated'].astype(str).str.lower() == parsed['is_restated'])]
    return int(statement.index.get_loc(match.index[0])) if len(match) else len(statement) - 1


def evidence_children(row: dict, dark: bool) -> list:
    """Evidence for one anomaly row, dispatched on its source table."""
    if row['SrcTable'] in _PRICE_TABLES:
        return price_evidence(row, dark)
    return fundamentals_evidence(row, dark)


def safe_evidence_children(row: dict, dark: bool) -> list:
    """evidence_children, but a locked DB shows a busy notice instead of a traceback."""
    try:
        return evidence_children(row, dark)
    except duckdb.Error:
        return [html.Div(BUSY, style={'color': 'var(--mantine-color-yellow-9)', 'fontSize': '13px', 'padding': '12px'})]
