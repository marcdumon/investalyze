"""Line-item structure and view transforms for the ticker page's financial statements table.

A statement frame is one DuckDB fundamentals table's rows for a ticker (income, balance or
cashflow), one row per fiscal period, oldest first. `table_data` turns it into AG Grid
columnDefs + rowData for one combination of depth (summary/detail) and view ($ / YoY % /
common-size % / per share). Pure frames-in/dicts-out, no Dash, no DB.
"""

import numpy as np
import pandas as pd

# leading identity/meta columns shared by all three statement tables; line items follow them
META_COLUMNS = ['Ticker', 'SrcId', 'Src', 'Market', 'Period', 'IsRestated', 'Currency', 'Fiscal Year',
                'Fiscal Period', 'Report Date', 'Publish Date', 'Restated Date', 'Shares (Basic)', 'Shares (Diluted)']

STATEMENT_LABELS = {'income': 'Income', 'balance': 'Balance', 'cashflow': 'Cash flow'}

# statement totals: bold with a separator line, always part of the summary view
TOTALS = {
    'income': ('Gross Profit', 'Operating Income (Loss)', 'Pretax Income (Loss)', 'Net Income', 'Net Income (Common)'),
    'balance': ('Total Current Assets', 'Total Noncurrent Assets', 'Total Assets', 'Total Current Liabilities',
                'Total Noncurrent Liabilities', 'Total Liabilities', 'Total Equity', 'Total Liabilities & Equity'),
    'cashflow': ('Net Cash from Operating Activities', 'Net Cash from Investing Activities',
                 'Net Cash from Financing Activities', 'Net Change in Cash'),
}

# the line items the summary depth shows (statement order); detail depth shows every reported item
SUMMARY = {
    'income': ('Revenue', 'Cost of Revenue', 'Gross Profit', 'Operating Expenses', 'Operating Income (Loss)',
               'Non-Operating Income (Loss)', 'Pretax Income (Loss), Adj.', 'Abnormal Gains (Losses)',
               'Pretax Income (Loss)', 'Income Tax (Expense) Benefit, Net', 'Income (Loss) from Continuing Operations',
               'Net Extraordinary Gains (Losses)', 'Net Income', 'Net Income (Common)'),
    'balance': ('Cash, Cash Equivalents & Short Term Investments', 'Accounts & Notes Receivable', 'Inventories',
                'Other Short Term Assets', 'Total Current Assets', 'Property, Plant & Equipment, Net',
                'Long Term Investments & Receivables', 'Intangible Assets', 'Other Long Term Assets',
                'Total Noncurrent Assets', 'Total Assets', 'Payables & Accruals', 'Short Term Debt',
                'Other Short Term Liabilities', 'Total Current Liabilities', 'Long Term Debt',
                'Other Long Term Liabilities', 'Total Noncurrent Liabilities', 'Total Liabilities',
                'Preferred Equity', 'Share Capital & Additional Paid-In Capital', 'Treasury Stock',
                'Retained Earnings', 'Other Equity', 'Minority Interest', 'Total Equity',
                'Total Liabilities & Equity'),
    'cashflow': ('Net Income/Starting Line', 'Depreciation & Amortization', 'Non-Cash Items',
                 'Change in Working Capital', 'Net Cash from Operating Activities',
                 'Change in Fixed Assets & Intangibles', 'Net Change in Long Term Investment',
                 'Net Cash from Acquisitions & Divestitures', 'Other Investing Activities',
                 'Net Cash from Investing Activities', 'Dividends Paid', 'Cash from (Repayment of) Debt',
                 'Cash from (Repurchase of) Equity', 'Other Financing Activities',
                 'Net Cash from Financing Activities', 'Effect of Foreign Exchange Rates', 'Net Change in Cash'),
}

# validated status poles (dataviz palette): negatives / declines red, YoY gains green
_RED = '#d03b3b'
_GREEN = '#0ca30c'


def latest_restated(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per fiscal period (the latest by Restated Date), ordered by Report Date."""
    ordered = frame.sort_values(['Fiscal Year', 'Fiscal Period', 'Restated Date'])
    deduped = ordered.groupby(['Fiscal Year', 'Fiscal Period'], as_index=False).tail(1)
    return deduped.sort_values('Report Date', ignore_index=True)


def line_items(frame: pd.DataFrame) -> list[str]:
    """The statement's line-item columns, in statement order."""
    return [column for column in frame.columns if column not in META_COLUMNS]


def period_label(row: pd.Series) -> str:
    """Column header for one fiscal period: 'FY 2024' or '2024 Q3'."""
    if row['Fiscal Period'] == 'FY':
        return f"FY {row['Fiscal Year']}"
    return f"{row['Fiscal Year']} {row['Fiscal Period']}"


def _fmt_millions(value: float) -> str:
    """A value in millions with separators, gaining decimals as the magnitude shrinks."""
    if pd.isna(value):
        return ''
    m = value / 1e6
    if abs(m) >= 100:
        return f'{m:,.0f}'
    if abs(m) >= 10:
        return f'{m:,.1f}'
    return f'{m:,.2f}'


def _fmt_signed_pct(value: float) -> str:
    """A YoY percentage with an explicit sign."""
    return '' if pd.isna(value) else f'{value:+,.1f}%'


def _fmt_pct(value: float) -> str:
    """A common-size percentage."""
    return '' if pd.isna(value) else f'{value:,.1f}%'


def _fmt_per_share(value: float) -> str:
    """A per-share value in currency units."""
    return '' if pd.isna(value) else f'{value:,.2f}'


def _yoy(matrix: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Percent change of each period column vs the same fiscal period one year earlier."""
    position = {(fy, fp): i for i, (fy, fp) in enumerate(zip(frame['Fiscal Year'], frame['Fiscal Period']))}
    out = pd.DataFrame(np.nan, index=matrix.index, columns=matrix.columns)
    for i, (fy, fp) in enumerate(zip(frame['Fiscal Year'], frame['Fiscal Period'])):
        prev = position.get((fy - 1, fp))
        if prev is not None:
            base = matrix.iloc[:, prev]
            out.iloc[:, i] = (matrix.iloc[:, i] - base) / base.abs() * 100
    return out


def _transform(matrix: pd.DataFrame, frame: pd.DataFrame, statement: str, view: str,
               revenue: pd.Series | None) -> pd.DataFrame:
    """The numeric matrix (items x periods) for one view; $ stays raw, ratios are percent/per-share."""
    if view == 'yoy':
        return _yoy(matrix, frame)
    if view == 'common':
        if statement == 'balance':
            base = frame['Total Assets']
        elif statement == 'income':
            base = frame['Revenue']
        else:
            base = revenue if revenue is not None else pd.Series(np.nan, index=frame.index)
        return matrix / base.where(base != 0).abs().to_numpy() * 100
    if view == 'ps':
        shares = frame['Shares (Diluted)']
        return matrix / shares.where(shares > 0).to_numpy()
    return matrix


def _header_unit(statement: str, view: str, currency: str) -> str:
    """The pinned column's header, naming the unit the cells are in."""
    if view == 'yoy':
        return 'YoY %'
    if view == 'common':
        return '% of total assets' if statement == 'balance' else '% of revenue'
    if view == 'ps':
        return f'{currency} per share'
    return f'{currency} millions'


def _view_matrix(frame: pd.DataFrame, items: list[str], statement: str, view: str,
                 revenue: pd.Series | None) -> pd.DataFrame:
    """The view-transformed numeric matrix (items x periods) for one statement frame."""
    matrix = frame[items].apply(pd.to_numeric, errors='coerce').astype(float).T
    return _transform(matrix, frame, statement, view, revenue)


def table_data(frame: pd.DataFrame, statement: str, depth: str, view: str,
               revenue: pd.Series | None = None, last: int = 10,
               filed: pd.DataFrame | None = None,
               filed_revenue: pd.Series | None = None) -> tuple[list[dict], list[dict]]:
    """(columnDefs, rowData) for the statements grid: `last` periods newest first, one row per line item.

    `frame` holds all deduped periods oldest first (YoY needs the year before the window);
    `revenue` supplies the common-size base for cashflow frames. Passing `filed` (the as-filed
    frame) switches to diff mode: cells become restated minus as-filed per fiscal period,
    view-transformed first (percent views diff in points); unchanged cells stay blank and fully
    unchanged lines are dropped. Rows carry a `level` field: 0 statement totals (bold +
    separator), 1 summary lines, 2 detail lines (indented, dimmed).
    """
    labels = [period_label(row) for _, row in frame.iterrows()]
    items = line_items(frame)
    if depth == 'summary':
        items = [item for item in SUMMARY[statement] if item in items]
    values = _view_matrix(frame, items, statement, view, revenue)
    diff = filed is not None
    if diff:
        filed_values = _view_matrix(filed, items, statement, view, filed_revenue)
        filed_position = {(fy, fp): i for i, (fy, fp)
                          in enumerate(zip(filed['Fiscal Year'], filed['Fiscal Period']))}
        aligned = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
        for i, (fy, fp) in enumerate(zip(frame['Fiscal Year'], frame['Fiscal Period'])):
            position = filed_position.get((fy, fp))
            if position is not None:
                aligned.iloc[:, i] = filed_values.iloc[:, position].to_numpy()
        delta = values - aligned
        values = delta.where(delta != 0)
    values = values.iloc[:, -last:].iloc[:, ::-1]
    shown = labels[-last:][::-1]

    formatter = {'usd': _fmt_millions, 'yoy': _fmt_signed_pct, 'common': _fmt_pct, 'ps': _fmt_per_share}[view]
    totals = TOTALS[statement]
    summary = SUMMARY[statement]
    rows = []
    for item in items:
        cells = values.loc[item]
        if cells.isna().all():
            continue
        level = 0 if item in totals else 1 if item in summary else 2
        row = {'item': item, 'level': level}
        for label, value in zip(shown, cells):
            text = formatter(value)
            if diff and text and not text.startswith(('+', '-')):
                text = '+' + text
            row[label] = text
        rows.append(row)

    currency = str(frame['Currency'].iloc[-1]) if len(frame) else ''
    if diff or view == 'yoy':
        value_style = {'styleConditions': [
            {'condition': "params.value && params.value.startsWith('+')", 'style': {'color': _GREEN}},
            {'condition': "params.value && params.value.startsWith('-')", 'style': {'color': _RED}},
        ]}
    else:
        value_style = {'styleConditions': [
            {'condition': "params.value && params.value.startsWith('-')", 'style': {'color': _RED}},
        ]}
    unit = _header_unit(statement, view, currency)
    columns = [{'field': 'item', 'headerName': f'diff {unit}' if diff else unit, 'pinned': 'left',
                'width': 260, 'cellStyle': {'styleConditions': [
                    {'condition': 'params.data.level === 2', 'style': {'paddingLeft': '24px'}}]}}]
    for label in shown:
        columns.append({'field': label, 'width': 92, 'type': 'rightAligned', 'cellStyle': value_style})
    return columns, rows
