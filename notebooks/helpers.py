"""Helpers for the investalyze notebooks.

Layout:
  - SHARED: used by more than one notebook (connection + generic display helpers).
  - One section per notebook below the shared block, separated by banners.

Notebooks import what they need (kernel CWD is this folder) and turn on autoreload:

    %load_ext autoreload
    %autoreload 2
    from helpers import connect_readonly, show_ticker_profile

    con = connect_readonly()
"""
import math
import numbers
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display


# ======================================================================
# SHARED — used by more than one notebook
# ======================================================================

def fmt_number(x, sig=4):
    """Render a number readably: thousands separator for the big, significant figures for the small.

    Values >= 1 get a thousands separator and at most 4 decimals (1,234,567 / 593.57); values
    below 1 keep `sig` significant figures so tiny rates survive (0.00002, 0.000369) instead of
    rounding to zero. Trailing zeros and bare decimal points are trimmed (603 not 603.0000).
    Bools and non-numbers (strings, dates) pass through untouched — safe on mixed/transposed frames.
    """
    if isinstance(x, bool) or not isinstance(x, numbers.Number):
        return '' if x is pd.NA else x
    if pd.isna(x):
        return ''
    ax = abs(x)
    if ax == 0:
        return '0'
    if ax >= 1:
        s = f'{x:,.4f}'
    else:
        decimals = -math.floor(math.log10(ax)) + (sig - 1)
        s = f'{x:.{decimals}f}'
    return s.rstrip('0').rstrip('.') if '.' in s else s


def _fmt_cell(x):
    """Cell formatter: dates as YYYY-MM-DD, numbers via `fmt_number`, everything else untouched."""
    if x is pd.NaT:
        return ''
    if isinstance(x, pd.Timestamp):
        return '' if pd.isna(x) else x.strftime('%Y-%m-%d')
    return fmt_number(x)


def _is_numeric_col(col):
    """True if a column holds numbers — including object columns of mixed Int/None (financials)."""
    if pd.api.types.is_numeric_dtype(col):
        return not pd.api.types.is_bool_dtype(col)
    non_null = col.dropna()
    return len(non_null) > 0 and all(isinstance(v, numbers.Number) and not isinstance(v, bool) for v in non_null)


MONO_FONT = '"JetBrainsMono Nerd Font Mono", monospace'


def show_df(df):
    """Display a DataFrame: dates as YYYY-MM-DD, numbers formatted + right-aligned, text left-aligned.

    Numeric columns (incl. object columns of mixed Int/None) are right-aligned; everything else
    (text, dates, the object-typed transposed identity cards) is left-aligned. Rendered monospace
    (Styler tables carry no `dataframe` class, so a global `.dataframe` font rule can't reach them).
    """
    num_cols = [c for c in df.columns if _is_numeric_col(df[c])]
    styler = (df.style.format(_fmt_cell)
              .set_properties(**{'white-space': 'nowrap', 'text-align': 'left'})
              .set_table_styles([{'selector': '', 'props': [('font-family', MONO_FONT)]},
                                 {'selector': 'th', 'props': [('text-align', 'left')]}]))
    if num_cols:
        styler = styler.set_properties(subset=num_cols, **{'text-align': 'right'})
    display(styler)


def connect_readonly():
    """Open a read-only DuckDB connection at the configured data root, display defaults applied.

    Walks up from the kernel CWD to the repo root (the dir holding `ingest.toml`) so it
    works regardless of where the notebook is launched from. Never mutates the DB.
    """
    from investalyze.ingest import config, storage

    display(HTML('<style>table.dataframe td {white-space: nowrap;}</style>'))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', fmt_number)
    root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'ingest.toml').exists())
    cfg = config.load(root / 'ingest.toml')
    return storage.connect(root / cfg.data_root, read_only=True)


def list_tables(con):
    """Set of table names present in the DB."""
    return {name for (name,) in con.execute('SHOW TABLES').fetchall()}


def load_ticker_rows(con, table, ticker):
    """All rows for `ticker` in `table` as a DataFrame (empty if the table is absent)."""
    if table not in list_tables(con):
        return pd.DataFrame()
    return con.execute(f'SELECT * FROM {table} WHERE Ticker = ?', [ticker]).df()


def head_and_tail(df, n=5):
    """First `n` + last `n` rows; the whole frame if it has <= 2n rows."""
    if len(df) <= 2 * n:
        return df
    return pd.concat([df.head(n), df.tail(n)])


def show_section_header(title, level=2):
    """Render a markdown header at the given level (h2 by default)."""
    display(Markdown(f'{"#" * level} {title}'))


def show_note(text):
    """Render an italic one-line note."""
    display(Markdown(f'*{text}*'))


# ======================================================================
# 1_explore_db — random-sample browser
# ======================================================================

PROVIDER_TABLES = {
    'stooq': ['market_data'],
    'yahoo': ['prices', 'dividends', 'splits'],
    'yahoo-meta': ['company_profile', 'company_officers'],
    'simfin': ['income', 'balance', 'cashflow', 'companies'],
}

# Fundamentals tables holding both as-reported and restated rows (split by the IsRestated flag).
RESTATED_TABLES = {'income', 'balance', 'cashflow'}


def sample_rows(con, table, n=15, where=None):
    """`n` random rows from `table` as a DataFrame, optionally filtered by a SQL `where` clause."""
    clause = f' WHERE {where}' if where else ''
    return con.execute(f'SELECT * FROM {table}{clause} ORDER BY random() LIMIT {n}').df()


def show_provider_samples(con, provider):
    """Display an n-row sample of every table the provider owns; split restated fundamentals."""
    present = list_tables(con)
    for table in PROVIDER_TABLES[provider]:
        if table in RESTATED_TABLES and table in present:
            for is_restated, label in ((False, 'as-reported'), (True, 'restated')):
                show_section_header(f'{provider} — {table} ({label})', level=3)
                show_df(sample_rows(con, table, where=f'IsRestated = {is_restated}'))
            continue
        show_section_header(f'{provider} — {table}', level=3)
        if table not in present:
            show_note('not in the DB yet')
            continue
        show_df(sample_rows(con, table))


# ======================================================================
# 2_ticker_profile — per-ticker drill-down
# ======================================================================

def timeseries_stats(df):
    """count/min/max/mean/std per numeric column of a time series (Date span shown separately)."""
    return df.select_dtypes('number').describe().T[['count', 'min', 'max', 'mean', 'std']]


def fundamentals_coverage(df):
    """One-line coverage summary for a fundamentals slice."""
    fy = f"{int(df['Fiscal Year'].min())} .. {int(df['Fiscal Year'].max())}"
    periods = ' '.join(f'{p}={n}' for p, n in df['Period'].value_counts().sort_index().items())
    currency = ', '.join(sorted(df['Currency'].dropna().unique()))
    return f'Fiscal Year {fy}  |  {periods}  |  {currency}  |  {len(df)} rows'


def fundamentals_stats(df, min_fill):
    """count/min/max/mean/std for the well-populated numeric columns of a fundamentals slice.

    Importance heuristic: a column's NaN fraction. Columns filled in fewer than `min_fill` of
    the rows (e.g. 'Sales & Services Revenue', 'Other Revenue') are dropped; the core line
    items (Revenue, Cost of Revenue, ...) survive. Ordered most-populated first.
    """
    num = df.select_dtypes('number').drop(columns=['SrcId', 'Fiscal Year'], errors='ignore')
    fill = num.notna().mean()
    keep = fill[fill >= min_fill].sort_values(ascending=False).index
    return num[keep].describe().T[['count', 'min', 'max', 'mean', 'std']]


def show_timeseries_section(con, table, ticker):
    """A time-series table: date span + per-column stats + head/tail preview."""
    show_section_header(table)
    df = load_ticker_rows(con, table, ticker).sort_values('Date')
    if df.empty:
        show_note('no rows')
        return
    show_note(f'{df.Date.min()} .. {df.Date.max()}  |  {len(df)} rows')
    show_df(timeseries_stats(df))
    show_df(head_and_tail(df))


def show_ticker_profile(con, ticker, min_fill=0.5):
    """Everything we hold for `ticker`: metadata + a head/tail preview, table by table."""
    display(Markdown(f'# {ticker}'))

    # identity card — one row, transposed
    show_section_header('companies')
    companies = load_ticker_rows(con, 'companies', ticker)
    show_note('no rows') if companies.empty else show_df(companies.T)

    show_section_header('company_profile')
    profile = load_ticker_rows(con, 'company_profile', ticker)
    show_note('no rows') if profile.empty else show_df(profile.T)

    show_section_header('company_officers')
    officers = load_ticker_rows(con, 'company_officers', ticker)
    show_note('no rows') if officers.empty else show_df(officers)

    show_timeseries_section(con, 'prices', ticker)
    show_timeseries_section(con, 'dividends', ticker)
    show_timeseries_section(con, 'splits', ticker)

    # fundamentals — split each statement into as-reported vs restated
    for table in ('income', 'balance', 'cashflow'):
        show_section_header(table)
        df = load_ticker_rows(con, table, ticker).sort_values(['Fiscal Year', 'Fiscal Period'])
        if df.empty:
            show_note('no rows')
            continue
        for is_restated, label in ((False, 'as-reported'), (True, 'restated')):
            show_section_header(f'{table} ({label})', level=3)
            part = df[df['IsRestated'] == is_restated]
            if part.empty:
                show_note('no rows')
                continue
            show_note(fundamentals_coverage(part))
            show_df(fundamentals_stats(part, min_fill))
            show_df(head_and_tail(part))

    show_timeseries_section(con, 'market_data', ticker)  # populates for non-equity tickers
