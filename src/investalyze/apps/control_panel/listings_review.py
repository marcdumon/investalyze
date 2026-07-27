"""Control-panel section: review the last listings scan and add candidates to ticker.csv.

Reads `data/yahoo/state/listing_candidates.csv` (written by `investalyze.ingest.listings.scan`)
and lets the user append new/reactivated-safe tickers to `ticker.csv`. Reactivated-collision rows
are never added here; see `rename_entries` (added in the next task) for staging the rename that
must happen first.
"""

from pathlib import Path

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash_iconify import DashIconify
from pandas.errors import EmptyDataError

CANDIDATES_PATH = Path('data/yahoo/state/listing_candidates.csv')
TICKER_CSV_PATH = Path('data/yahoo/raw/ticker.csv')

CANDIDATE_COLUMNS = [
    {'field': 'ticker', 'checkboxSelection': True, 'headerCheckboxSelection': True, 'width': 100},
    {'field': 'market', 'width': 90},
    {'field': 'company_name', 'headerName': 'company', 'flex': 1},
    {'field': 'classification', 'width': 170},
    {'field': 'suggested_new_ticker', 'headerName': 'suggested key', 'width': 130},
]


def read_candidates(path: Path = CANDIDATES_PATH) -> list[dict]:
    """The last scan's candidates, or an empty list if no scan has run yet or none were found."""
    if not path.exists():
        return []
    try:
        return pd.read_csv(path, keep_default_na=False).to_dict('records')
    except EmptyDataError:
        return []


def add_tickers(rows: list[dict], ticker_csv_path: Path = TICKER_CSV_PATH) -> int:
    """Append rows classified new/reactivated_safe to ticker.csv (ticker, market); returns rows processed.

    Rows classified reactivated_collision are skipped; those need a rename staged first.
    """
    addable = [r for r in rows if r['classification'] in ('new', 'reactivated_safe')]
    if not addable:
        return 0
    existing = pd.read_csv(ticker_csv_path)
    new_rows = pd.DataFrame({'ticker': [r['ticker'] for r in addable], 'market': [r['market'] for r in addable]})
    combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates('ticker', keep='first')
    combined.to_csv(ticker_csv_path, index=False)
    return len(addable)


def listings_card() -> dmc.Card:
    """New-listings review card: scan-result grid + Add / Stage-rename actions."""
    return dmc.Card([
        dmc.Group([dmc.Text('New listings review', fw=700, size='sm'),
                  dmc.Text('run a scan above, then review candidates here', size='xs', c='dimmed')], gap=8, mb=8),
        dag.AgGrid(
            id='cp-listings-grid', columnDefs=CANDIDATE_COLUMNS, rowData=[],
            defaultColDef={'sortable': True, 'resizable': True},
            dashGridOptions={'animateRows': False, 'theme': 'legacy', 'rowSelection': 'multiple',
                             'suppressRowClickSelection': True},
            className='ag-theme-alpine-dark', style={'height': '260px', 'width': '100%'},
        ),
        dmc.Group([
            dmc.Button('Add selected to ticker.csv', id='cp-btn-listings-add', size='xs', color='orange',
                       leftSection=DashIconify(icon='tabler:playlist-add')),
        ], gap=8, mt=8),
        dmc.Text(id='cp-listings-status', size='xs', c='dimmed', mt=4),
    ], withBorder=True, radius='md', padding='sm')
