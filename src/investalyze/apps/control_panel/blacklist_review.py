"""Control-panel section: review and lift manual ticker blacklists.

Reads `data/manual_blacklist.csv` (written by `investalyze.ingest.listings.blacklist_tickers`).
Unblacklisting only removes the entry so a future scan can offer the ticker again - it never
restores data already deleted, and it's a plain file write, not a DB write, so it runs directly
in-process rather than through the MANAGER subprocess (see `apps.universes`'s save/delete for the
same reasoning).
"""

from pathlib import Path

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash_iconify import DashIconify
from pandas.errors import EmptyDataError

BLACKLIST_PATH = Path('data/manual_blacklist.csv')

BLACKLIST_COLUMNS = [
    {'field': 'ticker', 'headerName': 'Ticker', 'width': 110},
    {'field': 'market', 'headerName': 'Market', 'width': 100},
    {'field': 'company_name', 'headerName': 'Company', 'flex': 1},
    {'field': 'reason', 'headerName': 'Reason', 'flex': 1},
    {'field': 'blacklisted_on', 'headerName': 'Blacklisted On', 'width': 140},
]

_OPTIONAL_COLUMNS = ('market', 'company_name')


def read_blacklist(path: Path = BLACKLIST_PATH) -> list[dict]:
    """Every manually blacklisted ticker with its market, company name, reason and date, or an
    empty list if none yet.

    Market and company name are blank wherever they were unknown at blacklist time, and the whole
    column is missing from a file written before they were recorded at all - both read back as ''.
    """
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except EmptyDataError:
        return []
    for column in _OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ''
    return frame.to_dict('records')


def filter_blacklist(rows: list[dict], search: str | None) -> list[dict]:
    """Rows whose ticker contains `search` (case-insensitive), or every row when `search` is blank."""
    if not search:
        return rows
    needle = search.strip().upper()
    return [row for row in rows if needle in row['ticker'].upper()]


def blacklist_review_card() -> dmc.Card:
    """Manually blacklisted tickers grid + a ticker search box + an Unblacklist action."""
    return dmc.Card([
        dmc.Group([dmc.Text('Blacklisted tickers', fw=700, size='sm'),
                  dmc.Text('Select tickers and click Unblacklist to let a future scan offer them again.',
                          size='xs', c='dimmed')], gap=8, mb=4),
        dmc.TextInput(id='cp-blacklist-search', placeholder='ticker', size='xs', style={'width': '130px'}, mb=6),
        dag.AgGrid(
            id='cp-blacklist-grid', columnDefs=BLACKLIST_COLUMNS, rowData=[],
            defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
            dashGridOptions={'animateRows': False, 'theme': 'legacy',
                             'rowSelection': {'mode': 'multiRow', 'checkboxes': True, 'headerCheckbox': True,
                                              'enableClickSelection': False}},
            className='ag-theme-alpine-dark', style={'height': '300px', 'width': '100%'},
        ),
        dmc.Group([
            dmc.Button('Unblacklist selected', id='cp-btn-unblacklist', size='xs', variant='default',
                       leftSection=DashIconify(icon='tabler:restore')),
            dmc.Text(id='cp-unblacklist-status', size='xs', c='teal'),
        ], gap=8, mt=8),
    ], withBorder=True, radius='md', padding='sm')
