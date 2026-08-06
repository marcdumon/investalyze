"""Control-panel section: review and lift manual ticker blacklists.

Reads `data/yahoo/state/manual_blacklist.csv` (written by `investalyze.ingest.listings.blacklist_tickers`).
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

BLACKLIST_PATH = Path('data/yahoo/state/manual_blacklist.csv')

BLACKLIST_COLUMNS = [
    {'field': 'ticker', 'headerName': 'Ticker', 'width': 140},
    {'field': 'reason', 'headerName': 'Reason', 'flex': 1},
    {'field': 'blacklisted_on', 'headerName': 'Blacklisted On', 'width': 140},
]


def read_blacklist(path: Path = BLACKLIST_PATH) -> list[dict]:
    """Every manually blacklisted ticker with its reason and date, or an empty list if none yet."""
    if not path.exists():
        return []
    try:
        return pd.read_csv(path, keep_default_na=False).to_dict('records')
    except EmptyDataError:
        return []


def blacklist_review_card() -> dmc.Card:
    """Manually blacklisted tickers grid + an Unblacklist action."""
    return dmc.Card([
        dmc.Group([dmc.Text('Blacklisted tickers', fw=700, size='sm'),
                  dmc.Text('Select tickers and click Unblacklist to let a future scan offer them again.',
                          size='xs', c='dimmed')], gap=8, mb=4),
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
