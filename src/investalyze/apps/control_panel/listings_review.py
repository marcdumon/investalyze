"""Control-panel section: review the last listings scan and add tickers still needing a decision.

Reads `data/yahoo/state/listing_candidates.csv` (written by `investalyze.ingest.listings.scan`,
which already adds 'new' candidates to `ticker.csv` on its own). Reactivated-safe candidates are
added to `ticker.csv` directly; reactivated-collision candidates are renamed first (old company's
data moved to the row's editable "suggested ticker" key), then added, all from one Add action.
"""

from pathlib import Path

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash_iconify import DashIconify
from pandas.errors import EmptyDataError

CANDIDATES_PATH = Path('data/yahoo/state/listing_candidates.csv')

# raw classification value -> label matching the legend text in listings_card()
_CLASSIFICATION_LABELS = {
    'reactivated_safe': 'Reactivated (safe)',
    'reactivated_collision': 'Reactivated (collision)',
}

CANDIDATE_COLUMNS = [
    {'field': 'ticker', 'headerName': 'Ticker', 'checkboxSelection': True, 'headerCheckboxSelection': True, 'width': 140},
    {'field': 'market', 'headerName': 'Market', 'width': 90},
    {'field': 'company_name', 'headerName': 'Live Listing', 'flex': 1},
    {'field': 'classification_label', 'headerName': 'Classification', 'width': 190},
    {'field': 'previous_company_name', 'headerName': 'Previously Traded As', 'flex': 1},
    {'field': 'suggested_new_ticker', 'headerName': 'Suggested Ticker', 'width': 150, 'editable': True},
]


def _normalize_company_name(name: str) -> str:
    """Title-case every company name, regardless of how the source stored it."""
    return name.title()


def read_candidates(path: Path = CANDIDATES_PATH) -> list[dict]:
    """The last scan's candidates, or an empty list if no scan has run yet or none were found.

    Each row also gets a `classification_label` (the human-readable form shown in the grid); the
    raw `classification` field stays untouched, since request_run's Add logic matches on it.
    `company_name`/`previous_company_name` are title-cased, since source formatting is inconsistent
    (some records store an all-caps name, e.g. older SimFin data).
    """
    if not path.exists():
        return []
    try:
        rows = pd.read_csv(path, keep_default_na=False).to_dict('records')
    except EmptyDataError:
        return []
    for row in rows:
        row['classification_label'] = _CLASSIFICATION_LABELS.get(row['classification'], row['classification'])
        row['company_name'] = _normalize_company_name(row['company_name'])
        row['previous_company_name'] = _normalize_company_name(row['previous_company_name'])
    return rows


def listings_card() -> dmc.Card:
    """New-listings review card: scan-result grid + one Add action."""
    return dmc.Card([
        dmc.Group([dmc.Text('New listings review', fw=700, size='sm'),
                  dmc.Text('Run a scan above, then review candidates here.', size='xs', c='dimmed')], gap=8, mb=4),
        dmc.Text(
            "New tickers are added to ticker.csv automatically and don't show up below. This table "
            'only lists tickers with some stored data already on file for this symbol: Reactivated '
            "(safe) means it's the same company as the live listing. Reactivated (collision) means a "
            'different company now trades under this symbol than the one whose data is stored (see '
            'the "previously traded as" column); its data is moved to the "suggested ticker" key '
            '(editable, clear it to skip that row) before the new company is added. Select rows and '
            'click Add.',
            size='xs', c='dimmed', mb=8,
        ),
        dag.AgGrid(
            id='cp-listings-grid', columnDefs=CANDIDATE_COLUMNS, rowData=[],
            defaultColDef={'sortable': True, 'resizable': True},
            dashGridOptions={'animateRows': False, 'theme': 'legacy', 'rowSelection': 'multiple',
                             'suppressRowClickSelection': True},
            className='ag-theme-alpine-dark', style={'height': '260px', 'width': '100%'},
        ),
        dmc.Button('Add selected', id='cp-btn-listings-add', size='xs', color='orange', mt=8,
                   leftSection=DashIconify(icon='tabler:playlist-add')),
    ], withBorder=True, radius='md', padding='sm')
