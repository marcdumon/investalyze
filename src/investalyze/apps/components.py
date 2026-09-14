"""Building blocks the Dash pages share: data grids, inline notes and empty charts."""

from typing import Literal

import dash_ag_grid as dag
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash.development.base_component import Component

from investalyze.apps import theming

SINGLE_ROW = {'mode': 'singleRow', 'checkboxes': False, 'enableClickSelection': True}
# the checkbox column comes free with this selection mode; clicking the row still selects it
MULTI_ROW = {'mode': 'multiRow', 'checkboxes': True, 'headerCheckbox': True, 'enableClickSelection': True}

ColumnSize = Literal['autoSize', 'sizeToFit'] | None
# what a callback may put in a `children` property
Children = Component | list | str | None


def grid(grid_id: str, columns: list[dict], height: int, *, selection: dict | None = None, row_id: str | None = None,
         column_size: ColumnSize = 'sizeToFit', row_height: int | None = None, min_width: int | None = 70,
         filterable: bool = False) -> dag.AgGrid:
    """One AG Grid in the app's dark theme.

    Args:
        selection: `SINGLE_ROW`, `MULTI_ROW`, or None for a grid without row selection.
        row_id: a getRowId expression, which keeps the selection when the rows reload.
        column_size: how columns fit the grid; None keeps the widths the columns declare.
        row_height: a row height in pixels, with the header two pixels taller.
    """
    options: dict = {'theme': 'legacy'}   # without this the className theme below is ignored
    if selection:
        options['rowSelection'] = selection
    if row_height:
        options['rowHeight'] = row_height
        options['headerHeight'] = row_height + 2
    default_column: dict = {'sortable': True, 'resizable': True}
    if min_width:
        default_column['minWidth'] = min_width
    if filterable:
        default_column['filter'] = True
    extra: dict = {}
    if row_id:
        extra['getRowId'] = row_id
    if column_size == 'autoSize':
        extra |= {'columnSize': 'autoSize', 'columnSizeOptions': {'skipHeader': False}}
    elif column_size == 'sizeToFit':
        extra['columnSize'] = 'sizeToFit'
    return dag.AgGrid(id=grid_id, columnDefs=columns, rowData=[], defaultColDef=default_column, dashGridOptions=options,
                      style={'height': height}, className='ag-theme-alpine-dark', **extra)


def note(text: str, colour: str = 'gray') -> dmc.Alert:
    """A short inline note for an empty, unavailable or refused panel."""
    return dmc.Alert(text, color=colour, variant='light')


def empty_figure(dark: bool, message: str | None = None) -> go.Figure:
    """A blank chart in the current theme, carrying one centred message when given."""
    figure = go.Figure()
    figure.update_layout(**theming.figure_layout(bool(dark)), margin={'l': 40, 'r': 20, 't': 20, 'b': 30})
    if message:
        figure.add_annotation(text=message, showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        figure.update_layout(xaxis={'visible': False}, yaxis={'visible': False})
    return figure
