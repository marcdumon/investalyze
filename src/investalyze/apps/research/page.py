"""Factors page: the factor library, factor definitions, diagnostics and IC analysis.

The factor library on the left lists every factor with its research status. The right side holds the
selected factor's definition, the data fields its expression can use, its diagnostics and its IC
analysis. Results are cached under `data/cache` by factor fingerprint, so editing a factor clears the
results computed from its previous definition.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from investalyze.apps import components, theming
from investalyze.apps.research import data, logic
from investalyze.research import factors

BUSY = 'The database is busy with a running job. Try again once it finishes.'
NO_SELECTION = 'Select a factor, or define a new one.'
TEXT_COLUMNS = ('Ticker', 'sector', 'industry', 'name')

_T_FORMAT = {'function': 'params.value == null ? "" : d3.format("+.2f")(params.value)'}
_RATIO_FORMAT = {'function': 'params.value == null ? "" : d3.format(".2f")(params.value)'}

LIBRARY_COLUMNS = [
    {'field': 'name', 'headerName': 'Factor', 'flex': 1, 'minWidth': 150},
    {'field': 'status', 'headerName': 'Status', 'width': 92},
    {'field': 't_stat', 'headerName': 't-stat', 'width': 72, 'type': 'rightAligned', 'valueFormatter': _T_FORMAT},
    {'field': 'oos_t_stat', 'headerName': 'OOS t', 'width': 72, 'type': 'rightAligned', 'valueFormatter': _T_FORMAT},
    {'field': 'sharpe', 'headerName': 'Sharpe', 'width': 76, 'type': 'rightAligned', 'valueFormatter': _RATIO_FORMAT},
]

YEAR_COLUMNS = [
    {'field': 'year', 'headerName': 'Year', 'width': 80},
    {'field': 'mean_ic', 'headerName': 'Mean IC', 'width': 110, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.4f")(params.value)'}},
    {'field': 't_stat', 'headerName': 't-stat', 'width': 90, 'type': 'rightAligned', 'valueFormatter': _T_FORMAT},
    {'field': 'n_periods', 'headerName': 'Dates', 'width': 80, 'type': 'rightAligned'},
]


DISTRIBUTION_COLUMNS = [
    {'field': 'statistic', 'headerName': 'Statistic', 'width': 110},
    {'field': 'value', 'headerName': 'Value', 'flex': 1, 'type': 'rightAligned'},
]

DECAY_COLUMNS = [
    {'field': 'horizon', 'headerName': 'Horizon (days)', 'width': 130},
    {'field': 'mean_ic', 'headerName': 'Mean IC', 'width': 110, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.4f")(params.value)'}},
    {'field': 't_stat', 'headerName': 't-stat', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.2f")(params.value)'}},
    {'field': 'names_median', 'headerName': 'Names', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.0f")(params.value)'}},
]

QUANTILE_COLUMNS = [
    {'field': 'Quantile', 'headerName': 'Quantile', 'width': 100},
    {'field': 'mean_return', 'headerName': 'Mean return', 'width': 130, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.2%")(params.value)'}},
    {'field': 'median_return', 'headerName': 'Median return', 'width': 140, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.2%")(params.value)'}},
    {'field': 'names', 'headerName': 'Names', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.0f")(params.value)'}},
]


def _grid(grid_id: str, columns: list[dict], height: int) -> dag.AgGrid:
    """One of the page's result grids."""
    return components.grid(grid_id, columns, height, row_height=28)


def _library_panel() -> dmc.Paper:
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Group(justify='space-between', children=[
            dmc.Title('Factor library', order=5),
            dmc.Button('New factor', id='rs-btn-new', size='xs', variant='light'),
        ]),
        components.grid('rs-library', LIBRARY_COLUMNS, 430, selection=components.SINGLE_ROW, row_id='params.data.name', row_height=28),
        dmc.Text('', id='rs-library-description', size='xs', c='dimmed'),
    ]))


def _definition_panel() -> dmc.Paper:
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Title('Factor definition', order=5),
        dmc.Group(grow=True, children=[
            dmc.TextInput(id='rs-name', label='Name', placeholder='op_margin_sector_neutral_rd'),
            dmc.Select(id='rs-status', label='Status', value=factors.Status.CANDIDATE,
                       data=[{'value': status, 'label': status.title()} for status in factors.Status]),
            dmc.Select(id='rs-direction', label='Direction', value='higher',
                       data=[{'value': 'higher', 'label': 'Higher is better'},
                             {'value': 'lower', 'label': 'Lower is better'}]),
        ]),
        dmc.TextInput(id='rs-description', label='Rationale',
                      placeholder='Sector-neutral operating margin, restricted to names with R&D above 2% of revenue.'),
        dmc.Textarea(id='rs-expression', label='Expression', autosize=True, minRows=2, maxRows=6,
                     placeholder='rank(mask(neutralize(op_margin, by=sector), rd_intensity > 0.02))'),
        html.Div(id='rs-validation'),
        dmc.Group(children=[
            dmc.Button('Run diagnostics', id='rs-btn-diagnostics', size='xs'),
            dmc.Button('Run IC analysis', id='rs-btn-ic', size='xs', variant='light'),
            dmc.Anchor(dmc.Button('Backtest this factor', size='xs', variant='light'), id='rs-backtest-link',
                       href='/backtest', underline='never'),
            dmc.Button('Save', id='rs-btn-save', size='xs', color='green', variant='light'),
            dmc.Button('Delete', id='rs-btn-delete', size='xs', color='red', variant='subtle'),
        ]),
        html.Div(id='rs-saved'),
    ]))


def _fields_panel() -> dmc.Paper:
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Group(justify='space-between', children=[
            dmc.Title('Data fields', order=5),
            dmc.Text('', id='rs-fields-count', size='xs', c='dimmed'),
        ]),
        dmc.TextInput(id='rs-fields-search', placeholder='Search data fields', debounce=300),
        _grid('rs-fields', logic.FIELD_COLUMNS, 260),
    ]))


def _diagnostics_panel() -> dmc.Paper:
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Title('Diagnostics', order=5),
        dcc.Loading(html.Div(id='rs-diagnostics-notes', children=components.note(NO_SELECTION)), type='dot'),
        dmc.Group(grow=True, align='flex-start', children=[
            dmc.Stack(gap=4, children=[
                dmc.Text('Coverage', size='xs', fw=600),
                dcc.Graph(id='rs-coverage', config={'displayModeBar': False}, style={'height': 290}),
            ]),
            dmc.Stack(gap=4, children=[
                dmc.Text('Distribution', size='xs', fw=600),
                _grid('rs-distribution', DISTRIBUTION_COLUMNS, 290),
            ]),
        ]),
        dmc.Group(grow=True, align='flex-start', children=[
            dmc.Stack(gap=4, children=[
                dmc.Text('Top names', size='xs', fw=600),
                _grid('rs-top', [{'field': 'Ticker', 'width': 90}], 210),
            ]),
            dmc.Stack(gap=4, children=[
                dmc.Text('Bottom names', size='xs', fw=600),
                _grid('rs-bottom', [{'field': 'Ticker', 'width': 90}], 210),
            ]),
        ]),
    ]))


def _ic_panel() -> dmc.Paper:
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Title('IC analysis', order=5),
        dcc.Loading(html.Div(id='rs-ic-notes', children=components.note('Save the factor, then run IC analysis.')), type='dot'),
        dmc.Group(grow=True, align='flex-start', children=[
            dmc.Stack(gap=4, children=[
                dmc.Text('IC decay', size='xs', fw=600),
                _grid('rs-decay', DECAY_COLUMNS, 190),
            ]),
            dmc.Stack(gap=4, children=[
                dmc.Text('Quantile returns', size='xs', fw=600),
                _grid('rs-quantiles', QUANTILE_COLUMNS, 190),
            ]),
        ]),
        dmc.Stack(gap=4, children=[
            dmc.Text('IC by year', size='xs', fw=600),
            _grid('rs-years', YEAR_COLUMNS, 300),
        ]),
    ]))


def layout(**_query: str) -> html.Div:
    return html.Div([
        dmc.Group(justify='space-between', mb='sm', children=[
            dmc.Title('Factors', order=3),
            dmc.Text('Define a factor, run diagnostics, run IC analysis.', size='sm', c='dimmed'),
        ]),
        dmc.Grid(gutter='sm', children=[
            dmc.GridCol(span=4, children=_library_panel()),
            dmc.GridCol(span=8, children=dmc.Stack(gap='sm', children=[
                _definition_panel(), _fields_panel(), _diagnostics_panel(), _ic_panel()])),
        ]),
    ], style={'padding': '4px'})


@callback(Output('rs-library', 'rowData'), Input('rs-saved', 'children'), Input('rs-diagnostics-notes', 'children'),
          Input('rs-ic-notes', 'children'))
def load_library(_saved: components.Children, _diagnostics: components.Children, _ic: components.Children) -> list[dict]:
    """The factor library with recorded results, reloaded after a save, delete or new result."""
    return logic.library_rows(data.library(), data.summaries())


@callback(Output('rs-fields', 'rowData'), Output('rs-fields-count', 'children'),
          Input('rs-fields-search', 'value'))
def load_fields(search: str | None) -> tuple:
    """The data fields, filtered by the search box."""
    try:
        rows = data.field_rows(search)
    except duckdb.Error:
        return [], 'Database busy'
    return rows, f'{len(rows)} fields'


@callback(Output('rs-name', 'value'), Output('rs-description', 'value'),
          Output('rs-expression', 'value'), Output('rs-direction', 'value'),
          Output('rs-status', 'value'), Output('rs-library-description', 'children'),
          Input('rs-library', 'selectedRows'), Input('rs-btn-new', 'n_clicks'))
def fill_definition(selected: list[dict] | dict | None, _new: int | None) -> tuple:
    """Load the selected factor into the definition form, or clear it for a new factor."""
    if ctx.triggered_id == 'rs-btn-new':
        return '', '', '', 'higher', factors.Status.CANDIDATE, NO_SELECTION
    if not selected:
        raise PreventUpdate
    row = selected[0]
    return (row['name'], row['description'], row['expression'], row['direction'], row['status'],
            row['description'] or 'No rationale recorded.')


@callback(Output('rs-validation', 'children'), Input('rs-expression', 'value'))
def validate(expression: str | None) -> components.Children:
    """Validate the expression as it is typed."""
    try:
        known = data.known_names()
    except duckdb.Error:
        return components.note(BUSY)
    ok, message = logic.validate_expression(expression or '', known)
    return components.note(message, 'green' if ok else 'orange')


def _empty_outputs(message: dmc.Alert, dark: bool) -> tuple:
    ticker_only = [{'field': 'Ticker'}]
    return message, components.empty_figure(dark), [], [], ticker_only, [], ticker_only


@callback(Output('rs-diagnostics-notes', 'children'), Output('rs-coverage', 'figure'),
          Output('rs-distribution', 'rowData'), Output('rs-top', 'rowData'),
          Output('rs-top', 'columnDefs'), Output('rs-bottom', 'rowData'),
          Output('rs-bottom', 'columnDefs'),
          Input('rs-btn-diagnostics', 'n_clicks'), State('rs-name', 'value'),
          State('rs-expression', 'value'), State('rs-direction', 'value'),
          State('theme-switch', 'checked'), prevent_initial_call=True)
def show_diagnostics(_clicks: int | None, name: str | None, expression: str | None, direction: str | None, dark: bool) -> tuple:
    """Run diagnostics on the sample and fill the diagnostics panel."""
    if not name or not expression:
        raise PreventUpdate
    try:
        result = data.run_diagnostics(name, expression, higher_is_better=direction == 'higher')
    except data.NoValuesError as error:
        return _empty_outputs(components.note(str(error), 'orange'), dark)
    except (ValueError, duckdb.Error) as error:
        return _empty_outputs(components.note(str(error), 'red'), dark)

    columns = []
    for column in result.top.columns:
        definition = {'field': column, 'flex': 1}
        if column not in TEXT_COLUMNS:
            definition['valueFormatter'] = {'function': 'd3.format(",.3f")(params.value)'}
        columns.append(definition)
    return (_diagnostics_notes(result), _coverage_figure(result.coverage, dark),
            logic.distribution_rows(result.distribution),
            result.top.to_dict('records'), columns, result.bottom.to_dict('records'), columns)


def _diagnostics_notes(result: data.Diagnostics) -> dmc.Stack:
    """The notes above the diagnostics tables: coverage, overlap, universe and turnover."""
    text, colour = logic.coverage_note(result.coverage)
    notes = [components.note(text, colour),
             components.note(f'Universe: top {result.universe_size:,} names by trailing dollar volume on each of '
                             f'{result.dates} month-ends.'),
             components.note(logic.turnover_note(result.autocorrelation))]
    found = logic.correlation_note(result.correlations)
    if found is not None:
        notes.insert(1, components.note(found[0], found[1]))
    return dmc.Stack(gap=4, children=notes)


def _coverage_figure(table: pd.DataFrame, dark: bool) -> go.Figure:
    """Names with a value per year, overlaid on the names in the universe."""
    figure = go.Figure()
    figure.add_bar(x=table['year'], y=table['universe'], name='Universe', marker_color='#8C8C8C', opacity=0.35)
    figure.add_bar(x=table['year'], y=table['names'], name='Covered', marker_color='#4C72B0')
    figure.update_layout(**theming.figure_layout(bool(dark)), barmode='overlay',
                         margin={'l': 40, 'r': 10, 't': 10, 'b': 30}, showlegend=True,
                         legend={'orientation': 'h', 'y': 1.1, 'x': 0})
    return figure


@callback(Output('rs-ic-notes', 'children'), Output('rs-decay', 'rowData'),
          Output('rs-quantiles', 'rowData'), Output('rs-years', 'rowData'),
          Input('rs-btn-ic', 'n_clicks'), State('rs-name', 'value'), prevent_initial_call=True)
def show_ic_analysis(_clicks: int | None, name: str | None) -> tuple:
    """Run IC analysis on a saved factor and fill the IC panel."""
    if not name:
        raise PreventUpdate
    if name not in data.library():
        return components.note('Save the factor before running IC analysis.', 'orange'), [], [], []
    try:
        result = data.run_tearsheet(name)
    except data.NoValuesError as error:
        return components.note(str(error), 'orange'), [], [], []
    except (ValueError, duckdb.Error) as error:
        return components.note(str(error), 'red'), [], [], []

    text, colour = logic.ic_note(result)
    notes = [components.note(text, colour)]
    holdout = logic.holdout_note(result.split)
    if holdout is not None:
        notes.append(components.note(holdout[0], holdout[1]))
    notes += [components.note(logic.ic_decay_note(result.decay, higher_is_better=result.higher_is_better)),
              components.note(f'Universe: top {data.RESEARCH_SIZE:,} names by trailing dollar volume on each month-end.'),
              components.note(logic.multiple_testing_note(result))]
    return (dmc.Stack(gap=4, children=notes), result.decay.to_dict('records'),
            result.quantiles.to_dict('records'), result.by_year.to_dict('records'))


@callback(Output('rs-backtest-link', 'href'), Input('rs-name', 'value'))
def backtest_link(name: str | None) -> str:
    """Open the Backtest page with this factor already selected."""
    return f'/backtest?factor={name}' if name else '/backtest'


@callback(Output('rs-saved', 'children'),
          Input('rs-btn-save', 'n_clicks'), Input('rs-btn-delete', 'n_clicks'),
          State('rs-name', 'value'), State('rs-description', 'value'),
          State('rs-expression', 'value'), State('rs-direction', 'value'),
          State('rs-status', 'value'), prevent_initial_call=True)
def save_factor(_save: int | None, _delete: int | None, name: str | None, description: str | None, expression: str | None,
                direction: str | None, status: str | None) -> components.Children:
    """Save or delete the factor, clearing cached results for its previous definition."""
    if not name:
        return components.note('A factor needs a name.', 'orange')
    if ctx.triggered_id == 'rs-btn-delete':
        data.delete(name)
        return components.note(f'Deleted {name} and its cached results.', 'gray')
    try:
        factor = factors.Factor(name=name, expression=expression or '', description=description or '',
                                higher_is_better=direction == 'higher', status=status)
        ok, message = logic.validate_expression(factor.expression, data.known_names() | {name})
        if not ok:
            return components.note(message, 'red')
        data.save(factor)
    except (ValueError, duckdb.Error) as error:
        return components.note(str(error), 'red')
    return components.note(f'Saved {name}. Cached results for the previous definition were cleared.', 'green')


dash.register_page(__name__, path='/factors', name='Factors', layout=layout)

# every grid on the page follows the theme switch
for _grid_id in ('rs-library', 'rs-fields', 'rs-distribution', 'rs-top', 'rs-bottom', 'rs-decay', 'rs-quantiles',
                 'rs-years'):
    clientside_callback(
        "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
        Output(_grid_id, 'className'),
        Input('theme-switch', 'checked'),
    )
