"""Factors page: write an idea, find out whether it is sane, then whether it predicts anything.

One page, because the work is one loop. The library on the left is the decision surface: every idea
with where it stands. The workbench on the right holds the expression, the palette it is written
against, and what the idea looks like once computed.

Nothing on this page stores an answer a user has to manage. Values and scores are cached under
`data/cache`, keyed by a fingerprint reaching through the idea's dependencies, so redefining an idea
throws away everything computed from it without anyone being asked.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import duckdb
import plotly.graph_objects as go
from dash import Input, Output, State, callback, clientside_callback, ctx, html
from dash.exceptions import PreventUpdate

from investalyze.apps import theming
from investalyze.apps.research import data, logic
from investalyze.research import ideas

BUSY = 'The database is busy with a running job. Try again once it finishes.'
NO_SELECTION = 'Pick an idea on the left, or write a new one.'

LIBRARY_COLUMNS = [
    {'field': 'name', 'headerName': 'Idea', 'flex': 1, 'minWidth': 150},
    {'field': 'status', 'headerName': 'Status', 'width': 88},
]

DISTRIBUTION_COLUMNS = [
    {'field': 'measure', 'headerName': 'Measure', 'width': 110},
    {'field': 'value', 'headerName': 'Value', 'flex': 1, 'type': 'rightAligned'},
]

DECAY_COLUMNS = [
    {'field': 'horizon', 'headerName': 'Sessions', 'width': 100},
    {'field': 'mean_ic', 'headerName': 'Mean IC', 'width': 110, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.4f")(params.value)'}},
    {'field': 't_stat', 'headerName': 't', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.2f")(params.value)'}},
    {'field': 'names_median', 'headerName': 'Names', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.0f")(params.value)'}},
]

QUANTILE_COLUMNS = [
    {'field': 'Quantile', 'headerName': 'Bucket', 'width': 90},
    {'field': 'mean_return', 'headerName': 'Mean return', 'width': 130, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.2%")(params.value)'}},
    {'field': 'median_return', 'headerName': 'Median return', 'width': 140, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format("+.2%")(params.value)'}},
    {'field': 'names', 'headerName': 'Names', 'width': 90, 'type': 'rightAligned',
     'valueFormatter': {'function': 'd3.format(",.0f")(params.value)'}},
]


def _grid(grid_id: str, columns: list, height: int, *, selectable: bool = False) -> dag.AgGrid:
    """One AG Grid with the page's shared styling."""
    options = {'theme': 'legacy', 'rowHeight': 28, 'headerHeight': 30}
    if selectable:
        options['rowSelection'] = {'mode': 'singleRow', 'checkboxes': False, 'enableClickSelection': True}
    return dag.AgGrid(
        id=grid_id, columnDefs=columns, rowData=[],
        defaultColDef={'sortable': True, 'resizable': True, 'minWidth': 70},
        dashGridOptions=options, style={'height': height},
        className='ag-theme-alpine-dark', columnSize='sizeToFit',
    )


def _note(text: str, colour: str = 'gray') -> dmc.Alert:
    return dmc.Alert(text, color=colour, variant='light')


def _library_panel() -> dmc.Paper:
    """Every idea and where it stands, which is what a keep or kill decision reads."""
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Group(justify='space-between', children=[
            dmc.Title('Your ideas', order=5),
            dmc.Button('New', id='rs-btn-new', size='xs', variant='light'),
        ]),
        _grid('rs-library', LIBRARY_COLUMNS, 430, selectable=True),
        dmc.Text('', id='rs-library-description', size='xs', c='dimmed'),
    ]))


def _editor_panel() -> dmc.Paper:
    """The idea itself: what it is called, what it believes, and how it is computed."""
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Title('The idea', order=5),
        dmc.Group(grow=True, children=[
            dmc.TextInput(id='rs-name', label='Name', placeholder='disciplined_margin'),
            dmc.Select(id='rs-status', label='Status', value=ideas.DRAFT,
                       data=[{'value': status, 'label': status.title()} for status in ideas.STATUSES]),
            dmc.Select(id='rs-direction', label='Good end', value='high',
                       data=[{'value': 'high', 'label': 'High is good'},
                             {'value': 'low', 'label': 'Low is good'}]),
        ]),
        dmc.TextInput(id='rs-description', label='What you believe',
                      placeholder='Sector-relative margin, among names still spending on research.'),
        dmc.Textarea(id='rs-expression', label='Expression', autosize=True, minRows=2, maxRows=6,
                     placeholder='rank(mask(neutralize(op_margin, by=sector), rd_intensity > 0.02))'),
        html.Div(id='rs-validation'),
        dmc.Group(children=[
            dmc.Button('Check it', id='rs-btn-check', size='xs'),
            dmc.Button('Does it predict?', id='rs-btn-predict', size='xs', variant='light'),
            dmc.Button('Save', id='rs-btn-save', size='xs', color='green', variant='light'),
            dmc.Button('Delete', id='rs-btn-delete', size='xs', color='red', variant='subtle'),
        ]),
        html.Div(id='rs-saved'),
    ]))


def _palette_panel() -> dmc.Paper:
    """The raw fields an expression may name, which is what makes composing possible at all."""
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Group(justify='space-between', children=[
            dmc.Title('The palette', order=5),
            dmc.Text('', id='rs-palette-count', size='xs', c='dimmed'),
        ]),
        dmc.TextInput(id='rs-palette-search', placeholder='Search the fields, for example: research',
                      debounce=300),
        _grid('rs-palette', logic.FIELD_COLUMNS, 260),
    ]))


def _check_panel() -> dmc.Paper:
    """Level one: everything that can kill an idea before anything expensive is asked of it."""
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Title('Is it sane?', order=5),
        html.Div(id='rs-check-notes', children=_note(NO_SELECTION)),
        dmc.Group(grow=True, align='flex-start', children=[
            dmc.Stack(gap=4, children=[
                dmc.Text('Coverage over time', size='xs', fw=600),
                dash.dcc.Graph(id='rs-coverage', config={'displayModeBar': False},
                               style={'height': 290}),
            ]),
            dmc.Stack(gap=4, children=[
                dmc.Text('Distribution', size='xs', fw=600),
                _grid('rs-distribution', DISTRIBUTION_COLUMNS, 290),
            ]),
        ]),
        dmc.Group(grow=True, align='flex-start', children=[
            dmc.Stack(gap=4, children=[
                dmc.Text('Highest now', size='xs', fw=600),
                _grid('rs-top', [{'field': 'Ticker', 'width': 90}], 210),
            ]),
            dmc.Stack(gap=4, children=[
                dmc.Text('Lowest now', size='xs', fw=600),
                _grid('rs-bottom', [{'field': 'Ticker', 'width': 90}], 210),
            ]),
        ]),
    ]))


def _predict_panel() -> dmc.Paper:
    """Level two: does the idea forecast the returns that followed it."""
    return dmc.Paper(withBorder=True, p='sm', children=dmc.Stack(gap='xs', children=[
        dmc.Title('Does it predict?', order=5),
        html.Div(id='rs-verdict', children=_note('Press "Does it predict?" once an idea is saved.')),
        dmc.Group(grow=True, align='flex-start', children=[
            dmc.Stack(gap=4, children=[
                dmc.Text('Edge by horizon', size='xs', fw=600),
                _grid('rs-decay', DECAY_COLUMNS, 190),
            ]),
            dmc.Stack(gap=4, children=[
                dmc.Text('Return by score bucket', size='xs', fw=600),
                _grid('rs-quantiles', QUANTILE_COLUMNS, 190),
            ]),
        ]),
    ]))


def layout() -> html.Div:
    return html.Div([
        dmc.Group(justify='space-between', mb='sm', children=[
            dmc.Title('Factors', order=3),
            dmc.Text('Write an idea, find out whether it is sane, then whether it works.',
                     size='sm', c='dimmed'),
        ]),
        dmc.Grid(gutter='sm', children=[
            dmc.GridCol(span=3, children=_library_panel()),
            dmc.GridCol(span=9, children=dmc.Stack(gap='sm', children=[
                _editor_panel(), _palette_panel(), _check_panel(), _predict_panel()])),
        ]),
    ], style={'padding': '4px'})


@callback(Output('rs-library', 'rowData'), Input('rs-saved', 'children'))
def load_library(_saved):
    """The library, reloaded whenever an idea is written or removed."""
    return logic.library_rows(data.library())


@callback(Output('rs-palette', 'rowData'), Output('rs-palette-count', 'children'),
          Input('rs-palette-search', 'value'))
def load_palette(search):
    """The raw fields, narrowed by whatever is typed in the search box."""
    try:
        rows = data.field_rows(search)
    except duckdb.Error:
        return [], 'database busy'
    return rows, f'{len(rows)} fields'


@callback(Output('rs-name', 'value'), Output('rs-description', 'value'),
          Output('rs-expression', 'value'), Output('rs-direction', 'value'),
          Output('rs-status', 'value'), Output('rs-library-description', 'children'),
          Input('rs-library', 'selectedRows'), Input('rs-btn-new', 'n_clicks'))
def fill_editor(selected, _new):
    """Load the selected idea into the editor, or clear it for a new one."""
    if ctx.triggered_id == 'rs-btn-new' or not selected:
        return '', '', '', 'high', ideas.DRAFT, NO_SELECTION
    row = selected[0]
    direction = 'high' if row['direction'] == 'high' else 'low'
    return (row['name'], row['description'], row['expression'], direction, row['status'],
            row['description'] or 'No description yet.')


@callback(Output('rs-validation', 'children'), Input('rs-expression', 'value'))
def validate(expression):
    """Say whether the expression is usable while it is being written, which costs no query."""
    try:
        known = data.known_names()
    except duckdb.Error:
        return _note(BUSY)
    ok, message = logic.validate_expression(expression or '', known)
    return _note(message, 'green' if ok else 'orange')


@callback(Output('rs-check-notes', 'children'), Output('rs-coverage', 'figure'),
          Output('rs-distribution', 'rowData'), Output('rs-top', 'rowData'),
          Output('rs-top', 'columnDefs'), Output('rs-bottom', 'rowData'),
          Output('rs-bottom', 'columnDefs'),
          Input('rs-btn-check', 'n_clicks'), State('rs-name', 'value'),
          State('rs-expression', 'value'), State('rs-direction', 'value'),
          State('theme-switch', 'checked'), prevent_initial_call=True)
def check_idea(_clicks, name, expression, direction, dark):
    """Compute the draft over a sample and report everything level one knows about it."""
    if not name or not expression:
        raise PreventUpdate
    try:
        result = data.preview(name, expression, higher_is_better=direction == 'high')
    except (ValueError, duckdb.Error) as error:
        return _note(str(error), 'red'), _empty_figure(dark), [], [], [{'field': 'Ticker'}], [], [{'field': 'Ticker'}]

    if 'error' in result:
        return (_note(result['error'], 'orange'), _empty_figure(dark), [], [],
                [{'field': 'Ticker'}], [], [{'field': 'Ticker'}])

    notes = _check_notes(result)
    top, bottom = result['extremes']['top'], result['extremes']['bottom']
    columns = [{'field': column, 'flex': 1,
                'valueFormatter': ({'function': 'd3.format(",.3f")(params.value)'}
                                   if column not in ('Ticker', 'sector', 'industry', 'name') else None)}
               for column in top.columns]
    return (notes, _coverage_figure(result['coverage'], dark),
            logic.distribution_rows(result['distribution']),
            top.to_dict('records'), columns, bottom.to_dict('records'), columns)


def _check_notes(result: dict) -> dmc.Stack:
    """The sentences that decide whether an idea is worth taking further."""
    text, colour = logic.coverage_headline(result['coverage'])
    sample = result['sample']
    notes = [_note(text, colour),
             _note(f'Sampled over {sample["names"]:,} of the most traded names on {sample["dates"]} dates.'),
             _note(logic.turnover_note(result['autocorrelation']))]
    found = logic.similarity(result['correlations'])
    if found is not None:
        notes.insert(1, _note(found[0], found[1]))
    return dmc.Stack(gap=4, children=notes)


def _empty_figure(dark: bool) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(**theming.figure_layout(bool(dark)), margin={'l': 40, 'r': 10, 't': 10, 'b': 30})
    return figure


def _coverage_figure(table, dark: bool) -> go.Figure:
    """How many names carry a value each year, against how many were there to carry one."""
    figure = go.Figure()
    figure.add_bar(x=table['year'], y=table['universe'], name='In the sample',
                   marker_color='#8C8C8C', opacity=0.35)
    figure.add_bar(x=table['year'], y=table['names'], name='With a value', marker_color='#4C72B0')
    figure.update_layout(**theming.figure_layout(bool(dark)), barmode='overlay',
                         margin={'l': 40, 'r': 10, 't': 10, 'b': 30}, showlegend=True,
                         legend={'orientation': 'h', 'y': 1.1, 'x': 0})
    figure.update_yaxes(title=None)
    return figure


@callback(Output('rs-verdict', 'children'), Output('rs-decay', 'rowData'),
          Output('rs-quantiles', 'rowData'),
          Input('rs-btn-predict', 'n_clicks'), State('rs-name', 'value'),
          prevent_initial_call=True)
def predict_idea(_clicks, name):
    """Score a saved idea against the returns that followed it."""
    if not name:
        raise PreventUpdate
    if name not in data.library():
        return _note('Save the idea first, then it can be scored.', 'orange'), [], []
    try:
        answer = data.verdict(name)
    except (ValueError, duckdb.Error) as error:
        return _note(str(error), 'red'), [], []
    if 'error' in answer:
        return _note(answer['error'], 'orange'), [], []

    text, colour = logic.verdict_headline(answer)
    notes = dmc.Stack(gap=4, children=[
        _note(text, colour),
        _note(logic.decay_note(answer['decay'])),
        _note(logic.multiple_testing_note(answer)),
    ])
    return notes, answer['decay'].to_dict('records'), answer['quantiles'].to_dict('records')


@callback(Output('rs-saved', 'children'),
          Input('rs-btn-save', 'n_clicks'), Input('rs-btn-delete', 'n_clicks'),
          State('rs-name', 'value'), State('rs-description', 'value'),
          State('rs-expression', 'value'), State('rs-direction', 'value'),
          State('rs-status', 'value'), prevent_initial_call=True)
def save_idea(_save, _delete, name, description, expression, direction, status):
    """Write the idea, or remove it, and throw away whatever the old definition had computed."""
    if not name:
        return _note('An idea needs a name before it can be saved.', 'orange')
    if ctx.triggered_id == 'rs-btn-delete':
        data.delete(name)
        return _note(f'Deleted {name}, and every answer computed from it.', 'gray')
    try:
        idea = ideas.Idea(name=name, expression=expression or '', description=description or '',
                          higher_is_better=direction == 'high', status=status)
        known = data.known_names() | {name}
        ok, message = logic.validate_expression(idea.expression, known)
        if not ok:
            return _note(message, 'red')
        data.save(idea)
    except (ValueError, duckdb.Error) as error:
        return _note(str(error), 'red')
    return _note(f'Saved {name}. Anything computed from its previous definition has been discarded.',
                 'green')


dash.register_page(__name__, path='/factors', name='Factors', layout=layout)

# every grid on the page follows the switch, not just the first one
for _grid_id in ('rs-library', 'rs-palette', 'rs-distribution', 'rs-top', 'rs-bottom',
                 'rs-decay', 'rs-quantiles'):
    clientside_callback(
        "(dark) => dark ? 'ag-theme-alpine-dark' : 'ag-theme-alpine'",
        Output(_grid_id, 'className'),
        Input('theme-switch', 'checked'),
    )
