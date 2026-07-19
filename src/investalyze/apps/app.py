"""Multi-page Dash app: control panel + screener, one server, shared shell with a light/dark toggle.

Run with:  .venv/bin/python -m investalyze.apps  then open http://127.0.0.1:8050
"""

import logging

import dash
import dash_mantine_components as dmc
from dash import Dash, Input, Output, clientside_callback, dcc, html
from dash_ag_grid import themes as ag_grid_themes
from dash_iconify import DashIconify

# suppress_callback_exceptions: without it, Dash eagerly builds every registered page's layout
# (not just the requested one) on the first request, to validate callback IDs. Since the screener's
# layout hits the DB, that would make the control panel's first load fail whenever the DB happens
# to be locked by a running job: exactly when the panel most needs to still come up.
# dash_ag_grid ships no CSS of its own (AG Grid's newer Theming API replaces it); the legacy
# ag-theme-alpine[-dark] classes used across the grids need this stylesheet to render at all.
# update_title=None: the control panel polls every second to drive its live job log, which would
# otherwise flash the browser tab title to 'Updating...' on every tick.
app = Dash(__name__, use_pages=True, pages_folder='', suppress_callback_exceptions=True,
          external_stylesheets=[ag_grid_themes.BASE, ag_grid_themes.ALPINE], update_title=None)

# Registers each page (dash.register_page at import time) before the layout below reads dash.page_registry.
from investalyze.apps.control_panel import page as control_panel_page  # noqa: E402,F401
from investalyze.apps.data_quality import page as data_quality_page  # noqa: E402,F401
from investalyze.apps.screener import page as screener_page  # noqa: E402,F401
from investalyze.apps.ticker import page as ticker_page  # noqa: E402,F401

_NAV_ICONS = {'/': 'tabler:layout-dashboard', '/screener': 'tabler:list-search', '/quality': 'tabler:shield-check',
              '/ticker': 'tabler:chart-candle'}


def nav_link(page: dict) -> dmc.NavLink:
    """One sidebar link for a registered Dash page."""
    icon = _NAV_ICONS.get(page['path'], 'tabler:file')
    return dmc.NavLink(label=page['name'], href=page['path'], leftSection=DashIconify(icon=icon), variant='filled')


theme_switch = dmc.Switch(
    id='theme-switch', checked=True, persistence=True, size='md',
    onLabel=DashIconify(icon='tabler:moon', width=14), offLabel=DashIconify(icon='tabler:sun', width=14),
    style={'marginTop': 'auto'},
)

app.layout = dmc.MantineProvider(
    id='mantine-provider',
    forceColorScheme='dark',
    children=html.Div([
        dcc.Location(id='url'),
        dmc.AppShell(
            [
                dmc.AppShellNavbar(
                    [nav_link(p) for p in dash.page_registry.values()] + [theme_switch],
                    p='sm', style={'display': 'flex', 'flexDirection': 'column'},
                ),
                dmc.AppShellMain(dash.page_container),
            ],
            navbar={'width': 220, 'breakpoint': 'sm'},
            padding='md',
        ),
    ], style={'height': '100vh'}),
)

clientside_callback(
    "(dark) => dark ? 'dark' : 'light'",
    Output('mantine-provider', 'forceColorScheme'),
    Input('theme-switch', 'checked'),
)

class _QuietSuccessfulRequests(logging.Filter):
    """Drop werkzeug's access log line for successful requests; errors and startup messages still show.

    The control panel's 1s poll would otherwise print a request line every second, burying anything
    else in the terminal.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        parts = record.getMessage().rsplit(' ', 2)
        return not (len(parts) == 3 and parts[1] in ('200', '304'))


# Set up here, not behind `if __name__ == '__main__'`: `python -m investalyze.apps` runs this
# module's __main__.py, which imports `app` from here and calls app.run() itself.
logging.getLogger('werkzeug').addFilter(_QuietSuccessfulRequests())

if __name__ == '__main__':
    app.run(debug=True, dev_tools_silence_routes_logging=False)
