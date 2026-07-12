"""Multi-page Dash app: control panel + ticker selector, one server, shared dark shell.

Run with:  .venv/bin/python -m investalyze.apps  then open http://127.0.0.1:8050
"""

import dash
import dash_mantine_components as dmc
import plotly.io as pio
from dash import Dash, dcc, html
from dash_iconify import DashIconify

pio.templates.default = 'plotly_dark'

# suppress_callback_exceptions: without it, Dash eagerly builds every registered page's layout
# (not just the requested one) on the first request, to validate callback IDs. Since the ticker
# selector's layout hits the DB, that would make the control panel's first load fail whenever the
# DB happens to be locked by a running job — exactly when the panel most needs to still come up.
app = Dash(__name__, use_pages=True, pages_folder='', suppress_callback_exceptions=True)

# Registers each page (dash.register_page at import time) before the layout below reads dash.page_registry.
from investalyze.apps.control_panel import page as control_panel_page  # noqa: E402,F401
from investalyze.apps.screener import page as screener_page  # noqa: E402,F401
from investalyze.apps.ticker_selector import page as ticker_selector_page  # noqa: E402,F401

_NAV_ICONS = {'/': 'tabler:layout-dashboard', '/tickers': 'tabler:list-search', '/screener': 'tabler:filter'}


def nav_link(page: dict) -> dmc.NavLink:
    """One sidebar link for a registered Dash page."""
    icon = _NAV_ICONS.get(page['path'], 'tabler:file')
    return dmc.NavLink(label=page['name'], href=page['path'], leftSection=DashIconify(icon=icon), variant='filled')


app.layout = dmc.MantineProvider(
    forceColorScheme='dark',
    children=html.Div([
        dcc.Location(id='url'),
        dmc.AppShell(
            [
                dmc.AppShellNavbar(
                    [nav_link(p) for p in dash.page_registry.values()],
                    p='sm',
                ),
                dmc.AppShellMain(dash.page_container),
            ],
            navbar={'width': 220, 'breakpoint': 'sm'},
            padding='md',
        ),
    ], style={'height': '100vh'}),
)

if __name__ == '__main__':
    app.run(debug=False)
