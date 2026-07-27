"""Shared pytest setup for the apps test suite.

Every `apps/*/page.py` module calls `dash.register_page()` at import time, which requires a Dash
app to already exist in the process; without it, importing a page module directly (as tests do to
reach its pure helper functions) fails with `PageError: register_page() must be called after app
instantiation`. Instantiating one throwaway app here, before any test module imports a page,
satisfies that requirement for the whole session.
"""

import dash

dash.Dash(__name__, use_pages=True, pages_folder='')
