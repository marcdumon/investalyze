"""Shared pytest setup for the apps test suite.

Every `apps/*/page.py` module calls `dash.register_page()` at import time, which requires a Dash
app to already exist in the process; without it, importing a page module directly (as tests do to
reach its pure helper functions) fails with `PageError: register_page() must be called after app
instantiation`. Instantiating one throwaway app here, before any test module imports a page,
satisfies that requirement for the whole session.

The factors override is set before any import too. `analysis/factors.py` merges the definitions in
the repo's factors.toml into FACTORS at import time, so without this a factor someone adds to their
own config would change what the suite asserts. Tests that exercise user-defined factors build their
definitions explicitly instead.
"""

import os

os.environ.setdefault('INVESTALYZE_FACTORS_FILE', '')

import dash  # noqa: E402

dash.Dash(__name__, use_pages=True, pages_folder='')
