"""CLI entry: `python -m investalyze.apps`. Launch the control panel + ticker selector server."""

from investalyze.apps.app import app

if __name__ == '__main__':
    # Dash builds its callback map lazily on the first request, and the control panel's 1s poll
    # can race that setup when a browser is already sitting on the page as the server (re)starts.
    # Calling _setup_server directly builds it up front. A warmup request would too, but it marks
    # Flask setup as finished, and debug mode registers an error handler after this point.
    app._setup_server()
    # silence_routes_logging=False: Dash's own silencing mutes the werkzeug logger wholesale; the
    # filter in app.py is finer, it keeps 4xx/5xx lines while dropping the 200/304 poll noise.
    app.run(debug=True, dev_tools_silence_routes_logging=False)
