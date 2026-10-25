"""CLI entry: `python -m investalyze.apps` or `investalyze`. Launch the control panel + ticker selector server."""

from investalyze import paths
from investalyze.apps.app import app


def main() -> None:
    """Serve the app on localhost at the port of investalyze.toml, in debug mode only when it says so."""
    # Dash builds its callback map lazily on the first request, and the control panel's 1s poll
    # can race that setup when a browser is already sitting on the page as the server (re)starts.
    # Calling _setup_server directly builds it up front. A warmup request would too, but it marks
    # Flask setup as finished, and debug mode registers an error handler after this point.
    app._setup_server()
    # silence_routes_logging=False: Dash's own silencing mutes the werkzeug logger wholesale; the
    # filter in app.py is finer, it keeps 4xx/5xx lines while dropping the 200/304 poll noise.
    port, debug = paths.server()
    app.run(port=port, debug=debug, dev_tools_silence_routes_logging=False)


if __name__ == '__main__':
    main()
