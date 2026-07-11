"""CLI entry: `python -m investalyze.apps`. Launch the control panel + ticker selector server."""

from investalyze.apps.app import app

if __name__ == '__main__':
    app.run(debug=False)
