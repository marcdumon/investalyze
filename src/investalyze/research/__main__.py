"""CLI entry: `python -m investalyze.research`.

`list` shows the factor library, `fields` searches the data fields, `diagnostics` runs coverage,
distribution, top/bottom names, correlation and rank autocorrelation on a sample, and `tearsheet`
runs IC analysis. Results are cached under `data/cache` and recomputed when a factor definition
changes.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from investalyze.ingest import storage
from investalyze.research import cache, compute, diagnostics, factors, fields, tearsheet

ROOT = Path(__file__).resolve().parents[3]
FACTORS_DIR = ROOT / 'factors'
CACHE_DIR = ROOT / 'data' / 'cache'
COMMANDS = ('list', 'fields', 'diagnostics', 'tearsheet')


def _connect():
    """A read-only connection to the project database."""
    return storage.connect(ROOT / 'data', read_only=True)


def _grid(args) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(args.start, args.end, freq='ME'))


def _panel(con, name: str, library: dict, args) -> pd.DataFrame:
    tickers = None
    if args.names:
        tickers = [row[0] for row in con.execute(
            'SELECT DISTINCT Ticker FROM prices ORDER BY Ticker LIMIT ?', [args.names]).fetchall()]
    return compute.compute(con, [name], library, as_of=_grid(args), tickers=tickers)


def _show_library(library: dict) -> None:
    if not library:
        print(f'no factors yet; add one as a .toml file under {FACTORS_DIR}')
        return
    print(f'{"factor":30} {"status":10} {"direction":9} expression')
    for name in sorted(library):
        factor = library[name]
        direction = 'higher' if factor.higher_is_better else 'lower'
        print(f'{name:30} {factor.status:10} {direction:9} {factor.expression}')


def _show_fields(con, search: str | None) -> None:
    catalog = fields.catalog(con)
    matches = sorted(catalog) if not search else sorted(
        alias for alias in catalog if search.lower() in alias.lower())
    print(f'{len(matches)} of {len(catalog)} data fields')
    for alias in matches[:60]:
        field = catalog[alias]
        print(f'  {alias:44} {field.statement:9} {field.window.upper():4} {field.source}')
    if len(matches) > 60:
        print(f'  ... and {len(matches) - 60} more')


def _diagnostics(con, name: str, library: dict, args) -> None:
    frame = _panel(con, name, library, args)
    if frame.empty or name not in frame.columns:
        raise SystemExit(f'{name!r} has no values over this range')
    inputs = sorted(library[name].dependencies)

    print(f'=== {name} ===  {library[name].description}')
    print(f'    {library[name].expression}')
    table = diagnostics.coverage(frame, name)
    latest = table.iloc[-1]
    print(f'\ncoverage: {latest["names"]:,.0f} of {latest["universe"]:,.0f} names '
          f'({latest["share"]:.1f}%) in {latest["year"]:.0f}')
    print(table.tail(6).to_string(index=False))

    print('\ndistribution:')
    for key, value in diagnostics.distribution(frame[name]).items():
        print(f'  {key:8} {value:,.4f}' if isinstance(value, float) else f'  {key:8} {value:,}')

    ranked = diagnostics.top_bottom(frame, name, n=5, inputs=inputs)
    print('\ntop 5:')
    print(ranked['top'].to_string(index=False))
    print('\nbottom 5:')
    print(ranked['bottom'].to_string(index=False))

    others = [other for other in library if other != name and other in frame.columns]
    if others:
        print('\ncorrelation with other factors:')
        print(diagnostics.correlations(frame, name, others).to_string())
    print(f'\nrank autocorrelation: {diagnostics.rank_autocorrelation(frame, name):.3f}')


def _tearsheet(con, name: str, library: dict, args) -> None:
    frame = _panel(con, name, library, args)
    if frame.empty or name not in frame.columns:
        raise SystemExit(f'{name!r} has no values over this range')
    result = tearsheet.tearsheet(con, name, library, frame, horizon=args.horizon, store=cache.Cache(CACHE_DIR))
    summary = result['summary']

    print(f'=== {name} ===  horizon {result["horizon"]} trading days')
    print(f'  IC {summary["oriented_ic"]:+.4f}   t-stat {summary["oriented_t"]:+.2f}   '
          f'hit rate {summary["hit_rate"]:.0%}   {summary["n_periods"]} dates')
    print(f'  cross-section: median {summary["names_median"]:,.0f} names, minimum {summary["names_min"]:,.0f}')
    print(f'  monotonic quantile returns: {result["monotonic"]}')
    print(f'  multiple testing: {result["tested"]} factors, '
          f'~{result["expected_false_positives"]:.1f} expected false positives at p<0.05')
    print('\nIC decay:')
    print(result['decay'].to_string(index=False))
    print('\nquantile returns:')
    print(result['quantiles'].to_string(index=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m investalyze.research',
                                     description='Define factors and run diagnostics and IC analysis on them.')
    parser.add_argument('command', choices=COMMANDS, metavar='command', help=f'one of: {", ".join(COMMANDS)}')
    parser.add_argument('name', nargs='?', help='the factor name, or a search term for fields')
    parser.add_argument('--start', default='2015-01-31', help='first as-of date (default: 2015-01-31)')
    parser.add_argument('--end', default='2024-12-31', help='last as-of date (default: 2024-12-31)')
    parser.add_argument('--horizon', type=int, default=tearsheet.DEFAULT_HORIZON,
                        help=f'forward return horizon in trading days (default: {tearsheet.DEFAULT_HORIZON})')
    parser.add_argument('--names', type=int, default=0, help='limit to this many tickers (default: all)')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    library = factors.load(FACTORS_DIR)

    if args.command == 'list':
        _show_library(library)
        return

    con = _connect()
    try:
        if args.command == 'fields':
            _show_fields(con, args.name)
            return
        if not args.name:
            raise SystemExit(f'{args.command} needs a factor name')
        if args.name not in library:
            raise SystemExit(f'no factor named {args.name!r}; `list` shows the library')
        factors.validate_library(library, set(fields.catalog(con)) | set(fields.IDENTITY))
        if args.command == 'diagnostics':
            _diagnostics(con, args.name, library, args)
        else:
            _tearsheet(con, args.name, library, args)
    finally:
        con.close()


if __name__ == '__main__':
    sys.exit(main())
