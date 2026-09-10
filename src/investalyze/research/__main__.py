"""CLI entry: `python -m investalyze.research`.

Four commands, one per gate. `fields` browses the palette an idea is written against, `check`
answers whether a draft is sane, `predict` whether it forecasts anything, and `ideas` shows the
library. Nothing here stores an answer a user has to manage: everything computed is cached under
`data/cache` and rebuilt whenever the idea that produced it moves.
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

from investalyze.ingest import config as ingest_config
from investalyze.ingest import storage
from investalyze.research import cache, compute, diagnostics, fields, ideas, predict

ROOT = Path(__file__).resolve().parents[3]
IDEAS_DIR = ROOT / 'ideas'
CACHE_DIR = ROOT / 'data' / 'cache'
COMMANDS = ('ideas', 'fields', 'check', 'predict')


def _connect():
    """A read-only connection, since nothing here writes to the database."""
    data_root = ingest_config.read().data_root if hasattr(ingest_config.read(), 'data_root') else ROOT / 'data'
    return storage.connect(Path(data_root), read_only=True)


def _library() -> dict:
    return ideas.load(IDEAS_DIR)


def _grid(args) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(args.start, args.end, freq='ME'))


def _panel(con, name: str, library: dict, args) -> pd.DataFrame:
    tickers = None
    if args.names:
        tickers = [row[0] for row in con.execute(
            'SELECT DISTINCT Ticker FROM prices ORDER BY Ticker LIMIT ?', [args.names]).fetchall()]
    return compute.compute(con, [name], library, as_of=_grid(args), tickers=tickers)


def _show_ideas(library: dict) -> None:
    if not library:
        print(f'no ideas yet; add one as a .toml file under {IDEAS_DIR}')
        return
    print(f'{"idea":28} {"status":10} {"dir":4} expression')
    for name in sorted(library):
        idea = library[name]
        arrow = 'high' if idea.higher_is_better else 'low'
        print(f'{name:28} {idea.status:10} {arrow:4} {idea.expression}')


def _show_fields(con, search: str | None) -> None:
    catalog = fields.catalog(con)
    matches = sorted(catalog) if not search else sorted(
        alias for alias in catalog if search.lower() in alias.lower())
    print(f'{len(matches)} of {len(catalog)} fields')
    for alias in matches[:60]:
        field = catalog[alias]
        print(f'  {alias:44} {field.statement:9} {field.window:7} {field.source}')
    if len(matches) > 60:
        print(f'  ... and {len(matches) - 60} more')


def _check(con, name: str, library: dict, args) -> None:
    """Level one: is this idea sane, before anything expensive is asked of it."""
    frame = _panel(con, name, library, args)
    if frame.empty or name not in frame.columns:
        raise SystemExit(f'{name!r} produced no values over this range')
    inputs = sorted(library[name].dependencies)

    print(f'=== {name} ===  {library[name].description}')
    print(f'    {library[name].expression}')
    table = diagnostics.coverage(frame, name)
    latest = table.iloc[-1]
    print(f'\ncoverage: {latest["names"]:,.0f} of {latest["universe"]:,.0f} names '
          f'({latest["share"]:.1f}%) in {latest["year"]:.0f}')
    print(table.tail(6).to_string(index=False))

    print('\ndistribution:')
    summary = diagnostics.distribution(frame[name])
    for key, value in summary.items():
        print(f'  {key:8} {value:,.4f}' if isinstance(value, float) else f'  {key:8} {value:,}')

    ends = diagnostics.extremes(frame, name, n=5, inputs=inputs)
    print('\ntop 5 now:')
    print(ends['top'].to_string(index=False))
    print('\nbottom 5 now:')
    print(ends['bottom'].to_string(index=False))

    others = [other for other in library if other != name and other in frame.columns]
    if others:
        print('\ncorrelation to the rest of the library:')
        print(diagnostics.correlations(frame, name, others).to_string())
    print(f'\nrank autocorrelation: {diagnostics.rank_autocorrelation(frame, name):.3f}')


def _predict(con, name: str, library: dict, args) -> None:
    """Level two: does this idea forecast the returns that followed it."""
    frame = _panel(con, name, library, args)
    if frame.empty or name not in frame.columns:
        raise SystemExit(f'{name!r} produced no values over this range')
    store = cache.Cache(CACHE_DIR)
    answer = predict.verdict(con, name, library, frame, horizon=args.horizon, store=store)
    summary = answer['summary']

    print(f'=== {name} ===  horizon {answer["horizon"]} sessions')
    print(f'  oriented IC {summary["oriented_ic"]:+.4f}   t {summary["oriented_t"]:+.2f}   '
          f'hit {summary["hit_rate"]:.0%}   over {summary["n_periods"]} dates')
    print(f'  measured on {summary["names_median"]:,.0f} names '
          f'(smallest cross-section {summary["names_min"]:,.0f})')
    print(f'  monotonic quantiles: {answer["monotonic"]}')
    print(f'  {answer["tested"]} ideas in the library, so expect about '
          f'{answer["expected_false_positives"]:.1f} to look significant by luck')
    print('\ndecay:')
    print(answer['decay'].to_string(index=False))
    print('\nquantiles:')
    print(answer['quantiles'].to_string(index=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m investalyze.research',
                                     description='Build factor ideas and find out whether they work.')
    parser.add_argument('command', choices=COMMANDS, metavar='command',
                        help=f'one of: {", ".join(COMMANDS)}')
    parser.add_argument('name', nargs='?', help='the idea to work on, or a search term for fields')
    parser.add_argument('--start', default='2015-01-31', help='first as-of date (default: 2015-01-31)')
    parser.add_argument('--end', default='2024-12-31', help='last as-of date (default: 2024-12-31)')
    parser.add_argument('--horizon', type=int, default=predict.DEFAULT_HORIZON,
                        help=f'forward-return horizon in sessions (default: {predict.DEFAULT_HORIZON})')
    parser.add_argument('--names', type=int, default=0,
                        help='limit to this many tickers, for a quick look (default: every name)')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    library = _library()

    if args.command == 'ideas':
        _show_ideas(library)
        return

    con = _connect()
    try:
        if args.command == 'fields':
            _show_fields(con, args.name)
            return
        if not args.name:
            raise SystemExit(f'{args.command} needs the name of an idea')
        if args.name not in library:
            raise SystemExit(f'no idea named {args.name!r}; `ideas` lists what there is')
        ideas.validate_library(library, set(fields.catalog(con)) | set(fields.IDENTITY))
        if args.command == 'check':
            _check(con, args.name, library, args)
        else:
            _predict(con, args.name, library, args)
    finally:
        con.close()


if __name__ == '__main__':
    sys.exit(main())
