"""CLI entry: `python -m investalyze.research`.

`list` shows the factor library, `fields` searches the data fields, `diagnostics` runs coverage,
distribution, top/bottom names, correlation and rank autocorrelation, `tearsheet` runs IC analysis,
`backtest` backtests the top quantile of one factor or a composite, and `timing` backtests a timing rule:
entry and exit factors, a trailing stop and a position cap, decided daily. The universe on each month-end is the most liquid
names by trailing dollar volume known on that date. Results are cached under `data/cache` and
recomputed when a factor definition changes.
"""

import argparse
import sys

import duckdb
import pandas as pd

from investalyze import paths
from investalyze.analysis import factors_pit
from investalyze.apps import universes as saved_universes
from investalyze.backtest import config as backtest_config
from investalyze.backtest import report, writer
from investalyze.ingest import storage
from investalyze.research import backtest, cache, compute, diagnostics, factors, fields, tearsheet, timing, universe

COMMANDS = ('list', 'fields', 'diagnostics', 'tearsheet', 'backtest', 'timing')


def _grid(con: duckdb.DuckDBPyConnection, args: argparse.Namespace) -> pd.DatetimeIndex:
    """The last trading session of each month in the range."""
    return factors_pit.rebalance_dates(con, start=args.start, end=args.end, freq='M')


def _panel(con: duckdb.DuckDBPyConnection, name: str, library: dict, args: argparse.Namespace) -> pd.DataFrame:
    grid = _grid(con, args)
    members = universe.liquid_universe(con, grid, size=args.size)
    return compute.compute(con, [name], library, as_of=grid, members=members)


def _holdout(args: argparse.Namespace) -> pd.Timestamp | None:
    return pd.Timestamp(args.holdout) if args.holdout else None


def _show_library(library: dict) -> None:
    if not library:
        print(f'no factors yet; add one as a .toml file under {paths.FACTORS_DIR}')
        return
    print(f'{"factor":30} {"status":10} {"direction":9} expression')
    for name in sorted(library):
        factor = library[name]
        direction = 'higher' if factor.higher_is_better else 'lower'
        print(f'{name:30} {factor.status:10} {direction:9} {factor.expression}')


def _show_fields(con: duckdb.DuckDBPyConnection, search: str | None) -> None:
    catalog = fields.catalog(con)
    matches = sorted(catalog) if not search else sorted(
        alias for alias in catalog if search.lower() in alias.lower())
    print(f'{len(matches)} of {len(catalog)} data fields')
    for alias in matches[:60]:
        field = catalog[alias]
        print(f'  {alias:44} {field.statement:9} {field.window.upper():4} {field.source}')
    if len(matches) > 60:
        print(f'  ... and {len(matches) - 60} more')


def _diagnostics(con: duckdb.DuckDBPyConnection, name: str, library: dict, args: argparse.Namespace) -> None:
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


def _tearsheet(con: duckdb.DuckDBPyConnection, name: str, library: dict, args: argparse.Namespace) -> None:
    frame = _panel(con, name, library, args)
    if frame.empty or name not in frame.columns:
        raise SystemExit(f'{name!r} has no values over this range')
    result = tearsheet.tearsheet(con, name, library, frame, horizon=args.horizon, store=cache.Cache(paths.CACHE_DIR),
                                 holdout_start=_holdout(args))
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
    if not result['split'].empty:
        print('\nin-sample and out-of-sample:')
        print(result['split'].to_string(index=False))
    print('\nIC by year:')
    print(result['by_year'].to_string(index=False))


def _backtest(name_arg: str, library: dict, args: argparse.Namespace) -> None:
    """Simulate a factor portfolio and store it as a backtest run, with its report."""
    names = [name.strip() for name in name_arg.split(',') if name.strip()]
    unknown = [name for name in names if name not in library]
    if unknown:
        raise SystemExit(f'no factor named {", ".join(unknown)}; `list` shows the library')
    cfg = backtest_config.read()
    with storage.connect() as con:
        factors.validate_library(library, set(fields.catalog(con)) | set(fields.IDENTITY))
        grid = factors_pit.rebalance_dates(con, start=args.start, end=args.end, freq=args.rebalance)
        members = universe.liquid_universe(con, grid, size=args.size)
        frame = compute.compute(con, names, library, as_of=grid, members=members)
        missing = [name for name in names if name not in frame.columns]
        if frame.empty or missing:
            raise SystemExit(f'no values over this range for {", ".join(missing or names)}')

        result = backtest.simulate(con, frame, names, library, n_quantiles=args.quantiles, cfg=cfg)
        params = backtest.run_params(names, library, n_quantiles=args.quantiles, rebalance=args.rebalance,
                                     universe_size=args.size, start=args.start, end=args.end,
                                     holdout=args.holdout or None)
        finished = backtest.build_report(result, params, cfg)
        writer.ensure_tables(con)
        writer.replace_run(con, finished.run_id, finished.run_row(), finished.equity_rows(), finished.trade_rows())
        path = report.write_report(finished, paths.DATA_ROOT)

        periods = backtest.period_stats(result['equity'], result['benchmark_equity'], holdout_start=_holdout(args))
        if len(names) == 1:
            split = periods.set_index('period')
            cache.Cache(paths.CACHE_DIR).put_summary(names[0], factors.fingerprint(names[0], library), {
                'sharpe': result['stats']['sharpe'], 'cagr': result['stats']['cagr'],
                'oos_sharpe': float(split.loc['Out-of-sample', 'sharpe']) if 'Out-of-sample' in split.index else None,
            })

    print(finished.render())
    print('periods:')
    print(periods.to_string(index=False))
    print(f'run {finished.run_id} stored; report written to {path}; median {result["names_median"]:,.0f} names held')


def _timing(entry_factor: str, library: dict, args: argparse.Namespace) -> None:
    """Simulate a timing rule on the daily grid and store it as a backtest run, with its report."""
    used = [name for name in (entry_factor, args.exit_factor, args.priority_factor) if name]
    unknown = [name for name in used if name not in library]
    if unknown:
        raise SystemExit(f'no factor named {", ".join(unknown)}; `list` shows the library')
    trailing_stop = args.trailing_stop or None
    cfg = backtest_config.read()
    with storage.connect() as con:
        factors.validate_library(library, set(fields.catalog(con)) | set(fields.IDENTITY))
        sessions = factors_pit.rebalance_dates(con, start=args.start, end=args.end, freq='D')
        if args.universe:
            tickers = saved_universes.load_universe(args.universe)
            members = pd.DataFrame({'Date': sessions[0], 'Ticker': tickers})
        else:
            first = (pd.Timestamp(args.start) - pd.Timedelta(days=40)).date().isoformat()
            members = universe.liquid_universe(con, factors_pit.rebalance_dates(con, start=first, end=args.end, freq='M'),
                                               size=args.size)
        frame = compute.compute(con, used, library, as_of=sessions, members=members)
        if frame.empty or entry_factor not in frame.columns:
            raise SystemExit(f'no values over this range for {entry_factor}')

        result = timing.simulate(con, frame, library, entry_factor=entry_factor, exit_factor=args.exit_factor,
                                 exit_on_entry_false=not args.keep_on_entry_false, trailing_stop=trailing_stop,
                                 max_positions=args.max_positions, priority_factor=args.priority_factor,
                                 start=args.start, end=args.end, cfg=cfg)
        params = timing.run_params(library, entry_factor=entry_factor, exit_factor=args.exit_factor,
                                   exit_on_entry_false=not args.keep_on_entry_false, trailing_stop=trailing_stop,
                                   max_positions=args.max_positions, priority_factor=args.priority_factor,
                                   universe=args.universe, universe_size=args.size, start=args.start, end=args.end,
                                   holdout=args.holdout or None)
        finished = timing.build_report(result, params, cfg)
        writer.ensure_tables(con)
        writer.replace_run(con, finished.run_id, finished.run_row(), finished.equity_rows(), finished.trade_rows())
        path = report.write_report(finished, paths.DATA_ROOT)
        periods = backtest.period_stats(result['equity'], result['benchmark_equity'], holdout_start=_holdout(args))

    print(finished.render())
    print('periods:')
    print(periods.to_string(index=False))
    print(f'run {finished.run_id} stored; report written to {path}; median {result["names_median"]:,.0f} names held')


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m investalyze.research',
                                     description='Define factors, run diagnostics and IC analysis, and backtest them.')
    parser.add_argument('command', choices=COMMANDS, metavar='command', help=f'one of: {", ".join(COMMANDS)}')
    parser.add_argument('name', nargs='?',
                        help='the factor name (several separated by commas for backtest), or a search term for fields')
    parser.add_argument('--start', default='2015-01-31', help='first as-of date (default: 2015-01-31)')
    parser.add_argument('--end', default='2024-12-31', help='last as-of date (default: 2024-12-31)')
    parser.add_argument('--horizon', type=int, default=tearsheet.DEFAULT_HORIZON,
                        help=f'forward return horizon in trading days (default: {tearsheet.DEFAULT_HORIZON})')
    parser.add_argument('--size', type=int, default=1000,
                        help='names in the universe on each date, by trailing dollar volume (default: 1000)')
    parser.add_argument('--quantiles', type=int, default=backtest.DEFAULT_QUANTILES,
                        help=f'quantiles the universe is split into; the top one is held (default: {backtest.DEFAULT_QUANTILES})')
    parser.add_argument('--rebalance', choices=sorted(backtest.REBALANCE), default='M',
                        help='rebalance at each month-end (M) or quarter-end (Q) session (default: M)')
    parser.add_argument('--exit-factor', help='timing: exit a position on a session where this factor is true')
    parser.add_argument('--keep-on-entry-false', action='store_true',
                        help='timing: keep a position after its entry factor turns false (default: exit then)')
    parser.add_argument('--trailing-stop', type=float, default=0.0,
                        help='timing: exit after this fraction below the highest close since entry, e.g. 0.15 (default: none)')
    parser.add_argument('--max-positions', type=int, default=timing.DEFAULT_MAX_POSITIONS,
                        help=f'timing: most names held at once, each at 1 / this weight (default: {timing.DEFAULT_MAX_POSITIONS})')
    parser.add_argument('--priority-factor', help='timing: when entries exceed free positions, take the highest values of '
                        'this factor first (default: the most liquid)')
    parser.add_argument('--universe', help='timing: a saved universe to trade (default: the most liquid names, see --size)')
    parser.add_argument('--holdout', default='2022-01-01',
                        help='first date of the out-of-sample period (default: 2022-01-01; empty for none)')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    library = factors.load(paths.FACTORS_DIR)

    if args.command == 'list':
        _show_library(library)
        return
    if args.command == 'backtest':
        if not args.name:
            raise SystemExit('backtest needs a factor name, or several separated by commas')
        _backtest(args.name, library, args)
        return
    if args.command == 'timing':
        if not args.name:
            raise SystemExit('timing needs the entry factor name')
        _timing(args.name, library, args)
        return

    with storage.connect(read_only=True) as con:
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


if __name__ == '__main__':
    sys.exit(main())
