"""CLI entry: `python -m investalyze.backtest`. Build factor panels, evaluate signals, run backtests.

Commands split along what they need: `factors` writes the point-in-time factor panel, `evaluate`
scores a factor against forward returns, `run` walks one signal through the engine, `walkforward`
repeats that fold by fold, `report` prints what a finished run wrote, `list` shows the signals, and
`delete` removes stored runs and `delete-evaluation` removes a stored evaluation, or one period of it
when `--start`/`--end` are given. Only the commands that store results open the DB for writing, so an
evaluation can run while a job holds the write lock.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from investalyze.analysis import custom_factors, factors_pit, panels
from investalyze.analysis.factors import CUSTOM, FACTORS, HIGHER_IS_BETTER
from investalyze.apps import universes
from investalyze.backtest import config as backtest_config
from investalyze.backtest import engine, evaluation, market as market_data, performance, registry, report
from investalyze.backtest import walkforward, writer
from investalyze.ingest import config, storage
from investalyze.ingest.logging import configure_logging

log = logging.getLogger('investalyze.backtest')

COMMANDS = ('factors', 'evaluate', 'run', 'walkforward', 'report', 'list', 'delete',
            'delete-evaluation')
FREQUENCY_WORDS = {'D': 'daily', 'W': 'weekly', 'M': 'monthly', 'Q': 'quarterly'}


def _epilog() -> str:
    """Render the signal catalog (name, kind, description) shown at the end of `--help`."""
    width = max(len(name) for name in registry.SIGNALS)
    kind_width = max(len(kind) for kind, _ in registry.SIGNALS.values())
    lines = ['signals:']
    for name in sorted(registry.SIGNALS):
        kind, _ = registry.SIGNALS[name]
        lines.append(f'  {name:<{width}}  {kind:<{kind_width}}  {registry.SIGNAL_DESCRIPTIONS[name]}')
    return '\n'.join(lines)


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser, kept separate so tests can inspect it without running a command."""
    parser = argparse.ArgumentParser(
        prog='python -m investalyze.backtest',
        description='Build point-in-time factor panels, evaluate signals, and run backtests over them.',
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('command', choices=COMMANDS, metavar='command',
                        help=f'one of: {", ".join(COMMANDS)}')
    parser.add_argument('--signal', choices=sorted(registry.SIGNALS), metavar='signal',
                        help='signal to run; see the list below')
    parser.add_argument('--factor', help='factor to evaluate, for the evaluate command')
    parser.add_argument('--horizon', type=int, default=21, help='forward-return horizon in sessions (default: 21)')
    parser.add_argument('--train', type=int, default=756, help='fitting sessions per walk-forward fold (default: 756)')
    parser.add_argument('--test', type=int, default=252, help='out-of-sample sessions per fold (default: 252)')
    parser.add_argument('--fold-step', type=int, default=252, help='sessions between fold starts (default: 252)')
    parser.add_argument('--embargo', type=int, default=0, help='sessions left empty between fitting and test (default: 0)')
    parser.add_argument('--anchored', action='store_true', help='grow the fitting span from the first session')
    parser.add_argument('--start', help='first date of the run, ISO format')
    parser.add_argument('--end', help='last date of the run, ISO format')
    parser.add_argument('--freq', choices=backtest_config.REBALANCE_FREQUENCIES, default='M',
                        help='how often a run rebalances its portfolio (default: M)')
    parser.add_argument('--panel-freq', choices=backtest_config.REBALANCE_FREQUENCIES, default='M',
                        help='how often the factor panel recomputes its values (default: M)')
    parser.add_argument('--rebuild', action='store_true',
                        help='recompute every date in the range, not only the ones not stored yet')
    parser.add_argument('--universe', help='saved universe to limit the run to (default: the config value)')
    parser.add_argument('--n-top', type=int, help='how many names to keep (default: the config value)')
    parser.add_argument('--rank-by', choices=('liquidity', 'mcap', 'ticker'),
                        help='how the universe is cut to size (default: the config value)')
    # --version is argparse's own convention for the tool's version number, so this flag is
    # spelled --data-version; dest='version' keeps every args.version reference unchanged
    parser.add_argument('--data-version', dest='version', choices=('as_reported', 'restated'),
                        default='as_reported',
                        help='fundamentals version the factor panel is built from (default: as_reported)')
    parser.add_argument('--build-panel', action='store_true',
                        help='build the factor panel first when the stored one does not cover the request')
    parser.add_argument('--run-id', nargs='+', metavar='run_id',
                        help='runs to delete, or the one run to print, for the delete and report commands')
    parser.add_argument('--config', type=Path, default=Path('backtest.toml'),
                        help='backtest TOML giving costs, limits and universe (default: ./backtest.toml)')
    parser.add_argument('--ingest-config', type=Path, default=Path('ingest.toml'),
                        help='ingest TOML giving the DB location (default: ./ingest.toml)')
    return parser


def _list_signals() -> None:
    """Print the signal catalog."""
    print(_epilog())


def _resolve_universe(args: argparse.Namespace, run_cfg: backtest_config.Config) -> list[str] | None:
    """The ticker list a command runs over: the flag, else the config, else every priced ticker."""
    name = args.universe or run_cfg.universe.name
    if name is None:
        return None
    tickers = universes.load_universe(name)
    log.info(f'universe {name}: {len(tickers)} tickers')
    return tickers


def _panel_universe(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> list[str] | None:
    """The names a panel is built for: the named universe as it stands, else the database cut to size.

    A named universe is used whole. The completeness filter a run applies belongs to a run, which
    needs one tradable set across its whole span; a panel row exists as soon as a ticker has a close,
    so a name that listed halfway through belongs in it. Putting seven solar names through that
    filter over a span starting in 2000 leaves nothing, which is not what the person picking them
    meant.
    """
    named = _resolve_universe(args, run_cfg)
    if named is not None:
        return named
    if args.n_top:
        return _run_universe(con, args, run_cfg, args.start, args.end)
    return None


def _build_factors(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Build the point-in-time factor panel over the requested range and merge it into `factor_panel`."""
    if args.start is None or args.end is None:
        raise SystemExit('factors needs --start and --end')
    freq = args.panel_freq
    asked = factors_pit.rebalance_dates(con, start=args.start, end=args.end, freq=freq)
    tickers = _panel_universe(con, args, run_cfg)
    if tickers is not None and not tickers:
        raise SystemExit('the chosen universe resolved to no tickers; nothing to build a panel over')
    grid = factors_pit.buildable_dates(con, asked, tickers)
    if len(grid) < len(asked):
        log.info(f'{len(asked) - len(grid)} of the {len(asked)} dates asked for are before any of '
                 'these names traded, so there is nothing to build on them')
    scope = f'{len(tickers):,} tickers' if tickers is not None else 'every priced ticker'

    building, why = _dates_to_build(con, args, grid, tickers)
    log.info(f'{args.version} panel, {args.start} to {args.end}, {scope}: '
             f'{len(grid)} {FREQUENCY_WORDS[freq]} dates requested, {len(building)} to build ({why})')
    if not len(building):
        return

    began = time.monotonic()
    panel = factors_pit.build_factor_panel(con, building, tickers=tickers, version=args.version,
                                           benchmark=run_cfg.benchmark)
    stored = factors_pit.store_factor_panel(con, panel)
    writer.record_panel_build(con, version=args.version,
                              universe=args.universe or run_cfg.universe.name or writer.WHOLE_DATABASE,
                              start=args.start, end=args.end, frequency=freq,
                              n_tickers=int(panel['Ticker'].nunique()), n_dates=len(building),
                              n_rows=len(panel),
                              seconds=time.monotonic() - began,
                              definitions=custom_factors.fingerprint(CUSTOM))
    log.info(f'this build wrote {len(panel):,} rows for {panel["Ticker"].nunique():,} tickers; '
             f'factor_panel now holds {stored:,} rows across every build so far')


def _dates_to_build(con, args: argparse.Namespace, grid: pd.DatetimeIndex,
                    tickers: list[str] | None) -> tuple[pd.DatetimeIndex, str]:
    """The dates a build should actually compute, and the reason in words.

    A date already stored for these names holds the same values a rebuild would produce, so it is
    skipped. Two things override that: `--rebuild`, and a factor definition edited since the last
    build, whose stored values came from a formula that no longer exists.
    """
    if args.rebuild:
        return grid, 'rebuilding every one'
    build = writer.latest_panel_build(con, args.version) \
        if storage.table_exists(con, 'factor_panel_builds') else None
    changed = custom_factors.changed_since(build['definitions'] if build else None, CUSTOM)
    if changed:
        return grid, f'rebuilding every one, {", ".join(changed)} was redefined'
    held = factors_pit.stored_dates(con, args.version, tickers)
    missing = pd.DatetimeIndex([date for date in grid if date not in held])
    return missing, 'the ones not stored yet'


def _read_factor_panel(con, args: argparse.Namespace) -> pd.DataFrame:
    """The stored factor panel for one version, narrowed to the requested date range."""
    clauses = ['Version = ?']
    params: list = [args.version]
    if args.start is not None:
        clauses.append('Date >= ?')
        params.append(args.start)
    if args.end is not None:
        clauses.append('Date <= ?')
        params.append(args.end)
    panel = con.execute(f'SELECT * FROM factor_panel WHERE {" AND ".join(clauses)}', params).df()
    if panel.empty:
        raise SystemExit(f'factor_panel holds no {args.version} rows for this range; run the factors command first')
    panel['Date'] = pd.to_datetime(panel['Date'])
    # ranked over the names being evaluated, which is the population the reading is about
    return factors_pit.apply_derived(panel)


def _median_spread(quantiles: pd.DataFrame) -> float:
    """Mean gap between the top and bottom bucket's median return, the spread a few names cannot move."""
    if quantiles.empty:
        return float('nan')
    medians = quantiles.pivot(index='Date', columns='Quantile', values='MedianReturn')
    return float((medians[medians.columns.max()] - medians[medians.columns.min()]).mean())


def _evaluate(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Score factors against forward returns and store their IC and quantile series.

    The universe is recorded with every row, so evaluating a factor over one set of names and then
    over another leaves two readings to compare rather than one overwriting the other.
    """
    if args.build_panel:
        _ensure_panel(con, args, run_cfg)
    if not storage.table_exists(con, 'factor_panel'):
        raise SystemExit('factor_panel does not exist; run the factors command first')
    panel = _read_factor_panel(con, args)
    selected = [args.factor] if args.factor else FACTORS
    unknown = [factor for factor in selected if factor not in panel.columns]
    if unknown:
        raise SystemExit(f'not factors of the stored panel: {", ".join(unknown)}')

    named = _resolve_universe(args, run_cfg)
    universe = args.universe or run_cfg.universe.name or writer.WHOLE_DATABASE
    if named is not None:
        panel = panel[panel['Ticker'].isin(named)]
        if panel.empty:
            raise SystemExit(f'the stored panel holds none of the {len(named)} names in {universe}')

    tickers = sorted(panel['Ticker'].unique())
    first, last = panel['Date'].min(), panel['Date'].max()
    horizon = args.horizon
    calendar = panels.trading_calendar(con, start=first.date().isoformat(),
                                       end=(last + pd.Timedelta(days=2 * horizon + 10)).date().isoformat())
    closes = panels.close_panel(con, tickers, calendar, min_obs_frac=factors_pit.MIN_PRICE_COVERAGE)
    forwards = evaluation.forward_returns(closes, horizon=horizon)
    log.info(f'{len(selected)} factor(s) over {universe}, {len(tickers):,} tickers, horizon {horizon}, '
             f'{first.date()} to {last.date()}')

    for factor in selected:
        scores = evaluation.factor_scores(panel, factor)
        ic = evaluation.rank_ic(scores, forwards)
        counts = evaluation.names_per_date(scores, forwards)
        quantiles = evaluation.quantile_frame(scores, forwards)

        if ic.empty:
            log.info(f'{factor:<18} no dates paired with a forward return over this range; skipped')
            continue

        ic_rows = pd.DataFrame({'Version': args.version, 'Universe': universe, 'Factor': factor,
                                'Date': ic.index, 'Horizon': horizon, 'Ic': ic.to_numpy(),
                                'NNames': counts.to_numpy()})
        quantile_rows = quantiles.assign(Version=args.version, Universe=universe, Factor=factor,
                                         Horizon=horizon)
        quantile_rows = quantile_rows[writer.QUANTILE_COLS]
        writer.replace_evaluation(con, args.version, universe, factor, ic_rows[writer.IC_COLS],
                                  quantile_rows, first.date(), last.date())
        decay = evaluation.ic_decay(scores, closes, horizons=writer.DECAY_HORIZONS)
        writer.replace_ic_decay(con, args.version, universe, factor, first.date(), last.date(), decay)

        summary = evaluation.ic_summary(ic)
        spread = evaluation.quantile_spread(evaluation.quantile_returns(scores, forwards))
        # the stored IC is on raw values; orienting it by HIGHER_IS_BETTER makes factors comparable
        orientation = 1 if HIGHER_IS_BETTER.get(factor, True) else -1
        log.info(f'{factor:<18} IC {summary["mean"]:+.4f}  t {summary["t_stat"]:+6.2f}  '
                 f'oriented t {orientation * summary["t_stat"]:+6.2f}  hit {summary["hit_rate"]:.0%}  '
                 f'spread {spread.mean():+.4f}  median spread {_median_spread(quantiles):+.4f}  '
                 f'n {summary["n_periods"]}')


def _run_universe(con, args: argparse.Namespace, run_cfg: backtest_config.Config,
                  calendar_start: str, calendar_end: str) -> list[str]:
    """The names a run trades: every one in the universe that traded during the range.

    Traded during, not throughout. A name that listed partway in cannot be bought before it exists,
    and one that stops trading is liquidated at its last price, so neither has to be excluded. The
    old rule dropped both, which is the shape survivorship bias takes when it is built into a filter.
    """
    named = _resolve_universe(args, run_cfg)
    calendar = panels.trading_calendar(con, start=calendar_start, end=calendar_end)
    kept = panels.traded_universe(con, calendar,
                                    n_top=args.n_top or run_cfg.universe.n_top,
                                    rank_by=args.rank_by or run_cfg.universe.rank_by,
                                    require_sector=run_cfg.universe.require_sector,
                                    candidates=named)
    if not kept:
        source = f'the {len(named)} names in {args.universe or run_cfg.universe.name}' if named is not None \
            else 'the database'
        raise SystemExit(f'none of {source} traded across the whole of {calendar_start} to '
                         f'{calendar_end}{_earliest_workable(con, named)}')
    return kept


def _earliest_workable(con, named: list[str] | None) -> str:
    """The soonest start that keeps a name, since telling someone to shorten the range begs the question.

    A name qualifies only if it traded across the whole span, so the earliest usable start is the
    first date any of them traded.
    """
    if named is None:
        return '; shorten the range or pick another universe'
    row = con.execute('SELECT min(Date), arg_min(Ticker, Date) FROM prices '
                      'WHERE Ticker IN (SELECT unnest(?::VARCHAR[]))', [sorted(named)]).fetchone()
    if row is None or row[0] is None:
        return '; none of them has any prices at all'
    return f'. The earliest start that keeps one is {row[0]}, when {row[1]} first traded'


def _ensure_panel(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Build the factor panel when the stored one does not reach across the requested dates.

    Reached only through `--build-panel`, never on its own. A command that quietly builds what it
    needs is a command whose cost and effect the caller cannot predict, so the decision is the
    caller's and this is what carries it out once they have made it.
    """
    coverage = factors_pit.panel_coverage(con, args.version)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if coverage is not None and coverage['start'] <= start and coverage['end'] >= end:
        log.info(f'factor panel covers {coverage["start"].date()} to {coverage["end"].date()}, '
                 f'{coverage["n_dates"]} dates')
        return
    reach = 'nothing stored' if coverage is None else \
        f'{coverage["start"].date()} to {coverage["end"].date()}'
    log.info(f'factor panel reaches {reach}, which does not cover {start.date()} to {end.date()}; '
             'building it first')
    _build_factors(con, args, run_cfg)


def _benchmark_curve(con, name: str, dates: pd.Index, starting_cash: float) -> tuple[pd.Series | None, str | None]:
    """The benchmark rebased to the run's starting cash, plus a note when it stops early."""
    rows = con.execute('SELECT Date, C FROM market_data WHERE Ticker = ? ORDER BY Date', [name]).df()
    if rows.empty:
        return None, market_data.benchmark_caveat(name, None, dates[-1])
    series = pd.Series(rows['C'].to_numpy(), index=pd.DatetimeIndex(rows['Date']))
    note = market_data.benchmark_caveat(name, series.index.max(), dates[-1])
    aligned = series.reindex(dates).ffill()
    if aligned.dropna().empty:
        return None, note
    first = aligned.dropna().iloc[0]
    return aligned / first * starting_cash, note


def _run(con, args: argparse.Namespace, run_cfg: backtest_config.Config, data_root: Path) -> None:
    """Walk one signal through the engine over the requested range, then store and print the report."""
    if args.signal is None:
        raise SystemExit('run needs --signal')
    if args.start is None or args.end is None:
        raise SystemExit('run needs --start and --end')

    kind, factory = registry.SIGNALS[args.signal]
    if args.build_panel:
        _ensure_panel(con, args, run_cfg)
    tickers = _run_universe(con, args, run_cfg, args.start, args.end)
    log.info(f'universe: {len(tickers):,} names priced across the whole range')

    market = market_data.build_market(con, tickers, start=args.start, end=args.end,
                                      exclude_checks=run_cfg.exclude_checks)
    params = run_cfg.signal(args.signal) | {'rebalance': args.freq,
                                            'version': args.version,
                                            'max_weight': run_cfg.limits.max_weight}
    step = factory(con, market, params)
    book = engine.run_backtest(market, step, starting_cash=run_cfg.starting_cash,
                               costs=run_cfg.costs, limits=run_cfg.limits)

    equity = pd.Series(book.equity, index=market.dates)
    fills = book.blotter()
    benchmark_equity, benchmark_note = _benchmark_curve(con, run_cfg.benchmark, market.dates,
                                                        run_cfg.starting_cash)
    benchmark_returns = benchmark_equity.pct_change() if benchmark_equity is not None else None
    stats = performance.summary(equity, fills, benchmark_returns)

    caveats = [benchmark_note] if benchmark_note else []

    identifier = writer.run_id({'signal': args.signal, 'start': args.start, 'end': args.end,
                                'params': params, 'universe': sorted(tickers),
                                'costs': vars(run_cfg.costs), 'limits': vars(run_cfg.limits),
                                'starting_cash': run_cfg.starting_cash})
    finished = report.BacktestReport(
        run_id=identifier, signal=args.signal, kind=kind, config=run_cfg, params=params,
        start=market.dates[0], end=market.dates[-1], tickers=tickers, benchmark=run_cfg.benchmark,
        caveats=caveats, stats=stats, equity=equity,
        gross=pd.Series(book.gross, index=market.dates), net=pd.Series(book.net, index=market.dates),
        cash=pd.Series(book.equity - book.net, index=market.dates),   # equity less the marked positions
        fills=fills, benchmark_equity=benchmark_equity,
        exposure=performance.exposures(pd.Series(book.gross, index=market.dates),
                                       pd.Series(book.net, index=market.dates), equity),
    )
    writer.replace_run(con, identifier, finished.run_row(), finished.equity_rows(), finished.trade_rows())
    path = report.write_report(finished, data_root)
    for caveat in caveats:
        log.warning(caveat)
    print(finished.render())
    log.info(f'run {identifier} stored; report written to {path}')


def _report(args: argparse.Namespace, data_root: Path) -> None:
    """Print the report a run wrote when it finished."""
    if args.run_id is None or len(args.run_id) != 1:
        raise SystemExit('report needs one --run-id')
    path = report.report_dir(args.run_id[0], data_root) / 'report.md'
    if not path.is_file():
        raise SystemExit(f'no report at {path}; the run may have been deleted')
    print(path.read_text())


def _delete_evaluation(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Remove one version and universe's stored evaluation, or one period of it when given."""
    universe = args.universe or run_cfg.universe.name
    if not universe:
        raise SystemExit('delete-evaluation needs --universe, the scope the rows were stored under')
    if bool(args.start) != bool(args.end):
        raise SystemExit('delete-evaluation needs both --start and --end, or neither')
    removed = writer.delete_evaluation(con, args.version, universe, args.start, args.end)
    span = f' ({args.start} to {args.end})' if args.start else ''
    if not any(removed.values()):
        over = f' over {args.start} to {args.end}' if args.start else ''
        raise SystemExit(f'no {args.version} evaluation stored for {universe}{over}')
    log.info(f'deleted the {args.version} evaluation for {universe}{span}: '
             f'{removed["factor_ic"]:,} IC rows, {removed["factor_quantiles"]:,} quantile rows, '
             f'{removed["factor_ic_decay"]:,} decay rows')


def _delete(con, args: argparse.Namespace, data_root: Path) -> None:
    """Remove stored runs: their summary rows, equity curves, blotters and rendered reports.

    Every run named is dealt with before an unknown one is reported, so one bad id in a selection
    does not leave the rest of it half deleted.
    """
    if not args.run_id:
        raise SystemExit('delete needs --run-id')
    missing = []
    for identifier in args.run_id:
        removed = writer.delete_run(con, identifier)
        if not any(removed.values()):
            missing.append(identifier)
            continue
        written = 'report removed' if report.delete_report(identifier, data_root) else 'no report on disk'
        log.info(f'deleted run {identifier}: {removed["backtest_equity"]:,} equity rows, '
                 f'{removed["backtest_trades"]:,} fills, {written}')
    if missing:
        raise SystemExit(f'no stored run: {", ".join(missing)}')


def _walk_forward(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Run one signal fold by fold, printing each fold's out-of-sample statistics and the whole track."""
    if args.signal is None:
        raise SystemExit('walkforward needs --signal')
    if args.start is None or args.end is None:
        raise SystemExit('walkforward needs --start and --end')

    _, factory = registry.SIGNALS[args.signal]
    calendar = panels.trading_calendar(con, start=args.start, end=args.end)
    windows = walkforward.walk_forward_windows(calendar, train=args.train, test=args.test,
                                               step=args.fold_step, embargo=args.embargo,
                                               anchored=args.anchored)
    log.info(f'{len(windows)} folds of {args.train} fitting and {args.test} test sessions, '
             f'embargo {args.embargo}')

    params = run_cfg.signal(args.signal) | {'rebalance': args.freq,
                                            'version': args.version,
                                            'max_weight': run_cfg.limits.max_weight}
    result = walkforward.run_walk_forward(con, factory, windows, run_cfg, params)

    columns = ['fold', 'n_tickers', 'cagr', 'vol', 'sharpe', 'max_drawdown', 'turnover', 'n_trades']
    print(result.folds[columns].to_string(index=False, float_format=lambda value: f'{value:.4f}'))
    combined = performance.summary(result.equity, result.fills)
    print()
    print(f'out of sample overall: CAGR {combined["cagr"]:.2%}  Sharpe {combined["sharpe"]:.3f}  '
          f'max drawdown {combined["max_drawdown"]:.2%}  turnover {combined["turnover"]:.2f}x')


def main() -> None:
    """Parse CLI args and dispatch to the selected command."""
    args = _parser().parse_args()

    if args.command == 'list':
        _list_signals()
        return

    cfg = config.read(args.ingest_config)
    run_cfg = backtest_config.read(args.config)
    configure_logging(cfg.log_level)

    if args.command == 'factors':
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            writer.ensure_tables(con)
            _build_factors(con, args, run_cfg)
        finally:
            con.close()
        return

    if args.command == 'evaluate':
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            writer.ensure_tables(con)
            _evaluate(con, args, run_cfg)
        finally:
            con.close()
        return

    if args.command == 'run':
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            writer.ensure_tables(con)
            _run(con, args, run_cfg, cfg.data_root)
        finally:
            con.close()
        return

    if args.command == 'walkforward':
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            writer.ensure_tables(con)
            _walk_forward(con, args, run_cfg)
        finally:
            con.close()
        return

    if args.command == 'delete':
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            writer.ensure_tables(con)
            _delete(con, args, cfg.data_root)
        finally:
            con.close()
        return

    if args.command == 'delete-evaluation':
        con = storage.connect(cfg.data_root, cfg.db)
        try:
            writer.ensure_tables(con)
            _delete_evaluation(con, args, run_cfg)
        finally:
            con.close()
        return

    if args.command == 'report':
        _report(args, cfg.data_root)
        return


if __name__ == '__main__':
    main()
