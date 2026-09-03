"""CLI entry: `python -m investalyze.backtest`. Build factor panels, evaluate signals, run backtests.

Commands split along what they need: `factors` writes the point-in-time factor panel, `evaluate`
scores a factor against forward returns, `run` walks one signal through the engine, `walkforward`
repeats that fold by fold, and `report` and `list` read back what earlier runs stored. Only the
commands that store results open the DB for writing, so an evaluation can run while a job holds the
write lock.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from investalyze.analysis import factors_pit, panels
from investalyze.analysis.factors import FACTORS, HIGHER_IS_BETTER
from investalyze.apps import universes
from investalyze.backtest import config as backtest_config
from investalyze.backtest import engine, evaluation, market as market_data, performance, registry, report
from investalyze.backtest import walkforward, writer
from investalyze.ingest import config, storage
from investalyze.ingest.logging import configure_logging

log = logging.getLogger('investalyze.backtest')

COMMANDS = ('factors', 'evaluate', 'run', 'walkforward', 'report', 'list')


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
    parser.add_argument('--freq', choices=backtest_config.REBALANCE_FREQUENCIES,
                        help='rebalance frequency (default: the config value)')
    parser.add_argument('--universe', help='saved universe to limit the run to (default: the config value)')
    parser.add_argument('--n-top', type=int, help='how many names to keep (default: the config value)')
    parser.add_argument('--rank-by', choices=('liquidity', 'mcap', 'ticker'),
                        help='how the universe is cut to size (default: the config value)')
    parser.add_argument('--vintage', choices=('as_reported', 'restated'), default='as_reported',
                        help='fundamentals vintage the factor panel is built from (default: as_reported)')
    parser.add_argument('--run-id', help='run to read back, for the report command')
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


def _build_factors(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Build the point-in-time factor panel over the requested range and merge it into `factor_panel`."""
    if args.start is None or args.end is None:
        raise SystemExit('factors needs --start and --end')
    freq = args.freq or run_cfg.rebalance
    grid = factors_pit.rebalance_dates(con, start=args.start, end=args.end, freq=freq)
    tickers = _run_universe(con, args, run_cfg, args.start, args.end) if (args.universe or args.n_top) \
        else _resolve_universe(args, run_cfg)
    log.info(f'{args.vintage} panel over {len(grid)} {freq} dates, {args.start} to {args.end}')

    panel = factors_pit.build_factor_panel(con, grid, tickers=tickers, vintage=args.vintage,
                                           benchmark=run_cfg.benchmark)
    stored = factors_pit.store_factor_panel(con, panel)
    log.info(f'built {len(panel):,} rows over {panel["Ticker"].nunique():,} tickers; factor_panel holds {stored:,}')


def _read_factor_panel(con, args: argparse.Namespace) -> pd.DataFrame:
    """The stored factor panel for one vintage, narrowed to the requested date range."""
    clauses = ['Vintage = ?']
    params: list = [args.vintage]
    if args.start is not None:
        clauses.append('Date >= ?')
        params.append(args.start)
    if args.end is not None:
        clauses.append('Date <= ?')
        params.append(args.end)
    panel = con.execute(f'SELECT * FROM factor_panel WHERE {" AND ".join(clauses)}', params).df()
    if panel.empty:
        raise SystemExit(f'factor_panel holds no {args.vintage} rows for this range; run the factors command first')
    panel['Date'] = pd.to_datetime(panel['Date'])
    return panel


def _evaluate(con, args: argparse.Namespace, run_cfg: backtest_config.Config) -> None:
    """Score factors against forward returns and store their IC and quantile series."""
    if not storage.table_exists(con, 'factor_panel'):
        raise SystemExit('factor_panel does not exist; run the factors command first')
    panel = _read_factor_panel(con, args)
    selected = [args.factor] if args.factor else FACTORS
    unknown = [factor for factor in selected if factor not in panel.columns]
    if unknown:
        raise SystemExit(f'not factors of the stored panel: {", ".join(unknown)}')

    tickers = sorted(panel['Ticker'].unique())
    first, last = panel['Date'].min(), panel['Date'].max()
    horizon = args.horizon
    calendar = panels.trading_calendar(con, start=first.date().isoformat(),
                                       end=(last + pd.Timedelta(days=2 * horizon + 10)).date().isoformat())
    closes = panels.close_panel(con, tickers, calendar, min_obs_frac=factors_pit.MIN_PRICE_COVERAGE)
    forwards = evaluation.forward_returns(closes, horizon=horizon)
    log.info(f'{len(selected)} factor(s), {len(tickers):,} tickers, horizon {horizon}, {first.date()} to {last.date()}')

    for factor in selected:
        scores = evaluation.factor_scores(panel, factor)
        ic = evaluation.rank_ic(scores, forwards)
        counts = evaluation.names_per_date(scores, forwards)
        quantiles = evaluation.quantile_frame(scores, forwards)

        ic_rows = pd.DataFrame({'Vintage': args.vintage, 'Factor': factor, 'Date': ic.index,
                                'Horizon': horizon, 'Ic': ic.to_numpy(), 'NNames': counts.to_numpy()})
        quantile_rows = quantiles.assign(Vintage=args.vintage, Factor=factor, Horizon=horizon)
        quantile_rows = quantile_rows[writer.QUANTILE_COLS]
        writer.replace_evaluation(con, args.vintage, factor, ic_rows[writer.IC_COLS], quantile_rows)

        summary = evaluation.ic_summary(ic)
        spread = evaluation.quantile_spread(evaluation.quantile_returns(scores, forwards))
        # the stored IC is on raw values; orienting it by HIGHER_IS_BETTER makes factors comparable
        orientation = 1 if HIGHER_IS_BETTER.get(factor, True) else -1
        log.info(f'{factor:<18} IC {summary["mean"]:+.4f}  t {summary["t_stat"]:+6.2f}  '
                 f'oriented t {orientation * summary["t_stat"]:+6.2f}  hit {summary["hit_rate"]:.0%}  '
                 f'spread {spread.mean():+.4f}  n {summary["n_periods"]}')


def _run_universe(con, args: argparse.Namespace, run_cfg: backtest_config.Config,
                  calendar_start: str, calendar_end: str) -> list[str]:
    """The names a run trades: the named universe, else the most liquid complete names in the range."""
    named = _resolve_universe(args, run_cfg)
    calendar = panels.trading_calendar(con, start=calendar_start, end=calendar_end)
    return panels.complete_universe(con, calendar,
                                    n_top=args.n_top or run_cfg.universe.n_top,
                                    rank_by=args.rank_by or run_cfg.universe.rank_by,
                                    require_sector=run_cfg.universe.require_sector,
                                    candidates=named)


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
    tickers = _run_universe(con, args, run_cfg, args.start, args.end)
    log.info(f'universe: {len(tickers):,} names priced across the whole range')

    market = market_data.build_market(con, tickers, start=args.start, end=args.end,
                                      exclude_checks=run_cfg.exclude_checks)
    params = run_cfg.signal(args.signal) | {'rebalance': args.freq or run_cfg.rebalance,
                                            'vintage': args.vintage,
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

    caveats = market_data.universe_caveats(con, tickers, args.end)
    if benchmark_note:
        caveats.append(benchmark_note)

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

    params = run_cfg.signal(args.signal) | {'rebalance': args.freq or run_cfg.rebalance,
                                            'vintage': args.vintage,
                                            'max_weight': run_cfg.limits.max_weight}
    result = walkforward.run_walk_forward(con, factory, windows, run_cfg, params)

    for caveat in market_data.universe_caveats(con, [], args.end):
        log.warning(caveat)
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

    if args.command == 'report':
        con = storage.connect(cfg.data_root, cfg.db, read_only=True)
        try:
            raise NotImplementedError('reporting lands in phase 8')
        finally:
            con.close()


if __name__ == '__main__':
    main()
