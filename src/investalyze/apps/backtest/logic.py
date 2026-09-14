"""Pure shaping for the Backtest page: run rows, curves, job commands and run settings.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import json

import pandas as pd

from investalyze.research import backtest as factor_backtest
from investalyze.research import timing

# the grid autosizes these to their contents; minWidth is a floor under that, since
# autoSize's canvas measurement has been seen to under-size a column when browser zoom or
# DPI scaling changes rendered text width independently of it
RUN_COLUMNS = [
    {'field': 'Created', 'headerName': 'Run made', 'minWidth': 150},
    {'field': 'Signal', 'headerName': 'Strategy', 'minWidth': 180},
    {'field': 'StartDate', 'headerName': 'From', 'minWidth': 100},
    {'field': 'EndDate', 'headerName': 'To', 'minWidth': 100},
    {'field': 'NTickers', 'headerName': 'Names', 'minWidth': 90},
    {'field': 'Cagr', 'headerName': 'CAGR', 'minWidth': 90,
     'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
    {'field': 'Sharpe', 'headerName': 'Sharpe', 'minWidth': 90,
     'valueFormatter': {'function': 'd3.format(".2f")(params.value)'}},
    {'field': 'MaxDrawdown', 'headerName': 'Max DD', 'minWidth': 95,
     'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
    {'field': 'Turnover', 'headerName': 'Turnover', 'minWidth': 95,
     'valueFormatter': {'function': 'd3.format(".2f")(params.value)'}},
    {'field': 'NTrades', 'headerName': 'Fills', 'minWidth': 90},
    {'field': 'RunId', 'headerName': 'Run id', 'minWidth': 130},
]

QUANTILE_CHOICES = (3, 5, 10)
UNIVERSE_SIZES = (500, 1000, 2000)


def rebase(curve: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rebase the named columns to 100 at the first session each of them has a value for."""
    out = curve.copy()
    for column in columns:
        if column not in out.columns:
            continue
        series = out[column]
        first = series.dropna()
        out[column] = series / first.iloc[0] * 100 if len(first) else series
    return out


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """Fractional drawdown of a curve, for the band under the equity chart."""
    return equity / equity.cummax() - 1


def factor_portfolio_argv(factor_names: list[str], start: str, end: str, rebalance: str, quantiles: int,
                          universe_size: int, holdout: str | None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.research backtest` on one or more factors."""
    argv = ['-m', 'investalyze.research', 'backtest', ','.join(factor_names), '--start', start, '--end', end,
            '--rebalance', rebalance, '--quantiles', str(int(quantiles)), '--size', str(int(universe_size)),
            '--holdout', holdout or '']
    return f'backtest {factor_backtest.signal_label(factor_names)} ({start} to {end})', argv


def timing_rule_argv(entry_factor: str, *, exit_factor: str | None, exit_on_entry_false: bool,
                     trailing_stop_percent: float | None, max_positions: int, priority_factor: str | None,
                     universe: str | None, universe_size: int, start: str, end: str,
                     holdout: str | None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.research timing` on one timing rule."""
    argv = ['-m', 'investalyze.research', 'timing', entry_factor, '--start', start, '--end', end,
            '--max-positions', str(int(max_positions)), '--size', str(int(universe_size)), '--holdout', holdout or '']
    if exit_factor:
        argv += ['--exit-factor', exit_factor]
    if not exit_on_entry_false:
        argv += ['--keep-on-entry-false']
    trailing_stop = trailing_stop_percent / 100 if trailing_stop_percent else None
    if trailing_stop:
        argv += ['--trailing-stop', f'{trailing_stop:g}']
    if priority_factor:
        argv += ['--priority-factor', priority_factor]
    if universe:
        argv += ['--universe', universe]
    label = timing.signal_label(entry_factor, exit_factor=exit_factor, exit_on_entry_false=exit_on_entry_false,
                                trailing_stop=trailing_stop)
    return f'backtest {label} ({start} to {end})', argv


def timing_rule_problem(*, exit_factor: str | None, exit_on_entry_false: bool,
                        trailing_stop_percent: float | None) -> str | None:
    """Why a timing rule cannot run, or None when it can."""
    if not (exit_factor or exit_on_entry_false or trailing_stop_percent):
        return 'A timing rule needs at least one exit: the entry factor turning false, an exit factor, or a trailing stop.'
    if trailing_stop_percent is not None and not 0 <= trailing_stop_percent < 100:
        return 'The trailing stop is a percentage between 0 and 100.'
    return None


def delete_argv(run_ids: list[str]) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest delete` on one or more stored runs."""
    argv = ['-m', 'investalyze.backtest', 'delete', '--run-id', *run_ids]
    scope = run_ids[0] if len(run_ids) == 1 else f'{len(run_ids)} runs'
    return f'backtest delete ({scope})', argv


def delete_prompt(runs: list[dict]) -> str:
    """The confirmation text for deleting the selected runs, naming each one and what it holds."""
    total_fills = 0
    lines = []
    for run in runs:
        fills = int(run['NTrades'] or 0)
        total_fills += fills
        lines.append(f'{run["Created"]}  {run["Signal"]}, {run["StartDate"]} to {run["EndDate"]}, '
                     f'{fills:,} fills, run {run["RunId"]}')
    if len(runs) == 1:
        head = 'Delete this run? Its equity curve, fills and written report go with it.'
    else:
        head = (f'Delete these {len(runs)} runs? Their equity curves, {total_fills:,} fills in total '
                'and written reports go with them.')
    return '\n'.join([head, ''] + lines)


def run_settings_rows(config_json: str | None) -> list[dict]:
    """The settings a run was made with, as label and value rows."""
    if not config_json:
        return []
    config = json.loads(config_json)
    params = config.get('params', {})
    rows = []
    for name, expression in params.get('expressions', {}).items():
        rows.append({'setting': f'Factor {name}', 'value': expression})
    labels = (('entry_factor', 'Entry factor'), ('exit_on_entry_false', 'Exit when entry is false'),
              ('exit_factor', 'Exit factor'), ('trailing_stop', 'Trailing stop'), ('max_positions', 'Max positions'),
              ('priority_factor', 'Priority factor'), ('n_quantiles', 'Quantiles'), ('rebalance', 'Rebalance'),
              ('universe', 'Universe'), ('universe_size', 'Universe size'), ('holdout', 'Out-of-sample from'))
    for key, label in labels:
        if params.get(key) not in (None, ''):
            value = params[key]
            if key == 'rebalance':
                value = {'D': 'daily'}.get(value, factor_backtest.REBALANCE.get(value, value))
            elif key == 'trailing_stop':
                value = f'{value:.0%}'
            elif key == 'exit_on_entry_false':
                value = 'yes' if value else 'no'
            rows.append({'setting': label, 'value': str(value)})
    rows.append({'setting': 'Benchmark', 'value': str(config.get('benchmark', ''))})
    costs = config.get('costs', {})
    if costs:
        rows.append({'setting': 'Costs', 'value': f'{costs.get("slippage_bps", 0):g} bps slippage, '
                                                  f'{costs.get("commission_bps", 0):g} bps commission, '
                                                  f'impact coefficient {costs.get("impact_coef", 0):g}'})
    return rows


def holdout_of(config_json: str | None) -> pd.Timestamp | None:
    """The out-of-sample start date recorded with a run, if it has one."""
    if not config_json:
        return None
    holdout = json.loads(config_json).get('params', {}).get('holdout')
    return pd.Timestamp(holdout) if holdout else None


def run_periods(curve: pd.DataFrame, holdout: pd.Timestamp | None) -> pd.DataFrame:
    """Full sample, in-sample and out-of-sample statistics of a stored equity curve."""
    if curve.empty:
        return pd.DataFrame()
    dates = pd.DatetimeIndex(curve['Date'])
    equity = pd.Series(curve['Equity'].to_numpy(), index=dates)
    benchmark = pd.Series(curve['BenchmarkEquity'].to_numpy(), index=dates)
    return factor_backtest.period_stats(equity, None if benchmark.isna().all() else benchmark, holdout_start=holdout)
