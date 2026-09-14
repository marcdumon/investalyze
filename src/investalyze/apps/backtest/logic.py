"""Pure shaping for the Backtest page: run rows, curves, job commands and run settings.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import json

import pandas as pd

from investalyze.analysis import factors_pit
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


def _scope_arguments(start: str, end: str, universe_size: int, holdout: str | None) -> list[str]:
    """The date range, universe size and out-of-sample arguments every research run takes."""
    return ['--start', start, '--end', end, '--size', str(int(universe_size)), '--holdout', holdout or '']


def factor_portfolio_argv(portfolio: factor_backtest.FactorPortfolio, *, start: str, end: str, universe_size: int,
                          holdout: str | None) -> tuple[str, list[str]]:
    """Job title and argv for `python -m investalyze.research backtest` on one factor portfolio."""
    argv = ['-m', 'investalyze.research', 'backtest', *portfolio.arguments(),
            *_scope_arguments(start, end, universe_size, holdout)]
    return f'backtest {portfolio.label} ({start} to {end})', argv


def timing_rule_argv(rule: timing.TimingRule, *, universe: str | None, universe_size: int, start: str, end: str,
                     holdout: str | None) -> tuple[str, list[str]]:
    """Job title and argv for `python -m investalyze.research timing` on one timing rule."""
    argv = ['-m', 'investalyze.research', 'timing', *rule.arguments(), *_scope_arguments(start, end, universe_size, holdout)]
    if universe:
        argv += ['--universe', universe]
    return f'backtest {rule.label} ({start} to {end})', argv


def trailing_stop_fraction(percent: float | str | None) -> float | None:
    """The trailing stop typed on the page, in percent, as a fraction; None when it is left blank or zero."""
    if percent in (None, ''):
        return None
    value = float(percent)
    if not 0 <= value < 100:
        raise ValueError('The trailing stop is a percentage between 0 and 100.')
    return value / 100 or None


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
                value = factors_pit.Rebalance(value).label
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
