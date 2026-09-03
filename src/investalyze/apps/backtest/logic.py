"""Pure shaping for the backtest page: run rows, curves and factor tables.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import numpy as np
import pandas as pd

from investalyze.analysis import custom_factors
from investalyze.analysis.factors import BUILTIN_FACTORS, HIGHER_IS_BETTER

RUN_COLUMNS = [
    {'field': 'RunId', 'headerName': 'Run', 'width': 130},
    {'field': 'Signal', 'headerName': 'Signal', 'width': 160},
    {'field': 'StartDate', 'headerName': 'From', 'width': 110},
    {'field': 'EndDate', 'headerName': 'To', 'width': 110},
    {'field': 'NTickers', 'headerName': 'Names', 'width': 90},
    {'field': 'Cagr', 'headerName': 'CAGR', 'width': 100, 'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
    {'field': 'Sharpe', 'headerName': 'Sharpe', 'width': 100, 'valueFormatter': {'function': 'd3.format(".2f")(params.value)'}},
    {'field': 'MaxDrawdown', 'headerName': 'Max DD', 'width': 100, 'valueFormatter': {'function': 'd3.format(".1%")(params.value)'}},
    {'field': 'Turnover', 'headerName': 'Turnover', 'width': 110, 'valueFormatter': {'function': 'd3.format(".2f")(params.value)'}},
    {'field': 'NTrades', 'headerName': 'Fills', 'width': 90},
]

FACTOR_COLUMNS = [
    {'field': 'Factor', 'headerName': 'Factor', 'width': 170},
    {'field': 'Horizon', 'headerName': 'Horizon', 'width': 100},
    {'field': 'MeanIc', 'headerName': 'Mean IC', 'width': 110, 'valueFormatter': {'function': 'd3.format("+.4f")(params.value)'}},
    {'field': 'OrientedIc', 'headerName': 'Oriented IC', 'width': 120,
     'valueFormatter': {'function': 'd3.format("+.4f")(params.value)'}},
    {'field': 'TStat', 'headerName': 't', 'width': 90, 'valueFormatter': {'function': 'd3.format("+.2f")(params.value)'}},
    {'field': 'HitRate', 'headerName': 'Hit rate', 'width': 100, 'valueFormatter': {'function': 'd3.format(".0%")(params.value)'}},
    {'field': 'Periods', 'headerName': 'Periods', 'width': 100},
]


def orient_ic(table: pd.DataFrame) -> pd.DataFrame:
    """Add the orientation-adjusted IC and its t-statistic, so factors can be ranked against each other.

    The stored coefficient is on raw factor values, where a negative reading is the expected one for
    a factor whose lower values are the better ones. Flipping those makes one column comparable
    across the whole library.
    """
    if table.empty:
        return table
    out = table.copy()
    orientation = out['Factor'].map(lambda name: 1 if HIGHER_IS_BETTER.get(name, True) else -1)
    out['OrientedIc'] = out['MeanIc'] * orientation
    standard_error = out['StdIc'] / np.sqrt(out['Periods'].clip(lower=1))
    out['TStat'] = (out['OrientedIc'] / standard_error).replace([np.inf, -np.inf], np.nan)
    return out.sort_values('TStat', ascending=False, ignore_index=True)


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


def run_summary_text(run: pd.Series) -> str:
    """The one-line description of a run shown above its chart."""
    return (f'{run["Signal"]} on {run["NTickers"]:,} names, {run["StartDate"]} to {run["EndDate"]}, '
            f'benchmark {run["Benchmark"]}')


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """Fractional drawdown of a curve, for the band under the equity chart."""
    return equity / equity.cummax() - 1


DEFINITION_COLUMNS = [
    {'field': 'name', 'headerName': 'Factor', 'width': 190},
    {'field': 'kind', 'headerName': 'Kind', 'width': 110},
    {'field': 'formula', 'headerName': 'Formula', 'flex': 1},
    {'field': 'direction', 'headerName': 'Good end', 'width': 120},
    {'field': 'description', 'headerName': 'Note', 'flex': 1},
]


def definition_rows(customs: dict[str, custom_factors.CustomFactor]) -> list[dict]:
    """The defined factors as grid rows."""
    rows = []
    for name, factor in customs.items():
        rows.append({
            'name': name,
            'kind': factor.kind,
            'formula': custom_factors.describe(factor),
            'direction': 'higher' if factor.higher_is_better else 'lower',
            'description': factor.description,
        })
    return rows


def definition_fields(kind: str, expression: str | None, numerator: str | None,
                      denominator: str | None, higher_is_better: bool,
                      description: str | None) -> dict:
    """The form's values as the fields a definition is written from."""
    if kind == custom_factors.EXPRESSION:
        return {'expression': expression, 'higher_is_better': higher_is_better,
                'description': description}
    return {'numerator': numerator, 'denominator': denominator,
            'higher_is_better': higher_is_better, 'description': description}


def validate_definition(name: str | None, fields: dict,
                        existing: dict[str, custom_factors.CustomFactor]) -> str | None:
    """The reason a definition cannot be saved, or None when it is good.

    The candidate is serialized and parsed exactly as the file would be, so what the form accepts and
    what a run accepts can never drift apart.
    """
    name = (name or '').strip()
    if not name:
        return 'give the factor a name'
    if name in existing:
        return f'{name} is already defined; delete it first to redefine it'
    try:
        custom_factors.parse(custom_factors.serialize_factor(name, fields), BUILTIN_FACTORS)
    except ValueError as failure:
        return str(failure)
    return None


def preview_definition(name: str | None, fields: dict) -> str:
    """The TOML the form would write, shown so the file stays legible rather than magic."""
    return custom_factors.serialize_factor((name or 'unnamed').strip() or 'unnamed', fields)


# --- launching a run ---

def _universe_argv(universe: str | None, n_top: int | None, rank_by: str | None) -> list[str]:
    """The flags that narrow a command to one universe, left out where the card said nothing."""
    argv = []
    if universe:
        argv += ['--universe', universe]
    if n_top:
        argv += ['--n-top', str(int(n_top))]
    if rank_by:
        argv += ['--rank-by', rank_by]
    return argv


def factors_argv(start: str, end: str, freq: str, vintage: str,
                                universe: str | None = None, n_top: int | None = None,
                                rank_by: str | None = None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest factors` over one date range."""
    argv = (['-m', 'investalyze.backtest', 'factors', '--start', start, '--end', end,
             '--freq', freq, '--vintage', vintage] + _universe_argv(universe, n_top, rank_by))
    scope = universe or 'whole database'
    return f'backtest factors ({vintage}, {freq}, {scope}, {start} to {end})', argv


def evaluate_argv(horizon: int, vintage: str) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest evaluate` across every factor."""
    argv = ['-m', 'investalyze.backtest', 'evaluate', '--horizon', str(horizon), '--vintage', vintage]
    return f'backtest evaluate ({vintage}, horizon {horizon})', argv


def run_argv(signal: str, start: str, end: str, freq: str,
                            universe: str | None = None, n_top: int | None = None,
                            rank_by: str | None = None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest run` on one signal."""
    argv = (['-m', 'investalyze.backtest', 'run', '--signal', signal, '--start', start, '--end', end,
             '--freq', freq] + _universe_argv(universe, n_top, rank_by))
    scope = universe or 'whole database'
    return f'backtest run ({signal}, {scope}, {start} to {end})', argv
