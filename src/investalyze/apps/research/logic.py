"""Text and thresholds for the Factors page.

Pure functions, no database and no Dash, so every threshold and message is testable on its own.
"""

import numpy as np
import pandas as pd

from investalyze.research import formula

# at or above this absolute correlation a new factor is redundant with an existing one
REDUNDANT_CORRELATION = 0.9
# at or above this the two factors select largely the same names
OVERLAP_CORRELATION = 0.7
# below this share of the universe, results describe a subset rather than the market
LOW_COVERAGE = 25.0
SIGNIFICANT_T = 2.0

FIELD_COLUMNS = [
    {'field': 'name', 'headerName': 'Field', 'width': 240},
    {'field': 'statement', 'headerName': 'Statement', 'width': 110},
    {'field': 'period', 'headerName': 'Period', 'width': 90},
    {'field': 'description', 'headerName': 'Description', 'flex': 1},
]


def validate_expression(expression: str, known: set) -> tuple[bool, str]:
    """Whether an expression is valid, with the message to show."""
    if not expression or not expression.strip():
        return False, 'Enter an expression.'
    try:
        formula.validate(expression, known)
    except ValueError as error:
        return False, str(error)
    return True, 'Expression is valid.'


def library_rows(library: dict, summaries: dict | None = None) -> list[dict]:
    """One row per factor with its recorded results: accepted first, then candidates, then rejected."""
    order = {'accepted': 0, 'candidate': 1, 'rejected': 2}
    summaries = summaries or {}
    rows = []
    for name in sorted(library):
        factor = library[name]
        recorded = summaries.get(name, {})
        rows.append({'name': name, 'status': factor.status, 'description': factor.description,
                     'direction': 'higher' if factor.higher_is_better else 'lower',
                     'expression': factor.expression, 'coverage': recorded.get('coverage'),
                     't_stat': recorded.get('t_stat'), 'oos_t_stat': recorded.get('oos_t_stat'),
                     'sharpe': recorded.get('sharpe'), 'oos_sharpe': recorded.get('oos_sharpe'),
                     '_order': order.get(factor.status, 3)})
    rows.sort(key=lambda row: (row['_order'], row['name']))
    for row in rows:
        row.pop('_order')
    return rows


def coverage_note(table: pd.DataFrame) -> tuple[str, str]:
    """Coverage on the latest year, and the alert colour."""
    if table.empty:
        return 'No coverage.', 'red'
    latest = table.iloc[-1]
    share = float(latest['share'])
    text = (f'Coverage: {latest["names"]:,.0f} of {latest["universe"]:,.0f} names per date '
            f'in {int(latest["year"])} ({share:.0f}%).')
    if share < LOW_COVERAGE:
        return text + ' Low coverage: results describe a subset of the universe.', 'orange'
    return text, 'green'


def correlation_note(scores: pd.Series) -> tuple[str, str] | None:
    """The highest absolute correlation with an existing factor, and the alert colour."""
    if scores is None or scores.empty:
        return None
    strongest = scores.abs().idxmax()
    value = float(scores.loc[strongest])
    lead = f'Highest correlation: {strongest} ({value:+.2f}).'
    if abs(value) >= REDUNDANT_CORRELATION:
        return f'{lead} Redundant with an existing factor.', 'red'
    if abs(value) >= OVERLAP_CORRELATION:
        return f'{lead} High overlap with an existing factor.', 'orange'
    return f'{lead} Low overlap with existing factors.', 'green'


def redundancy_warning(scores: pd.Series) -> str | None:
    """The correlation note, only when the overlap is high or the factor is redundant."""
    found = correlation_note(scores)
    if found is None or found[1] == 'green':
        return None
    return found[0]


def turnover_note(autocorrelation: float) -> str:
    """Rank autocorrelation, described as a turnover level."""
    if autocorrelation != autocorrelation:
        return 'Rank autocorrelation: not enough dates.'
    if autocorrelation > 0.95:
        return f'Rank autocorrelation {autocorrelation:.2f}: low turnover.'
    if autocorrelation > 0.8:
        return f'Rank autocorrelation {autocorrelation:.2f}: moderate turnover.'
    return f'Rank autocorrelation {autocorrelation:.2f}: high turnover, transaction costs will be significant.'


def distribution_rows(summary: dict) -> list[dict]:
    """Summary statistics as label and value rows."""
    labels = {'count': 'Count', 'missing': 'Missing', 'mean': 'Mean', 'std': 'Std dev',
              'p1': '1st pct', 'p25': '25th pct', 'median': 'Median', 'p75': '75th pct', 'p99': '99th pct'}
    rows = []
    for key, label in labels.items():
        if key in summary:
            value = summary[key]
            shown = f'{value:,}' if isinstance(value, int) else f'{value:,.4f}'
            rows.append({'statistic': label, 'value': shown})
    return rows


def ic_note(result: dict) -> tuple[str, str]:
    """The IC summary line, and the alert colour."""
    summary = result['summary']
    t_stat = summary.get('oriented_t', np.nan)
    names = summary.get('names_median', np.nan)
    if t_stat != t_stat:
        return 'Not enough dates for IC analysis.', 'gray'

    text = f'IC {summary["oriented_ic"]:+.4f}, t-stat {t_stat:+.2f}, {summary["n_periods"]} dates'
    if names == names:
        text += f', median {names:,.0f} names'
    if abs(t_stat) < SIGNIFICANT_T:
        return text + '. Not significant (|t| < 2).', 'gray'
    if not result.get('monotonic'):
        return text + '. Significant, but quantile returns are not monotonic.', 'orange'
    return text + '. Significant, with monotonic quantile returns.', 'green'


def multiple_testing_note(result: dict) -> str:
    """Expected false positives for the size of the factor library."""
    return (f'Multiple testing: {result["tested"]} factors tested, about '
            f'{result["expected_false_positives"]:.1f} expected false positives at p<0.05.')


def ic_decay_note(decay: pd.DataFrame, *, higher_is_better: bool = True) -> str:
    """The horizon with the highest IC in the factor's direction, and whether IC reverses sign."""
    usable = decay.dropna(subset=['mean_ic'])
    if usable.empty:
        return 'IC decay: no readable horizon.'
    oriented = usable['mean_ic'] if higher_is_better else -usable['mean_ic']
    best = usable.loc[oriented.idxmax()]
    if oriented.max() <= 0:
        return 'IC is negative at every horizon: the factor predicts in the opposite direction.'
    text = f'IC peaks at the {int(best["horizon"])}-day horizon.'
    reversed_at = usable.loc[oriented < 0, 'horizon'].astype(int).tolist()
    if reversed_at:
        text += f' IC turns negative at {", ".join(f"{h}-day" for h in reversed_at)}.'
    return text


def holdout_note(split: pd.DataFrame) -> tuple[str, str] | None:
    """In-sample against out-of-sample t-statistic, and whether the IC holds out of sample."""
    if split is None or split.empty or set(split['period']) != {'In-sample', 'Out-of-sample'}:
        return None
    table = split.set_index('period')
    inside, outside = float(table.loc['In-sample', 't_stat']), float(table.loc['Out-of-sample', 't_stat'])
    text = (f'In-sample t-stat {inside:+.2f} ({table.loc["In-sample", "start"]} to {table.loc["In-sample", "end"]}), '
            f'out-of-sample t-stat {outside:+.2f} ({table.loc["Out-of-sample", "start"]} to '
            f'{table.loc["Out-of-sample", "end"]}).')
    if inside != inside or outside != outside:
        return text, 'gray'
    if outside <= 0 < inside:
        return text + ' IC reverses sign out of sample.', 'red'
    if abs(inside) >= SIGNIFICANT_T and abs(outside) < SIGNIFICANT_T:
        return text + ' Significant in-sample only.', 'orange'
    if abs(outside) >= SIGNIFICANT_T:
        return text + ' Significant out of sample.', 'green'
    return text, 'gray'
