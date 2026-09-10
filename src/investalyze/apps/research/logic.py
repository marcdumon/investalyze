"""Turning what the workbench computes into what it says.

Pure functions, no database and no Dash, so the wording and the thresholds behind every judgement
the page makes are testable on their own.
"""

import numpy as np
import pandas as pd

from investalyze.research import formula

# above this, a new idea is a rewording of one already in the library rather than a new one
DUPLICATE_CORRELATION = 0.9
# and above this it overlaps enough that the two will mostly pick the same names
OVERLAP_CORRELATION = 0.7
# below this share of the universe, a reading describes a corner of the market, not the market
THIN_COVERAGE = 25.0
# the conventional bar, kept explicit so the page can say what it is applying
SIGNIFICANT_T = 2.0

FIELD_COLUMNS = [
    {'field': 'name', 'headerName': 'Name', 'width': 240},
    {'field': 'statement', 'headerName': 'Statement', 'width': 110},
    {'field': 'window', 'headerName': 'Window', 'width': 90},
    {'field': 'source', 'headerName': 'Reads', 'flex': 1},
]

EXTREME_COLUMNS = [
    {'field': 'Ticker', 'width': 90},
]


def validate_expression(expression: str, known: set) -> tuple[bool, str]:
    """Whether an expression is usable, and what to say when it is not."""
    if not expression or not expression.strip():
        return False, 'Write an expression to see what it does.'
    try:
        formula.validate(expression, known)
    except ValueError as error:
        return False, str(error)
    return True, 'Expression is valid.'


def library_rows(library: dict) -> list[dict]:
    """One row per idea, in the order a decision reads them: kept first, then drafts."""
    order = {'kept': 0, 'draft': 1, 'archived': 2}
    rows = []
    for name in sorted(library):
        idea = library[name]
        rows.append({'name': name, 'status': idea.status, 'description': idea.description,
                     'direction': 'high' if idea.higher_is_better else 'low',
                     'expression': idea.expression, '_order': order.get(idea.status, 3)})
    rows.sort(key=lambda row: (row['_order'], row['name']))
    for row in rows:
        row.pop('_order')
    return rows


def coverage_headline(table: pd.DataFrame) -> tuple[str, str]:
    """How much of the universe this idea actually speaks about, and how alarmed to be."""
    if table.empty:
        return 'No coverage at all.', 'red'
    latest = table.iloc[-1]
    share = float(latest['share'])
    text = (f'{int(latest["names"]):,} of {int(latest["universe"]):,} names carry a value '
            f'in {int(latest["year"])} ({share:.0f}%).')
    if share < THIN_COVERAGE:
        return text + ' That is a corner of the market, so read every later number as being about it.', 'orange'
    return text, 'green'


def similarity(scores: pd.Series) -> tuple[str, str] | None:
    """How much this idea overlaps the closest one already in the library, and how much to worry.

    A sign flip does not make an idea new, so the comparison is on absolute correlation.
    """
    if scores is None or scores.empty:
        return None
    strongest = scores.abs().idxmax()
    value = float(scores.loc[strongest])
    lead = f'Closest existing idea: {strongest} at {value:+.2f}.'
    if abs(value) >= DUPLICATE_CORRELATION:
        return f'{lead} That is close to a rewording of an idea you already have.', 'red'
    if abs(value) >= OVERLAP_CORRELATION:
        return (f'{lead} They will pick largely the same names, so check it earns a place beside it.',
                'orange')
    return f'{lead} That is different enough to be worth its own place.', 'green'


def duplicate_warning(scores: pd.Series) -> str | None:
    """The overlap message, but only when it is worth interrupting for."""
    found = similarity(scores)
    if found is None or found[1] == 'green':
        return None
    return found[0]


def turnover_note(autocorrelation: float) -> str:
    """What a ranking's persistence implies for what trading it will cost."""
    if autocorrelation != autocorrelation:
        return 'Not enough dates to say how fast the ranking moves.'
    if autocorrelation > 0.95:
        return f'Rank autocorrelation {autocorrelation:.2f}: very slow moving, so it will trade rarely.'
    if autocorrelation > 0.8:
        return f'Rank autocorrelation {autocorrelation:.2f}: moves gradually, so turnover stays modest.'
    return (f'Rank autocorrelation {autocorrelation:.2f}: the ranking churns, so costs will eat a '
            f'large share of any edge.')


def distribution_rows(summary: dict) -> list[dict]:
    """The shape of an idea's values, as label and value pairs."""
    labels = {'count': 'Values', 'missing': 'Missing', 'mean': 'Mean', 'std': 'Std dev',
              'p1': '1st pct', 'p25': '25th pct', 'median': 'Median', 'p75': '75th pct', 'p99': '99th pct'}
    rows = []
    for key, label in labels.items():
        if key in summary:
            value = summary[key]
            shown = f'{value:,}' if isinstance(value, int) else f'{value:,.4f}'
            rows.append({'measure': label, 'value': shown})
    return rows


def verdict_headline(answer: dict) -> tuple[str, str]:
    """The one sentence a quant reads first, and the colour that carries its weight."""
    summary = answer['summary']
    t_stat = summary.get('oriented_t', np.nan)
    names = summary.get('names_median', np.nan)
    if t_stat != t_stat:
        return 'Not enough overlapping dates to judge this idea.', 'gray'

    verdict = f'IC {summary["oriented_ic"]:+.4f}, t {t_stat:+.2f}, over {summary["n_periods"]} dates'
    if names == names:
        verdict += f', measured on {names:,.0f} names'
    if abs(t_stat) < SIGNIFICANT_T:
        return verdict + '. That is inside the range luck produces.', 'gray'
    if not answer.get('monotonic'):
        return verdict + '. Significant, but the quantiles are not ordered, so treat it carefully.', 'orange'
    return verdict + '. Significant, and the quantiles are ordered.', 'green'


def multiple_testing_note(answer: dict) -> str:
    """How many of a library this size look significant by luck alone."""
    tested, expected = answer['tested'], answer['expected_false_positives']
    return (f'You have {tested} ideas. At p<0.05 about {expected:.1f} of them will look significant '
            f'by chance, so a borderline t-statistic on the {tested}th idea is weaker than on the first.')


def decay_note(decay: pd.DataFrame) -> str:
    """What the decay curve argues for as a rebalance frequency."""
    usable = decay.dropna(subset=['mean_ic'])
    if usable.empty:
        return 'No decay reading.'
    best = usable.loc[usable['mean_ic'].abs().idxmax()]
    return (f'The edge is strongest at {int(best["horizon"])} sessions, which is the rebalance '
            f'frequency this argues for.')
