"""Pure shaping for the backtest page: run rows, curves and factor tables.

Kept free of Dash imports so it is testable standalone; the page module wires it to callbacks.
"""

import numpy as np
import pandas as pd

from investalyze.analysis import custom_factors
from investalyze.analysis.factors import BUILTIN_FACTORS, FACTORS, HIGHER_IS_BETTER
from investalyze.backtest.engine import CROSS_SECTIONAL

# the grid autosizes these to their contents; minWidth is a floor under that, since
# autoSize's canvas measurement has been seen to under-size a column when browser zoom or
# DPI scaling changes rendered text width independently of it, which otherwise reads as
# truncated data rather than a rendering quirk. The run id sits last because it identifies a
# run for the CLI rather than describing it; what a reader recognises a run by is when it was
# made and what it ran.
RUN_COLUMNS = [
    {'field': 'Created', 'headerName': 'Run made', 'minWidth': 150},
    {'field': 'Signal', 'headerName': 'Signal', 'minWidth': 130},
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


# --- what pressing a button will do ---

FREQUENCY_WORDS = {'D': 'daily', 'W': 'weekly', 'M': 'monthly', 'Q': 'quarterly'}


def _estimate_words(seconds: float | None) -> str:
    """A rough duration in words, or nothing when no build has been timed to estimate from."""
    if not seconds:
        return ''
    if seconds < 90:
        whole = max(1, round(seconds))
        return f', about {whole} second' + ('' if whole == 1 else 's')
    minutes = round(seconds / 60)
    return f', about {minutes} minute' + ('' if minutes == 1 else 's')


def _gap_reasons(gaps: dict) -> list[str]:
    """Why the stored panel cannot answer a request, each clause naming a concrete thing."""
    cadence = FREQUENCY_WORDS.get(gaps.get('build_frequency'), '')
    reasons = []
    if gaps['missing_dates'] and gaps['missing_dates'] == gaps['requested_dates']:
        reasons.append('the panel holds nothing for this version and these names yet')
    elif gaps['missing_dates']:
        reasons.append(f'{gaps["missing_dates"]} of the {gaps["requested_dates"]} {cadence} dates '
                       'it needs do not exist yet')
    if gaps['changed_factors']:
        reasons.append(f'{", ".join(gaps["changed_factors"])} was redefined since the last build')
    return reasons


def _scope_words(universe: str | None, n_tickers: int | None) -> str:
    """How the names a request covers are described in a sentence.

    An unknown or zero count names the scope in words instead. Saying a request covers no names when
    it covers every one of them is worse than not counting.
    """
    if not n_tickers:
        return f'the names in {universe}' if universe else 'the whole database'
    return f'{n_tickers:,} names in {universe}' if universe else f'{n_tickers:,} names'


def _build_words(gaps: dict, universe: str | None, version: str) -> str:
    """What a build would actually produce, in the terms the fields on the card are set in."""
    cadence = FREQUENCY_WORDS.get(gaps.get('build_frequency'), '')
    return (f'the {version.replace("_", " ")} panel: {gaps["missing_dates"]} {cadence} dates for '
            f'{_scope_words(universe, gaps["n_tickers"])}')


def _empty_note(gaps: dict) -> str:
    """What has no values for these names however often the panel is built.

    Kept apart from the reasons to build. A factor with no inputs for a name is not a gap anything
    can close, and putting it in a list headed by a button implies otherwise.
    """
    empty = gaps.get('empty_factors') or []
    if not empty:
        return ''
    names = ', '.join(empty)
    verb = 'has' if len(empty) == 1 else 'have'
    return (f' {names} {verb} no values for these names, and building will not change that: '
            'the inputs do not exist.')


def _needs_sentence(gaps: dict, universe: str | None, version: str) -> str:
    """The amber line: what pressing builds, why it has to, and what it still will not fix."""
    return (f'Builds {_build_words(gaps, universe, version)}'
            f'{_estimate_words(gaps["seconds_estimate"])}. '
            f'{"; ".join(_gap_reasons(gaps)).capitalize()}, and pressing this is what builds them.'
            f'{_empty_note(gaps)}')


def research_plan(gaps: dict, universe: str | None, version: str = 'as_reported') -> tuple[bool, str]:
    """Whether Evaluate can run from what is stored, and the sentence saying so."""
    if not _gap_reasons(gaps):
        return True, (f'Scores {len(FACTORS)} factors over {_scope_words(universe, gaps["n_tickers"])} '
                      f'from the stored panel.{_empty_note(gaps)}')
    return False, _needs_sentence(gaps, universe, version)


def _cadence_note(gaps: dict) -> str:
    """What a rebalance finer than the stored panel's cadence actually trades on.

    Not a fault: a session between panel dates reads the most recent one before it, which is what a
    reader could have known. Worth saying, because the signal then updates on the panel's schedule
    rather than the one the card asks to rebalance on.
    """
    panel, asked = gaps.get('panel_frequency'), gaps.get('rebalance')
    if not panel or not asked or panel == asked:
        return ''
    return (f' The panel updates {FREQUENCY_WORDS.get(panel, panel)}, so rebalances between its '
            'dates repeat the last scores.')


def _tradable_note(tradable: tuple[int, int | None] | None, universe: str | None) -> str:
    """How much of a universe a run will actually trade, said only when it is not all of it.

    A name that listed partway through the range is dropped, since a run needs one tradable set
    across its whole span. Over a long enough range that quietly turns a universe into one stock.
    """
    if tradable is None:
        return ''
    kept, total = tradable
    if not kept:
        return (f' None of the names in {universe} traded across this whole range, so there is '
                'nothing to trade; start later.')
    if total is None or kept == total:
        return ''
    where = f' in {universe}' if universe else ''
    dropped = total - kept
    verb = 'was' if dropped == 1 else 'were'
    return (f' Trades {kept} of the {total} names{where}: the other {dropped} {verb} not priced '
            'across the whole range.')


def backtest_plan(gaps: dict, kind: str, signal: str, universe: str | None,
                  version: str = 'as_reported',
                  tradable: tuple[int, int | None] | None = None) -> tuple[bool, str]:
    """Whether Run can go straight to the engine, and the sentence saying so."""
    if kind != CROSS_SECTIONAL:
        return True, f'{signal} reads prices only. Nothing to build.{_tradable_note(tradable, universe)}'
    if not _gap_reasons(gaps):
        return True, (f'Runs {signal} over {_scope_words(universe, gaps["n_tickers"])} from the '
                      f'stored panel.{_tradable_note(tradable, universe)}{_cadence_note(gaps)}'
                      f'{_empty_note(gaps)}')
    return False, _needs_sentence(gaps, universe, version)


def _universe_argv(universe: str | None, n_top: int | None, rank_by: str | None) -> list[str]:
    """The flags that narrow a run to one universe, left out where the card said nothing."""
    argv = []
    if universe:
        argv += ['--universe', universe]
    if n_top:
        argv += ['--n-top', str(int(n_top))]
    if rank_by:
        argv += ['--rank-by', rank_by]
    return argv


def factors_argv(start: str, end: str, freq: str, version: str, universe: str | None = None,
                 n_top: int | None = None, rank_by: str | None = None) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest factors` over one date range.

    A named universe is built whole, so Top N and Ranked by are left out rather than passed and
    ignored: the command line should say what the build will actually do.
    """
    argv = ['-m', 'investalyze.backtest', 'factors', '--start', start, '--end', end,
            '--freq', freq, '--data-version', version]
    argv += ['--universe', universe] if universe else _universe_argv(None, n_top, rank_by)
    scope = universe or 'whole database'
    return f'backtest factors ({version}, {freq}, {scope}, {start} to {end})', argv


def evaluate_argv(horizon: int, version: str, universe: str | None = None,
                  start: str | None = None, end: str | None = None,
                  build_panel: bool = False, panel_freq: str = 'M') -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest evaluate` across every factor.

    The dates narrow which stored panel rows are scored, so the range on the form means the same
    thing here as it does for the other two buttons.
    """
    argv = ['-m', 'investalyze.backtest', 'evaluate', '--horizon', str(horizon), '--data-version', version,
            '--panel-freq', panel_freq]
    if start:
        argv += ['--start', start]
    if end:
        argv += ['--end', end]
    if universe:
        argv += ['--universe', universe]
    if build_panel:
        argv += ['--build-panel']
    scope = universe or 'whole database'
    prefix = 'build panel, then evaluate' if build_panel else 'evaluate'
    return f'backtest {prefix} ({version}, {scope}, horizon {horizon})', argv


def delete_evaluation_argv(version: str, universe: str) -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest delete-evaluation` on one stored evaluation."""
    argv = ['-m', 'investalyze.backtest', 'delete-evaluation', '--data-version', version,
            '--universe', universe]
    return f'backtest delete-evaluation ({version}, {universe})', argv


def delete_evaluation_prompt(scope: dict) -> str:
    """The confirmation text for deleting one stored evaluation, naming what goes with it."""
    return '\n'.join([
        f'Delete the {scope["version"].replace("_", " ")} evaluation stored under '
        f'{scope["universe"]}?',
        '',
        f'  {scope["factors"]} factors, {scope["first"]} to {scope["last"]}',
        f'  {scope["ic_rows"]:,} IC rows and {scope["quantile_rows"]:,} quantile rows',
        '',
        'Evaluate factors over the same universe rebuilds it.',
    ])


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


def run_argv(signal: str, start: str, end: str, freq: str, universe: str | None = None,
             n_top: int | None = None, rank_by: str | None = None, version: str = 'as_reported',
             build_panel: bool = False, panel_freq: str = 'M') -> tuple[str, list[str]]:
    """Argv for `python -m investalyze.backtest run` on one signal.

    The version travels with the run because it decides which panel a cross-sectional signal ranks;
    without it the choice on the form went nowhere and every run scored as-reported values. The two
    frequencies travel separately: `--freq` is how often the portfolio trades, `--panel-freq` how
    often the factor values behind it are recomputed.
    """
    argv = (['-m', 'investalyze.backtest', 'run', '--signal', signal, '--start', start, '--end', end,
             '--freq', freq, '--data-version', version, '--panel-freq', panel_freq]
            + _universe_argv(universe, n_top, rank_by))
    if build_panel:
        argv += ['--build-panel']
    scope = universe or 'whole database'
    prefix = 'build panel, then run' if build_panel else 'run'
    return f'backtest {prefix} ({signal}, {scope}, {start} to {end})', argv
