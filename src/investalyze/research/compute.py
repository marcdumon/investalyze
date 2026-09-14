"""Computing factor values.

A factor reads data fields and other factors, so computing one means resolving its dependencies,
fetching only those fields, and evaluating the library in dependency order. Only the fields an
expression names are queried, which keeps a sampled diagnostic run fast.

Time-series function windows count trading days. When any factor involved uses one, the panel is
evaluated on every trading session, starting early enough to fill the longest windows, and the values
are then read on the requested dates. The same expression therefore gives the same value whether it is
asked for monthly or daily.
"""

import pandas as pd

from investalyze.analysis import factors_pit, panels
from investalyze.research import factors, fields, formula


def required_fields(names: list[str], library: dict, catalog: dict) -> set[str]:
    """Every data field the given factors need, following references through other factors."""
    needed: set[str] = set()
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dependency in sorted(library[name].dependencies):
            if dependency in library:
                visit(dependency)
            elif dependency in catalog or dependency in fields.IDENTITY:
                needed.add(dependency)
            else:
                raise ValueError(f'factor {name!r} reads unknown name {dependency!r}')

    for name in names:
        visit(name)
    return needed


def apply_universe(frame: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """The panel with an `in_universe` column: whether the name was in the universe on each row's date.

    Membership is set on the dates `members` lists and holds until the next of them, so a daily panel
    between two rebalance dates uses the universe chosen on the earlier one. Rows before the first
    membership date are outside the universe.
    """
    marked = members[['Date', 'Ticker']].copy()
    marked['Date'] = pd.to_datetime(marked['Date'])
    member_dates = pd.DatetimeIndex(sorted(marked['Date'].unique()))
    dates = pd.DatetimeIndex(pd.to_datetime(frame['Date']))
    positions = member_dates.searchsorted(dates, side='right') - 1
    effective = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns]')
    known = positions >= 0
    effective[known] = member_dates[positions[known]]
    keyed = frame.assign(Date=dates, _member_date=effective.to_numpy())
    marked = marked.rename(columns={'Date': '_member_date'}).assign(**{formula.IN_UNIVERSE: True})
    joined = keyed.merge(marked, on=['_member_date', 'Ticker'], how='left').drop(columns=['_member_date'])
    joined[formula.IN_UNIVERSE] = joined[formula.IN_UNIVERSE].fillna(False).astype(bool)
    return joined


def lookback_days(names: list[str], library: dict) -> int:
    """Trading days of history the given factors need, as the sum of every time-series window involved.

    Summing overstates the need of windows used side by side, and is exact for windows nested inside
    each other, which is the case that must not run short.
    """
    total = 0
    seen: set[str] = set()

    def visit(name: str) -> None:
        nonlocal total
        if name in seen:
            return
        seen.add(name)
        total += sum(formula.time_series_windows(library[name].expression))
        for dependency in library[name].dependencies:
            if dependency in library:
                visit(dependency)

    for name in names:
        visit(name)
    return total


def compute(con, names: list[str], library: dict, *, as_of: pd.DatetimeIndex,
            tickers: list[str] | None = None, version: str = factors_pit.AS_REPORTED,
            members: pd.DataFrame | None = None) -> pd.DataFrame:
    """A long panel of `Ticker`, `Date` and one column per requested factor.

    Factors are evaluated in dependency order, so a factor reading another sees its column already
    computed. Intermediate factors are kept in the frame, so every input to a value can be inspected.
    """
    catalog = fields.catalog(con)
    needed = required_fields(names, library, catalog)
    if members is not None:
        tickers = sorted(members['Ticker'].unique())
    requested = pd.DatetimeIndex(pd.to_datetime(as_of))
    history = lookback_days(names, library)
    grid = requested
    if history:
        # about 1.5 calendar days per trading day, plus slack for holidays around the start
        start = (requested.min() - pd.Timedelta(days=int(history * 1.5) + 14)).date().isoformat()
        grid = panels.trading_calendar(con, start=start, end=requested.max().date().isoformat())
    frame = fields.panel(con, needed, as_of=grid, version=version, tickers=tickers)
    if frame.empty:
        return frame
    if members is not None:
        frame = apply_universe(frame, members)

    wanted = set(names)
    for name in factors.evaluation_order(library, set(catalog) | set(fields.IDENTITY)):
        if name in wanted or _feeds(name, names, library):
            frame[name] = formula_evaluate(library[name].expression, frame)
    if members is not None:
        for name in library:
            if name in frame.columns:
                frame[name] = frame[name].where(frame[formula.IN_UNIVERSE])
    if history:
        frame = frame[pd.to_datetime(frame['Date']).isin(requested)].reset_index(drop=True)
    return frame


def _feeds(candidate: str, targets: list[str], library: dict) -> bool:
    """Whether a factor is needed, directly or indirectly, by any of the target factors."""
    for target in targets:
        if candidate == target:
            return True
        for dependency in library[target].dependencies:
            if dependency in library and _feeds(candidate, [dependency], library):
                return True
    return False


def formula_evaluate(expression: str, frame: pd.DataFrame) -> pd.Series:
    """Evaluate one expression against a frame, kept here so callers need one import."""
    from investalyze.research import formula
    return formula.evaluate(expression, frame)
