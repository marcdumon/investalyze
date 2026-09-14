"""Computing factor values.

A factor reads data fields and other factors, so computing one means resolving its dependencies,
fetching only those fields, and evaluating the library in dependency order. Only the fields an
expression names are queried, which keeps a sampled diagnostic run fast.

Time-series function windows count trading days. Factors that use one are evaluated on every trading
session, starting early enough to fill their longest window, and their values are then read on the
requested dates. The same expression therefore gives the same value whether it is asked for monthly or
daily. Factors without one are evaluated on the requested dates only.
"""

import duckdb
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
    """Trading days of history the most demanding of the given factors needs.

    One factor needs the sum of every time-series window in it and in the factors it reads. Summing
    overstates the need of windows used side by side, and is exact for windows nested inside each other,
    which is the case that must not run short.
    """
    return max((_own_lookback(name, library) for name in names), default=0)


def _own_lookback(name: str, library: dict) -> int:
    """The sum of every time-series window in one factor and in the factors it reads."""
    total = 0
    seen: set[str] = set()

    def visit(current: str) -> None:
        nonlocal total
        if current in seen:
            return
        seen.add(current)
        total += sum(formula.time_series_windows(library[current].expression))
        for dependency in library[current].dependencies:
            if dependency in library:
                visit(dependency)

    visit(name)
    return total


def compute(con: duckdb.DuckDBPyConnection, names: list[str], library: dict, *, as_of: pd.DatetimeIndex,
            tickers: list[str] | None = None, version: str = factors_pit.AS_REPORTED,
            members: pd.DataFrame | None = None) -> pd.DataFrame:
    """A long panel of `Ticker`, `Date` and one column per requested factor.

    Factors are evaluated in dependency order, so a factor reading another sees its column already
    computed. Intermediate factors are kept in the frame, so every input to a value can be inspected.
    """
    with_history = [name for name in names if _own_lookback(name, library)]
    without_history = [name for name in names if name not in with_history]
    if with_history and without_history:
        daily = compute(con, with_history, library, as_of=as_of, tickers=tickers, version=version, members=members)
        dated = compute(con, without_history, library, as_of=as_of, tickers=tickers, version=version, members=members)
        return _combine(daily, dated)

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
    if history:
        frame = complete_grid(frame, grid)
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


def complete_grid(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """The panel with a row for every name on every session, so a time-series window counts sessions.

    A session without data for a name becomes a row of missing values rather than no row, which keeps a
    252-day window from reaching back further for a name that did not trade every day.
    """
    names = sorted(frame['Ticker'].unique())
    full = pd.MultiIndex.from_product([names, pd.DatetimeIndex(sessions)], names=['Ticker', 'Date'])
    completed = frame.assign(Date=pd.to_datetime(frame['Date'])).set_index(['Ticker', 'Date']).reindex(full)
    return completed.reset_index()


def _combine(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Two panels over the same dates joined on `Ticker` and `Date`, keeping every row of both."""
    if first.empty or second.empty:
        return second if first.empty else first
    keys = ['Ticker', 'Date']
    combined = first.set_index(keys).combine_first(second.set_index(keys)).reset_index()
    if formula.IN_UNIVERSE in combined.columns:
        combined[formula.IN_UNIVERSE] = combined[formula.IN_UNIVERSE].astype('boolean').fillna(False).astype(bool)
    return combined.sort_values(['Date', 'Ticker']).reset_index(drop=True)


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
