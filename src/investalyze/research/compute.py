"""Turning ideas into values.

An idea reads raw fields and other ideas, so computing one means resolving what it needs, fetching
only those fields, and evaluating the library in dependency order. Only the fields an expression
actually names are queried, which is what makes drafting cheap enough to be a loop.
"""

import pandas as pd

from investalyze.analysis import factors_pit
from investalyze.research import fields, ideas


def required_fields(names: list[str], library: dict, catalog: dict) -> set[str]:
    """Every raw field the given ideas need, following references through other ideas."""
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
                raise ValueError(f'idea {name!r} reads unknown name {dependency!r}')

    for name in names:
        visit(name)
    return needed


def compute(con, names: list[str], library: dict, *, as_of: pd.DatetimeIndex,
            tickers: list[str] | None = None, version: str = factors_pit.AS_REPORTED) -> pd.DataFrame:
    """A long panel of `Ticker`, `Date` and one column per requested idea.

    Ideas are evaluated in dependency order, so one reading another sees a column already computed.
    Intermediate ideas needed only as inputs are computed too, and stay in the frame, since seeing
    what fed a value is most of what makes it checkable.
    """
    catalog = fields.catalog(con)
    needed = required_fields(names, library, catalog)
    frame = fields.panel(con, needed, as_of=as_of, version=version, tickers=tickers)
    if frame.empty:
        return frame

    wanted = set(names)
    for name in ideas.evaluation_order(library, set(catalog) | set(fields.IDENTITY)):
        if name in wanted or _feeds(name, names, library):
            frame[name] = formula_evaluate(library[name].expression, frame)
    return frame


def _feeds(candidate: str, targets: list[str], library: dict) -> bool:
    """Whether an idea is needed, directly or not, by any of the target ideas."""
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
