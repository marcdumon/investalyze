"""Computing factor values.

A factor reads data fields and other factors, so computing one means resolving its dependencies,
fetching only those fields, and evaluating the library in dependency order. Only the fields an
expression names are queried, which keeps a sampled diagnostic run fast.
"""

import pandas as pd

from investalyze.analysis import factors_pit
from investalyze.research import factors, fields


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


def compute(con, names: list[str], library: dict, *, as_of: pd.DatetimeIndex,
            tickers: list[str] | None = None, version: str = factors_pit.AS_REPORTED) -> pd.DataFrame:
    """A long panel of `Ticker`, `Date` and one column per requested factor.

    Factors are evaluated in dependency order, so a factor reading another sees its column already
    computed. Intermediate factors are kept in the frame, so every input to a value can be inspected.
    """
    catalog = fields.catalog(con)
    needed = required_fields(names, library, catalog)
    frame = fields.panel(con, needed, as_of=as_of, version=version, tickers=tickers)
    if frame.empty:
        return frame

    wanted = set(names)
    for name in factors.evaluation_order(library, set(catalog) | set(fields.IDENTITY)):
        if name in wanted or _feeds(name, names, library):
            frame[name] = formula_evaluate(library[name].expression, frame)
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
