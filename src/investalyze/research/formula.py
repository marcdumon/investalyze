"""The factor expression language.

A factor is one expression over a long panel of `Ticker`, `Date` and data columns. Expressions are
parsed to a syntax tree and walked against an allow list, so a name that does not exist or a call to
something that is not a function fails when the factor is saved, naming the offending term, rather than
producing a column of NaN discovered much later.

Three kinds of function, distinguished by what they group over:

- cross-sectional, comparing names to each other on one date: `rank`, `zscore`, `winsorize`,
  `neutralize`
- time-series, comparing a name to its own past: `lag`, `delta`, `pct_change`, `mean`, `std`, `max`,
  `min`. Windows count trading days: an expression is evaluated on the daily grid whenever it uses one.
- elementwise, row by row: `where`, `mask`, `safe_div`, `abs`, `log`, `clip`, `sign`

`mask(x, condition)` restricts a factor to names meeting a condition. It sets excluded names to NaN,
which removes them from the cross-section. `where(condition, x, 0)` instead assigns them zero, which
still takes part in every ranking and biases the factor's correlations.

Division yields NaN where the denominator is zero, in place of the infinity numpy would give, since
an infinite factor value silently dominates every ranking it takes part in.
"""

import ast

import numpy as np
import pandas as pd

TICKER, DATE = 'Ticker', 'Date'

# what evaluating part of an expression yields: a column, or a constant it is broadcast from
Value = pd.Series | float | str

# boolean column marking the names in the universe on each date, when the panel has one
IN_UNIVERSE = 'in_universe'

CROSS_SECTIONAL = ('rank', 'zscore', 'winsorize', 'neutralize')
TIME_SERIES = ('lag', 'delta', 'pct_change', 'mean', 'std', 'max', 'min')
ELEMENTWISE = ('where', 'mask', 'safe_div', 'abs', 'log', 'clip', 'sign')
FUNCTIONS = frozenset(CROSS_SECTIONAL + TIME_SERIES + ELEMENTWISE)

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp, ast.Call,
    ast.Name, ast.Constant, ast.keyword, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.And, ast.Or,
)


def _parse(expression: str) -> ast.Expression:
    """The syntax tree for an expression, with a parse failure reported as a ValueError."""
    try:
        return ast.parse(expression, mode='eval')
    except SyntaxError as error:
        raise ValueError(f'expression {expression!r} could not be parsed: {error.msg}') from error


def _called_names(tree: ast.Expression) -> set[str]:
    """The names used in function position, which are functions rather than columns."""
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    return called


def dependencies(expression: str) -> set[str]:
    """Every data field or factor name an expression reads.

    Names in function position are excluded, so `rank(a)` depends on `a` alone. A keyword argument
    naming a column, as in `neutralize(a, by=sector)`, is a dependency like any other.
    """
    tree = _parse(expression)
    called = _called_names(tree)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in called:
            names.add(node.id)
    return names


def time_series_windows(expression: str) -> list[int]:
    """The window length of every time-series function call in an expression, in trading days."""
    windows = []
    for node in ast.walk(_parse(expression)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in TIME_SERIES:
            window = 1
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, (int, float)):
                window = int(node.args[1].value)
            windows.append(window)
    return windows


def validate(expression: str, known: set[str]) -> None:
    """Raise unless an expression uses only allowed syntax, known functions and known names."""
    tree = _parse(expression)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f'{type(node).__name__} is not allowed in a factor expression')
    for name in sorted(_called_names(tree)):
        if name not in FUNCTIONS:
            raise ValueError(f'unknown function {name!r}; known functions are {", ".join(sorted(FUNCTIONS))}')
    unknown = sorted(name for name in dependencies(expression) if name not in known)
    if unknown:
        raise ValueError(f'unknown name(s) in expression: {", ".join(unknown)}')


def _as_series(value: Value, frame: pd.DataFrame) -> pd.Series:
    """A result broadcast to one value per row, so a constant expression still returns a column."""
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=frame.index)


def _finite(series: pd.Series) -> pd.Series:
    """The series with infinities replaced by NaN."""
    return series.replace([np.inf, -np.inf], np.nan)


def _percentile(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Position within each date's cross-section, from 0 for the lowest to 100 for the highest."""
    by_date = values.groupby(dates)
    span = by_date.transform('count') - 1
    return ((by_date.rank(method='average') - 1.0) / span.where(span > 0)) * 100.0


def _zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Distance from each date's cross-sectional mean, in standard deviations."""
    by_date = values.groupby(dates)
    return (values - by_date.transform('mean')) / by_date.transform('std')


def _winsorize(values: pd.Series, dates: pd.Series, share: float) -> pd.Series:
    """Each date's values with both tails pulled in to the given quantile."""
    by_date = values.groupby(dates)
    lower = dates.map(by_date.quantile(share))
    upper = dates.map(by_date.quantile(1.0 - share))
    return values.clip(lower=lower, upper=upper)


def _by_ticker(frame: pd.DataFrame, values: pd.Series, statistic: str, periods: int) -> pd.Series:
    """A shift, or a rolling `statistic`, over each name's own history in date order, in the frame's row order."""
    ordered = frame[[TICKER, DATE]].assign(value=values).sort_values([TICKER, DATE])
    by_ticker = ordered.groupby(TICKER, sort=False)['value']
    if statistic == 'shift':
        result = by_ticker.shift(periods)
    else:
        result = getattr(by_ticker.rolling(periods), statistic)().droplevel(0)
    return result.reindex(frame.index)


class _Evaluator(ast.NodeVisitor):
    """Walks a validated syntax tree, returning a column for every node."""

    _BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
               ast.Div: np.divide, ast.Mod: np.mod, ast.Pow: np.power}
    _COMPARE = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
                ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def visit_Expression(self, node: ast.Expression) -> Value:
        """The value of the whole expression."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Value:
        """A literal number or string."""
        return node.value

    def visit_Name(self, node: ast.Name) -> Value:
        """The panel column a name refers to."""
        return self.frame[node.id]

    def visit_BinOp(self, node: ast.BinOp) -> Value:
        """Arithmetic between two values."""
        left, right = self.visit(node.left), self.visit(node.right)
        result = self._BINARY[type(node.op)](left, right)
        # an infinite factor value dominates every ranking it enters, so division that
        # reached a zero denominator is reported as missing instead
        if isinstance(node.op, (ast.Div, ast.Mod)):
            return _finite(_as_series(result, self.frame))
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Value:
        """Negation, unary plus, or logical not."""
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return ~_as_series(operand, self.frame).astype(bool)

    def visit_Compare(self, node: ast.Compare) -> Value:
        """One comparison, as a 1 or 0 column."""
        if len(node.ops) != 1:
            raise ValueError('chained comparisons are not allowed in a factor expression')
        left, right = self.visit(node.left), self.visit(node.comparators[0])
        return self._COMPARE[type(node.ops[0])](left, right)

    def visit_BoolOp(self, node: ast.BoolOp) -> Value:
        """`and` and `or`, elementwise."""
        # `and` and `or` cannot vectorize, so they are read as elementwise & and |
        result = _as_series(self.visit(node.values[0]), self.frame).astype(bool)
        for value in node.values[1:]:
            other = _as_series(self.visit(value), self.frame).astype(bool)
            result = result & other if isinstance(node.op, ast.And) else result | other
        return result

    def visit_IfExp(self, node: ast.IfExp) -> Value:
        """`a if cond else b`, row by row."""
        condition = _as_series(self.visit(node.test), self.frame)
        return self._where(condition, self.visit(node.body), self.visit(node.orelse))

    def visit_Call(self, node: ast.Call) -> Value:
        """One allowed function call."""
        name = node.func.id
        arguments = [self.visit(argument) for argument in node.args]
        keywords = {keyword.arg: self.visit(keyword.value) for keyword in node.keywords}
        return self._call(name, arguments, keywords)

    def generic_visit(self, node: ast.AST) -> Value:
        """Refuse any syntax the expression language leaves out."""
        raise ValueError(f'{type(node).__name__} is not allowed in a factor expression')

    def _where(self, condition: Value, when_true: Value, when_false: Value) -> pd.Series:
        """Pick between two branches row by row."""
        chosen = np.where(_as_series(condition, self.frame).fillna(False).astype(bool), when_true, when_false)
        return pd.Series(chosen, index=self.frame.index, dtype=float)

    def _call(self, name: str, arguments: list, keywords: dict) -> pd.Series:
        """Dispatch one function call to its implementation."""
        if name == 'where':
            return self._where(arguments[0], arguments[1], arguments[2])
        if name == 'mask':
            # excluding a name is not the same as scoring it zero: a zero competes in every
            # ranking it enters, while a missing value steps out of the cross-section
            values = _as_series(arguments[0], self.frame)
            keep = _as_series(arguments[1], self.frame).fillna(False).astype(bool)
            return values.where(keep)
        if name == 'safe_div':
            numerator, denominator = _as_series(arguments[0], self.frame), _as_series(arguments[1], self.frame)
            return _finite(numerator / denominator.replace(0.0, np.nan))

        values = _as_series(arguments[0], self.frame)
        if name in ELEMENTWISE:
            return self._elementwise(name, values, arguments)
        if name in CROSS_SECTIONAL:
            return self._cross_sectional(name, values, arguments, keywords)
        return self._time_series(name, values, arguments)

    def _elementwise(self, name: str, values: pd.Series, arguments: list) -> pd.Series:
        if name == 'abs':
            return values.abs()
        if name == 'log':
            return _finite(pd.Series(np.log(values.where(values > 0)), index=self.frame.index))
        if name == 'sign':
            return pd.Series(np.sign(values), index=self.frame.index)
        return values.clip(lower=arguments[1], upper=arguments[2])

    def _cross_sectional(self, name: str, values: pd.Series, arguments: list, keywords: dict) -> pd.Series:
        # a cross-section compares the names in that date's universe only; time-series
        # functions still read a name's full history
        if IN_UNIVERSE in self.frame.columns:
            values = values.where(self.frame[IN_UNIVERSE].astype(bool))
        dates = self.frame[DATE]
        if name == 'rank':
            return _percentile(values, dates)
        if name == 'zscore':
            return _zscore(values, dates)
        if name == 'winsorize':
            share = float(arguments[1]) if len(arguments) > 1 else 0.01
            return _winsorize(values, dates, share)
        groups = keywords.get('by') if 'by' in keywords else arguments[1]
        return values - values.groupby([dates, _as_series(groups, self.frame)], dropna=False).transform('mean')

    def _time_series(self, name: str, values: pd.Series, arguments: list) -> pd.Series:
        periods = int(arguments[1]) if len(arguments) > 1 else 1
        if name == 'lag':
            return _by_ticker(self.frame, values, 'shift', periods)
        if name == 'delta':
            return values - _by_ticker(self.frame, values, 'shift', periods)
        if name == 'pct_change':
            previous = _by_ticker(self.frame, values, 'shift', periods)
            return _finite(values / previous.replace(0.0, np.nan) - 1.0)
        return _by_ticker(self.frame, values, name, periods)


def evaluate(expression: str, frame: pd.DataFrame) -> pd.Series:
    """Compute an expression over a long panel carrying `Ticker`, `Date` and its data columns."""
    tree = _parse(expression)
    result = _Evaluator(frame).visit(tree)
    return _as_series(result, frame)
