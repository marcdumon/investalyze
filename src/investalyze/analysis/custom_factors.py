"""User-defined factors, read from `factors.toml`.

Two kinds share one file. An **expression** factor combines columns the factor panel already holds,
which covers weighted blends and ratios between existing factors. A **ratio** factor names a
numerator and a denominator straight from the financial statements, which reaches quantities the
built-in library does not contain.

Expressions are evaluated with `DataFrame.eval`, so they take arithmetic and comparisons over column
names and numbers. Alongside each built-in factor `x`, the name `rank_x` holds that factor's
cross-sectional percentile rank from 0 to 100, oriented so 100 is always the good end. Ranking before
combining is what puts factors measured in different units on one scale.

Every name a definition references is checked when the file is read, so a typo fails at load with the
offending term named rather than producing a column of NaN much later.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FACTORS_FILE = 'factors.toml'
IDENTITY_COLUMNS = ('last_close', 'mcap', 'mcap_bn')
EXPRESSION = 'expression'
RATIO = 'ratio'

# statements a ratio term may come from, and the window each one is summed or sampled over
STATEMENT_WINDOWS = {'income': 'ttm', 'cashflow': 'ttm', 'balance': 'latest'}


@dataclass(frozen=True)
class CustomFactor:
    """One user-defined factor: how to compute it, which way is good, and what it is."""

    name: str
    kind: str
    higher_is_better: bool
    description: str
    expression: str | None = None
    numerator: str | None = None
    denominator: str | None = None

    @property
    def terms(self) -> tuple[str, ...]:
        """The statement columns a ratio factor reads, empty for an expression factor."""
        if self.kind != RATIO:
            return ()
        return tuple(strip_sign(term)[0] for term in (self.numerator, self.denominator) if term)


def strip_sign(term: str) -> tuple[str, float]:
    """A term split into its column name and the sign to apply.

    A leading minus flips the term, which is what turns a statement line stored as a negative
    expense, such as `Research & Development`, into the positive magnitude a ratio usually wants.
    """
    if term.startswith('-'):
        return term[1:].strip(), -1.0
    return term.strip(), 1.0


def available_names(factors: list[str]) -> set[str]:
    """Every name an expression may reference: the given factors, their ranks, and identity columns."""
    return set(factors) | {f'rank_{factor}' for factor in factors} | set(IDENTITY_COLUMNS)


# the only bare words an expression may use besides column names
OPERATOR_WORDS = frozenset({'and', 'or', 'not', 'True', 'False'})

_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def _validate_expression(name: str, expression: str, known: set[str]) -> None:
    """Raise when an expression references something the panel will not hold.

    Every bare word must be a column name or a logical operator, so a function call or a stray
    identifier is refused here rather than evaluating to NaN much later.
    """
    words = set(_IDENTIFIER.findall(expression))
    unknown = sorted(word for word in words if word not in known and word not in OPERATOR_WORDS)
    if unknown:
        raise ValueError(f'factor {name!r} references unknown column(s): {", ".join(unknown)}')


def parse(text: str, factors: list[str], *, known: set[str] | None = None) -> dict[str, CustomFactor]:
    """Parse factor definitions from TOML text, validating every reference against `factors`.

    Kept separate from `read` so definitions can be checked without a file on disk.
    """
    known = available_names(factors) if known is None else known
    raw = tomllib.loads(text)
    sections = raw.get('factors', {})
    if not isinstance(sections, dict):
        raise ValueError('factors.toml must hold a [factors.<name>] table per factor')

    defined: dict[str, CustomFactor] = {}
    for name, body in sections.items():
        if name in factors:
            raise ValueError(f'factor {name!r} is already a built-in factor; choose another name')
        if not name.isidentifier():
            raise ValueError(f'factor name {name!r} must be usable as a column name')

        expression = body.get('expression')
        numerator, denominator = body.get('numerator'), body.get('denominator')
        higher = bool(body.get('higher_is_better', True))
        description = body.get('description', '')

        if expression and (numerator or denominator):
            raise ValueError(f'factor {name!r} sets both an expression and a numerator/denominator')
        if expression:
            _validate_expression(name, expression, known)
            defined[name] = CustomFactor(name=name, kind=EXPRESSION, higher_is_better=higher,
                                         description=description, expression=expression)
        elif numerator and denominator:
            defined[name] = CustomFactor(name=name, kind=RATIO, higher_is_better=higher,
                                         description=description, numerator=numerator,
                                         denominator=denominator)
        else:
            raise ValueError(f'factor {name!r} needs either an expression, or a numerator and a denominator')
    return defined


def read(path: Path | None, factors: list[str]) -> dict[str, CustomFactor]:
    """Read factor definitions from a TOML file, returning an empty mapping when there is none."""
    if path is None or not path.exists():
        return {}
    return parse(path.read_text(), factors)


def expression_factors(customs: dict[str, CustomFactor]) -> list[CustomFactor]:
    """The definitions computed from panel columns."""
    return [factor for factor in customs.values() if factor.kind == EXPRESSION]


def ratio_factors(customs: dict[str, CustomFactor]) -> list[CustomFactor]:
    """The definitions computed from statement line items."""
    return [factor for factor in customs.values() if factor.kind == RATIO]


def orientation(customs: dict[str, CustomFactor], builtin: dict[str, bool]) -> dict[str, bool]:
    """Which way is good for every factor, built-in and custom together."""
    return builtin | {name: factor.higher_is_better for name, factor in customs.items()}


def with_ranks(frame: pd.DataFrame, factors: list[str], orientations: dict[str, bool], *,
               by: str | None = None) -> pd.DataFrame:
    """Add a `rank_<factor>` column per factor, 0 to 100, oriented so 100 is best.

    `by` names a column to rank within, so a panel spanning many dates ranks each date's
    cross-section separately rather than pooling them.
    """
    ranked = frame.copy()
    for factor in factors:
        if factor not in ranked.columns:
            continue
        values = ranked[factor] if orientations[factor] else -ranked[factor]
        if by is None:
            ranked[f'rank_{factor}'] = values.rank(pct=True) * 100
        else:
            ranked[f'rank_{factor}'] = values.groupby(ranked[by]).rank(pct=True) * 100
    return ranked


def apply_expressions(frame: pd.DataFrame, customs: dict[str, CustomFactor], factors: list[str],
                      orientations: dict[str, bool], *, by: str | None = None) -> pd.DataFrame:
    """Add one column per expression factor, computed from the frame's own columns.

    The rank columns the expressions may reference are built here and dropped again, so the returned
    frame gains only the custom factor columns.
    """
    defined = expression_factors(customs)
    if not defined:
        return frame
    working = with_ranks(frame, factors, orientations, by=by)
    added = frame.copy()
    for factor in defined:
        try:
            added[factor.name] = working.eval(factor.expression)
        except Exception as failure:
            raise ValueError(f'factor {factor.name!r} failed to evaluate: {failure}') from failure
        added[factor.name] = pd.to_numeric(added[factor.name], errors='coerce').replace(
            [np.inf, -np.inf], np.nan)
    return added


def column_alias(column: str) -> str:
    """A statement column name turned into something usable as a pandas column and a SQL alias."""
    return 'term_' + re.sub(r'[^a-z0-9]+', '_', column.lower()).strip('_')


def statement_schema(con) -> dict[str, set[str]]:
    """Which columns each financial statement holds, for resolving a ratio factor's terms."""
    schema = {}
    for statement in STATEMENT_WINDOWS:
        rows = con.execute(f'DESCRIBE {statement}').fetchall()
        schema[statement] = {row[0] for row in rows}
    return schema


def resolve_terms(customs: dict[str, CustomFactor], schema: dict[str, set[str]]) -> dict[str, str]:
    """Map every ratio term to the statement it comes from, raising when a term cannot be placed.

    A term found in more than one statement is ambiguous, so it is refused with the candidates named
    rather than resolved by an arbitrary order.
    """
    resolved: dict[str, str] = {}
    for factor in ratio_factors(customs):
        for term in factor.terms:
            if term in resolved:
                continue
            holders = [statement for statement, columns in schema.items() if term in columns]
            if not holders:
                raise ValueError(f'factor {factor.name!r} names {term!r}, which is not a column of '
                                 f'{", ".join(sorted(schema))}')
            if len(holders) > 1:
                raise ValueError(f'factor {factor.name!r} names {term!r}, which appears in '
                                 f'{", ".join(sorted(holders))}; the term is ambiguous')
            resolved[term] = holders[0]
    return resolved


def term_aliases(customs: dict[str, CustomFactor], resolved: dict[str, str],
                 existing: dict[str, dict[str, str]]) -> dict[str, str]:
    """The frame column each ratio term will arrive under.

    A term the built-in factors already pull reuses their alias, so asking for `Revenue` reads the
    column the library computed rather than adding a second copy of it under a different name.
    """
    aliases: dict[str, str] = {}
    for factor in ratio_factors(customs):
        for term in factor.terms:
            statement = resolved[term]
            aliases[term] = existing.get(statement, {}).get(term) or column_alias(term)
    return aliases


def extra_columns(customs: dict[str, CustomFactor], resolved: dict[str, str], aliases: dict[str, str],
                  existing: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """The columns each statement must supply beyond what the built-in factors already ask for."""
    needed: dict[str, dict[str, str]] = {statement: {} for statement in STATEMENT_WINDOWS}
    for factor in ratio_factors(customs):
        for term in factor.terms:
            statement = resolved[term]
            if term not in existing.get(statement, {}):
                needed[statement][term] = aliases[term]
    return needed


def apply_ratios(frame: pd.DataFrame, customs: dict[str, CustomFactor], resolved: dict[str, str],
                 aliases: dict[str, str]) -> pd.DataFrame:
    """Add one column per ratio factor, from the statement terms already merged onto the frame.

    A denominator of zero or less yields NaN, matching how every built-in ratio guards its scale
    term, so a negative book value never turns into a large positive-looking factor.
    """
    defined = ratio_factors(customs)
    if not defined:
        return frame
    added = frame.copy()
    for factor in defined:
        numerator = _term_values(added, factor.numerator, resolved, aliases)
        denominator = _term_values(added, factor.denominator, resolved, aliases)
        added[factor.name] = np.where(denominator > 0, numerator / denominator, np.nan)
    return added


def _term_values(frame: pd.DataFrame, term: str, resolved: dict[str, str],
                 aliases: dict[str, str]) -> pd.Series:
    """One ratio term's values, read from its merged column and signed as the definition asked."""
    column, sign = strip_sign(term)
    suffix = '_ttm' if STATEMENT_WINDOWS[resolved[column]] == 'ttm' else ''
    return frame[aliases[column] + suffix] * sign


# --- editing the definitions file ---

def _quote(text: str) -> str:
    """A TOML string literal: single-quoted, switching to double quotes only when it holds one."""
    if "'" not in text and '\n' not in text and '\r' not in text:
        return f"'{text}'"
    escaped = (text.replace('\\', '\\\\').replace('"', '\\"')
               .replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'))
    return f'"{escaped}"'


def serialize_factor(name: str, fields: dict) -> str:
    """Render one `[factors.<name>]` table; fields that are None or blank are left out."""
    lines = [f'[factors.{name}]']
    for key in ('expression', 'numerator', 'denominator'):
        if fields.get(key):
            lines.append(f'{key} = {_quote(str(fields[key]).strip())}')
    lines.append(f'higher_is_better = {"true" if fields.get("higher_is_better", True) else "false"}')
    if fields.get('description'):
        lines.append(f'description = {_quote(str(fields["description"]).strip())}')
    return '\n'.join(lines) + '\n'


def add_factor(path: Path, name: str, fields: dict, factors: list[str]) -> str:
    """Append one definition to the file, writing only once the whole file still parses.

    A definition that fails validation raises with the reason and leaves the file untouched, so a
    bad entry can never be the thing that stops the next run from starting.
    """
    block = serialize_factor(name, fields)
    current = path.read_text() if path.exists() else ''
    if current and not current.endswith('\n'):
        current += '\n'
    combined = f'{current}\n{block}' if current.strip() else block
    parse(combined, factors)
    path.write_text(combined)
    return block


def remove_factor(path: Path, name: str) -> bool:
    """Delete one `[factors.<name>]` table from the file, keeping every comment around it.

    Returns whether the table was found. The block runs from its header to the next table header or
    the end of the file, and trailing blank lines go with it.
    """
    if not path.exists():
        return False
    lines = path.read_text().splitlines()
    header = f'[factors.{name}]'

    start = next((index for index, line in enumerate(lines) if line.strip() == header), None)
    if start is None:
        return False
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].lstrip().startswith('['):
            end = index
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    while start > 0 and not lines[start - 1].strip():
        start -= 1

    remaining = lines[:start] + lines[end:]
    path.write_text('\n'.join(remaining).rstrip('\n') + '\n')
    return True


def fingerprint(customs: dict[str, CustomFactor]) -> str:
    """A stable one-line signature of a set of definitions, for spotting one that has been edited.

    A panel row holds the value a formula produced, not the formula, so a redefined factor leaves
    the stored dates intact and the numbers wrong. Comparing this against what the last build
    recorded is the only way to notice.
    """
    parts = []
    for name in sorted(customs):
        factor = customs[name]
        parts.append(f'{name}={describe(factor)}:{"high" if factor.higher_is_better else "low"}')
    return '|'.join(parts)


def changed_since(recorded: str | None, customs: dict[str, CustomFactor]) -> list[str]:
    """Definitions that differ from a fingerprint taken earlier, named.

    A definition that is new or edited since the fingerprint was taken is one whose stored values
    were computed from something else. Nothing recorded means nothing can be said, so nothing is.
    """
    if not recorded:
        return []
    was = dict(part.split('=', 1) for part in recorded.split('|') if '=' in part)
    now = dict(part.split('=', 1) for part in fingerprint(customs).split('|') if '=' in part)
    return sorted(name for name, formula in now.items() if was.get(name) != formula)


def describe(factor: CustomFactor) -> str:
    """The formula of one definition, as the UI and the factor docs show it."""
    if factor.kind == EXPRESSION:
        return factor.expression
    return f'{factor.numerator} / {factor.denominator}'
