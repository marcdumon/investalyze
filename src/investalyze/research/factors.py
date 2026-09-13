"""The factor library: factor definitions stored as text.

One factor is one file under `factors/`, holding its expression, its rationale, its direction and its
status. One file each rather than one shared file, so a diff shows which factor changed and two edits
never collide.

A factor may read raw data fields or other factors, which makes the library a graph.
`evaluation_order` returns the order that graph has to be computed in, and refuses a cycle.

`fingerprint` is the cache key. It covers the expression and, through its dependencies, every
expression upstream of it, so redefining a factor changes the fingerprint of every factor built on it.
It ignores rationale, status and notes, since none of them change a value.
"""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from investalyze.research import formula

CANDIDATE, ACCEPTED, REJECTED = 'candidate', 'accepted', 'rejected'
STATUSES = (CANDIDATE, ACCEPTED, REJECTED)


@dataclass(frozen=True)
class Factor:
    """One factor definition: its expression, rationale, direction and research status."""

    name: str
    expression: str
    description: str = ''
    higher_is_better: bool = True
    status: str = CANDIDATE
    notes: str = ''

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f'factor name {self.name!r} must be usable as a column name')
        if self.status not in STATUSES:
            raise ValueError(f'factor {self.name!r} has status {self.status!r}; expected one of {", ".join(STATUSES)}')
        if not self.expression.strip():
            raise ValueError(f'factor {self.name!r} has no expression')

    @property
    def dependencies(self) -> set[str]:
        """Every data field or factor this one reads."""
        return formula.dependencies(self.expression)


def _path_for(name: str, directory: Path) -> Path:
    return Path(directory) / f'{name}.toml'


def save(factor: Factor, directory: Path) -> Path:
    """Write one factor to its own file, creating the directory when needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        f'name = {factor.name!r}',
        f'expression = {factor.expression!r}',
        f'description = {factor.description!r}',
        f'higher_is_better = {str(factor.higher_is_better).lower()}',
        f'status = {factor.status!r}',
        f'notes = {factor.notes!r}',
    ]
    path = _path_for(factor.name, directory)
    path.write_text('\n'.join(lines) + '\n')
    return path


def load(directory: Path) -> dict:
    """Every factor stored under a directory, keyed by name.

    The name inside the file must match the file name. A mismatch is refused, since one of the two is
    a typo.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    library = {}
    for path in sorted(directory.glob('*.toml')):
        body = tomllib.loads(path.read_text())
        factor = Factor(**body)
        if factor.name != path.stem:
            raise ValueError(f'{path.name} holds a factor named {factor.name!r}; the two must agree')
        library[factor.name] = factor
    return library


def validate_library(library: dict, fields: set[str]) -> None:
    """Raise unless every factor parses, reads only known names, and the library holds no cycle."""
    known = set(fields) | set(library)
    for name in sorted(library):
        formula.validate(library[name].expression, known)
    evaluation_order(library, fields)


def evaluation_order(library: dict, fields: set[str]) -> list[str]:
    """The order factors have to be computed in, dependencies first.

    Raises on a cycle, naming the factors in it.
    """
    ordered: list[str] = []
    settled: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        if name in settled:
            return
        if name in visiting:
            cycle = ' -> '.join(visiting[visiting.index(name):] + [name])
            raise ValueError(f'factors form a cycle: {cycle}')
        visiting.append(name)
        for dependency in sorted(library[name].dependencies):
            if dependency in library:
                visit(dependency)
            elif dependency not in fields:
                raise ValueError(f'factor {name!r} reads unknown name {dependency!r}')
        visiting.pop()
        settled.add(name)
        ordered.append(name)

    for name in sorted(library):
        visit(name)
    return ordered


def fingerprint(name: str, library: dict) -> str:
    """A hash of a factor's expression and every expression upstream of it."""
    factor = library[name]
    parts = [factor.expression]
    for dependency in sorted(factor.dependencies):
        if dependency in library:
            parts.append(f'{dependency}={fingerprint(dependency, library)}')
        else:
            parts.append(dependency)
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]
