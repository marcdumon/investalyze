"""The ideas a user owns: what they believe, written down.

One idea is one file under `ideas/`, holding the expression that computes it, the belief behind it,
which end is good, and where it stands. One file each rather than one shared file, so a diff shows
which idea moved and two edits never collide.

An idea may read raw data fields or other ideas, which makes the library a graph. `evaluation_order`
returns the order that graph has to be computed in, and refuses a cycle.

`fingerprint` is what makes stale answers impossible. It covers the expression and, through its
dependencies, every expression upstream of it, so redefining an idea moves the fingerprint of
everything built on it. It deliberately ignores description, status and verdict: prose does not
change a number, and changing prose must not discard an expensive computation.
"""

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from investalyze.research import formula

DRAFT, KEPT, ARCHIVED = 'draft', 'kept', 'archived'
STATUSES = (DRAFT, KEPT, ARCHIVED)


@dataclass(frozen=True)
class Idea:
    """One idea: how to compute it, what it means, which way is good, and where it stands."""

    name: str
    expression: str
    description: str = ''
    higher_is_better: bool = True
    status: str = DRAFT
    verdict: str = ''

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f'idea name {self.name!r} must be usable as a column name')
        if self.status not in STATUSES:
            raise ValueError(f'idea {self.name!r} has status {self.status!r}; expected one of {", ".join(STATUSES)}')
        if not self.expression.strip():
            raise ValueError(f'idea {self.name!r} has no expression')

    @property
    def dependencies(self) -> set[str]:
        """Every field or idea this one reads."""
        return formula.dependencies(self.expression)


def _path_for(name: str, directory: Path) -> Path:
    return Path(directory) / f'{name}.toml'


def save(idea: Idea, directory: Path) -> Path:
    """Write one idea to its own file, creating the directory when it is not there yet."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        f'name = {idea.name!r}',
        f'expression = {idea.expression!r}',
        f'description = {idea.description!r}',
        f'higher_is_better = {str(idea.higher_is_better).lower()}',
        f'status = {idea.status!r}',
        f'verdict = {idea.verdict!r}',
    ]
    path = _path_for(idea.name, directory)
    path.write_text('\n'.join(lines) + '\n')
    return path


def load(directory: Path) -> dict[Idea]:
    """Every idea stored under a directory, keyed by name.

    The file name is not trusted as the idea's name: the name inside the file wins, and a mismatch
    is refused rather than silently resolved, since the two disagreeing means one of them is a typo.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    library = {}
    for path in sorted(directory.glob('*.toml')):
        body = tomllib.loads(path.read_text())
        idea = Idea(**body)
        if idea.name != path.stem:
            raise ValueError(f'{path.name} holds an idea named {idea.name!r}; the two must agree')
        library[idea.name] = idea
    return library


def validate_library(library: dict, fields: set[str]) -> None:
    """Raise unless every idea parses, reads only known names, and the library holds no cycle."""
    known = set(fields) | set(library)
    for name in sorted(library):
        formula.validate(library[name].expression, known)
    evaluation_order(library, fields)


def evaluation_order(library: dict, fields: set[str]) -> list[str]:
    """The order ideas have to be computed in, dependencies first.

    Raises when the library holds a cycle, naming the ideas caught in it, since no order exists.
    """
    ordered: list[str] = []
    settled: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        if name in settled:
            return
        if name in visiting:
            cycle = ' -> '.join(visiting[visiting.index(name):] + [name])
            raise ValueError(f'ideas form a cycle: {cycle}')
        visiting.append(name)
        for dependency in sorted(library[name].dependencies):
            if dependency in library:
                visit(dependency)
            elif dependency not in fields:
                raise ValueError(f'idea {name!r} reads unknown name {dependency!r}')
        visiting.pop()
        settled.add(name)
        ordered.append(name)

    for name in sorted(library):
        visit(name)
    return ordered


def fingerprint(name: str, library: dict) -> str:
    """A hash of an idea's expression and every expression upstream of it.

    This is the cache key. Because it reaches through dependencies, redefining one idea moves the
    fingerprint of everything built on it, so no answer can outlive the definition that produced it.
    """
    idea = library[name]
    parts = [idea.expression]
    for dependency in sorted(idea.dependencies):
        if dependency in library:
            parts.append(f'{dependency}={fingerprint(dependency, library)}')
        else:
            parts.append(dependency)
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]
