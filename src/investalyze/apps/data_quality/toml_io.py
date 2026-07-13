"""Serialize form dicts to TOML blocks and append them to a TOML file (validate-then-write).

Pure (no Dash). There is no stdlib TOML writer, so blocks are hand-rolled: strings single-quoted
(double-quoted with escaping only when they contain an apostrophe or newline), dates bare, numbers
bare, lists inline. `append_block` validates the *whole* resulting file through the caller's parser
before writing, so a bad entry raises and leaves the file untouched rather than half-written.
"""

from collections.abc import Callable
from datetime import date
from pathlib import Path


def _quote_string(text: str) -> str:
    """TOML string literal for `text`: single-quoted, or double-quoted+escaped if it holds a quote/newline."""
    if "'" not in text and '\n' not in text and '\r' not in text:
        return f"'{text}'"
    escaped = text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return f'"{escaped}"'


def _format_value(value: object) -> str:
    """Render a Python value as its TOML scalar or inline-array form."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return _quote_string(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_format_value(item) for item in value) + ']'
    raise TypeError(f'cannot serialize {type(value).__name__} to TOML: {value!r}')


def serialize_block(section: str, fields: dict) -> str:
    """Render a `[[section]]` array-of-tables block; fields with value None are omitted, order kept."""
    lines = [f'[[{section}]]']
    for key, value in fields.items():
        if value is None:
            continue
        lines.append(f'{key} = {_format_value(value)}')
    return '\n'.join(lines) + '\n'


def append_block(path: Path, block: str, parse: Callable[[str], list]) -> str:
    """Append `block` to `path` only if `parse` accepts the combined file; otherwise raise, unchanged."""
    current = path.read_text() if path.exists() else ''
    if current and not current.endswith('\n'):
        current += '\n'
    combined = f'{current}\n{block}' if current.strip() else block
    parse(combined)
    path.write_text(combined)
    return block
