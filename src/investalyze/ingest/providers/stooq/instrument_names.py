"""Decode Stooq market_data tickers (indices, bonds, currencies) into (name, country).

Indices are arbitrary vendor symbols with no decodable structure, looked up from the `[indices]`
table in `stooq_tickers.toml` at the repo root (manually curated as each one gets identified, see
notebooks/9999_data_quirks.ipynb). Bonds and currencies follow systematic patterns instead and are
decoded here; `[bond_countries]`, `[bond_types]`, `[currency_codes]` in the same TOML file hold the
lookup data the decoder needs, not the decoding logic itself.
"""

import re
import tomllib
from functools import lru_cache
from pathlib import Path

_REFERENCE_FILE = Path(__file__).parents[5] / 'stooq_tickers.toml'

_BOND_PATTERN = re.compile(r'^(\d+)([MY])([A-Z]{2})([YP])$')
_PAIR_PATTERN = re.compile(r'^([A-Z]{3})([A-Z]{3})$')
_SUFFIXED_PATTERN = re.compile(r'^([A-Z]{3})_(I|B|B50)$')


@lru_cache
def _reference() -> dict:
    """Load `stooq_tickers.toml`, cached for repeated lookups."""
    with _REFERENCE_FILE.open('rb') as f:
        return tomllib.load(f)


def describe(ticker: str) -> tuple[str, str | None] | None:
    """Decode a market_data ticker into `(name, country)`; `country` is None where not applicable.

    Returns None for a ticker this cannot identify at all (unknown index, or matching none of the
    bond/currency patterns).
    """
    ref = _reference()

    index = ref['indices'].get(ticker)
    if index is not None:
        return index['name'], index['country']

    bond_match = _BOND_PATTERN.match(ticker)
    if bond_match:
        tenor, unit, country_code, kind = bond_match.groups()
        unit_name = 'month' if unit == 'M' else 'year'
        kind_name = ref['bond_types'].get(kind, kind)
        country = ref['bond_countries'].get(country_code, country_code)
        return f'{tenor}-{unit_name} government bond {kind_name}', country

    suffixed_match = _SUFFIXED_PATTERN.match(ticker)
    if suffixed_match:
        code, suffix = suffixed_match.groups()
        currency_name = ref['currency_codes'].get(code, code)
        if suffix == 'I':
            return f'{currency_name} index (likely trade-weighted), suffix meaning unconfirmed', None
        return f'{currency_name} special variant (_{suffix}), meaning unconfirmed', None

    pair_match = _PAIR_PATTERN.match(ticker)
    if pair_match:
        base, quote = pair_match.groups()
        currencies = ref['currency_codes']
        base_name = currencies.get(base, base)
        quote_name = currencies.get(quote, quote)
        return f'{base_name}/{quote_name} exchange rate', None

    return None
