"""Data fields available to factor expressions: statement line items and market data.

Every line item on the three financial statements becomes a name an expression can use, so the
whole of what has been ingested is reachable without anyone hand-listing it. Aliases are derived
from the column headings themselves: `Research & Development` reads as `research_development`.

A heading that appears on more than one statement, such as `Net Income`, gets no bare alias, only
the statement-qualified `income_net_income` and `cashflow_net_income`. Silently picking one of the
two would be a number whose origin the reader cannot see.

Flow items on the income and cash flow statements are summed over the trailing four quarters;
balance sheet items use the most recent quarter (MRQ). Both read through the
point-in-time joins in `analysis.factors_pit`, so a value is only ever visible on a date it had
already been published.
"""

import re
from dataclasses import dataclass

import pandas as pd

from investalyze.analysis import factors_pit
from investalyze.research import market

TTM, MRQ, DAILY = 'ttm', 'mrq', 'daily'
MARKET = 'market'
STATEMENT_WINDOWS = {'income': TTM, 'cashflow': TTM, 'balance': MRQ}

# bookkeeping columns present on every statement, which describe the filing rather than
# the business, so they are not quantities anyone builds a factor from
METADATA = frozenset({
    'Ticker', 'SrcId', 'Src', 'Market', 'Period', 'IsRestated', 'Currency',
    'Fiscal Year', 'Fiscal Period', 'Report Date', 'Publish Date', 'Restated Date',
})

IDENTITY = ('sector', 'industry', 'name')


@dataclass(frozen=True)
class Field:
    """One raw statement column: where it comes from and how it is read through time."""

    alias: str
    source: str
    statement: str

    @property
    def window(self) -> str:
        """TTM for flow items summed over four quarters, MRQ for balance sheet items, daily for market data."""
        return STATEMENT_WINDOWS.get(self.statement, DAILY)


def to_alias(heading: str) -> str:
    """A statement heading reduced to a usable column name."""
    cleaned = re.sub(r'[^0-9A-Za-z]+', '_', heading).strip('_').lower()
    if cleaned and cleaned[0].isdigit():
        cleaned = f'f_{cleaned}'
    return cleaned


def build_catalog(schema: dict) -> dict:
    """Every data field, keyed by alias, given each statement's column headings."""
    by_alias: dict[str, list[Field]] = {}
    for statement, headings in schema.items():
        for heading in headings:
            if heading in METADATA:
                continue
            field = Field(to_alias(heading), heading, statement)
            by_alias.setdefault(field.alias, []).append(field)

    catalog: dict[str, Field] = {}
    for alias, found in by_alias.items():
        # a heading on one statement only keeps the short name; a shared one does not, so
        # that reading it always says which statement was meant
        if len(found) == 1:
            catalog[alias] = found[0]
        for field in found:
            qualified = f'{field.statement}_{alias}'
            catalog[qualified] = Field(qualified, field.source, field.statement)
    return catalog


def read_schema(con) -> dict:
    """The column headings of each statement table in the database."""
    schema = {}
    for statement in STATEMENT_WINDOWS:
        schema[statement] = [row[0] for row in con.execute(f'DESCRIBE {statement}').fetchall()]
    return schema


def catalog(con) -> dict:
    """Every data field: statement line items from the live schema, plus the market data fields."""
    available = build_catalog(read_schema(con))
    for name, description in market.FIELDS.items():
        available[name] = Field(name, description, MARKET)
    return available


def _requested_by_statement(names: set[str], available: dict) -> dict:
    """The requested aliases grouped by the statement they read, as source column to alias."""
    grouped: dict[str, dict[str, str]] = {statement: {} for statement in STATEMENT_WINDOWS}
    for name in sorted(names):
        field = available.get(name)
        if field is not None and field.statement in grouped:
            grouped[field.statement][field.source] = field.alias
    return grouped


def panel(con, names: set[str], *, as_of: pd.DatetimeIndex, version: str = factors_pit.AS_REPORTED,
          tickers: list[str] | None = None) -> pd.DataFrame:
    """A long panel of the requested fields, one row per name per as-of date.

    Only the requested fields are queried, so a two-field expression reads two columns rather than
    every statement column. Trailing-year items also
    return a `<alias>_prior` column holding the year before, so a growth rate needs no second query.
    """
    available = catalog(con)
    grouped = _requested_by_statement(set(names), available)
    frames = []

    for statement in ('income', 'cashflow'):
        columns = grouped[statement]
        if columns:
            frame = factors_pit.quarterly_ttm(con, statement, columns, as_of=as_of, version=version,
                                              tickers=tickers)
            # quarterly_ttm has already blanked windows holding fewer than four quarters
            renamed = {f'{alias}_ttm': alias for alias in columns.values()}
            frames.append(frame.rename(columns=renamed))

    if grouped['balance']:
        frame = factors_pit.balance_panel(con, as_of=as_of, version=version, tickers=tickers,
                                          extras=grouped['balance'])
        frames.append(frame[['Date', 'Ticker'] + list(grouped['balance'].values())])

    if names & set(IDENTITY):
        identity = factors_pit.identity_panel(con, as_of=as_of, tickers=tickers)
        wanted = [column for column in IDENTITY if column in names]
        frames.append(identity[['Date', 'Ticker'] + wanted])

    if names & set(market.FIELDS):
        frames.append(market.panel(con, names, as_of=as_of, tickers=tickers))

    if not frames:
        return pd.DataFrame(columns=['Ticker', 'Date'])

    merged = None
    for frame in frames:
        frame = frame.assign(Date=pd.to_datetime(frame['Date']))
        merged = frame if merged is None else merged.merge(frame, on=['Date', 'Ticker'], how='outer')
    return merged.sort_values(['Date', 'Ticker']).reset_index(drop=True)
