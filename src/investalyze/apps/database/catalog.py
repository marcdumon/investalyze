"""What each table in the database is for, in words no schema can supply.

A column list says a table has a `Date` and a `Ticker`. It cannot say that a factor panel row holds
every factor as it was knowable on that date, or that `market_data` is indices and currencies while
`prices` is equities. That is what this file holds: one sentence per table, and the command that
writes it, so a reader can go from "what is this" to "how do I fill it" without leaving the page.

Pure data. Nothing here imports Dash or DuckDB, so the page and the manual can both read it and a
test can check it against the live database without starting an app.
"""

from typing import NamedTuple

MARKET = 'Market data'
FUNDAMENTALS = 'Fundamentals'
REFERENCE = 'Reference'
QUALITY = 'Quality'
BACKTEST = 'Backtest'

AREAS = (MARKET, FUNDAMENTALS, REFERENCE, QUALITY, BACKTEST)


class Table(NamedTuple):
    """One table's purpose, the command that fills it, and what makes a row unique."""

    area: str
    purpose: str
    written_by: str
    key: str
    internal: bool = False


TABLES: dict[str, Table] = {
    'prices': Table(
        MARKET,
        'Daily equity bars. O/H/L/C/V are split-adjusted; AC is adjusted for dividends too, which is '
        'why a total-return calculation uses AC and a share count does not.',
        'python -m investalyze.ingest yahoo',
        'Ticker, Date'),
    'market_data': Table(
        MARKET,
        'Daily bars for everything that is not an equity: indices, currencies, commodities. The '
        'benchmark a backtest measures against is read from here.',
        'python -m investalyze.ingest stooq',
        'Ticker, Date'),
    'market_instruments': Table(
        MARKET,
        'What each non-equity ticker in market_data is, by name, country and asset class.',
        'python -m investalyze.ingest stooq',
        'Ticker'),
    'dividends': Table(
        MARKET,
        'Cash dividends per share, dated on the ex-date. A backtest credits these to cash rather '
        'than assuming they are reinvested.',
        'python -m investalyze.ingest yahoo',
        'Ticker, Date'),
    'splits': Table(
        MARKET,
        'Share splits, as the ratio applied on that date. Used when checking that a price series '
        'moves for a reason.',
        'python -m investalyze.ingest yahoo',
        'Ticker, Date'),

    'income': Table(
        FUNDAMENTALS,
        'Income statements, quarterly and annual, in both versions: as first published and as later '
        'restated. `IsRestated` separates them, and `Publish Date` is what makes point-in-time work.',
        'python -m investalyze.ingest simfin',
        'Ticker, Report Date, Period, IsRestated'),
    'balance': Table(
        FUNDAMENTALS,
        'Balance sheets, same shape and same two versions as the income statements.',
        'python -m investalyze.ingest simfin',
        'Ticker, Report Date, Period, IsRestated'),
    'cashflow': Table(
        FUNDAMENTALS,
        'Cash flow statements, same shape and same two versions as the income statements.',
        'python -m investalyze.ingest simfin',
        'Ticker, Report Date, Period, IsRestated'),

    'companies': Table(
        REFERENCE,
        'One row per ticker: sector, industry, name, exchange, employee count. The sector a screener '
        'filters on and a universe is built from comes from here.',
        'python -m investalyze.ingest housekeeping',
        'Ticker'),
    'company_officers': Table(
        REFERENCE,
        'Named officers and their reported pay, one row per person per company.',
        'python -m investalyze.ingest yahoo-profile',
        'Ticker, Name'),
    '_yahoo_companies': Table(
        REFERENCE,
        'What Yahoo said about each company, kept as landed. Merged into companies rather than read '
        'directly, so a provider changing its mind does not silently change a result.',
        'python -m investalyze.ingest yahoo-profile',
        'Ticker', internal=True),
    '_simfin_companies': Table(
        REFERENCE,
        'The same for SimFin. Merged into companies alongside the Yahoo rows.',
        'python -m investalyze.ingest simfin',
        'Ticker', internal=True),

    'anomalies': Table(
        QUALITY,
        'Every finding the quality checks have raised, with the check that raised it and enough of '
        'the row to go and look. A finding stays here until it is logged or fixed.',
        'python -m investalyze.quality',
        'CheckName, SrcTable, Ticker, Date, Key'),

    'factor_panel': Table(
        BACKTEST,
        'The precomputed factor table: one row per ticker per as-of date, holding each factor as it '
        'was knowable on that date. This is what a cross-sectional signal ranks. Rank-based factors '
        'are absent on purpose, since their value depends on which names were in the cross-section.',
        'python -m investalyze.backtest factors',
        'Ticker, Date, Version'),
    'factor_panel_builds': Table(
        BACKTEST,
        'One row per build of the factor panel: what was asked for, how long it took, and a '
        'fingerprint of the factor formulas in force, which is how a redefined factor is noticed.',
        'python -m investalyze.backtest factors',
        'append-only'),
    'factor_ic': Table(
        BACKTEST,
        'Information coefficient per date: the rank correlation between a factor and the return that '
        'followed it. Averaged, this is the Mean IC column on the Factors tab.',
        'python -m investalyze.backtest evaluate',
        'Version, Universe, Factor, Date, Horizon'),
    'factor_quantiles': Table(
        BACKTEST,
        'Forward return of each score bucket per date, as a mean and as a median. The two part '
        'company on a universe that was never filtered for price.',
        'python -m investalyze.backtest evaluate',
        'Version, Universe, Factor, Date, Horizon, Quantile'),
    'factor_ic_decay': Table(
        BACKTEST,
        'Mean IC and its t-statistic at each of a fixed set of forward horizons, one row per horizon: '
        "how quickly a factor loses its edge, which is what the Factors tab's IC 5d/21d/63d/126d "
        'columns show.',
        'python -m investalyze.backtest evaluate',
        'Version, Universe, Factor, StartDate, EndDate, Horizon'),
    'backtest_runs': Table(
        BACKTEST,
        'One row per stored run: the settings behind it and its headline statistics.',
        'python -m investalyze.backtest run',
        'RunId'),
    'backtest_equity': Table(
        BACKTEST,
        'A stored run day by day: equity, gross and net exposure, cash, and the benchmark beside it.',
        'python -m investalyze.backtest run',
        'RunId, Date'),
    'backtest_trades': Table(
        BACKTEST,
        'Every fill a stored run made, with the price it paid and what the trade cost.',
        'python -m investalyze.backtest run',
        'RunId, Date, Ticker'),
}


def described(name: str) -> Table | None:
    """The catalogue entry for a table, or None when it has none."""
    return TABLES.get(name)


def by_area() -> dict[str, list[str]]:
    """Table names grouped by area, in the order the areas are meant to be read."""
    grouped = {}
    for area in AREAS:
        members = [name for name, table in TABLES.items() if table.area == area]
        if members:
            grouped[area] = members
    return grouped
