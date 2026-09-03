"""What each table in the database is for, written down because nothing can derive it.

A schema says a column is called `Vintage` and holds VARCHAR. It cannot say that the row is one
ticker's factor values as they were knowable on one past date, which is the thing a reader needs.
So the descriptions are written by hand, and a test fails when a table exists in the database
without an entry here, so the two cannot drift apart in silence.
"""

from dataclasses import dataclass

MARKET = 'Market data'
FUNDAMENTALS = 'Fundamentals'
REFERENCE = 'Reference'
QUALITY = 'Quality'
BACKTEST = 'Backtest'

AREAS = (MARKET, FUNDAMENTALS, REFERENCE, QUALITY, BACKTEST)


@dataclass(frozen=True)
class TableInfo:
    """One table's purpose, the command that fills it, and how a row is identified."""

    area: str
    purpose: str
    written_by: str
    key: tuple[str, ...] = ()
    internal: bool = False


TABLES: dict[str, TableInfo] = {
    'prices': TableInfo(
        MARKET,
        'Daily equity bars: open, high, low, close, volume, and a dividend-adjusted close. '
        'Split-adjusted, so O/H/L/C are comparable through time but exclude dividends; AC includes them.',
        'python -m investalyze.ingest ingest -p yahoo',
        ('Ticker', 'Date')),
    'market_data': TableInfo(
        MARKET,
        'Daily bars for everything that is not an equity: indices, bonds, commodities, currencies. '
        'No volume, and AssetClass says which kind of series a row belongs to.',
        'python -m investalyze.ingest ingest -p stooq',
        ('Ticker', 'Date')),
    'dividends': TableInfo(
        MARKET,
        'Cash dividends per share, dated on the ex-date. The engine credits these to cash rather than '
        'assuming they are reinvested.',
        'python -m investalyze.ingest ingest -p yahoo',
        ('Ticker', 'Date')),
    'splits': TableInfo(
        MARKET,
        'Share splits and their ratios, dated on the effective date. A 4:1 split reads as 4.',
        'python -m investalyze.ingest ingest -p yahoo',
        ('Ticker', 'Date')),
    'income': TableInfo(
        FUNDAMENTALS,
        'Income statements, one row per ticker per fiscal period. Both vintages live here: IsRestated '
        'FALSE is the figure as first published, TRUE is the figure as it reads today.',
        'python -m investalyze.ingest ingest -p simfin',
        ('Ticker', 'Fiscal Year', 'Fiscal Period', 'IsRestated')),
    'balance': TableInfo(
        FUNDAMENTALS,
        'Balance sheets, same shape and vintages as income. Point-in-time balance items come from the '
        'latest period published by a given date.',
        'python -m investalyze.ingest ingest -p simfin',
        ('Ticker', 'Fiscal Year', 'Fiscal Period', 'IsRestated')),
    'cashflow': TableInfo(
        FUNDAMENTALS,
        'Cash flow statements, same shape and vintages as income. Capex is stored negative, which is '
        'why free cash flow adds it rather than subtracting it.',
        'python -m investalyze.ingest ingest -p simfin',
        ('Ticker', 'Fiscal Year', 'Fiscal Period', 'IsRestated')),
    'companies': TableInfo(
        REFERENCE,
        'One row per ticker: name, sector, industry, country, CIK, business summary. Built by merging '
        'the two provider tables, with Yahoo winning where both carry a field.',
        'python -m investalyze.ingest housekeeping -t companies',
        ('Ticker',)),
    'company_officers': TableInfo(
        REFERENCE,
        'Named officers per company with title, age and compensation, as far as the provider reports it.',
        'python -m investalyze.ingest ingest -p yahoo_info',
        ('Ticker', 'Src', 'Name')),
    'market_instruments': TableInfo(
        REFERENCE,
        'What each non-equity ticker in market_data actually is: its name, country and asset class.',
        'python -m investalyze.ingest ingest -p stooq',
        ('Ticker',)),
    '_yahoo_companies': TableInfo(
        REFERENCE,
        'Yahoo profile fields exactly as fetched. Staging only: read companies instead.',
        'python -m investalyze.ingest ingest -p yahoo_info',
        ('Ticker',), internal=True),
    '_simfin_companies': TableInfo(
        REFERENCE,
        'SimFin company fields exactly as fetched. Staging only: read companies instead.',
        'python -m investalyze.ingest ingest -p simfin',
        ('Ticker',), internal=True),
    'anomalies': TableInfo(
        QUALITY,
        'Every finding the quality checks raised: which check, how severe, which table and row. '
        'The Data Quality page reads this, and logging a finding drops it off that list.',
        'python -m investalyze.quality',
        ('CheckName', 'SrcTable', 'Ticker', 'Date', 'Key')),
    'factor_panel': TableInfo(
        BACKTEST,
        'The precomputed factor table. One row per ticker per as-of date, holding every factor value '
        'as it was knowable on that date. This is what cross-sectional signals rank and what Evaluate '
        'scores; it exists because recomputing factors from prices and filings on every rebalance '
        'would be far too slow.',
        'Backtest page, or python -m investalyze.backtest factors',
        ('Ticker', 'Date', 'Vintage')),
    'factor_panel_builds': TableInfo(
        BACKTEST,
        'One row per factor panel build: what was asked for, how long it took, and a fingerprint of '
        'the factor formulas used. The fingerprint is how a redefined factor is noticed, since the '
        'panel rows hold the number and not the formula behind it.',
        'Backtest page, or python -m investalyze.backtest factors',
        ()),
    'factor_ic': TableInfo(
        BACKTEST,
        'Information coefficient per factor and date: the rank correlation between a factor\'s scores '
        'and the returns that followed over Horizon sessions. +1 is a perfect ordering, 0 is none. '
        'Kept per universe, so the same factor scored over two sets of names gives two readings.',
        'Backtest page, or python -m investalyze.backtest evaluate',
        ('Vintage', 'Universe', 'Factor', 'Date', 'Horizon')),
    'factor_quantiles': TableInfo(
        BACKTEST,
        'Forward return of each score bucket per factor and date, as a mean and as a median. The two '
        'part company on an unfiltered universe, where a few sub-dollar names carry the mean.',
        'Backtest page, or python -m investalyze.backtest evaluate',
        ('Vintage', 'Universe', 'Factor', 'Date', 'Horizon', 'Quantile')),
    'backtest_runs': TableInfo(
        BACKTEST,
        'One row per stored backtest: its settings, its headline statistics, and the survivorship note '
        'that travels with them. RunId is derived from the configuration, so the same run overwrites '
        'itself rather than accumulating.',
        'Backtest page, or python -m investalyze.backtest run',
        ('RunId',)),
    'backtest_equity': TableInfo(
        BACKTEST,
        'A stored run\'s equity curve, one row per session: equity, gross and net exposure, cash, and '
        'the benchmark rebased alongside it.',
        'Backtest page, or python -m investalyze.backtest run',
        ('RunId', 'Date')),
    'backtest_trades': TableInfo(
        BACKTEST,
        'Every fill a stored run made: side, shares, price, notional and the cost charged for it.',
        'Backtest page, or python -m investalyze.backtest run',
        ()),
}


def area_of(table: str) -> str:
    """Which group a table belongs to, or Reference for one nothing has been written about yet."""
    entry = TABLES.get(table)
    return entry.area if entry else REFERENCE


def undocumented(tables: list[str]) -> list[str]:
    """Tables present in the database that nothing here describes."""
    return sorted(table for table in tables if table not in TABLES)
