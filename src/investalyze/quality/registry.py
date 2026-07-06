"""Registry of data-quality checks.

Each check is a pure producer `(con) -> DataFrame` with `writer.FINDING_COLS` columns;
the CLI runs them and stores findings via `writer.replace_findings`. Adding a check is
one function in a check module plus a `CHECKS` line.
"""

from collections.abc import Callable

import duckdb
import pandas as pd

from investalyze.quality import (
    corporate_actions,
    fundamentals_identities,
    fundamentals_qsum,
    fundamentals_sanity,
    prices_ohlc,
    prices_series,
)

CheckFn = Callable[[duckdb.DuckDBPyConnection], pd.DataFrame]

# name -> (severity, check function); 'error' = hard invariant broken, 'warn' = tunable threshold exceeded
CHECKS: dict[str, tuple[str, CheckFn]] = {
    'nonpositive_price': ('error', prices_ohlc.nonpositive_price),
    'ohlc_inconsistent': ('error', prices_ohlc.ohlc_inconsistent),
    'negative_volume': ('error', prices_ohlc.negative_volume),
    'bond_yield_bound': ('warn', prices_ohlc.bond_yield_bound),
    'extreme_return': ('warn', prices_series.extreme_return),
    'stale_run': ('warn', prices_series.stale_run),
    'date_gap': ('warn', prices_series.date_gap),
    'nonpositive_dividend': ('error', corporate_actions.nonpositive_dividend),
    'oversized_dividend': ('warn', corporate_actions.oversized_dividend),
    'invalid_split_ratio': ('error', corporate_actions.invalid_split_ratio),
    'balance_identity': ('warn', fundamentals_identities.balance_identity),
    'balance_subtotals': ('warn', fundamentals_identities.balance_subtotals),
    'income_chain': ('warn', fundamentals_identities.income_chain),
    'cashflow_identity': ('warn', fundamentals_identities.cashflow_identity),
    'hard_invariants': ('error', fundamentals_sanity.hard_invariants),
    'negative_revenue': ('warn', fundamentals_sanity.negative_revenue),
    'quarters_vs_fy': ('warn', fundamentals_qsum.quarters_vs_fy),
}

# name -> one-line description, shown in `--help`; keys must match CHECKS exactly
CHECK_DESCRIPTIONS: dict[str, str] = {
    'nonpositive_price': 'O/H/L/C/AC <= 0 in prices; O/H/L/C <= 0 in market_data (non-bonds)',
    'ohlc_inconsistent': 'H < L, or O/C outside [L, H]',
    'negative_volume': 'Volume < 0',
    'bond_yield_bound': 'bond |C| > 50',
    'extreme_return': 'close doubles/halves overnight with no same-day split',
    'stale_run': '20+ consecutive identical closes',
    'date_gap': 'more than 30 days between consecutive rows',
    'nonpositive_dividend': 'Dividend <= 0',
    'oversized_dividend': 'Dividend above 25% of the same-day close',
    'invalid_split_ratio': 'split ratio <= 0 or = 1',
    'balance_identity': 'Total Liabilities + Total Equity vs Total Assets',
    'balance_subtotals': 'current + noncurrent vs total, both sides',
    'income_chain': 'Revenue -> Gross Profit -> Operating Income -> Pretax, link by link',
    'cashflow_identity': 'operating + investing + financing (+ FX, disc. ops) vs Net Change in Cash',
    'hard_invariants': 'Shares <= 0 on any statement; negative Total Assets',
    'negative_revenue': 'Revenue < 0 (legitimate for some financials)',
    'quarters_vs_fy': 'sum of 4 quarters vs the FY row',
}
