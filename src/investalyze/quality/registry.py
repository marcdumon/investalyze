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
