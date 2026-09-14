"""Point-in-time universe: the most liquid names as known on each as-of date.

On each date the universe is the `size` names with the highest average dollar volume over the
`lookback` sessions ending that date. Only trading on or before the date is used, so a name that became
liquid later is not in the universe earlier. Selecting on liquidity measured today and testing over the
past would hold only names already known to have grown, which is look-ahead bias.
"""

import pandas as pd

DEFAULT_LOOKBACK = 63


def liquid_universe(con, as_of: pd.DatetimeIndex, *, size: int, lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    """`Date` and `Ticker` rows: the top `size` names by trailing average dollar volume on each date."""
    dates = [pd.Timestamp(when).date() for when in as_of]
    if not dates:
        return pd.DataFrame(columns=['Date', 'Ticker'])
    first = (pd.Timestamp(min(dates)) - pd.Timedelta(days=lookback * 2 + 14)).date()
    members = con.execute(f"""
        WITH dollar_volume AS (
            SELECT Ticker, Date,
                   avg(C * V) OVER (PARTITION BY Ticker ORDER BY Date
                                    ROWS BETWEEN {lookback - 1} PRECEDING AND CURRENT ROW) AS adv
            FROM prices
            WHERE Date BETWEEN ? AND ? AND C > 0 AND V IS NOT NULL
        )
        SELECT Date, Ticker FROM dollar_volume
        WHERE Date = ANY(?::DATE[]) AND adv > 0
        QUALIFY row_number() OVER (PARTITION BY Date ORDER BY adv DESC, Ticker) <= ?
        ORDER BY Date, Ticker
    """, [first, max(dates), dates, size]).df()
    members['Date'] = pd.to_datetime(members['Date'])
    return members
