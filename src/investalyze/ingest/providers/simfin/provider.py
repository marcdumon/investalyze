"""SimFin provider — fundamentals (income/balance/cashflow/derived), as-filed +
restated, annual + quarterly, plus company metadata. Bulk REST download.

Owns its whole flow: setup dirs -> fetch -> transform -> save (through storage.write).
Split into more files inside this folder if it grows.
"""
