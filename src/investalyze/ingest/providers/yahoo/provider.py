"""Yahoo provider — stock prices (OHLCV), dividends, splits via yfinance.

Owns its whole flow: setup dirs -> fetch -> transform -> save (through storage.write).
Split into more files inside this folder if it grows.
"""
