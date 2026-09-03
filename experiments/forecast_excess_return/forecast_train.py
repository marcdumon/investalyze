"""Scoring for the excess-return forecast model, re-exported from the backtest evaluation module.

The implementation moved to `investalyze.backtest.evaluation` when the backtest subsystem needed the
same per-date rank correlation; this module keeps the name the notebooks import.
"""

from investalyze.backtest.evaluation import spearman_by_group

__all__ = ['spearman_by_group']
