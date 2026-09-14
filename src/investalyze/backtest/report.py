"""What a finished run looks like on the page, and the rows it stores.

The caveats print above the numbers, not beneath them. A reader who stops after the headline Sharpe
should already have been told what the universe cannot tell them, and the cost assumptions behind
the figure should be visible next to it.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from investalyze.backtest import performance
from investalyze.backtest.config import Config

HEADLINE = ['cagr', 'vol', 'sharpe', 'sortino', 'max_drawdown', 'calmar', 'hit_rate', 'turnover']
PERCENT_STATS = {'cagr', 'vol', 'max_drawdown', 'hit_rate', 'mean_gross', 'mean_net',
                 'max_gross', 'max_net', 'alpha'}


@dataclass
class BacktestReport:
    """One run's identity, caveats, statistics and curves, ready to render or store."""

    run_id: str
    signal: str
    kind: str
    config: Config
    params: dict
    start: pd.Timestamp
    end: pd.Timestamp
    tickers: list[str]
    benchmark: str
    caveats: list[str]
    stats: dict[str, float]
    equity: pd.Series
    gross: pd.Series
    net: pd.Series
    cash: pd.Series
    fills: pd.DataFrame
    benchmark_equity: pd.Series | None = None
    universe_equity: pd.Series | None = None
    exposure: dict[str, float] = field(default_factory=dict)

    def config_json(self) -> str:
        """The resolved settings behind this run, as the stored run row keeps them.

        The rebalance is inside `params`, where the run put it, rather than read from the config
        file: it is a property of this run, not of the machine it ran on.
        """
        return json.dumps({
            'signal': self.signal, 'params': self.params, 'benchmark': self.benchmark,
            'long_only': self.config.long_only,
            'starting_cash': self.config.starting_cash,
            'costs': vars(self.config.costs), 'limits': vars(self.config.limits),
        }, sort_keys=True, default=str)

    def render(self) -> str:
        """The run as markdown: caveats, then costs, then the numbers, then the trading."""
        lines = [f'# Backtest {self.run_id}', '']
        lines += [f'`{self.signal}` ({self.kind}), {self.start.date()} to {self.end.date()}, '
                  f'{len(self.tickers):,} names, rebalanced {self.params.get("rebalance", "?")}', '']

        if self.caveats:
            lines += ['## Read this first', '']
            for caveat in self.caveats:
                lines.append(f'- {caveat}')
            lines.append('')
        lines += ['## Cost assumptions', '']
        costs = self.config.costs
        lines += [f'- slippage {costs.slippage_bps:g} bps, commission {costs.commission_bps:g} bps '
                  f'(minimum {costs.min_commission:g}), impact coefficient {costs.impact_coef:g}',
                  f'- borrow {costs.borrow_bps_annual:g} bps a year, cash interest '
                  f'{costs.cash_bps_annual:g} bps a year',
                  f'- participation cap {self.config.limits.participation:.0%} of session volume, '
                  f'position cap {self.config.limits.max_weight:.0%} of the book', '']

        lines += ['## Performance', '', '| statistic | value |', '| --- | ---: |']
        for name in HEADLINE:
            lines.append(f'| {name.replace("_", " ")} | {_format(name, self.stats.get(name))} |')
        for name in ('beta', 'alpha', 'information_ratio', 'up_capture', 'down_capture'):
            if name in self.stats:
                lines.append(f'| {name.replace("_", " ")} | {_format(name, self.stats[name])} |')
        lines.append('')

        if self.exposure:
            lines += ['## Exposure', '', '| statistic | value |', '| --- | ---: |']
            for name, value in self.exposure.items():
                lines.append(f'| {name.replace("_", " ")} | {_format(name, value)} |')
            lines.append('')

        lines += ['## Trading', '',
                  f'- {self.stats.get("n_trades", 0):,} fills, {self.stats.get("turnover", float("nan")):.2f}x '
                  'the book traded a year',
                  f'- total costs {self.stats.get("total_costs", 0):,.0f}, '
                  f'final equity {self.stats.get("final_equity", float("nan")):,.0f}', '']

        worst = performance.drawdown_table(self.equity, top_n=3)
        if not worst.empty:
            lines += ['## Deepest drawdowns', '', '| peak | trough | recovered | depth | sessions |',
                      '| --- | --- | --- | ---: | ---: |']
            for row in worst.to_dict('records'):
                recovered = 'not yet' if pd.isna(row['recovered']) else str(row['recovered'].date())
                lines.append(f'| {row["peak"].date()} | {row["trough"].date()} | {recovered} | '
                             f'{row["depth"]:.1%} | {row["sessions"]} |')
            lines.append('')
        return '\n'.join(lines)

    def run_row(self) -> pd.DataFrame:
        """The single row `backtest_runs` stores for this run."""
        return pd.DataFrame([{
            'RunId': self.run_id, 'Signal': self.signal, 'Kind': self.kind,
            'ConfigJson': self.config_json(), 'StartDate': self.start.date(), 'EndDate': self.end.date(),
            'NTickers': len(self.tickers), 'Benchmark': self.benchmark,
            'Cagr': self.stats.get('cagr'), 'Vol': self.stats.get('vol'),
            'Sharpe': self.stats.get('sharpe'), 'Sortino': self.stats.get('sortino'),
            'MaxDrawdown': self.stats.get('max_drawdown'), 'Calmar': self.stats.get('calmar'),
            'Turnover': self.stats.get('turnover'), 'HitRate': self.stats.get('hit_rate'),
            'NTrades': self.stats.get('n_trades'), 'FinalEquity': self.stats.get('final_equity'),
            'TotalCosts': self.stats.get('total_costs'),
        }])

    def equity_rows(self) -> pd.DataFrame:
        """The per-session curve `backtest_equity` stores for this run."""
        benchmark = (self.benchmark_equity.reindex(self.equity.index)
                     if self.benchmark_equity is not None else pd.Series(np.nan, index=self.equity.index))
        return pd.DataFrame({
            'RunId': self.run_id, 'Date': [date.date() for date in self.equity.index],
            'Equity': self.equity.to_numpy(), 'Gross': self.gross.to_numpy(), 'Net': self.net.to_numpy(),
            'Cash': self.cash.to_numpy(), 'BenchmarkEquity': benchmark.to_numpy(),
        })

    def trade_rows(self) -> pd.DataFrame:
        """The blotter `backtest_trades` stores for this run."""
        if self.fills.empty:
            return pd.DataFrame(columns=['RunId', 'Date', 'Ticker', 'Side', 'Shares', 'Price',
                                         'Notional', 'Cost', 'Kind'])
        rows = self.fills.copy()
        rows.insert(0, 'RunId', self.run_id)
        rows['Date'] = [date.date() for date in pd.DatetimeIndex(rows['Date'])]
        return rows[['RunId', 'Date', 'Ticker', 'Side', 'Shares', 'Price', 'Notional', 'Cost', 'Kind']]


def _format(name: str, value: float | None) -> str:
    """One statistic as it appears in the report table."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'n/a'
    if name in PERCENT_STATS:
        return f'{value:.2%}'
    if name == 'n_trades':
        return f'{int(value):,}'
    return f'{value:.3f}'


def report_dir(run_id: str, data_root: Path) -> Path:
    """Where one run's rendered report lives, so writing and deleting agree on the path."""
    return data_root / 'backtests' / run_id


def write_report(report: BacktestReport, data_root: Path) -> Path:
    """Write the rendered report under `data_root/backtests/<run id>/report.md` and return its path."""
    directory = report_dir(report.run_id, data_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'report.md'
    path.write_text(report.render())
    return path


def delete_report(run_id: str, data_root: Path) -> bool:
    """Remove one run's report directory, returning whether there was one to remove."""
    directory = report_dir(run_id, data_root)
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True
