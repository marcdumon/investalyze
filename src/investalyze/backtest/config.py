"""Configuration reading for the backtest subsystem.

Reads `backtest.toml` into a `Config`, with built-in defaults so a run works with no config file at
all. The DB location is not here: that stays in `ingest.toml` and reaches the CLI through
`investalyze.ingest.config`.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_STARTING_CASH = 1_000_000.0
_DEFAULT_BENCHMARK = '^NDX'  # current to the last price date, unlike ^SPX and ^DJI
_DEFAULT_LONG_ONLY = True
_DEFAULT_EXCLUDE_CHECKS = ['ohlc_inconsistent', 'nonpositive_price']

_DEFAULT_SLIPPAGE_BPS = 5.0
_DEFAULT_COMMISSION_BPS = 2.0
_DEFAULT_MIN_COMMISSION = 1.0
_DEFAULT_IMPACT_COEF = 0.1
_DEFAULT_BORROW_BPS_ANNUAL = 50.0
_DEFAULT_CASH_BPS_ANNUAL = 0.0

_DEFAULT_MAX_WEIGHT = 0.05
_DEFAULT_MAX_GROSS = 1.0
_DEFAULT_MAX_NET = 1.0
_DEFAULT_MAX_POSITION_ADV = 0.25
_DEFAULT_PARTICIPATION = 0.05
_DEFAULT_MAX_ORDER_AGE = 5

@dataclass(frozen=True)
class Costs:
    """What a fill costs, in basis points of notional unless the name says otherwise."""

    slippage_bps: float
    commission_bps: float
    min_commission: float
    impact_coef: float
    borrow_bps_annual: float
    cash_bps_annual: float


@dataclass(frozen=True)
class Limits:
    """Bounds the engine holds a book and its orders to."""

    max_weight: float
    max_gross: float
    max_net: float
    max_position_adv: float
    participation: float
    max_order_age: int


@dataclass(frozen=True)
class Config:
    """Resolved backtest settings."""

    starting_cash: float
    benchmark: str
    long_only: bool
    exclude_checks: list[str]
    costs: Costs
    limits: Limits


def read(path: Path | None = None) -> Config:
    """Read `Config` from a TOML file, falling back to defaults.

    Top-level keys set the run-wide settings; the `[costs]` and `[limits]` tables set their matching
    groups. A missing or unspecified file yields all defaults.
    """
    raw = tomllib.loads(path.read_text()) if path is not None and path.exists() else {}
    costs = raw.get('costs', {})
    limits = raw.get('limits', {})
    return Config(
        starting_cash=float(raw.get('starting_cash', _DEFAULT_STARTING_CASH)),
        benchmark=raw.get('benchmark', _DEFAULT_BENCHMARK),
        long_only=bool(raw.get('long_only', _DEFAULT_LONG_ONLY)),
        exclude_checks=list(raw.get('exclude_checks', _DEFAULT_EXCLUDE_CHECKS)),
        costs=Costs(
            slippage_bps=float(costs.get('slippage_bps', _DEFAULT_SLIPPAGE_BPS)),
            commission_bps=float(costs.get('commission_bps', _DEFAULT_COMMISSION_BPS)),
            min_commission=float(costs.get('min_commission', _DEFAULT_MIN_COMMISSION)),
            impact_coef=float(costs.get('impact_coef', _DEFAULT_IMPACT_COEF)),
            borrow_bps_annual=float(costs.get('borrow_bps_annual', _DEFAULT_BORROW_BPS_ANNUAL)),
            cash_bps_annual=float(costs.get('cash_bps_annual', _DEFAULT_CASH_BPS_ANNUAL)),
        ),
        limits=Limits(
            max_weight=float(limits.get('max_weight', _DEFAULT_MAX_WEIGHT)),
            max_gross=float(limits.get('max_gross', _DEFAULT_MAX_GROSS)),
            max_net=float(limits.get('max_net', _DEFAULT_MAX_NET)),
            max_position_adv=float(limits.get('max_position_adv', _DEFAULT_MAX_POSITION_ADV)),
            participation=float(limits.get('participation', _DEFAULT_PARTICIPATION)),
            max_order_age=int(limits.get('max_order_age', _DEFAULT_MAX_ORDER_AGE)),
        ),
    )
