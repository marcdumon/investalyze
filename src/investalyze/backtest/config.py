"""Configuration reading for the backtest subsystem.

Reads `backtest.toml` into a `Config`, with built-in defaults so a run works with no config file at
all. The DB location is not here: that stays in `ingest.toml` and reaches the CLI through
`investalyze.ingest.config`.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_STARTING_CASH = 1_000_000.0
_DEFAULT_BENCHMARK = '^NDX'  # current to the last price date, unlike ^SPX and ^DJI
_DEFAULT_LONG_ONLY = True
_DEFAULT_EXCLUDE_CHECKS = ['ohlc_inconsistent', 'nonpositive_price']

_DEFAULT_UNIVERSE_NAME = None
_DEFAULT_N_TOP = 500
_DEFAULT_RANK_BY = 'liquidity'
_DEFAULT_REQUIRE_SECTOR = True

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

# the shared vocabulary for both frequencies; neither has a home in this file, since
# a cadence that lives here would change a result without appearing on the page that asked
REBALANCE_FREQUENCIES = ('D', 'W', 'M', 'Q')


@dataclass(frozen=True)
class Universe:
    """How a run's tradable name list is chosen."""

    name: str | None
    n_top: int
    rank_by: str
    require_sector: bool


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
    universe: Universe
    costs: Costs
    limits: Limits
    signals: dict[str, dict] = field(default_factory=dict)

    def signal(self, name: str) -> dict:
        """Parameters for one signal (empty dict if unconfigured)."""
        return self.signals.get(name, {})


def read(path: Path | None = None) -> Config:
    """Read `Config` from a TOML file, falling back to defaults.

    Top-level keys set the run-wide settings; the `[universe]`, `[costs]` and `[limits]` tables set
    their matching groups, and every table under `[signals]` holds one signal's parameters. An empty
    `universe.name` ranks the whole DB. A missing or unspecified file yields all defaults.
    """
    raw = tomllib.loads(path.read_text()) if path is not None and path.exists() else {}
    universe = raw.get('universe', {})
    costs = raw.get('costs', {})
    limits = raw.get('limits', {})
    return Config(
        starting_cash=float(raw.get('starting_cash', _DEFAULT_STARTING_CASH)),
        benchmark=raw.get('benchmark', _DEFAULT_BENCHMARK),
        long_only=bool(raw.get('long_only', _DEFAULT_LONG_ONLY)),
        exclude_checks=list(raw.get('exclude_checks', _DEFAULT_EXCLUDE_CHECKS)),
        universe=Universe(
            name=universe.get('name') or _DEFAULT_UNIVERSE_NAME,
            n_top=int(universe.get('n_top', _DEFAULT_N_TOP)),
            rank_by=universe.get('rank_by', _DEFAULT_RANK_BY),
            require_sector=bool(universe.get('require_sector', _DEFAULT_REQUIRE_SECTOR)),
        ),
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
        signals=raw.get('signals', {}),
    )
