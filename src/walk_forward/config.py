from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class FeatureConfig:
    """Settings used to build point-in-time features and training labels."""

    holding_days: int = 20
    min_history_days: int = 504
    min_price: float = 1.0
    liquidity_lookback_days: int = 60
    min_median_dollar_volume: float = 2_000_000.0
    target_winsor_quantile: float = 0.01

    def __post_init__(self) -> None:
        _require_integer("holding_days", self.holding_days, minimum=1)
        _require_integer("min_history_days", self.min_history_days, minimum=1)
        _require_integer("liquidity_lookback_days", self.liquidity_lookback_days, minimum=2)
        if not isfinite(self.min_price) or self.min_price <= 0:
            raise ValueError("min_price must be finite and positive.")
        if not isfinite(self.min_median_dollar_volume) or self.min_median_dollar_volume < 0:
            raise ValueError("min_median_dollar_volume must be finite and non-negative.")
        if not isfinite(self.target_winsor_quantile) or not 0 <= self.target_winsor_quantile < 0.5:
            raise ValueError("target_winsor_quantile must be in [0, 0.5).")


@dataclass(frozen=True)
class ModelConfig:
    """Settings for purged walk-forward model fitting."""

    validation_years: int = 3
    max_train_years: int | None = 12
    refit_every_signals: int = 3
    ensemble_size: int = 3
    min_train_rows: int = 1_000
    min_validation_dates: int = 12
    minimum_validation_rank_ic: float = 0.01
    calibration_buckets: int = 5
    calibration_shrinkage_dates: int = 24
    random_state: int = 0

    def __post_init__(self) -> None:
        _require_integer("validation_years", self.validation_years, minimum=1)
        if self.max_train_years is not None:
            _require_integer("max_train_years", self.max_train_years, minimum=1)
        if self.max_train_years is not None and self.max_train_years <= self.validation_years:
            raise ValueError("max_train_years must exceed validation_years.")
        _require_integer("refit_every_signals", self.refit_every_signals, minimum=1)
        _require_integer("ensemble_size", self.ensemble_size, minimum=1)
        _require_integer("min_train_rows", self.min_train_rows, minimum=1)
        _require_integer("min_validation_dates", self.min_validation_dates, minimum=2)
        if not isfinite(self.minimum_validation_rank_ic) or self.minimum_validation_rank_ic < 0:
            raise ValueError("minimum_validation_rank_ic must be finite and non-negative.")
        _require_integer("calibration_buckets", self.calibration_buckets, minimum=2)
        _require_integer("calibration_shrinkage_dates", self.calibration_shrinkage_dates, minimum=0)
        _require_integer("random_state", self.random_state, minimum=0)


@dataclass(frozen=True)
class BacktestConfig:
    """Execution, cost, capacity, and portfolio-construction settings."""

    initial_capital: float = 1_000_000.0
    top_n: int = 10
    buffer_n: int = 15
    max_weight: float = 0.15
    commission_bps: float = 5.0
    half_spread_bps: float = 10.0
    impact_bps: float = 25.0
    max_participation_rate: float = 0.01
    max_stale_days: int = 5
    stale_position_recovery: float = 0.0
    annual_risk_free_rate: float = 0.0
    minimum_expected_edge: float = 0.0
    rebalance_tolerance: float = 0.01
    weighting_method: str = "inverse_volatility"

    def __post_init__(self) -> None:
        _require_integer("top_n", self.top_n, minimum=1)
        _require_integer("buffer_n", self.buffer_n, minimum=1)
        _require_integer("max_stale_days", self.max_stale_days, minimum=1)
        finite_values = {
            "initial_capital": self.initial_capital,
            "max_weight": self.max_weight,
            "commission_bps": self.commission_bps,
            "half_spread_bps": self.half_spread_bps,
            "impact_bps": self.impact_bps,
            "max_participation_rate": self.max_participation_rate,
            "stale_position_recovery": self.stale_position_recovery,
            "annual_risk_free_rate": self.annual_risk_free_rate,
            "minimum_expected_edge": self.minimum_expected_edge,
            "rebalance_tolerance": self.rebalance_tolerance,
        }
        for name, value in finite_values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.buffer_n < self.top_n:
            raise ValueError("buffer_n must be at least top_n.")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1].")
        if min(self.commission_bps, self.half_spread_bps, self.impact_bps) < 0:
            raise ValueError("Cost assumptions cannot be negative.")
        if not 0 < self.max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1].")
        if not 0 <= self.stale_position_recovery <= 1:
            raise ValueError("stale_position_recovery must be in [0, 1].")
        if self.annual_risk_free_rate <= -1:
            raise ValueError("annual_risk_free_rate must be greater than -1.")
        if self.minimum_expected_edge < 0:
            raise ValueError("minimum_expected_edge cannot be negative.")
        if not 0 <= self.rebalance_tolerance < 1:
            raise ValueError("rebalance_tolerance must be in [0, 1).")
        if self.weighting_method not in {"equal", "inverse_volatility"}:
            raise ValueError("weighting_method must be 'equal' or 'inverse_volatility'.")


def _require_integer(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer no smaller than {minimum}.")
