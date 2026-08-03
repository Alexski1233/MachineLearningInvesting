from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import BacktestConfig, FeatureConfig, ModelConfig
from .data import attach_universe_membership, validate_prices
from .features import LABEL_DATE_COLUMN, TARGET_COLUMN, build_feature_panel
from .model import fit_latest_and_predict, walk_forward_predict


@dataclass(frozen=True)
class ResearchResult:
    panel: pd.DataFrame
    predictions: pd.DataFrame
    momentum_predictions: pd.DataFrame
    model_diagnostics: pd.DataFrame
    strategy: BacktestResult
    momentum_baseline: BacktestResult
    survivorship_safe: bool

    def metric_comparison(self) -> pd.DataFrame:
        results = (
            ("walk_forward_ml", self.strategy),
            ("momentum_12_1", self.momentum_baseline),
        )
        rows = [{"strategy": name, **result.metrics.as_dict()} for name, result in results]
        return pd.DataFrame(rows).set_index("strategy")


def make_signal_dates(prices: pd.DataFrame, start_date: object, every_n_sessions: int) -> tuple[pd.Timestamp, ...]:
    """Create one fixed rebalance phase from a market-session calendar."""
    if every_n_sessions < 1:
        raise ValueError("every_n_sessions must be positive.")
    start = _normalize_timestamp(start_date)
    dates = pd.DatetimeIndex(pd.to_datetime(prices["date"]).drop_duplicates().sort_values())
    selected = dates[dates >= start][::every_n_sessions]
    if selected.empty:
        raise ValueError(f"No market dates are available on or after {start.date()}.")
    return tuple(pd.Timestamp(date).normalize() for date in selected)


def run_research(
    prices: pd.DataFrame,
    start_date: object,
    feature_config: FeatureConfig | None = None,
    model_config: ModelConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    universe_membership: pd.DataFrame | None = None,
) -> ResearchResult:
    """Build features, create walk-forward forecasts, and backtest net returns."""
    features = feature_config or FeatureConfig()
    models = model_config or ModelConfig()
    portfolio = backtest_config or BacktestConfig()
    had_membership = universe_membership is not None or "in_universe" in prices.columns
    validated = (
        attach_universe_membership(prices, universe_membership)
        if universe_membership is not None
        else validate_prices(prices)
    )
    _require_single_exchange_calendar(validated)
    panel = build_feature_panel(validated, features)
    signal_dates = make_signal_dates(validated, start_date, features.holding_days)
    predictions, diagnostics = walk_forward_predict(panel, signal_dates, models)
    strategy = run_backtest(validated, predictions, portfolio)

    comparable_dates = tuple(pd.to_datetime(predictions["date"].drop_duplicates()).sort_values())
    momentum_signals = _momentum_signals(
        panel,
        comparable_dates,
        max_history_years=models.max_train_years,
        bucket_count=models.calibration_buckets,
        minimum_calibration_dates=models.min_validation_dates,
        shrinkage_dates=models.calibration_shrinkage_dates,
    )
    momentum_baseline = run_backtest(validated, momentum_signals, portfolio)
    return ResearchResult(
        panel=panel,
        predictions=predictions,
        momentum_predictions=momentum_signals,
        model_diagnostics=diagnostics,
        strategy=strategy,
        momentum_baseline=momentum_baseline,
        survivorship_safe=had_membership,
    )


def blend_ranked_signals(
    ml_predictions: pd.DataFrame,
    momentum_predictions: pd.DataFrame,
    ml_weight: float = 0.25,
) -> pd.DataFrame:
    """Blend point-in-time ML and momentum ranks without changing the trade horizon."""
    if not np.isfinite(ml_weight) or not 0 <= ml_weight <= 1:
        raise ValueError("ml_weight must be between 0 and 1.")
    merged = ml_predictions.merge(
        momentum_predictions[["date", "ticker", "score", "expected_return"]],
        on=["date", "ticker"],
        how="inner",
        suffixes=("_ml", "_momentum"),
    )
    merged["score"] = ml_weight * merged["score_ml"] + (1 - ml_weight) * merged["score_momentum"]
    merged["expected_return"] = (
        ml_weight * merged["expected_return_ml"] + (1 - ml_weight) * merged["expected_return_momentum"]
    )
    columns = [
        "date",
        "ticker",
        "expected_return",
        "score",
        "horizon_sessions",
        "vol_60d",
        "turnover_60d_median",
        "model_refit_date",
    ]
    return merged[columns].sort_values(["date", "ticker"]).reset_index(drop=True)


def momentum_predictions_for_dates(
    panel: pd.DataFrame,
    dates: Iterable[object],
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Generate momentum forecasts using only outcomes known before each date."""
    models = config or ModelConfig()
    return _momentum_signals(
        panel,
        dates,
        max_history_years=models.max_train_years,
        bucket_count=models.calibration_buckets,
        minimum_calibration_dates=models.min_validation_dates,
        shrinkage_dates=models.calibration_shrinkage_dates,
    )


def latest_predictions(
    prices: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    model_config: ModelConfig | None = None,
    universe_membership: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Refit on all currently known labels and score the latest eligible date."""
    features = feature_config or FeatureConfig()
    models = model_config or ModelConfig()
    validated = (
        attach_universe_membership(prices, universe_membership)
        if universe_membership is not None
        else validate_prices(prices)
    )
    _require_single_exchange_calendar(validated)
    panel = build_feature_panel(validated, features)
    eligible_dates = panel.loc[panel["eligible"], "date"]
    if eligible_dates.empty:
        raise ValueError("No date has a complete, investable feature cross-section.")
    signal_date = pd.Timestamp(eligible_dates.max())
    predictions, diagnostics = fit_latest_and_predict(panel, signal_date, models)
    predictions = predictions.sort_values(["score", "expected_return"], ascending=False).reset_index(drop=True)
    return signal_date, predictions, diagnostics


def _momentum_signals(
    panel: pd.DataFrame,
    dates: Iterable[object],
    max_history_years: int | None,
    bucket_count: int,
    minimum_calibration_dates: int,
    shrinkage_dates: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for signal_date in pd.to_datetime(list(dates)):
        current = panel[panel["date"].eq(signal_date) & panel["eligible"] & panel["momentum_12_1_xrank"].notna()].copy()
        if current.empty:
            continue
        history = panel[
            panel["eligible"]
            & panel[TARGET_COLUMN].notna()
            & panel[LABEL_DATE_COLUMN].notna()
            & panel[LABEL_DATE_COLUMN].lt(signal_date)
            & panel["momentum_12_1_xrank"].notna()
        ].copy()
        if max_history_years is not None:
            history_start = signal_date - pd.DateOffset(years=max_history_years)
            history = history[history["date"] >= history_start]
        if history["date"].nunique() < minimum_calibration_dates:
            continue

        history["_bucket"] = _rank_bucket(history["momentum_12_1_xrank"], bucket_count)
        by_date_bucket = history.groupby(["date", "_bucket"])[TARGET_COLUMN].mean()
        bucket_statistics = by_date_bucket.groupby("_bucket").agg(["mean", "count"])
        global_return = float(history.groupby("date")[TARGET_COLUMN].mean().mean())
        expected_by_bucket = (
            bucket_statistics["mean"] * bucket_statistics["count"] + global_return * shrinkage_dates
        ) / (bucket_statistics["count"] + shrinkage_dates)

        current["score"] = current["momentum_12_1_xrank"]
        current["_bucket"] = _rank_bucket(current["score"], bucket_count)
        current["expected_return"] = current["_bucket"].map(expected_by_bucket)
        current["expected_return"] = current["expected_return"].fillna(global_return)
        current["model_refit_date"] = signal_date
        frames.append(
            current[
                [
                    "date",
                    "ticker",
                    "expected_return",
                    "score",
                    "horizon_sessions",
                    "vol_60d",
                    "turnover_60d_median",
                    "model_refit_date",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "expected_return",
                "score",
                "horizon_sessions",
                "vol_60d",
                "turnover_60d_median",
                "model_refit_date",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _rank_bucket(ranks: pd.Series, bucket_count: int) -> pd.Series:
    return (np.ceil(ranks * bucket_count) - 1).clip(0, bucket_count - 1).astype(int)


def _normalize_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("start_date cannot be missing.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _require_single_exchange_calendar(prices: pd.DataFrame) -> None:
    if "exchange" not in prices.columns:
        return
    exchanges = prices["exchange"].drop_duplicates()
    if len(exchanges) > 1:
        raise ValueError(
            "One research run can contain only one exchange calendar; "
            "run each exchange separately so next-open execution stays exact."
        )

