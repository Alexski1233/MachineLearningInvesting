"""Leak-aware model selection and walk-forward prediction.

The public functions in this module preserve the temporal ordering of the
panel.  A model fitted at time ``t`` can only see observations whose complete
forward-return label was available strictly before ``t``.  Candidate models
are selected on a trailing, chronological validation window and are then
refitted on all labels known at the refit date.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .features import FEATURE_COLUMNS, LABEL_DATE_COLUMN, TARGET_COLUMN

DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
PASSTHROUGH_COLUMNS = ("horizon_sessions", "vol_60d", "turnover_60d_median")


@dataclass(frozen=True)
class _FittedEnsemble:
    """Models and audit metadata from one point-in-time refit."""

    models: tuple[tuple[str, Any], ...]
    refit_date: pd.Timestamp
    validation_start: pd.Timestamp
    selected_models: tuple[str, ...]
    max_label_date: pd.Timestamp
    calibration_by_bucket: tuple[float, ...]


def model_candidates(random_state: int = 0) -> dict[str, BaseEstimator]:
    """Return a small, fixed set of complementary regularized regressors.

    The candidates intentionally cover a stable linear baseline and two
    nonlinear tree families.  Hyperparameters are fixed here; the walk-forward
    routine selects model families using only its trailing validation sample.
    """

    return {
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=20,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def date_balanced_sample_weights(dates: pd.Series) -> pd.Series:
    """Return row weights that give every cross-sectional date equal mass.

    Within a date, weight is divided equally among available securities.  The
    resulting weights are normalized to have a row-wise mean of one, which
    keeps estimator regularization scales easy to interpret.
    """

    if dates.empty:
        return pd.Series(dtype=float, index=dates.index)
    if dates.isna().any():
        raise ValueError("Training dates cannot be missing.")

    counts = dates.groupby(dates, sort=False).transform("size").astype(float)
    weights = 1.0 / counts
    return weights * (len(weights) / weights.sum())


def mean_rank_ic(
    frame: pd.DataFrame,
    prediction_column: str = "pred",
) -> float:
    """Compute the equally weighted mean cross-sectional Spearman IC by date."""

    required = {DATE_COLUMN, TARGET_COLUMN, prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Rank-IC frame is missing columns: {sorted(missing)}")

    correlations: list[float] = []
    for _, cross_section in frame.groupby(DATE_COLUMN, sort=True):
        valid = cross_section[[TARGET_COLUMN, prediction_column]].dropna()
        if len(valid) < 2:
            continue
        target_rank = valid[TARGET_COLUMN].rank(method="average")
        prediction_rank = valid[prediction_column].rank(method="average")
        if target_rank.nunique() < 2 or prediction_rank.nunique() < 2:
            continue
        correlation = prediction_rank.corr(target_rank)
        if pd.notna(correlation):
            correlations.append(float(correlation))
    return float(np.mean(correlations)) if correlations else float("nan")


def walk_forward_predict(
    panel: pd.DataFrame,
    signal_dates: Iterable[object],
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate purged, periodically refitted predictions for signal dates.

    Signal dates are deduplicated and evaluated chronologically.  On the first
    date and every ``config.refit_every_signals`` dates thereafter, candidate
    models are selected on a trailing historical validation window.  The
    selected models are then refitted on all complete labels with
    ``label_date < refit_date``.  Between refits, the prior ensemble is reused.

    Returns:
        A pair ``(predictions, diagnostics)``.  Predictions contain ``date``,
        ``ticker``, mean predicted holding-period return in ``expected_return``, a cross-sectional
        rank-mean ``score``, the model refit date, and available signal columns.
        Diagnostics contain one row per candidate and refit, including
        validation rank IC and label cutoffs.
    """

    resolved_config = config or ModelConfig()
    prepared = _prepare_panel(panel)
    dates = _normalize_signal_dates(signal_dates)
    if not dates:
        return _empty_predictions(prepared), _empty_diagnostics()

    prediction_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    ensemble: _FittedEnsemble | None = None

    for signal_index, signal_date in enumerate(dates):
        needs_refit = ensemble is None or signal_index % resolved_config.refit_every_signals == 0
        if needs_refit:
            ensemble, diagnostics = _fit_ensemble(
                prepared,
                refit_date=signal_date,
                config=resolved_config,
            )
            diagnostic_frames.append(diagnostics)
        if ensemble is None:
            raise RuntimeError("Walk-forward refit did not produce an ensemble state.")
        prediction_frames.append(_predict_for_date(prepared, signal_date, ensemble))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    return predictions, diagnostics


def fit_latest_and_predict(
    panel: pd.DataFrame,
    signal_date: object,
    config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a production ensemble using all labels known before ``signal_date``.

    Model-family selection is confined to the historical trailing validation
    window.  After selection, each chosen model is refitted on every eligible
    labeled row in the configured training window and used to score the signal
    date's cross-section.
    """

    resolved_config = config or ModelConfig()
    prepared = _prepare_panel(panel)
    normalized_signal_date = _normalize_timestamp(signal_date)
    ensemble, diagnostics = _fit_ensemble(
        prepared,
        refit_date=normalized_signal_date,
        config=resolved_config,
    )
    predictions = _predict_for_date(prepared, normalized_signal_date, ensemble)
    return predictions, diagnostics


def _fit_ensemble(
    panel: pd.DataFrame,
    refit_date: pd.Timestamp,
    config: ModelConfig,
) -> tuple[_FittedEnsemble, pd.DataFrame]:
    eligible = _eligible_training_rows(panel, refit_date, config)
    validation_start = refit_date - pd.DateOffset(years=config.validation_years)

    development = eligible[
        (eligible[DATE_COLUMN] < validation_start) & (eligible[LABEL_DATE_COLUMN] < validation_start)
    ].copy()
    validation = eligible[eligible[DATE_COLUMN] >= validation_start].copy()

    if len(development) < config.min_train_rows:
        raise ValueError(
            f"Only {len(development):,} purged training rows are available at "
            f"{refit_date.date()}; need at least {config.min_train_rows:,}."
        )
    validation_dates = validation[DATE_COLUMN].nunique()
    if validation_dates < config.min_validation_dates:
        raise ValueError(
            f"Only {validation_dates} validation dates are available at "
            f"{refit_date.date()}; need at least {config.min_validation_dates}."
        )

    train_weights = date_balanced_sample_weights(development[DATE_COLUMN])
    candidates = model_candidates(config.random_state)
    report_rows: list[dict[str, object]] = []
    validation_predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in candidates.items():
        fitted = _fit_estimator(clone(estimator), development, train_weights)
        validation_scored = validation[[DATE_COLUMN, TARGET_COLUMN]].copy()
        candidate_predictions = fitted.predict(validation[list(FEATURE_COLUMNS)])
        validation_predictions[model_name] = candidate_predictions
        validation_scored["pred"] = candidate_predictions
        report_rows.append(
            {
                "model": model_name,
                "validation_rank_ic": mean_rank_ic(validation_scored),
            }
        )

    report = pd.DataFrame(report_rows)
    finite_scores = report[np.isfinite(report["validation_rank_ic"])].copy()
    if finite_scores.empty:
        raise ValueError(f"No candidate produced a finite validation rank IC at {refit_date.date()}.")
    finite_scores = finite_scores.sort_values(
        ["validation_rank_ic", "model"],
        ascending=[False, True],
    )
    positive_scores = finite_scores[finite_scores["validation_rank_ic"] >= config.minimum_validation_rank_ic]
    if positive_scores.empty:
        selected_models = ()
        selection_rule = "cash_no_validated_edge"
    else:
        ensemble_size = min(config.ensemble_size, len(positive_scores))
        selected_models = tuple(positive_scores.head(ensemble_size)["model"].astype(str))
        selection_rule = "positive_rank_ic_ensemble"

    calibration_by_bucket = _calibrate_expected_returns(
        validation,
        validation_predictions,
        selected_models,
        config,
    )

    final_weights = date_balanced_sample_weights(eligible[DATE_COLUMN])
    fitted_models: list[tuple[str, Any]] = []
    for model_name in selected_models:
        final_model = _fit_estimator(
            clone(candidates[model_name]),
            eligible,
            final_weights,
        )
        fitted_models.append((model_name, final_model))

    max_label_date = pd.Timestamp(eligible[LABEL_DATE_COLUMN].max())
    report["selected"] = report["model"].isin(selected_models)
    report["refit_date"] = refit_date
    report["validation_start"] = validation_start
    report["development_rows"] = len(development)
    report["validation_rows"] = len(validation)
    report["validation_dates"] = validation_dates
    report["final_fit_rows"] = len(eligible)
    report["max_label_date"] = max_label_date
    report["selected_models"] = ",".join(selected_models)
    report["selection_rule"] = selection_rule
    report["calibration_top_bucket_return"] = calibration_by_bucket[-1] if calibration_by_bucket else np.nan
    report = report.sort_values(
        ["selected", "validation_rank_ic", "model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    ensemble = _FittedEnsemble(
        models=tuple(fitted_models),
        refit_date=refit_date,
        validation_start=validation_start,
        selected_models=selected_models,
        max_label_date=max_label_date,
        calibration_by_bucket=calibration_by_bucket,
    )
    return ensemble, report


def _eligible_training_rows(
    panel: pd.DataFrame,
    refit_date: pd.Timestamp,
    config: ModelConfig,
) -> pd.DataFrame:
    feature_complete = panel[list(FEATURE_COLUMNS)].notna().all(axis=1)
    mask = (
        feature_complete
        & panel[TARGET_COLUMN].notna()
        & panel[LABEL_DATE_COLUMN].notna()
        & (panel[DATE_COLUMN] < refit_date)
        & (panel[LABEL_DATE_COLUMN] < refit_date)
    )
    if "eligible" in panel.columns:
        mask &= panel["eligible"]
    eligible = panel.loc[mask].copy()
    if config.max_train_years is not None:
        train_start = refit_date - pd.DateOffset(years=config.max_train_years)
        eligible = eligible[eligible[DATE_COLUMN] >= train_start].copy()
    if eligible.empty:
        raise ValueError(f"No complete labels are known before {refit_date.date()}.")
    return eligible


def _fit_estimator(
    estimator: BaseEstimator,
    frame: pd.DataFrame,
    sample_weights: pd.Series,
) -> Any:
    features = frame[list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]
    weights = sample_weights.to_numpy(dtype=float)
    if isinstance(estimator, Pipeline):
        estimator.fit(features, target, model__sample_weight=weights)
    else:
        estimator.fit(features, target, sample_weight=weights)
    return estimator


def _predict_for_date(
    panel: pd.DataFrame,
    signal_date: pd.Timestamp,
    ensemble: _FittedEnsemble,
) -> pd.DataFrame:
    signal = panel[panel[DATE_COLUMN] == signal_date].copy()
    if "eligible" in signal.columns:
        signal = signal[signal["eligible"]].copy()
    signal = signal[signal[list(FEATURE_COLUMNS)].notna().all(axis=1)].copy()
    output_columns = _prediction_columns(panel)
    if signal.empty:
        return pd.DataFrame(columns=output_columns)
    if not ensemble.models:
        passthrough = [column for column in PASSTHROUGH_COLUMNS if column in signal.columns]
        predictions = signal[[DATE_COLUMN, TICKER_COLUMN, *passthrough]].copy()
        predictions.insert(2, "expected_return", -1.0)
        predictions.insert(3, "score", 0.0)
        predictions["model_refit_date"] = ensemble.refit_date
        predictions["validated_edge"] = False
        return predictions[output_columns]

    raw_predictions = np.column_stack([model.predict(signal[list(FEATURE_COLUMNS)]) for _, model in ensemble.models])
    ranked_predictions = pd.DataFrame(raw_predictions).rank(
        axis=0,
        method="average",
        pct=True,
    )
    ensemble_score = ranked_predictions.mean(axis=1).to_numpy(dtype=float)
    calibration = np.asarray(ensemble.calibration_by_bucket, dtype=float)
    buckets = _rank_bucket(ensemble_score, len(calibration))
    expected_return = calibration[buckets]

    passthrough = [column for column in PASSTHROUGH_COLUMNS if column in signal.columns]
    predictions = signal[[DATE_COLUMN, TICKER_COLUMN, *passthrough]].copy()
    predictions.insert(2, "expected_return", expected_return)
    predictions.insert(3, "score", ensemble_score)
    predictions["model_refit_date"] = ensemble.refit_date
    predictions["validated_edge"] = True
    return predictions[output_columns]


def _calibrate_expected_returns(
    validation: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    selected_models: tuple[str, ...],
    config: ModelConfig,
) -> tuple[float, ...]:
    if not selected_models:
        return ()
    scored = validation[[DATE_COLUMN, TARGET_COLUMN]].copy()
    rank_columns: list[str] = []
    for model_name in selected_models:
        prediction_column = f"_{model_name}_prediction"
        rank_column = f"_{model_name}_rank"
        scored[prediction_column] = predictions[model_name]
        scored[rank_column] = scored.groupby(DATE_COLUMN)[prediction_column].rank(
            method="average",
            pct=True,
        )
        rank_columns.append(rank_column)
    scored["_score"] = scored[rank_columns].mean(axis=1)
    scored["_bucket"] = _rank_bucket(
        scored["_score"].to_numpy(dtype=float),
        config.calibration_buckets,
    )
    by_date_bucket = scored.groupby([DATE_COLUMN, "_bucket"])[TARGET_COLUMN].mean()
    statistics = by_date_bucket.groupby("_bucket").agg(["mean", "count"])
    global_return = float(scored.groupby(DATE_COLUMN)[TARGET_COLUMN].mean().mean())
    calibrated: list[float] = []
    for bucket in range(config.calibration_buckets):
        if bucket not in statistics.index:
            calibrated.append(global_return)
            continue
        bucket_mean = float(statistics.loc[bucket, "mean"])
        bucket_dates = int(statistics.loc[bucket, "count"])
        weight = config.calibration_shrinkage_dates
        calibrated.append((bucket_mean * bucket_dates + global_return * weight) / (bucket_dates + weight))
    return tuple(calibrated)


def _rank_bucket(scores: np.ndarray, bucket_count: int) -> np.ndarray:
    return np.clip(np.ceil(scores * bucket_count).astype(int) - 1, 0, bucket_count - 1)


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        DATE_COLUMN,
        TICKER_COLUMN,
        TARGET_COLUMN,
        LABEL_DATE_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    prepared = panel.copy()
    prepared[DATE_COLUMN] = _normalize_datetime_series(prepared[DATE_COLUMN], DATE_COLUMN)
    prepared[LABEL_DATE_COLUMN] = _normalize_datetime_series(
        prepared[LABEL_DATE_COLUMN],
        LABEL_DATE_COLUMN,
        allow_missing=True,
    )
    prepared[TICKER_COLUMN] = prepared[TICKER_COLUMN].astype(str)
    if "eligible" in prepared.columns:
        prepared["eligible"] = prepared["eligible"].fillna(False).astype(bool)
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")
    prepared.loc[~np.isfinite(prepared[TARGET_COLUMN]), TARGET_COLUMN] = np.nan
    for feature in FEATURE_COLUMNS:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
        prepared.loc[~np.isfinite(prepared[feature]), feature] = np.nan

    duplicate_keys = prepared.duplicated([DATE_COLUMN, TICKER_COLUMN], keep=False)
    if duplicate_keys.any():
        examples = prepared.loc[duplicate_keys, [DATE_COLUMN, TICKER_COLUMN]].head(5)
        raise ValueError(
            f"Panel must contain one row per (date, ticker); duplicate examples: {examples.to_dict(orient='records')}"
        )
    return prepared.sort_values([DATE_COLUMN, TICKER_COLUMN]).reset_index(drop=True)


def _normalize_datetime_series(
    values: pd.Series,
    column_name: str,
    allow_missing: bool = False,
) -> pd.Series:
    original_missing = values.isna()
    normalized = values.map(_normalize_datetime_value)
    normalized = pd.to_datetime(normalized)
    invalid = normalized.isna() & ~original_missing
    if invalid.any():
        raise ValueError(f"Column {column_name!r} contains invalid dates.")
    if not allow_missing and normalized.isna().any():
        raise ValueError(f"Column {column_name!r} cannot contain missing dates.")
    return normalized


def _normalize_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Signal dates cannot be missing.")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalize_datetime_value(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalize_signal_dates(values: Iterable[object]) -> list[pd.Timestamp]:
    return sorted({_normalize_timestamp(value) for value in values})


def _prediction_columns(panel: pd.DataFrame) -> list[str]:
    passthrough = [column for column in PASSTHROUGH_COLUMNS if column in panel.columns]
    return [
        DATE_COLUMN,
        TICKER_COLUMN,
        "expected_return",
        "score",
        *passthrough,
        "model_refit_date",
        "validated_edge",
    ]


def _empty_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=_prediction_columns(panel))


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "validation_rank_ic",
            "selected",
            "refit_date",
            "validation_start",
            "development_rows",
            "validation_rows",
            "validation_dates",
            "final_fit_rows",
            "max_label_date",
            "selected_models",
            "selection_rule",
            "calibration_top_bucket_return",
        ]
    )


__all__ = [
    "date_balanced_sample_weights",
    "fit_latest_and_predict",
    "mean_rank_ic",
    "model_candidates",
    "walk_forward_predict",
]
