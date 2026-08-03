import numpy as np
import pandas as pd
import pytest

from walk_forward.config import ModelConfig
from walk_forward.features import FEATURE_COLUMNS, LABEL_DATE_COLUMN, TARGET_COLUMN
from walk_forward.model import date_balanced_sample_weights, fit_latest_and_predict, mean_rank_ic, walk_forward_predict


def _synthetic_panel() -> pd.DataFrame:
    dates = pd.date_range("2014-01-31", periods=108, freq=pd.offsets.MonthEnd())
    tickers = [f"STOCK_{index:02d}" for index in range(9)]
    rows: list[dict[str, object]] = []

    for date_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            row: dict[str, object] = {
                "date": date,
                "ticker": ticker,
                LABEL_DATE_COLUMN: date + pd.offsets.MonthEnd(1),
                "eligible": True,
            }
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                cross_section = (ticker_index - 4) / 4
                slow_cycle = np.sin(date_index / (5 + feature_index % 7))
                interaction = np.cos((ticker_index + 1) * (feature_index + 1))
                row[feature] = (
                    cross_section * (0.4 + 0.03 * (feature_index % 5)) + 0.15 * slow_cycle + 0.02 * interaction
                )

            first_feature = float(row[FEATURE_COLUMNS[0]])
            row[TARGET_COLUMN] = (
                0.025 * first_feature + 0.006 * ((ticker_index - 4) / 4) + 0.003 * np.sin(date_index / 6)
            )
            row["vol_60d"] = 0.12 + 0.01 * ticker_index
            row["turnover_60d_median"] = 5_000_000 + 100_000 * ticker_index
            rows.append(row)
    return pd.DataFrame(rows)


def _test_config(refit_every_signals: int = 2) -> ModelConfig:
    return ModelConfig(
        validation_years=1,
        max_train_years=5,
        refit_every_signals=refit_every_signals,
        ensemble_size=2,
        min_train_rows=100,
        min_validation_dates=6,
        random_state=17,
    )


def test_date_balanced_weights_give_each_date_equal_mass() -> None:
    first = pd.Timestamp("2020-01-31")
    second = pd.Timestamp("2020-02-29")
    dates = pd.Series([first, first, first, second])

    weights = date_balanced_sample_weights(dates)

    assert weights.mean() == pytest.approx(1.0)
    mass_by_date = weights.groupby(dates).sum()
    assert mass_by_date.loc[first] == pytest.approx(mass_by_date.loc[second])


def test_mean_rank_ic_weights_dates_equally() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-31"] * 3 + ["2020-02-29"] * 3,
            TARGET_COLUMN: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "pred": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )

    assert mean_rank_ic(frame) == pytest.approx(0.0)


def test_walk_forward_refits_on_schedule_and_preserves_signal_columns() -> None:
    panel = _synthetic_panel()
    signal_dates = sorted(panel["date"].unique())[-4:]

    predictions, diagnostics = walk_forward_predict(
        panel,
        signal_dates,
        _test_config(refit_every_signals=2),
    )

    assert len(predictions) == 4 * panel["ticker"].nunique()
    assert {
        "date",
        "ticker",
        "expected_return",
        "score",
        "model_refit_date",
        "vol_60d",
        "turnover_60d_median",
    }.issubset(predictions.columns)
    assert predictions["expected_return"].abs().max() < 1.0
    assert predictions["score"].between(0.0, 1.0).all()

    refit_dates = diagnostics["refit_date"].drop_duplicates().sort_values().tolist()
    assert refit_dates == [pd.Timestamp(signal_dates[0]), pd.Timestamp(signal_dates[2])]
    assert (diagnostics["max_label_date"] < diagnostics["refit_date"]).all()
    assert diagnostics.groupby("refit_date")["selected"].sum().between(1, 2).all()
    threshold = _test_config().minimum_validation_rank_ic
    for _, refit_report in diagnostics.groupby("refit_date"):
        selected = refit_report[refit_report["selected"]]
        if (refit_report["validation_rank_ic"] >= threshold).any():
            assert (selected["validation_rank_ic"] >= threshold).all()
            assert set(refit_report["selection_rule"]) == {"positive_rank_ic_ensemble"}
        else:
            assert selected.empty
            assert set(refit_report["selection_rule"]) == {"cash_no_validated_edge"}


def test_latest_fit_ignores_targets_not_known_by_signal_date() -> None:
    panel = _synthetic_panel()
    signal_date = pd.Timestamp(sorted(panel["date"].unique())[-4])
    changed = panel.copy()
    unavailable = pd.to_datetime(changed[LABEL_DATE_COLUMN]) >= signal_date
    changed.loc[unavailable, TARGET_COLUMN] = changed.loc[unavailable, TARGET_COLUMN] + 1_000_000

    baseline_predictions, baseline_diagnostics = fit_latest_and_predict(
        panel,
        signal_date,
        _test_config(),
    )
    changed_predictions, changed_diagnostics = fit_latest_and_predict(
        changed,
        signal_date,
        _test_config(),
    )

    np.testing.assert_allclose(
        baseline_predictions[["expected_return", "score"]],
        changed_predictions[["expected_return", "score"]],
        rtol=1e-12,
        atol=1e-15,
    )
    assert (baseline_diagnostics["max_label_date"] < signal_date).all()
    pd.testing.assert_frame_equal(baseline_diagnostics, changed_diagnostics)


def test_ineligible_rows_are_neither_trained_nor_scored() -> None:
    panel = _synthetic_panel()
    signal_date = pd.Timestamp(sorted(panel["date"].unique())[-3])
    excluded_ticker = "STOCK_08"
    panel.loc[panel["ticker"] == excluded_ticker, "eligible"] = False

    predictions, diagnostics = fit_latest_and_predict(
        panel,
        signal_date,
        _test_config(),
    )

    assert excluded_ticker not in set(predictions["ticker"])
    eligible_training_rows = panel[
        panel["eligible"]
        & (pd.to_datetime(panel[LABEL_DATE_COLUMN]) < signal_date)
        & (panel["date"] >= signal_date - pd.DateOffset(years=5))
    ]
    assert diagnostics["final_fit_rows"].nunique() == 1
    assert diagnostics["final_fit_rows"].iloc[0] == len(eligible_training_rows)


def test_duplicate_panel_keys_are_rejected() -> None:
    panel = _synthetic_panel()
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    signal_date = sorted(panel["date"].unique())[-1]

    with pytest.raises(ValueError, match="one row per"):
        fit_latest_and_predict(duplicate, signal_date, _test_config())


def test_no_validated_edge_emits_cash_signal_instead_of_stale_holdings() -> None:
    panel = _synthetic_panel()
    panel["horizon_sessions"] = 20
    signal_date = pd.Timestamp(sorted(panel["date"].unique())[-3])
    config = ModelConfig(
        validation_years=1,
        max_train_years=5,
        min_train_rows=100,
        min_validation_dates=6,
        minimum_validation_rank_ic=1.1,
    )
    predictions, diagnostics = fit_latest_and_predict(panel, signal_date, config)

    assert not predictions.empty
    assert not predictions["validated_edge"].any()
    assert predictions["expected_return"].eq(-1.0).all()
    assert not diagnostics["selected"].any()
