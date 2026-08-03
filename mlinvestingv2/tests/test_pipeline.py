from __future__ import annotations

import pandas as pd
import pytest

from mlinvesting.features import LABEL_DATE_COLUMN, TARGET_COLUMN
from mlinvesting.pipeline import _momentum_signals, blend_ranked_signals, make_signal_dates, run_research


def test_signal_dates_use_one_fixed_session_phase() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=10),
            "ticker": ["A"] * 10,
        }
    )
    dates = make_signal_dates(prices, "2024-01-03", every_n_sessions=3)
    assert dates == (
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-11"),
    )


def test_signal_dates_reject_empty_range() -> None:
    prices = pd.DataFrame({"date": ["2024-01-01"], "ticker": ["A"]})
    with pytest.raises(ValueError, match="No market dates"):
        make_signal_dates(prices, "2025-01-01", every_n_sessions=20)


def test_research_rejects_mixed_exchange_calendars() -> None:
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": "2024-01-02",
                "open": 10.0,
                "close": 10.0,
                "adj_close": 10.0,
                "volume": 1_000_000,
                "exchange": exchange,
            }
            for ticker, exchange in (("A", "OSE"), ("B", "NYSE"))
        ]
    )
    with pytest.raises(ValueError, match="only one exchange calendar"):
        run_research(prices, "2024-01-02")


def test_momentum_calibration_uses_only_labels_known_before_signal() -> None:
    dates = pd.date_range("2020-01-31", periods=20, freq=pd.offsets.MonthEnd())
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index in range(5):
            rows.append(
                {
                    "date": date,
                    "ticker": f"S{ticker_index}",
                    "eligible": True,
                    "momentum_12_1_xrank": (ticker_index + 1) / 5,
                    TARGET_COLUMN: 0.001 * ticker_index + 0.0001 * date_index,
                    LABEL_DATE_COLUMN: date + pd.offsets.MonthEnd(1),
                    "horizon_sessions": 20,
                    "vol_60d": 0.2,
                    "turnover_60d_median": 1_000_000.0,
                }
            )
    panel = pd.DataFrame(rows)
    signal_date = dates[-1]
    unavailable = panel[LABEL_DATE_COLUMN] >= signal_date
    changed = panel.copy()
    changed.loc[unavailable, TARGET_COLUMN] += 1_000_000

    baseline = _momentum_signals(panel, [signal_date], 12, 5, 12, 24)
    altered = _momentum_signals(changed, [signal_date], 12, 5, 12, 24)
    pd.testing.assert_frame_equal(baseline, altered)


def test_rank_blend_preserves_ml_refit_date_and_trade_fields() -> None:
    common = {
        "date": [pd.Timestamp("2024-01-02")],
        "ticker": ["A"],
        "horizon_sessions": [20],
        "vol_60d": [0.2],
        "turnover_60d_median": [1_000_000.0],
        "model_refit_date": [pd.Timestamp("2023-12-01")],
    }
    ml = pd.DataFrame({**common, "score": [0.8], "expected_return": [0.04]})
    momentum = pd.DataFrame({**common, "score": [0.4], "expected_return": [0.02]})

    blended = blend_ranked_signals(ml, momentum, ml_weight=0.25)

    assert blended.loc[0, "score"] == pytest.approx(0.5)
    assert blended.loc[0, "expected_return"] == pytest.approx(0.025)
    assert blended.loc[0, "model_refit_date"] == pd.Timestamp("2023-12-01")
