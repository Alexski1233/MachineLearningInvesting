from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlinvesting.backtest import run_backtest
from mlinvesting.config import BacktestConfig


def _prices(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "ticker", "adj_open", "adj_close", "volume", "median_dollar_volume"],
    )


def _config(**overrides: object) -> BacktestConfig:
    defaults: dict[str, object] = {
        "initial_capital": 100.0,
        "top_n": 1,
        "buffer_n": 1,
        "max_weight": 1.0,
        "commission_bps": 0.0,
        "half_spread_bps": 0.0,
        "impact_bps": 0.0,
        "max_participation_rate": 1.0,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _signals(data: dict[str, list[object]]) -> pd.DataFrame:
    signals = pd.DataFrame(data)
    if "score" not in signals.columns:
        signals["score"] = signals.get("signal", signals.get("pred"))
    if "expected_return" not in signals.columns:
        signals["expected_return"] = signals.get("pred", signals.get("signal"))
    signals["horizon_sessions"] = 20
    signals["model_refit_date"] = signals["date"]
    return signals


def test_signal_trades_only_at_next_market_open() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 1_000, 10_000.0),
            ("2024-01-03", "A", 20.0, 20.0, 1_000, 10_000.0),
            ("2024-01-04", "A", 20.0, 22.0, 1_000, 10_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "signal": [0.1],
            "volatility": [0.2],
            "median_dollar_volume": [10_000.0],
        }
    )

    result = run_backtest(prices, signals, _config())

    assert result.trades.iloc[0]["date"] == pd.Timestamp("2024-01-03")
    assert result.trades.iloc[0]["price"] == 20.0
    assert result.daily_equity.iloc[-1]["equity"] == pytest.approx(110.0)
    assert result.metrics.turnover == pytest.approx(1.0)


def test_missing_future_price_does_not_replace_selected_name() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 1_000, 10_000.0),
            ("2024-01-02", "B", 10.0, 10.0, 1_000, 10_000.0),
            ("2024-01-03", "B", 10.0, 10.0, 1_000, 10_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "ticker": ["A", "B"],
            "signal": [0.2, 0.1],
            "volatility": [0.2, 0.2],
            "median_dollar_volume": [10_000.0, 10_000.0],
        }
    )

    result = run_backtest(prices, signals, _config())

    selected = result.selections[result.selections["selected"]]
    assert selected["ticker"].tolist() == ["A"]
    assert not bool(selected.iloc[0]["execution_price_available"])
    assert result.trades.empty


def test_rank_buffer_retains_incumbent() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = _prices([(date, ticker, 10.0, 10.0, 10_000, 100_000.0) for date in dates for ticker in ["A", "B"]])
    signals = _signals(
        {
            "date": [dates[0], dates[0], dates[2], dates[2]],
            "ticker": ["A", "B", "A", "B"],
            "signal": [0.2, 0.1, 0.1, 0.2],
            "volatility": [0.2] * 4,
            "median_dollar_volume": [100_000.0] * 4,
        }
    )

    result = run_backtest(prices, signals, _config(buffer_n=2))

    second = result.selections[result.selections["signal_date"] == dates[2]]
    retained = second[second["selected"]].iloc[0]
    assert retained["ticker"] == "A"
    assert bool(retained["retained_by_buffer"])


def test_inverse_volatility_caps_leave_cash_with_few_signals() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = _prices([(date, ticker, 10.0, 10.0, 100_000, 1_000_000.0) for date in dates for ticker in ["A", "B", "C"]])
    signals = _signals(
        {
            "date": [dates[0]] * 3,
            "ticker": ["A", "B", "C"],
            "signal": [0.3, 0.2, -0.1],
            "volatility": [0.1, 0.2, 0.3],
            "median_dollar_volume": [1_000_000.0] * 3,
        }
    )

    result = run_backtest(prices, signals, _config(top_n=3, buffer_n=3, max_weight=0.4))

    selected = result.selections[result.selections["selected"]]
    assert selected["target_weight"].sum() == pytest.approx(0.8)
    assert result.daily_equity.iloc[0]["cash"] == pytest.approx(20.0)


def test_equal_weighting_matches_concentrated_top_n_portfolio() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    tickers = ["A", "B", "C", "D", "E"]
    prices = _prices(
        [(date, ticker, 10.0, 10.0, 100_000, 1_000_000.0) for date in dates for ticker in tickers]
    )
    signals = _signals(
        {
            "date": [dates[0]] * 5,
            "ticker": tickers,
            "signal": [0.5, 0.4, 0.3, 0.2, 0.1],
            "volatility": [0.1, 0.2, 0.3, 0.4, 0.5],
            "median_dollar_volume": [1_000_000.0] * 5,
        }
    )

    result = run_backtest(
        prices,
        signals,
        _config(top_n=5, buffer_n=5, max_weight=0.2, weighting_method="equal"),
    )

    selected = result.selections[result.selections["selected"]]
    assert selected["target_weight"].tolist() == pytest.approx([0.2] * 5)


def test_capacity_and_square_root_impact_cost() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 1_000, 1_000.0),
            ("2024-01-03", "A", 10.0, 10.0, 1_000, 1_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "signal": [0.1],
            "volatility": [0.2],
            "median_dollar_volume": [1_000.0],
        }
    )
    config = _config(
        initial_capital=1_000.0, commission_bps=10.0, half_spread_bps=10.0, impact_bps=100.0, max_participation_rate=0.1
    )

    result = run_backtest(prices, signals, config)
    trade = result.trades.iloc[0]

    expected_bps = 20.0 + 100.0 * np.sqrt(0.1)
    assert trade["notional"] == pytest.approx(100.0)
    assert trade["participation"] == pytest.approx(0.1)
    assert trade["cost_bps"] == pytest.approx(expected_bps)
    assert result.metrics.total_costs == pytest.approx(100.0 * expected_bps / 10_000)


def test_stale_position_is_carried_then_conservatively_recovered() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-03", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-04", "MKT", 1.0, 1.0, 10_000, 100_000.0),
            ("2024-01-05", "MKT", 1.0, 1.0, 10_000, 100_000.0),
            ("2024-01-08", "MKT", 1.0, 1.0, 10_000, 100_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "signal": [0.1],
            "volatility": [0.2],
            "median_dollar_volume": [100_000.0],
        }
    )

    result = run_backtest(prices, signals, _config(max_stale_days=2, stale_position_recovery=0.0))

    carried = result.holdings[result.holdings["ticker"] == "A"]
    assert carried["stale_days"].tolist()[-2:] == [1, 2]
    assert "stale_liquidation" in result.trades["side"].tolist()
    assert result.daily_equity.iloc[-1]["equity"] == pytest.approx(0.0)


def test_delisting_return_is_applied_without_future_filtering() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-03", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-04", "A", 10.0, 10.0, 10_000, 100_000.0),
        ]
    )
    prices["delisting_return"] = [np.nan, np.nan, -0.5]
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "signal": [0.1],
            "volatility": [0.2],
            "median_dollar_volume": [100_000.0],
        }
    )

    result = run_backtest(prices, signals, _config())

    assert "delisting" in result.trades["side"].tolist()
    assert result.daily_equity.iloc[-1]["equity"] == pytest.approx(50.0)


def test_drawdown_includes_initial_capital() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-03", "A", 10.0, 8.0, 10_000, 100_000.0),
            ("2024-01-04", "A", 10.0, 10.0, 10_000, 100_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "signal": [0.1],
            "volatility": [0.2],
            "median_dollar_volume": [100_000.0],
        }
    )

    result = run_backtest(prices, signals, _config())

    assert result.metrics.max_drawdown == pytest.approx(-0.2)


def test_cash_accrues_risk_free_rate_and_has_zero_excess_return() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-03", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-04", "A", 10.0, 10.0, 10_000, 100_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "signal": [-0.1],
            "volatility": [0.2],
            "median_dollar_volume": [100_000.0],
        }
    )
    result = run_backtest(prices, signals, _config(annual_risk_free_rate=0.10))
    daily_rf = (1.10 ** (1 / 252)) - 1

    assert result.daily_equity.iloc[-1]["equity"] == pytest.approx(100.0 * (1 + daily_rf) ** 2)
    assert np.allclose(result.daily_equity["excess_return"], 0.0, atol=1e-12)


def test_expected_return_must_clear_round_trip_linear_cost() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = _prices([(date, ticker, 10.0, 10.0, 100_000, 1_000_000.0) for date in dates for ticker in ["A", "B"]])
    signals = _signals(
        {
            "date": [dates[0], dates[0]],
            "ticker": ["A", "B"],
            "pred": [0.001, 0.010],
            "score": [1.0, 0.5],
            "volatility": [0.2, 0.2],
            "median_dollar_volume": [1_000_000.0, 1_000_000.0],
        }
    )
    result = run_backtest(
        prices,
        signals,
        _config(commission_bps=10.0, half_spread_bps=10.0),
    )
    selected = result.selections[result.selections["selected"]]
    assert selected["ticker"].tolist() == ["B"]


def test_rebalance_tolerance_avoids_small_weight_churn() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = _prices([(date, ticker, 10.0, 10.0, 100_000, 1_000_000.0) for date in dates for ticker in ["A", "B"]])
    signals = _signals(
        {
            "date": [dates[0], dates[0], dates[2], dates[2]],
            "ticker": ["A", "B", "A", "B"],
            "pred": [0.02] * 4,
            "score": [1.0, 0.5, 1.0, 0.5],
            "volatility": [0.2, 0.2, 0.198, 0.202],
            "median_dollar_volume": [1_000_000.0] * 4,
        }
    )
    result = run_backtest(
        prices,
        signals,
        _config(top_n=2, buffer_n=2, max_weight=0.6, rebalance_tolerance=0.01),
    )
    assert len(result.trades) == 2


def test_model_prediction_is_preferred_to_rank_score_for_thresholding() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 10_000, 100_000.0),
            ("2024-01-03", "A", 10.0, 10.0, 10_000, 100_000.0),
        ]
    )
    signals = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "pred": [-0.01],
            "score": [1.0],
            "vol_60d": [0.2],
            "turnover_60d_median": [100_000.0],
        }
    )

    result = run_backtest(prices, signals, _config())

    assert not bool(result.selections.iloc[0]["threshold_eligible"])
    assert result.trades.empty


def test_expected_return_must_clear_impact_and_risk_free_hurdle() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    prices = _prices([(date, "A", 10.0, 10.0, 100_000, 1_000_000.0) for date in dates])
    signals = _signals(
        {
            "date": [dates[0]],
            "ticker": ["A"],
            "score": [1.0],
            "expected_return": [0.005],
            "volatility": [0.2],
            "median_dollar_volume": [1_000_000.0],
        }
    )
    result = run_backtest(
        prices,
        signals,
        _config(
            annual_risk_free_rate=0.10,
            impact_bps=100.0,
            max_participation_rate=0.01,
        ),
    )
    assert result.trades.empty
    assert result.selections.iloc[0]["entry_threshold"] > 0.005


def test_negative_expected_return_does_not_retain_incumbent() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    prices = _prices([(date, "A", 10.0, 10.0, 100_000, 1_000_000.0) for date in dates])
    signals = _signals(
        {
            "date": [dates[0], dates[2]],
            "ticker": ["A", "A"],
            "score": [1.0, 1.0],
            "expected_return": [0.02, -0.01],
            "volatility": [0.2, 0.2],
            "median_dollar_volume": [1_000_000.0, 1_000_000.0],
        }
    )
    result = run_backtest(prices, signals, _config())
    assert result.trades["side"].tolist() == ["buy", "sell"]


def test_signal_contract_rejects_rank_without_return_or_future_refit() -> None:
    prices = _prices(
        [
            ("2024-01-02", "A", 10.0, 10.0, 100_000, 1_000_000.0),
            ("2024-01-03", "A", 10.0, 10.0, 100_000, 1_000_000.0),
        ]
    )
    incomplete = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "score": [1.0],
            "horizon_sessions": [20],
            "model_refit_date": ["2024-01-02"],
        }
    )
    with pytest.raises(ValueError, match="expected_return"):
        run_backtest(prices, incomplete, _config())

    future_refit = _signals(
        {
            "date": ["2024-01-02"],
            "ticker": ["A"],
            "score": [1.0],
            "expected_return": [0.02],
        }
    )
    future_refit["model_refit_date"] = "2024-01-03"
    with pytest.raises(ValueError, match="cannot be later"):
        run_backtest(prices, future_refit, _config())


def test_backtest_config_rejects_non_finite_assumptions() -> None:
    with pytest.raises(ValueError, match="commission_bps must be finite"):
        _config(commission_bps=np.nan)
    with pytest.raises(ValueError, match="weighting_method"):
        _config(weighting_method="unknown")
