import numpy as np
import pandas as pd

from walk_forward.config import BacktestConfig, FeatureConfig, ModelConfig
from walk_forward.pipeline import run_research


def test_walk_forward_pipeline_runs_end_to_end_without_same_close_execution() -> None:
    dates = pd.bdate_range("2018-01-02", periods=950)
    tickers = [f"STOCK_{index}" for index in range(8)]
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers):
        drift = 0.00015 + ticker_index * 0.000025
        cycle = 0.0008 * np.sin(np.arange(len(dates)) / (25 + ticker_index))
        returns = drift + cycle
        close = (10 + ticker_index) * np.exp(np.cumsum(returns))
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": close[date_index] * 0.999,
                    "high": close[date_index] * 1.002,
                    "low": close[date_index] * 0.998,
                    "close": close[date_index],
                    "adj_close": close[date_index],
                    "volume": 2_000_000 + ticker_index * 100_000,
                    "in_universe": True,
                }
            )
    prices = pd.DataFrame(rows)

    result = run_research(
        prices,
        start_date=dates[900],
        feature_config=FeatureConfig(
            holding_days=20,
            min_history_days=252,
            liquidity_lookback_days=60,
            min_median_dollar_volume=1.0,
        ),
        model_config=ModelConfig(
            validation_years=1,
            max_train_years=3,
            refit_every_signals=3,
            ensemble_size=2,
            min_train_rows=500,
            min_validation_dates=20,
            random_state=7,
        ),
        backtest_config=BacktestConfig(
            initial_capital=100_000.0,
            top_n=3,
            buffer_n=5,
            max_weight=0.5,
            commission_bps=0.0,
            half_spread_bps=0.0,
            impact_bps=0.0,
            max_participation_rate=1.0,
        ),
    )

    assert not result.predictions.empty
    assert not result.strategy.daily_equity.empty
    assert result.survivorship_safe
    assert (result.model_diagnostics["max_label_date"] < result.model_diagnostics["refit_date"]).all()
    regular_trades = result.strategy.trades[result.strategy.trades["side"].isin(["buy", "sell"])]
    assert (regular_trades["date"] > regular_trades["signal_date"]).all()
