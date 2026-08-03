"""Run the historical simulation and latest stock selection."""

from pathlib import Path

import pandas as pd

from mlinvesting.backtest import BacktestResult, expected_return_hurdle, run_backtest
from mlinvesting.config import BacktestConfig, FeatureConfig, ModelConfig
from mlinvesting.data import drop_invalid_price_rows
from mlinvesting.io import load_price_directory
from mlinvesting.model import fit_latest_and_predict
from mlinvesting.pipeline import (
    blend_ranked_signals,
    momentum_predictions_for_dates,
    run_research,
)


PROJECT_DIR = Path(__file__).resolve().parent
PRICE_DIR = PROJECT_DIR.parent / "data" / "raw_prices"
OUTPUT_DIR = PROJECT_DIR / "output"
BACKTEST_START = "2020-01-02"
ML_WEIGHT = 0.25


def main() -> None:
    """Print a realistic backtest followed by the latest candidates."""
    prices, dropped = drop_invalid_price_rows(load_price_directory(PRICE_DIR))
    if not dropped.empty:
        print(
            f"Ignored {len(dropped)} invalid historical price rows "
            f"in {dropped['ticker'].nunique()} tickers."
        )

    feature_config = FeatureConfig()
    model_config = ModelConfig(refit_every_signals=12)
    backtest_config = BacktestConfig(
        top_n=5,
        buffer_n=8,
        max_weight=0.35,
        weighting_method="inverse_volatility",
    )

    print(f"Running historical simulation from {BACKTEST_START}. This normally takes 5-10 minutes...")
    research = run_research(
        prices,
        start_date=BACKTEST_START,
        feature_config=feature_config,
        model_config=model_config,
        backtest_config=backtest_config,
    )
    hybrid_signals = blend_ranked_signals(
        research.predictions,
        research.momentum_predictions,
        ML_WEIGHT,
    )
    hybrid = run_backtest(prices, hybrid_signals, backtest_config)
    comparison = _comparison(hybrid, research.strategy, research.momentum_baseline)
    _print_backtest(comparison)

    signal_date = pd.Timestamp(research.panel.loc[research.panel["eligible"], "date"].max())
    predictions, _ = fit_latest_and_predict(research.panel, signal_date, model_config)
    latest_momentum = momentum_predictions_for_dates(research.panel, [signal_date], model_config)
    predictions = blend_ranked_signals(predictions, latest_momentum, ML_WEIGHT)
    predictions = predictions.sort_values(["score", "expected_return"], ascending=False).reset_index(drop=True)
    predictions["entry_threshold"] = predictions["horizon_sessions"].map(
        lambda horizon: expected_return_hurdle(int(horizon), backtest_config)
    )
    candidates = predictions[predictions["expected_return"] > predictions["entry_threshold"]]
    _print_latest(signal_date, candidates, backtest_config.top_n)
    _save_results(comparison, predictions)


def _comparison(
    hybrid: BacktestResult,
    pure_ml: BacktestResult,
    momentum: BacktestResult,
) -> pd.DataFrame:
    rows = []
    for name, result in (
        ("hybrid_25_ml_75_momentum", hybrid),
        ("pure_ml", pure_ml),
        ("pure_momentum", momentum),
    ):
        rows.append({"strategy": name, **result.metrics.as_dict()})
    return pd.DataFrame(rows).set_index("strategy")


def _print_backtest(comparison: pd.DataFrame) -> None:
    strategy = comparison.loc["hybrid_25_ml_75_momentum"]
    pure_ml = comparison.loc["pure_ml"]
    momentum = comparison.loc["pure_momentum"]
    print("\n=== Historical simulation after trading costs ===")
    print(f"Total return:       {strategy['net_total_return']:.1%}")
    print(f"EAR / annual return:{strategy['net_cagr']:>9.1%}")
    print(f"Sharpe:             {strategy['daily_excess_sharpe']:>9.2f}")
    print(f"Maximum drawdown:   {strategy['max_drawdown']:>9.1%}")
    print(f"Trading costs:      {strategy['total_costs']:>12,.0f} NOK")
    print(f"Ending value:       {strategy['ending_equity']:>12,.0f} NOK")
    print(f"Pure ML EAR:        {pure_ml['net_cagr']:>9.1%}")
    print(f"Momentum EAR:       {momentum['net_cagr']:>9.1%}")
    print("Note: Historical membership data is missing, so survivorship bias may remain.")


def _print_latest(signal_date: pd.Timestamp, candidates: pd.DataFrame, top_n: int) -> None:
    print(f"\n=== Top {top_n} BUY candidates after close {signal_date.date()} ===")
    if candidates.empty:
        print("No stocks currently clear the estimated trading-cost threshold.")
        return
    display = candidates.head(top_n).copy()
    display["expected_20d"] = display["expected_return"].map(lambda value: f"{value:.2%}")
    display["minimum_needed"] = display["entry_threshold"].map(lambda value: f"{value:.2%}")
    display["score"] = display["score"].map(lambda value: f"{value:.3f}")
    print(
        display[["ticker", "expected_20d", "minimum_needed", "score"]]
        .rename(
            columns={
                "ticker": "Ticker",
                "expected_20d": "Expected 20d",
                "minimum_needed": "Minimum needed",
                "score": "Score",
            }
        )
        .to_string(index=False)
    )


def _save_results(comparison: pd.DataFrame, predictions: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "backtest_summary.csv")
    predictions.to_csv(OUTPUT_DIR / "latest_predictions.csv", index=False)
    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
