from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .backtest import expected_return_hurdle
from .config import BacktestConfig, FeatureConfig, ModelConfig
from .data import drop_invalid_price_rows
from .io import load_price_directory, load_universe_membership
from .pipeline import latest_predictions, run_research


def main(argv: Sequence[str] | None = None) -> int:
    """Run a research backtest or generate a freshly refitted signal snapshot."""
    parser = _parser()
    args = parser.parse_args(argv)
    prices, dropped = drop_invalid_price_rows(load_price_directory(args.prices_dir))
    if not dropped.empty:
        ticker_count = dropped["ticker"].nunique()
        print(f"Ignored {len(dropped)} invalid historical price rows in {ticker_count} tickers.")
    membership = load_universe_membership(args.universe) if args.universe else None
    feature_config = FeatureConfig(min_median_dollar_volume=args.min_turnover)
    model_config = ModelConfig()

    if args.command == "latest":
        signal_date, predictions, diagnostics = latest_predictions(
            prices,
            feature_config,
            model_config,
            membership,
        )
        screen_config = _backtest_config(args)
        predictions["entry_threshold"] = predictions["horizon_sessions"].map(
            lambda horizon: expected_return_hurdle(int(horizon), screen_config)
        )
        predictions["trade_eligible"] = predictions["expected_return"] > predictions["entry_threshold"]
        candidates = predictions[predictions["trade_eligible"]]
        print(f"Latest after-close signal date: {signal_date.date()}")
        display_columns = [
            "ticker",
            "expected_return",
            "entry_threshold",
            "score",
            "vol_60d",
            "turnover_60d_median",
        ]
        if candidates.empty:
            print("No candidate clears the risk-free and trading-cost hurdle.")
        else:
            print(candidates[display_columns].head(args.top_n).to_string(index=False))
        if args.output_dir:
            output = _output_directory(args.output_dir)
            predictions.to_csv(output / "latest_predictions.csv", index=False)
            diagnostics.to_csv(output / "latest_model_diagnostics.csv", index=False)
        return 0

    backtest_config = _backtest_config(args)
    result = run_research(
        prices,
        args.start,
        feature_config,
        model_config,
        backtest_config,
        membership,
    )
    print(result.metric_comparison().to_string(float_format=lambda value: f"{value:.4f}"))
    if not result.survivorship_safe:
        print("WARNING: no point-in-time universe was supplied; survivorship safety is not established.")
    if args.output_dir:
        output = _output_directory(args.output_dir)
        result.predictions.to_csv(output / "walk_forward_predictions.csv", index=False)
        result.model_diagnostics.to_csv(output / "model_diagnostics.csv", index=False)
        result.strategy.daily_equity.to_csv(output / "strategy_daily_equity.csv", index=False)
        result.strategy.trades.to_csv(output / "strategy_trades.csv", index=False)
        result.strategy.holdings.to_csv(output / "strategy_holdings.csv", index=False)
        result.strategy.selections.to_csv(output / "strategy_selections.csv", index=False)
        result.momentum_baseline.daily_equity.to_csv(
            output / "momentum_daily_equity.csv",
            index=False,
        )
        result.momentum_baseline.trades.to_csv(output / "momentum_trades.csv", index=False)
        result.metric_comparison().to_csv(output / "metric_comparison.csv")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cost-aware walk-forward equity ML research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("research", "latest"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--prices-dir", required=True)
        subparser.add_argument("--universe", help="CSV with ticker, listed_from, and listed_to")
        subparser.add_argument("--output-dir")
        subparser.add_argument("--top-n", type=int, default=10)
        subparser.add_argument("--min-turnover", type=float, default=2_000_000.0)
        subparser.add_argument("--commission-bps", type=float, default=5.0)
        subparser.add_argument("--half-spread-bps", type=float, default=10.0)
        subparser.add_argument("--impact-bps", type=float, default=25.0)
        subparser.add_argument("--max-participation", type=float, default=0.01)
        subparser.add_argument("--risk-free-rate", type=float, default=0.0)

    research = subparsers.choices["research"]
    research.add_argument("--start", required=True, help="First locked walk-forward signal date")
    research.add_argument("--capital", type=float, default=1_000_000.0)
    research.add_argument("--buffer-n", type=int, default=15)
    research.add_argument("--max-weight", type=float)
    return parser


def _backtest_config(args: argparse.Namespace) -> BacktestConfig:
    max_weight_arg = getattr(args, "max_weight", None)
    max_weight = max_weight_arg if max_weight_arg is not None else max(0.15, 1 / args.top_n)
    buffer_n = max(getattr(args, "buffer_n", args.top_n), args.top_n)
    return BacktestConfig(
        initial_capital=getattr(args, "capital", 1_000_000.0),
        top_n=args.top_n,
        buffer_n=buffer_n,
        max_weight=max_weight,
        commission_bps=args.commission_bps,
        half_spread_bps=args.half_spread_bps,
        impact_bps=args.impact_bps,
        max_participation_rate=args.max_participation,
        annual_risk_free_rate=args.risk_free_rate,
    )


def _output_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
