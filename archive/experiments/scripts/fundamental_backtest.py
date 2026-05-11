from __future__ import annotations

from pathlib import Path

import pandas as pd

from nordic_backtest import monthly_rebalance_dates, save_plot, summarize
from parse_fundamentals import build_quarterly_factors, latest_ranking, parse_all

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"


def load_close_prices() -> pd.DataFrame:
    yahoo_path = OUTPUT_DIR / "adjusted_close_yahoo.csv"
    fallback_path = OUTPUT_DIR / "close_prices.csv"
    path = yahoo_path if yahoo_path.exists() else fallback_path
    close = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    close = close.sort_index().ffill(limit=5)
    return close.dropna(how="all")


def build_fundamental_scores() -> pd.DataFrame:
    selected, _ = parse_all()
    factors = build_quarterly_factors(selected)
    ranked_rows = []
    for period_end_date, frame in factors.groupby("period_end_date"):
        ranked = latest_ranking(frame.copy())
        ranked_rows.append(ranked)
    scores = pd.concat(ranked_rows, ignore_index=True)
    return scores[["ticker", "period_end_date", "fundamental_score", "quality_score", "growth_score"]]


def score_asof(scores: pd.DataFrame, asof_date: pd.Timestamp) -> pd.Series:
    usable = scores[scores["period_end_date"] <= asof_date]
    if usable.empty:
        return pd.Series(dtype="float64")
    latest = usable.sort_values("period_end_date").groupby("ticker", as_index=False).tail(1)
    return latest.set_index("ticker")["fundamental_score"].dropna().sort_values(ascending=False)


def run_backtest(close: pd.DataFrame, scores: pd.DataFrame, top_n: int = 5, lag_days: int = 90, cost_bps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_returns = close.pct_change(fill_method=None)
    returns = raw_returns.fillna(0.0)
    rebalance_dates = monthly_rebalance_dates(close.index)
    cost_rate = cost_bps / 10_000.0
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    costs = pd.Series(0.0, index=close.index)
    current_weights = pd.Series(0.0, index=close.columns)
    ranking_rows = []

    for date in rebalance_dates:
        asof_date = date - pd.Timedelta(days=lag_days)
        available_scores = score_asof(scores, asof_date)
        tradable = close.loc[date].dropna().index
        available_scores = available_scores[available_scores.index.isin(tradable)]
        selected = available_scores.head(top_n).index

        new_weights = pd.Series(0.0, index=close.columns)
        if len(selected):
            new_weights.loc[selected] = 1.0 / len(selected)
        turnover = float((new_weights - current_weights).abs().sum())
        costs.loc[date] = turnover * cost_rate
        weights.loc[date] = new_weights
        current_weights = new_weights

        for symbol, score in available_scores.items():
            ranking_rows.append({"rebalance_date": date, "asof_date": asof_date, "symbol": symbol, "score": score, "selected": symbol in set(selected)})

    weights = weights.ffill().fillna(0.0)
    strategy_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1) - costs
    benchmark_returns = raw_returns.mean(axis=1, skipna=True).fillna(0.0)
    curve = pd.DataFrame(
        {
            "strategy": (1 + strategy_returns).cumprod(),
            "equal_weight_benchmark": (1 + benchmark_returns).cumprod(),
            "strategy_daily_return": strategy_returns,
            "benchmark_daily_return": benchmark_returns,
        }
    )
    return curve, pd.DataFrame(ranking_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    close = load_close_prices()
    scores = build_fundamental_scores()
    curve, ranking = run_backtest(close, scores, top_n=5, lag_days=90, cost_bps=10.0)
    summary = summarize(curve)

    scores.to_csv(OUTPUT_DIR / "fundamental_scores_history.csv", index=False)
    curve.to_csv(OUTPUT_DIR / "fundamental_equity_curve.csv")
    summary.to_csv(OUTPUT_DIR / "fundamental_backtest_summary.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "fundamental_backtest_ranking.csv", index=False)
    save_plot(curve, OUTPUT_DIR / "fundamental_equity_curve.png", title="Fundamental quality strategy vs benchmarks")

    print(summary.to_string(index=False))
    latest = ranking[ranking["rebalance_date"] == ranking["rebalance_date"].max()]
    print("\nSiste valgte aksjer:")
    print(latest[latest["selected"]][["rebalance_date", "symbol", "score"]].to_string(index=False))
    print("\nSkrev output/fundamental_backtest_summary.csv og output/fundamental_equity_curve.png")


if __name__ == "__main__":
    main()
