from collections import Counter
from pathlib import Path

import pandas as pd
import main as main_view
from load_prices import RAW_PRICES
from model import backtest_top_n, build_dataset, evaluate_predictions, holding_period_sweep, train_and_predict

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "universe_fondsfinans.csv"


def load_fondsfinans_universe() -> pd.DataFrame:
    return pd.read_csv(
        UNIVERSE,
        dtype={"ticker": str, "yahoo_symbol": str, "name": str, "market": str, "currency": str},
    )


def load_fondsfinans_prices(universe: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ticker in universe["ticker"]:
        path = RAW_PRICES / f"{ticker}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["date"], dtype={"ticker": str, "yahoo_symbol": str}))
    if not frames:
        message = f"No Fondsfinans price files found in {RAW_PRICES}. Run src/fetch_fondsfinans_prices.py first."
        raise FileNotFoundError(message)
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def print_fondsfinans_universe_summary(universe: pd.DataFrame, prices: pd.DataFrame, scored: pd.DataFrame) -> None:
    main_view.print_section("Fondsfinans Universe")
    scored_tickers = set(scored["ticker"].unique())
    price_tickers = set(prices["ticker"].unique())
    rows = [
        ("Holdings in list", f"{len(universe)}"),
        ("Price files available", f"{len(price_tickers)}"),
        ("Model-scoreable today", f"{len(scored_tickers)}"),
        ("Latest raw price bar", main_view.data_freshness_text(prices)),
    ]
    main_view.print_metric_table(rows)

    waiting_rows = []
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        if ticker in scored_tickers:
            continue
        if ticker in price_tickers:
            ticker_prices = prices[prices["ticker"] == ticker]
            price_range = f"{ticker_prices['date'].min().date()} -> {ticker_prices['date'].max().date()}"
            waiting_rows.append(
                (ticker, row["name"], f"{len(ticker_prices):,}", price_range, "needs more feature history")
            )
        else:
            waiting_rows.append((ticker, row["name"], "0", "", "missing price file"))

    if waiting_rows:
        main_view.print_section("Included But Not Scored Yet")
        align = {"Ticker": "l", "Name": "l", "Rows": "r", "Price range": "l", "Status": "l"}
        main_view.print_table(["Ticker", "Name", "Rows", "Price range", "Status"], waiting_rows, align=align)


def print_strategy_concentration(bt) -> None:
    main_view.print_section("Fondsfinans Pick Concentration")
    if bt.empty or "picks" not in bt:
        print("No backtest picks.")
        return
    counts = Counter()
    for picks in bt["picks"]:
        counts.update(str(picks).split(", "))
    rows = [
        (ticker, f"{count}", main_view.format_pct(count / len(bt), digits=1))
        for ticker, count in counts.most_common(15)
    ]
    align = {"Ticker": "l", "Selections": "r", "Periods": "r"}
    main_view.print_table(["Ticker", "Selections", "Periods"], rows, align=align)


def main() -> None:
    progress = main_view.ProgressPrinter()
    universe = load_fondsfinans_universe()
    prices = load_fondsfinans_prices(universe)

    progress.start("Building Fondsfinans features")
    df = build_dataset(prices, min_turnover_60d_median=None)
    progress.done("Building Fondsfinans features")
    print_fondsfinans_universe_summary(universe, prices, df)
    main_view.print_dataset_summary(df)

    progress.start("Training and selecting Fondsfinans model")
    _, test, info = train_and_predict(df, return_diagnostics=True, progress=progress)
    progress.done("Training and selecting Fondsfinans model")
    main_view.print_model_selection_summary(info)

    progress.start("Evaluating unseen Fondsfinans test data")
    metrics = evaluate_predictions(test, info["train_target_mean"])
    progress.done("Evaluating unseen Fondsfinans test data")
    main_view.print_unseen_accuracy_summary(metrics, info["split"])

    progress.start("Running Fondsfinans backtest")
    bt = backtest_top_n(test)
    progress.done("Running Fondsfinans backtest")
    main_view.print_backtest_summary(bt)
    print_strategy_concentration(bt)

    sweep = holding_period_sweep(test)
    print()
    main_view.print_holding_period_sweep(sweep)

    last_date, picks = main_view.latest_signals(test)
    main_view.print_latest_picks(last_date, picks)


if __name__ == "__main__":
    main()
