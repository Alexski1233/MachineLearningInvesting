from model import TOP_N, backtest_top_n, build_dataset, train_and_predict


def latest_signals(test, n: int = TOP_N):
    """Return (most_recent_date, top-n picks by predicted forward return).

    Used to produce the live BUY list to paste into the paper-trading site.
    """
    last_date = test["date"].max()
    snap = test[test["date"] == last_date]
    picks = snap.nlargest(n, "pred")[["ticker", "pred", "adj_close"]]
    return last_date, picks


def print_dataset_summary(df) -> None:
    """Print row/ticker/date-range summary of the assembled dataset."""
    print(
        f"Dataset: {len(df):,} rows | {df['ticker'].nunique()} tickers | "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )


def print_backtest_summary(bt) -> None:
    """Print headline backtest stats: periods, avg per-period return, total."""
    print("--- Backtest (out-of-sample) ---")
    print(
        f"Periods:       {len(bt)}  (20 trading-day rebalances)\n"
        f"Avg 20d ret:   {bt['ret'].mean():.2%}\n"
        f"Total return:  {bt['equity'].iloc[-1] - 1:.1%}"
    )


def main() -> None:
    """End-to-end demo: load -> features -> train -> backtest -> today's BUYs."""
    df = build_dataset()
    print_dataset_summary(df)

    _, test = train_and_predict(df)

    bt = backtest_top_n(test)
    print()
    print_backtest_summary(bt)

    last_date, picks = latest_signals(test)
    print()
    print(f"--- Top {TOP_N} BUY picks on {last_date.date()} ---")
    print(picks.to_string(index=False))


if __name__ == "__main__":
    main()
