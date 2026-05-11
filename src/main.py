import time

import pandas as pd
from prettytable import PrettyTable
from model import TOP_N, backtest_top_n, build_dataset, evaluate_predictions, summarize_backtest, train_and_predict


class ProgressPrinter:
    """Small terminal progress helper for the slow training path."""

    def __init__(self) -> None:
        self._starts = {}

    def start(self, label: str) -> None:
        self._starts[label] = time.perf_counter()
        print(f"> {label} ...", flush=True)

    def done(self, label: str) -> None:
        started = self._starts.pop(label, None)
        elapsed = time.perf_counter() - started if started else 0
        print(f"> Done in {elapsed:.1f}s", flush=True)

    def __call__(self, event: str, **data) -> None:
        if event == "split_ready":
            print(f"  split ready | train rows {data['train_rows']:,} | validation rows {data['validation_rows']:,}", flush=True)
        elif event == "candidate_start":
            print(f"  [{data['index']}/{data['total']}] training {data['model']} ...", flush=True)
        elif event == "candidate_done":
            print(f"      score {format_pct(data['score'])} | rank IC {format_float(data['rank_ic'])}", flush=True)
        elif event == "final_fit_start":
            print(f"  fitting final {data['model']} on {data['rows']:,} rows ...", flush=True)
        elif event == "final_fit_done":
            print(f"  predictions ready for {data['rows']:,} test rows", flush=True)


def latest_signals(test, n: int = TOP_N):
    """Return the most recent date and top-n picks by predicted return."""
    last_date = test["date"].max()
    snap = test[test["date"] == last_date]
    return last_date, snap.nlargest(n, "pred")[["ticker", "pred", "adj_close"]]


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def print_table(field_names, rows, align=None) -> None:
    table = PrettyTable()
    table.field_names = field_names
    table.add_rows(rows)
    for field, value in (align or {}).items():
        table.align[field] = value
    print(table)


def print_metric_table(rows) -> None:
    print_table(["Metric", "Value"], rows, align={"Metric": "l", "Value": "r"})


def data_freshness_text(df) -> str:
    latest = df["date"].max()
    days_old = (pd.Timestamp.today().normalize() - latest).days
    freshness = "today" if days_old == 0 else f"{days_old} day(s) ago"
    return f"{latest.date()} ({freshness})"


def print_dataset_summary(df) -> None:
    print_section("Data")
    rows = [
        ("Rows", f"{len(df):,}"),
        ("Tickers", f"{df['ticker'].nunique()}"),
        ("Date range", f"{df['date'].min().date()} to {df['date'].max().date()}"),
        ("Latest bar", data_freshness_text(df)),
    ]
    print_metric_table(rows)


def format_pct(value, digits: int = 2) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}%}"


def format_float(value, digits: int = 3) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def print_model_selection_summary(info) -> None:
    rows = []
    for _, row in info["validation_report"].iterrows():
        rows.append((
            "yes" if row["model"] == info["selected_model"] else "",
            row["model"],
            format_pct(row["avg_excess_20d_ret"]),
            format_pct(row["direction_accuracy"]),
            format_pct(row["top_n_hit_rate"]),
            format_float(row["rank_ic"]),
            format_pct(row["oos_r2"]),
        ))

    print_section("Model Selection")
    print_metric_table([("Validation window", f"{info['validation_start'].date()} to {info['split'].date()}"), ("Selected model", info["selected_model"])])
    fields = ["Selected", "Model", "Avg excess 20d", "Direction", f"Top {TOP_N} hit", "Rank IC", "OOS R2"]
    align = {"Selected": "c", "Model": "l", "Avg excess 20d": "r", "Direction": "r", f"Top {TOP_N} hit": "r", "Rank IC": "r", "OOS R2": "r"}
    print_table(fields, rows, align=align)


def print_unseen_accuracy_summary(metrics, split_date) -> None:
    print_section(f"Unseen Test Accuracy From {split_date.date()}")
    rows = [
        ("Labeled rows", f"{metrics['rows']:,}"),
        ("Direction accuracy", format_pct(metrics["direction_accuracy"])),
        (f"Top {TOP_N} hit rate", format_pct(metrics["top_n_hit_rate"])),
        (f"Top {TOP_N} period wins", format_pct(metrics["top_n_positive_period_rate"])),
        ("Rank IC", format_float(metrics["rank_ic"])),
        ("OOS R2 vs train mean", format_pct(metrics["oos_r2"])),
        ("RMSE 20d return", format_pct(metrics["rmse"])),
    ]
    print_metric_table(rows)


def print_backtest_summary(bt) -> None:
    summary = summarize_backtest(bt)
    print_section("Backtest")
    rows = [
        ("Periods", f"{summary['periods']}  20 trading-day rebalances"),
        ("Avg 20d ret", format_pct(summary["avg_20d_ret"])),
        ("Equal-weight avg 20d ret", format_pct(summary["avg_benchmark_20d_ret"])),
        ("Avg excess 20d ret", format_pct(summary["avg_excess_20d_ret"])),
        ("Total return", format_pct(summary["strategy_total_return"], digits=1)),
        ("Equal-weight total", format_pct(summary["benchmark_total_return"], digits=1)),
        ("Beat benchmark periods", format_pct(summary["beat_benchmark_rate"])),
        ("Max drawdown", format_pct(summary["max_drawdown"])),
        ("Approx annual Sharpe", format_float(summary["sharpe"], digits=2)),
    ]
    print_metric_table(rows)


def print_latest_picks(last_date, picks) -> None:
    rows = [(row["ticker"], format_pct(row["pred"]), f"{row['adj_close']:.2f}") for _, row in picks.iterrows()]
    print_section(f"Top {TOP_N} BUY Picks On {last_date.date()}")
    print_table(["Ticker", "Pred 20d", "Adj close"], rows, align={"Ticker": "l", "Pred 20d": "r", "Adj close": "r"})


def main() -> None:
    progress = ProgressPrinter()
    progress.start("Loading prices and building features")
    df = build_dataset()
    progress.done("Loading prices and building features")
    print_dataset_summary(df)

    progress.start("Training and selecting model")
    _, test, info = train_and_predict(df, return_diagnostics=True, progress=progress)
    progress.done("Training and selecting model")
    print_model_selection_summary(info)

    progress.start("Evaluating unseen test data")
    metrics = evaluate_predictions(test, info["train_target_mean"])
    progress.done("Evaluating unseen test data")
    print_unseen_accuracy_summary(metrics, info["split"])

    progress.start("Running backtest")
    bt = backtest_top_n(test)
    progress.done("Running backtest")
    print_backtest_summary(bt)

    last_date, picks = latest_signals(test)
    print_latest_picks(last_date, picks)


if __name__ == "__main__":
    main()
