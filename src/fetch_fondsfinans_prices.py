import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from fetch_prices import ROOT, RAW_PRICES, fetch_symbol

UNIVERSE = ROOT / "data" / "universe_fondsfinans.csv"
COVERAGE_FILE = ROOT / "data" / "processed" / "fondsfinans_price_coverage.csv"
FAILURE_FILE = ROOT / "data" / "processed" / "fondsfinans_price_failures.csv"


def fetch_fondsfinans_prices(
    universe_file: Path = UNIVERSE,
    output_dir: Path = RAW_PRICES,
    refresh: bool = True,
) -> pd.DataFrame:
    """Download/update price files for the Fondsfinans Utbytte holdings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(universe_file)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    rows = []
    failures = []
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        yahoo_symbol = row["yahoo_symbol"]
        out_file = output_dir / f"{ticker}.csv"
        try:
            if out_file.exists() and not refresh:
                frame = pd.read_csv(out_file, parse_dates=["date"])
            else:
                frame = fetch_symbol(session, ticker, yahoo_symbol)
                frame.to_csv(out_file, index=False)
                time.sleep(0.15)
            rows.append(
                {
                    "ticker": ticker,
                    "yahoo_symbol": yahoo_symbol,
                    "rows": len(frame),
                    "first_date": frame["date"].min().date(),
                    "last_date": frame["date"].max().date(),
                }
            )
            print(
                f"{ticker:10s} {len(frame):5d} rows  {frame['date'].min().date()} -> {frame['date'].max().date()}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "yahoo_symbol": yahoo_symbol, "error": str(exc)})
            print(f"{ticker:10s} failed  {exc}", flush=True)

    coverage = pd.DataFrame(rows).sort_values("ticker")
    coverage.to_csv(COVERAGE_FILE, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_FILE, index=False)
    elif FAILURE_FILE.exists():
        FAILURE_FILE.unlink()
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch prices for Fondsfinans Utbytte holdings.")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only download tickers that do not already have a raw price file.",
    )
    args = parser.parse_args()
    coverage = fetch_fondsfinans_prices(refresh=not args.missing_only)
    print()
    print(f"Updated {len(coverage)} Fondsfinans symbols into {RAW_PRICES}")
    print(f"Coverage written to {COVERAGE_FILE}")


if __name__ == "__main__":
    main()
