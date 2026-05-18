import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_PRICES = ROOT / "data" / "raw_prices"
UNIVERSE = ROOT / "data" / "universe_oslo.csv"


def unix_date(year: int, month: int, day: int) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp())


def fetch_symbol(session: requests.Session, ticker: str, yahoo_symbol: str, start_year: int = 2000) -> pd.DataFrame:
    """Download one Yahoo Finance daily price history in the project CSV format."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    params = {
        "period1": unix_date(start_year, 1, 1),
        "period2": int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)).timestamp()),
        "interval": "1d",
        "events": "history|div|split",
        "includeAdjustedClose": "true",
    }
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(error)

    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"No Yahoo result for {yahoo_symbol}")

    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps:
        raise RuntimeError(f"No timestamps for {yahoo_symbol}")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adjclose if adjclose is not None else quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    frame.insert(0, "ticker", ticker)
    frame.insert(1, "yahoo_symbol", yahoo_symbol)
    return frame.dropna(subset=["date", "close", "adj_close"]).sort_values("date")


def fetch_all(universe_file: Path = UNIVERSE, output_dir: Path = RAW_PRICES, refresh: bool = True) -> pd.DataFrame:
    """Download all symbols from the universe file into data/raw_prices."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
            rows.append({"ticker": ticker, "yahoo_symbol": yahoo_symbol, "rows": len(frame), "first_date": frame["date"].min().date(), "last_date": frame["date"].max().date()})
            print(f"{ticker:10s} {len(frame):5d} rows  {frame['date'].min().date()} -> {frame['date'].max().date()}", flush=True)
        except Exception as exc:
            failures.append({"ticker": ticker, "yahoo_symbol": yahoo_symbol, "error": str(exc)})
            print(f"{ticker:10s} failed  {exc}", flush=True)

    coverage = pd.DataFrame(rows).sort_values("ticker")
    coverage.to_csv(ROOT / "data" / "processed" / "price_coverage.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(ROOT / "data" / "processed" / "price_failures.csv", index=False)
    else:
        failure_file = ROOT / "data" / "processed" / "price_failures.csv"
        if failure_file.exists():
            failure_file.unlink()
    return coverage


def main() -> None:
    coverage = fetch_all()
    print()
    print(f"Downloaded {len(coverage)} symbols into {RAW_PRICES}")
    print(f"Coverage written to {ROOT / 'data' / 'processed' / 'price_coverage.csv'}")


if __name__ == "__main__":
    main()
