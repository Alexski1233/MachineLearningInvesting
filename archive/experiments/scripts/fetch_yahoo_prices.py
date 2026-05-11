from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
PRICE_DIR = ROOT / "data" / "yahoo_prices"
OUTPUT_DIR = ROOT / "output"

YAHOO_SYMBOLS = {
    "EQNR": "EQNR.OL",
    "DNB": "DNB.OL",
    "NHY": "NHY.OL",
    "TEL": "TEL.OL",
    "YAR": "YAR.OL",
    "ORK": "ORK.OL",
    "AKRBP": "AKRBP.OL",
    "MOWI": "MOWI.OL",
    "KOG": "KOG.OL",
    "GJF": "GJF.OL",
    "STB": "STB.OL",
    "SALM": "SALM.OL",
    "TOM": "TOM.OL",
    "SUBC": "SUBC.OL",
    "NOD": "NOD.OL",
    "FRO": "FRO.OL",
    "ELK": "ELK.OL",
    "BRG": "BRG.OL",
    "PROT": "PROT.OL",
    "VEI": "VEI.OL",
    "VAR": "VAR.OL",
    "MPCC": "MPCC.OL",
    "AFG": "AFG.OL",
    "ATEA": "ATEA.OL",
    "VEND": "VEND.OL",
}


def unix_date(year: int, month: int, day: int) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp())


def fetch_symbol(session: requests.Session, ticker: str, yahoo_symbol: str, start_year: int = 2000) -> pd.DataFrame:
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
    frame = frame.dropna(subset=["date", "close", "adj_close"]).sort_values("date")
    return frame


def fetch_all(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    frames = []
    failures = []
    for ticker, yahoo_symbol in YAHOO_SYMBOLS.items():
        cache_file = PRICE_DIR / f"{ticker}.csv"
        try:
            if cache_file.exists() and not refresh:
                frame = pd.read_csv(cache_file, parse_dates=["date"])
            else:
                frame = fetch_symbol(session, ticker, yahoo_symbol)
                frame.to_csv(cache_file, index=False)
                time.sleep(0.15)
            frames.append(frame)
        except Exception as exc:
            failures.append({"ticker": ticker, "yahoo_symbol": yahoo_symbol, "error": str(exc)})

    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "yahoo_price_failures.csv", index=False)
    else:
        failure_file = OUTPUT_DIR / "yahoo_price_failures.csv"
        if failure_file.exists():
            failure_file.unlink()

    prices = pd.concat(frames, ignore_index=True)
    adjusted_close = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = prices.pivot(index="date", columns="ticker", values="volume").sort_index()

    prices.to_csv(OUTPUT_DIR / "yahoo_prices_long.csv", index=False)
    adjusted_close.to_csv(OUTPUT_DIR / "adjusted_close_yahoo.csv")
    close.to_csv(OUTPUT_DIR / "close_yahoo.csv")
    volume.to_csv(OUTPUT_DIR / "volume_yahoo.csv")
    return prices, adjusted_close


def main() -> None:
    prices, adjusted_close = fetch_all(refresh=True)
    coverage = (
        prices.groupby("ticker")
        .agg(first_date=("date", "min"), last_date=("date", "max"), rows=("date", "size"))
        .reset_index()
        .sort_values("ticker")
    )
    coverage.to_csv(OUTPUT_DIR / "yahoo_price_coverage.csv", index=False)
    print(coverage.to_string(index=False))
    print(f"\nSkrev {OUTPUT_DIR / 'adjusted_close_yahoo.csv'}")


if __name__ == "__main__":
    main()
