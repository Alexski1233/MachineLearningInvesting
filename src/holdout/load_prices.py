from pathlib import Path
import pandas as pd

RAW_PRICES = Path(__file__).resolve().parents[2] / "data" / "raw_prices"


def load_all_prices() -> pd.DataFrame:
    """Read every CSV in data/raw_prices and return one tidy long-format frame.

    Each row is one (ticker, date) bar. 
    Rows are sorted by ticker then date so later groupby/rolling operations are stable.
    """
    frames = [pd.read_csv(csv, parse_dates=["date"], dtype={"ticker": str, "yahoo_symbol": str}) for csv in sorted(RAW_PRICES.glob("*.csv"))]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
