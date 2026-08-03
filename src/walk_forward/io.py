from pathlib import Path

import pandas as pd


def load_price_directory(price_dir: str | Path) -> pd.DataFrame:
    """Load all price CSV files into one frame."""
    directory = Path(price_dir)
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No price CSV files found in {directory.resolve()}.")

    frames = [pd.read_csv(path, parse_dates=["date"], dtype={"ticker": str}) for path in files]
    return pd.concat(frames, ignore_index=True)


def load_universe_membership(path: str | Path) -> pd.DataFrame:
    """Load point-in-time listing intervals used to prevent survivorship bias."""
    membership = pd.read_csv(path, dtype={"ticker": str})
    required = {"ticker", "listed_from", "listed_to"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Universe membership is missing columns: {sorted(missing)}")

    membership["listed_from"] = pd.to_datetime(membership["listed_from"], errors="raise")
    missing_end = membership["listed_to"].isna() | membership["listed_to"].astype("string").str.strip().eq("")
    parsed_end = pd.to_datetime(membership["listed_to"].where(~missing_end), errors="coerce")
    if (parsed_end.isna() & ~missing_end).any():
        raise ValueError("Universe membership contains invalid listed_to dates.")
    membership["listed_to"] = parsed_end
    return membership
