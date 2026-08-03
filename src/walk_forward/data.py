import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = ("ticker", "date", "open", "close", "adj_close", "volume")


def drop_invalid_price_rows(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove unusable price rows."""
    missing = set(REQUIRED_PRICE_COLUMNS).difference(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")

    out = prices.copy()
    invalid = pd.Series(False, index=out.index)
    numeric_columns = ["open", "close", "adj_close", "volume"]
    numeric_columns.extend(column for column in ("high", "low") if column in out.columns)
    for column in numeric_columns:
        converted = pd.to_numeric(out[column], errors="coerce")
        invalid |= converted.isna() | ~np.isfinite(converted)
        out[column] = converted

    for column in ("open", "close", "adj_close", "high", "low"):
        if column in out.columns:
            invalid |= out[column] <= 0
    invalid |= out["volume"] < 0

    if "high" in out.columns:
        invalid |= out["high"] < out[["open", "close"]].max(axis=1)
    if "low" in out.columns:
        invalid |= out["low"] > out[["open", "close"]].min(axis=1)
    if {"high", "low"}.issubset(out.columns):
        invalid |= out["high"] < out["low"]

    dropped = out.loc[invalid].copy()
    cleaned = out.loc[~invalid].copy()
    if cleaned.empty:
        raise ValueError("All price rows are invalid.")
    return cleaned, dropped


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a long price table."""
    missing = set(REQUIRED_PRICE_COLUMNS).difference(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("Price data is empty.")

    out = prices.copy()
    out["date"] = _normalize_dates(out["date"], "date")
    out["ticker"] = out["ticker"].astype("string").str.strip()
    if out["ticker"].isna().any() or out["ticker"].eq("").any():
        raise ValueError("Ticker values cannot be missing or blank.")

    numeric_columns = ["open", "close", "adj_close", "volume"]
    numeric_columns.extend(column for column in ("high", "low") if column in out.columns)
    for column in numeric_columns:
        converted = pd.to_numeric(out[column], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted)
        if invalid.any():
            examples = out.loc[invalid, ["ticker", "date", column]].head(5)
            raise ValueError(f"Column {column!r} contains missing or non-finite values: {examples.to_dict(orient='records')}")
        out[column] = converted.astype(float)

    for column in ("open", "close", "adj_close", "high", "low"):
        if column not in out.columns:
            continue
        if (out[column] <= 0).any():
            raise ValueError(f"Column {column!r} must be strictly positive.")
    if (out["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    if "high" in out.columns:
        below_bar = out["high"] < out[["open", "close"]].max(axis=1)
        if below_bar.any():
            raise ValueError("high cannot be below open or close.")
    if "low" in out.columns:
        above_bar = out["low"] > out[["open", "close"]].min(axis=1)
        if above_bar.any():
            raise ValueError("low cannot be above open or close.")
    if {"high", "low"}.issubset(out.columns) and (out["high"] < out["low"]).any():
        raise ValueError("high cannot be below low.")

    duplicate_keys = out.duplicated(["ticker", "date"], keep=False)
    if duplicate_keys.any():
        examples = out.loc[duplicate_keys, ["ticker", "date"]].head(5)
        raise ValueError(f"Price data must contain one row per (ticker, date); duplicate examples: {examples.to_dict(orient='records')}")

    if "exchange" in out.columns:
        out["exchange"] = out["exchange"].astype("string").str.strip()
        if out["exchange"].isna().any() or out["exchange"].eq("").any():
            raise ValueError("exchange values cannot be missing or blank.")
        exchange_counts = out.groupby("ticker")["exchange"].nunique(dropna=False)
        if (exchange_counts > 1).any():
            raise ValueError("Each ticker must map to one exchange calendar.")

    if "in_universe" in out.columns:
        out["in_universe"] = _normalize_boolean(out["in_universe"], "in_universe")
    if "delisting_return" in out.columns:
        missing_delisting_return = out["delisting_return"].isna() | out["delisting_return"].astype("string").str.strip().eq("")
        delisting_return = pd.to_numeric(out["delisting_return"], errors="coerce")
        invalid = (~missing_delisting_return & delisting_return.isna()) | (
            delisting_return.notna() & (~np.isfinite(delisting_return) | (delisting_return < -1))
        )
        if invalid.any():
            raise ValueError("delisting_return must be finite and no smaller than -1.")
        out["delisting_return"] = delisting_return.astype(float)
        delisting_events = out[out["delisting_return"].notna()]
        if delisting_events.groupby("ticker").size().gt(1).any():
            raise ValueError("Each stable ticker can contain only one delisting event.")
        last_dates = out.groupby("ticker")["date"].transform("max")
        event_not_last = out["delisting_return"].notna() & out["date"].ne(last_dates)
        if event_not_last.any():
            raise ValueError("A delisting event must be the ticker's final price row.")

    adjustment_factor = out["adj_close"] / out["close"]
    out["adj_open"] = out["open"] * adjustment_factor
    out["raw_dollar_volume"] = out["close"] * out["volume"]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def attach_universe_membership(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Attach inclusive point-in-time listing intervals to price rows."""
    required = {"ticker", "listed_from", "listed_to"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Universe membership is missing columns: {sorted(missing)}")

    out = validate_prices(prices)
    intervals = membership.copy()
    intervals["ticker"] = intervals["ticker"].astype("string").str.strip()
    if intervals["ticker"].isna().any() or intervals["ticker"].eq("").any():
        raise ValueError("Universe membership tickers cannot be missing or blank.")
    intervals["listed_from"] = _normalize_dates(intervals["listed_from"], "listed_from")
    intervals["listed_to"] = _normalize_optional_dates(intervals["listed_to"], "listed_to")
    invalid_interval = intervals["listed_to"].notna() & (intervals["listed_to"] < intervals["listed_from"])
    if invalid_interval.any():
        raise ValueError("listed_to cannot be earlier than listed_from.")
    _reject_overlapping_intervals(intervals)

    active_count = pd.Series(0, index=out.index, dtype=int)
    for interval in intervals.itertuples(index=False):
        active = (out["ticker"] == interval.ticker) & (out["date"] >= interval.listed_from)
        if pd.notna(interval.listed_to):
            active &= out["date"] <= interval.listed_to
        active_count.loc[active] += 1
    if (active_count > 1).any():
        raise ValueError("A price row matches overlapping universe intervals.")
    out["in_universe"] = active_count.eq(1)
    return out


def _normalize_dates(values: pd.Series, name: str) -> pd.Series:
    original_missing = values.isna()
    normalized = values.map(_normalize_calendar_value)
    normalized = pd.to_datetime(normalized)
    invalid = normalized.isna() & ~original_missing
    if invalid.any() or normalized.isna().any():
        raise ValueError(f"Column {name!r} contains missing or invalid dates.")
    return normalized


def _normalize_optional_dates(values: pd.Series, name: str) -> pd.Series:
    missing = values.isna() | values.astype("string").str.strip().eq("")
    normalized = values.where(~missing).map(_normalize_calendar_value)
    normalized = pd.to_datetime(normalized)
    if (normalized.isna() & ~missing).any():
        raise ValueError(f"Column {name!r} contains invalid dates.")
    return normalized


def _normalize_calendar_value(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _normalize_boolean(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"Column {name!r} cannot contain missing values.")
        return values.astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = values.astype("string").str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        raise ValueError(f"Column {name!r} must contain boolean values.")
    return normalized.astype(bool)


def _reject_overlapping_intervals(intervals: pd.DataFrame) -> None:
    for ticker, group in intervals.sort_values(["ticker", "listed_from"]).groupby("ticker"):
        previous_end: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            if previous_end is not None and row.listed_from <= previous_end:
                raise ValueError(f"Universe intervals overlap for ticker {ticker!r}.")
            previous_end = row.listed_to if pd.notna(row.listed_to) else pd.Timestamp.max.normalize()

