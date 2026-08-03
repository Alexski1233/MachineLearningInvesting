import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import validate_prices

LABEL_DATE_COLUMN = "label_date"
RAW_RETURN_COLUMN = "forward_return"
TARGET_COLUMN = "target_forward_return"

STOCK_FEATURE_COLUMNS = (
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_252d",
    "momentum_12_1",
    "vol_20d",
    "vol_60d",
    "downside_vol_60d",
    "ma_ratio_50_200",
    "drawdown_252d",
    "log_dollar_volume_20d",
    "dollar_volume_ratio_20_60",
    "amihud_20d",
    "beta_252d",
    "market_relative_momentum_120d",
)
MARKET_FEATURE_COLUMNS = (
    "market_ret_60d",
    "market_ret_252d",
    "market_ma_ratio_200",
)
RANK_FEATURE_COLUMNS = tuple(f"{column}_xrank" for column in STOCK_FEATURE_COLUMNS)
FEATURE_COLUMNS = STOCK_FEATURE_COLUMNS + RANK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS


def build_feature_panel(prices: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Build backward looking features and next open labels."""
    resolved = config or FeatureConfig()
    out = validate_prices(prices)
    out["horizon_sessions"] = resolved.holding_days
    out = _add_forward_labels(out, resolved.holding_days)
    out = _add_features(out)
    out = _add_eligibility(out, resolved)
    out = _add_cross_sectional_ranks(out)
    out = _add_training_target(out, resolved.target_winsor_quantile)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def _add_forward_labels(prices: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    out = prices.copy()
    calendar_column = "exchange" if "exchange" in out.columns else "_calendar"
    if calendar_column == "_calendar":
        out[calendar_column] = "all"

    calendar_rows: list[pd.DataFrame] = []
    for calendar, group in out.groupby(calendar_column, sort=False):
        dates = pd.Series(group["date"].drop_duplicates().sort_values().to_numpy())
        calendar_rows.append(pd.DataFrame({
            calendar_column: calendar,
            "date": dates,
            "entry_date": dates.shift(-1),
            "planned_exit_date": dates.shift(-(holding_days + 1)),
        }))
    calendar_map = pd.concat(calendar_rows, ignore_index=True)
    out = out.merge(calendar_map, on=[calendar_column, "date"], how="left", validate="many_to_one")

    price_lookup = out[["ticker", "date", "adj_open"]]
    entry_lookup = price_lookup.rename(columns={"date": "entry_date", "adj_open": "entry_adj_open"})
    exit_lookup = price_lookup.rename(columns={"date": "planned_exit_date", "adj_open": "planned_exit_adj_open"})
    out = out.merge(entry_lookup, on=["ticker", "entry_date"], how="left", validate="many_to_one")
    out = out.merge(exit_lookup, on=["ticker", "planned_exit_date"], how="left", validate="many_to_one")
    out[LABEL_DATE_COLUMN] = out["planned_exit_date"]
    out["realized_exit_value"] = out["planned_exit_adj_open"]
    delisting_events = (
        out.loc[out["delisting_return"].notna(), ["ticker", "date", "adj_close", "delisting_return"]].copy()
        if "delisting_return" in out.columns
        else pd.DataFrame()
    )
    if not delisting_events.empty:
        delisting_events["delisting_payoff"] = delisting_events["adj_close"] * (
            1 + delisting_events["delisting_return"]
        )
        delisting_events = delisting_events.rename(columns={"date": "delisting_date"})
        delisting_events = delisting_events[["ticker", "delisting_date", "delisting_payoff"]]
        out = out.merge(delisting_events, on="ticker", how="left", validate="many_to_one")
        use_delisting = (
            out["entry_date"].notna()
            & (out["delisting_date"] >= out["entry_date"])
            & (out["planned_exit_date"].isna() | (out["delisting_date"] < out["planned_exit_date"]))
        )
        out.loc[use_delisting, LABEL_DATE_COLUMN] = out.loc[
            use_delisting,
            "delisting_date",
        ]
        out.loc[use_delisting, "realized_exit_value"] = out.loc[
            use_delisting,
            "delisting_payoff",
        ]
    out[RAW_RETURN_COLUMN] = out["realized_exit_value"] / out["entry_adj_open"] - 1
    if calendar_column == "_calendar":
        out = out.drop(columns=[calendar_column])
    return out


def _add_features(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", group_keys=False)
    out["ret_1d"] = grouped["adj_close"].pct_change(fill_method=None)
    for days in (5, 20, 60, 120, 252):
        out[f"ret_{days}d"] = grouped["adj_close"].pct_change(days, fill_method=None)

    out["momentum_12_1"] = grouped["adj_close"].shift(21) / grouped["adj_close"].shift(252) - 1
    out["vol_20d"] = grouped["ret_1d"].transform(lambda values: values.rolling(20).std())
    out["vol_60d"] = grouped["ret_1d"].transform(lambda values: values.rolling(60).std())
    out["downside_vol_60d"] = grouped["ret_1d"].transform(lambda values: np.sqrt(values.clip(upper=0).pow(2).rolling(60).mean()))
    moving_average_50 = grouped["adj_close"].transform(lambda values: values.rolling(50).mean())
    moving_average_200 = grouped["adj_close"].transform(lambda values: values.rolling(200).mean())
    out["ma_ratio_50_200"] = moving_average_50 / moving_average_200
    rolling_high = grouped["adj_close"].transform(lambda values: values.rolling(252).max())
    out["drawdown_252d"] = out["adj_close"] / rolling_high - 1
    out["log_dollar_volume_20d"] = np.log1p(grouped["raw_dollar_volume"].transform(lambda values: values.rolling(20).mean()))
    dollar_volume_20 = grouped["raw_dollar_volume"].transform(lambda values: values.rolling(20).mean())
    dollar_volume_60 = grouped["raw_dollar_volume"].transform(lambda values: values.rolling(60).mean())
    out["dollar_volume_ratio_20_60"] = dollar_volume_20 / dollar_volume_60
    out["_amihud_daily"] = out["ret_1d"].abs() / out["raw_dollar_volume"].replace(0, np.nan)
    out["amihud_20d"] = grouped["_amihud_daily"].transform(lambda values: values.rolling(20).mean()) * 1_000_000

    market_members = out
    if "in_universe" in out.columns:
        market_members = out[out["in_universe"]]
    market_daily = market_members.groupby("date")["ret_1d"].mean().sort_index()
    market_index = (1 + market_daily.fillna(0)).cumprod()
    market_features = pd.DataFrame(
        {
            "market_ret_1d": market_daily,
            "market_ret_60d": market_index.pct_change(60),
            "market_ret_120d": market_index.pct_change(120),
            "market_ret_252d": market_index.pct_change(252),
            "market_ma_ratio_200": market_index / market_index.rolling(200).mean() - 1,
        }
    ).reset_index()
    out = out.merge(market_features, on="date", how="left", validate="many_to_one")
    out["market_relative_momentum_120d"] = out["ret_120d"] - out["market_ret_120d"]

    out["_ret_market"] = out["ret_1d"] * out["market_ret_1d"]
    out["_market_sq"] = out["market_ret_1d"].pow(2)
    grouped = out.groupby("ticker", group_keys=False)
    mean_product = grouped["_ret_market"].transform(lambda values: values.rolling(252).mean())
    mean_return = grouped["ret_1d"].transform(lambda values: values.rolling(252).mean())
    mean_market = grouped["market_ret_1d"].transform(lambda values: values.rolling(252).mean())
    mean_market_sq = grouped["_market_sq"].transform(lambda values: values.rolling(252).mean())
    covariance = mean_product - mean_return * mean_market
    market_variance = mean_market_sq - mean_market.pow(2)
    out["beta_252d"] = covariance / market_variance.replace(0, np.nan)

    helper_columns = ["_amihud_daily", "_ret_market", "_market_sq", "market_ret_120d"]
    return out.drop(columns=helper_columns)


def _add_eligibility(prices: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    out = prices.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", group_keys=False)
    out["history_days"] = grouped.cumcount() + 1
    out["turnover_60d_median"] = grouped["raw_dollar_volume"].transform(lambda values: values.rolling(config.liquidity_lookback_days).median())
    in_universe = out.get("in_universe", pd.Series(True, index=out.index)).astype(bool)
    feature_complete = out[list(STOCK_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS)].notna().all(axis=1)
    out["eligible"] = (
        in_universe
        & (out["history_days"] >= config.min_history_days)
        & (out["close"] >= config.min_price)
        & (out["turnover_60d_median"] >= config.min_median_dollar_volume)
        & feature_complete
    )
    return out


def _add_cross_sectional_ranks(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    for column in STOCK_FEATURE_COLUMNS:
        rank_column = f"{column}_xrank"
        out[rank_column] = np.nan
        eligible = out["eligible"] & out[column].notna()
        out.loc[eligible, rank_column] = out.loc[eligible].groupby("date")[column].rank(method="average", pct=True)
    return out


def _add_training_target(prices: pd.DataFrame, quantile: float) -> pd.DataFrame:
    out = prices.copy()
    usable = out["eligible"] & out[RAW_RETURN_COLUMN].notna()
    winsorized = pd.Series(np.nan, index=out.index, dtype=float)

    def clip_cross_section(values: pd.Series) -> pd.Series:
        if quantile == 0 or len(values) < 3:
            return values
        return values.clip(values.quantile(quantile), values.quantile(1 - quantile))

    winsorized.loc[usable] = out.loc[usable].groupby(["date", LABEL_DATE_COLUMN], dropna=False)[RAW_RETURN_COLUMN].transform(clip_cross_section)
    out[TARGET_COLUMN] = winsorized
    return out
