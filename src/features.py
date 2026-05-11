import pandas as pd
from load_prices import load_all_prices

FEATURE_COLS = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "vol_20d",
    "ma_ratio_50_200",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML features and the prediction label to a long-format price frame.

    Features (all per-ticker):
      * ret_{5,20,60,120}d  -- trailing returns over multiple windows (momentum)
      * vol_20d             -- rolling 20-day std of daily returns (risk proxy)
      * ma_ratio_50_200     -- 50-day MA / 200-day MA (trend indicator)

    Label:
      * fwd_ret_20d -- forward 20 trading-day return; what the model predicts.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = g["adj_close"].pct_change(1)
    df["ret_5d"] = g["adj_close"].pct_change(5)
    df["ret_20d"] = g["adj_close"].pct_change(20)
    df["ret_60d"] = g["adj_close"].pct_change(60)
    df["ret_120d"] = g["adj_close"].pct_change(120)

    df["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20).std())
    df["ma_ratio_50_200"] = (
        g["adj_close"].transform(lambda s: s.rolling(50).mean())
        / g["adj_close"].transform(lambda s: s.rolling(200).mean())
    )

    df["fwd_ret_20d"] = g["adj_close"].pct_change(20).shift(-20)
    return df


if __name__ == "__main__":
    prices = load_all_prices()
    feats = add_features(prices)
    cols = ["ticker", "date", "adj_close"] + FEATURE_COLS + ["fwd_ret_20d"]
    print(feats[cols].tail(10).to_string(index=False))
