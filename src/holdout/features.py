import numpy as np
import pandas as pd

BASE_FEATURE_COLS = ["ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d", "vol_20d", "vol_60d", "ma_ratio_50_200", "drawdown_252d", "log_dollar_volume_20d", "volume_ratio_20_60"]

RANK_FEATURE_COLS = [f"{col}_xrank" for col in BASE_FEATURE_COLS]
FEATURE_COLS = BASE_FEATURE_COLS + RANK_FEATURE_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML features and the prediction label to a long-format price frame.

    Features use only information known on the signal date.
    They cover momentum, trend, volatility, drawdown, and liquidity.
    Cross-sectional ranks help the model learn relative stock selection.

    The label is the next 20 trading-day return for the same ticker.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", group_keys=False)

    df["ret_1d"] = g["adj_close"].pct_change(1)
    df["ret_5d"] = g["adj_close"].pct_change(5)
    df["ret_20d"] = g["adj_close"].pct_change(20)
    df["ret_60d"] = g["adj_close"].pct_change(60)
    df["ret_120d"] = g["adj_close"].pct_change(120)
    df["ret_252d"] = g["adj_close"].pct_change(252)

    df["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20).std())
    df["vol_60d"] = g["ret_1d"].transform(lambda s: s.rolling(60).std())
    df["ma_ratio_50_200"] = g["adj_close"].transform(lambda s: s.rolling(50).mean()) / g["adj_close"].transform(lambda s: s.rolling(200).mean())
    df["drawdown_252d"] = (df["adj_close"] / g["adj_close"].transform(lambda s: s.rolling(252).max())) - 1
    dollar_volume = df["adj_close"] * df["volume"]
    df["log_dollar_volume_20d"] = np.log1p(dollar_volume.groupby(df["ticker"]).transform(lambda s: s.rolling(20).mean()))
    df["volume_ratio_20_60"] = g["volume"].transform(lambda s: s.rolling(20).mean()) / g["volume"].transform(lambda s: s.rolling(60).mean())

    for col in BASE_FEATURE_COLS:
        df[f"{col}_xrank"] = df.groupby("date")[col].rank(pct=True)

    df["label_date_20d"] = g["date"].shift(-20)
    df["fwd_ret_20d"] = g["adj_close"].transform(lambda s: s.shift(-20) / s - 1)
    return df
