import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import FEATURE_COLS, add_features
from load_prices import load_all_prices

TRAIN_TEST_SPLIT = "2020-01-01"
REBALANCE_EVERY_DAYS = 20
TOP_N = 5


def build_dataset() -> pd.DataFrame:
    """Load raw prices, build features, drop rows where any feature is missing."""
    prices = load_all_prices()
    feats = add_features(prices)
    return feats.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def train_and_predict(df: pd.DataFrame, split: str = TRAIN_TEST_SPLIT):
    """Fit a gradient-boosted regressor on pre-`split` data, predict the rest."""
    train = df[(df["date"] < split) & df["fwd_ret_20d"].notna()]
    test = df[df["date"] >= split].copy()
    model = GradientBoostingRegressor(random_state=0)
    model.fit(train[FEATURE_COLS], train["fwd_ret_20d"])
    test["pred"] = model.predict(test[FEATURE_COLS])
    return model, test


def backtest_top_n(test: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Simulate a long-only top-N strategy with periodic rebalancing.

    Every `REBALANCE_EVERY_DAYS` trading days, buy the n tickers with the
    highest predicted forward return, equal-weighted. Holding-period return is
    the average of their realized fwd_ret_20d. Returns a frame with one row per
    rebalance plus a cumulative equity curve.
    """
    labeled = test[test["fwd_ret_20d"].notna()]
    rebalance_dates = (
        labeled["date"].drop_duplicates().sort_values().iloc[::REBALANCE_EVERY_DAYS]
    )
    rows = []
    for d in rebalance_dates:
        snap = labeled[labeled["date"] == d]
        if len(snap) < n:
            continue
        picks = snap.nlargest(n, "pred")
        rows.append(
            {
                "date": d,
                "ret": picks["fwd_ret_20d"].mean(),
                "picks": ", ".join(picks["ticker"].tolist()),
            }
        )
    bt = pd.DataFrame(rows)
    bt["equity"] = (1 + bt["ret"]).cumprod()
    return bt


if __name__ == "__main__":
    df = build_dataset()
    _, test = train_and_predict(df)
    bt = backtest_top_n(test)
    print(bt.tail().to_string(index=False))
    print()
    print(f"Periods:       {len(bt)}")
    print(f"Avg 20d ret:   {bt['ret'].mean():.2%}")
    print(f"Total return:  {bt['equity'].iloc[-1] - 1:.1%}")
