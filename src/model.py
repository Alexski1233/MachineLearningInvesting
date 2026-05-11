import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from features import FEATURE_COLS, add_features
from load_prices import load_all_prices

TRAIN_TEST_SPLIT = "2020-01-01"
REBALANCE_EVERY_DAYS = 20
HOLDING_PERIODS = [10, 20, 30, 60]
TRADING_DAYS_PER_YEAR = 252
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


def add_forward_return(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Add a per-ticker forward return column for a given holding period."""
    out = df.sort_values(["ticker", "date"]).copy()
    col = f"fwd_ret_{days}d"
    out[col] = out.groupby("ticker")["adj_close"].pct_change(days).shift(-days)
    return out


def backtest_top_n(
    test: pd.DataFrame,
    n: int = TOP_N,
    holding_days: int = REBALANCE_EVERY_DAYS,
) -> pd.DataFrame:
    """Simulate a long-only top-N strategy with periodic rebalancing.

    Every `holding_days` trading days, buy the n tickers with the
    highest predicted forward return, equal-weighted. Holding-period return is
    the average of their realized forward return. Returns a frame with one row per
    rebalance plus a cumulative equity curve.
    """
    ret_col = f"fwd_ret_{holding_days}d"
    if ret_col not in test.columns:
        test = add_forward_return(test, holding_days)

    labeled = test[test[ret_col].notna()]
    rebalance_dates = (
        labeled["date"].drop_duplicates().sort_values().iloc[::holding_days]
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
                "ret": picks[ret_col].mean(),
                "picks": ", ".join(picks["ticker"].tolist()),
            }
        )
    bt = pd.DataFrame(rows)
    bt["equity"] = (1 + bt["ret"]).cumprod()
    return bt


def annualized_return(bt: pd.DataFrame, holding_days: int) -> float:
    """Return the effective annual return for a backtest equity curve."""
    years = len(bt) * holding_days / TRADING_DAYS_PER_YEAR
    if bt.empty or years <= 0:
        return float("nan")
    return bt["equity"].iloc[-1] ** (1 / years) - 1


def holding_period_sweep(
    test: pd.DataFrame,
    periods: list[int] = HOLDING_PERIODS,
    n: int = TOP_N,
) -> pd.DataFrame:
    """Compare performance when holding the same ranked picks for N days."""
    rows = []
    for days in periods:
        bt = backtest_top_n(test, n=n, holding_days=days)
        if bt.empty:
            continue
        rows.append(
            {
                "holding_days": days,
                "periods": len(bt),
                "avg_ret": bt["ret"].mean(),
                "total_return": bt["equity"].iloc[-1] - 1,
                "ear": annualized_return(bt, days),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    _, test = train_and_predict(df)
    bt = backtest_top_n(test)
    print(bt.tail().to_string(index=False))
    print()
    print(f"Periods:       {len(bt)}")
    print(f"Avg 20d ret:   {bt['ret'].mean():.2%}")
    print(f"Total return:  {bt['equity'].iloc[-1] - 1:.1%}")
