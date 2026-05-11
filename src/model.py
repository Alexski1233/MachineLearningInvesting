import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLS, add_features
from load_prices import load_all_prices

TRAIN_TEST_SPLIT = "2020-01-01"
VALIDATION_YEARS = 3
REBALANCE_EVERY_DAYS = 20
HOLDING_PERIODS = [10, 20, 30, 60]
TRADING_DAYS_PER_YEAR = 252
TOP_N = 5
PERIODS_PER_YEAR = 252 / REBALANCE_EVERY_DAYS
TARGET_COL = "fwd_ret_20d"
LABEL_DATE_COL = "label_date_20d"


def build_dataset() -> pd.DataFrame:
    """Load raw prices, build features, drop rows where any feature is missing."""
    prices = load_all_prices()
    feats = add_features(prices)
    return feats.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def model_candidates(random_state: int = 0) -> dict:
    """Return models that are defensible for a small return-prediction panel."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=50, max_features=0.8, random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=50, max_features=0.8, random_state=random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(learning_rate=0.03, max_iter=300, max_leaf_nodes=15, l2_regularization=0.01, early_stopping=False, random_state=random_state),
        "gbrt_huber": GradientBoostingRegressor(loss="huber", n_estimators=300, learning_rate=0.03, max_depth=3, min_samples_leaf=50, subsample=0.8, random_state=random_state),
    }


def train_validation_split(df: pd.DataFrame, split: str = TRAIN_TEST_SPLIT, validation_years: int = VALIDATION_YEARS) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split pre-test data into train and validation while preserving time order."""
    split_ts = pd.Timestamp(split)
    labeled = labeled_before(df, split_ts)
    dates = labeled["date"].drop_duplicates().sort_values()
    if len(dates) < 2:
        raise ValueError("Need at least two pre-test dates with labels.")

    validation_start = split_ts - pd.DateOffset(years=validation_years)
    train = labeled[labeled[LABEL_DATE_COL] < validation_start]
    validation = labeled[labeled["date"] >= validation_start]

    if train.empty or validation.empty:
        fallback_idx = max(1, int(len(dates) * 0.8))
        fallback_idx = min(fallback_idx, len(dates) - 1)
        validation_start = dates.iloc[fallback_idx]
        train = labeled[labeled[LABEL_DATE_COL] < validation_start]
        validation = labeled[labeled["date"] >= validation_start]

    if train.empty or validation.empty:
        raise ValueError("Could not build a non-empty train and validation split.")

    return train, validation, pd.Timestamp(validation_start)


def labeled_before(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Return rows whose full forward-return label is known before a cutoff."""
    cutoff_ts = pd.Timestamp(cutoff)
    return df[df[TARGET_COL].notna() & df[LABEL_DATE_COL].notna() & (df[LABEL_DATE_COL] < cutoff_ts)].copy()


def fit_model(estimator, train: pd.DataFrame):
    """Fit a fresh estimator clone on the supplied training frame."""
    model = clone(estimator)
    model.fit(train[FEATURE_COLS], train[TARGET_COL])
    return model


def select_model(train: pd.DataFrame, validation: pd.DataFrame, n: int = TOP_N, progress=None) -> tuple[str, pd.DataFrame]:
    """Choose the model with the best validation top-N excess return."""
    rows = []
    baseline_return = train[TARGET_COL].mean()
    candidates = model_candidates()

    for index, (name, estimator) in enumerate(candidates.items(), start=1):
        if progress:
            progress("candidate_start", model=name, index=index, total=len(candidates))
        model = fit_model(estimator, train)
        validation_pred = validation.copy()
        validation_pred["pred"] = model.predict(validation_pred[FEATURE_COLS])
        accuracy = evaluate_predictions(validation_pred, baseline_return, n=n)
        backtest = summarize_backtest(backtest_top_n(validation_pred, n=n))
        validation_score = backtest["avg_excess_20d_ret"]
        if pd.isna(validation_score):
            validation_score = accuracy["rank_ic"]
        if pd.isna(validation_score):
            validation_score = -np.inf

        rows.append(
            {
                "model": name,
                "validation_score": validation_score,
                "direction_accuracy": accuracy["direction_accuracy"],
                "top_n_hit_rate": accuracy["top_n_hit_rate"],
                "rank_ic": accuracy["rank_ic"],
                "oos_r2": accuracy["oos_r2"],
                "rmse": accuracy["rmse"],
                "avg_20d_ret": backtest["avg_20d_ret"],
                "avg_excess_20d_ret": backtest["avg_excess_20d_ret"],
                "beat_benchmark_rate": backtest["beat_benchmark_rate"],
            }
        )
        if progress:
            progress("candidate_done", model=name, score=validation_score, rank_ic=accuracy["rank_ic"])

    report = pd.DataFrame(rows).sort_values(["validation_score", "rank_ic"], ascending=False)
    return str(report.iloc[0]["model"]), report.reset_index(drop=True)


def train_and_predict(df: pd.DataFrame, split: str = TRAIN_TEST_SPLIT, return_diagnostics: bool = False, progress=None):
    """Tune on pre-test validation data, then predict the unseen test period."""
    split_ts = pd.Timestamp(split)
    train, validation, validation_start = train_validation_split(df, split)
    if progress:
        progress("split_ready", train_rows=len(train), validation_rows=len(validation), validation_start=validation_start, split=split_ts)
    selected_name, validation_report = select_model(train, validation, progress=progress)

    fit_frame = labeled_before(df, split_ts)
    test = df[df["date"] >= split_ts].copy()
    if progress:
        progress("final_fit_start", model=selected_name, rows=len(fit_frame))
    model = fit_model(model_candidates()[selected_name], fit_frame)
    test["pred"] = model.predict(test[FEATURE_COLS])
    if progress:
        progress("final_fit_done", model=selected_name, rows=len(test))

    diagnostics = {
        "selected_model": selected_name,
        "validation_report": validation_report,
        "split": split_ts,
        "validation_start": validation_start,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "fit_rows": len(fit_frame),
        "train_target_mean": fit_frame[TARGET_COL].mean(),
    }

    if return_diagnostics:
        return model, test, diagnostics
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
    the average realized forward return. The benchmark is the equal-weighted
    return of every ticker available on that rebalance date.
    """
    ret_col = f"fwd_ret_{holding_days}d"
    if ret_col not in test.columns:
        test = add_forward_return(test, holding_days)

    labeled = test[test[ret_col].notna()]
    rebalance_dates = labeled["date"].drop_duplicates().sort_values().iloc[::holding_days]
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
                "benchmark_ret": snap[ret_col].mean(),
                "excess_ret": picks[ret_col].mean() - snap[ret_col].mean(),
                "hit_rate": (picks[ret_col] > 0).mean(),
                "picks": ", ".join(picks["ticker"].tolist()),
            }
        )
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt
    bt["equity"] = (1 + bt["ret"]).cumprod()
    bt["benchmark_equity"] = (1 + bt["benchmark_ret"]).cumprod()
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
                "avg_benchmark_ret": bt["benchmark_ret"].mean(),
                "avg_excess_ret": bt["excess_ret"].mean(),
                "total_return": bt["equity"].iloc[-1] - 1,
                "benchmark_total_return": bt["benchmark_equity"].iloc[-1] - 1,
                "ear": annualized_return(bt, days),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(predictions: pd.DataFrame, baseline_return: float, n: int = TOP_N) -> dict:
    """Compute unseen-data accuracy and ranking metrics for return forecasts."""
    labeled = predictions[predictions[TARGET_COL].notna() & predictions["pred"].notna()].copy()
    if labeled.empty:
        return _empty_metrics()

    actual = labeled[TARGET_COL].to_numpy()
    pred = labeled["pred"].to_numpy()
    error = actual - pred
    sse = float(np.square(error).sum())
    sst = float(np.square(actual - baseline_return).sum())
    oos_r2 = 1 - sse / sst if sst > 0 else np.nan
    rank_ic = _mean_rank_ic(labeled)
    bt = backtest_top_n(labeled, n=n)

    return {
        "rows": len(labeled),
        "direction_accuracy": float((np.sign(actual) == np.sign(pred)).mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "oos_r2": float(oos_r2),
        "rank_ic": rank_ic,
        "top_n_hit_rate": _safe_mean(bt, "hit_rate"),
        "top_n_positive_period_rate": _safe_rate(bt, "ret", threshold=0.0),
    }


def summarize_backtest(bt: pd.DataFrame) -> dict:
    """Summarize strategy returns against the equal-weighted universe."""
    if bt.empty:
        return {
            "periods": 0,
            "avg_20d_ret": np.nan,
            "avg_benchmark_20d_ret": np.nan,
            "avg_excess_20d_ret": np.nan,
            "strategy_total_return": np.nan,
            "benchmark_total_return": np.nan,
            "excess_total_return": np.nan,
            "beat_benchmark_rate": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }

    strategy_total = float(bt["equity"].iloc[-1] - 1)
    benchmark_total = float(bt["benchmark_equity"].iloc[-1] - 1)
    return {
        "periods": len(bt),
        "avg_20d_ret": float(bt["ret"].mean()),
        "avg_benchmark_20d_ret": float(bt["benchmark_ret"].mean()),
        "avg_excess_20d_ret": float(bt["excess_ret"].mean()),
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "excess_total_return": strategy_total - benchmark_total,
        "beat_benchmark_rate": float((bt["ret"] > bt["benchmark_ret"]).mean()),
        "max_drawdown": _max_drawdown(bt["equity"]),
        "sharpe": _annualized_sharpe(bt["ret"]),
    }


def _mean_rank_ic(labeled: pd.DataFrame) -> float:
    rank_ics = []
    for _, snap in labeled.groupby("date"):
        if len(snap) < 2:
            continue
        if snap["pred"].nunique() < 2 or snap[TARGET_COL].nunique() < 2:
            continue
        corr = snap["pred"].corr(snap[TARGET_COL], method="spearman")
        if pd.notna(corr):
            rank_ics.append(corr)
    return float(np.mean(rank_ics)) if rank_ics else np.nan


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def _annualized_sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan
    return float((returns.mean() / std) * np.sqrt(PERIODS_PER_YEAR))


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df:
        return np.nan
    return float(df[col].mean())


def _safe_rate(df: pd.DataFrame, col: str, threshold: float) -> float:
    if df.empty or col not in df:
        return np.nan
    return float((df[col] > threshold).mean())


def _empty_metrics() -> dict:
    return {
        "rows": 0,
        "direction_accuracy": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "oos_r2": np.nan,
        "rank_ic": np.nan,
        "top_n_hit_rate": np.nan,
        "top_n_positive_period_rate": np.nan,
    }


if __name__ == "__main__":
    df = build_dataset()
    _, test, info = train_and_predict(df, return_diagnostics=True)
    bt = backtest_top_n(test)
    summary = summarize_backtest(bt)
    accuracy = evaluate_predictions(test, info["train_target_mean"])
    print(bt.tail().to_string(index=False))
    print()
    print(f"Selected model: {info['selected_model']}")
    print(f"Directional accuracy: {accuracy['direction_accuracy']:.2%}")
    print(f"Top-{TOP_N} hit rate: {accuracy['top_n_hit_rate']:.2%}")
    print(f"Rank IC: {accuracy['rank_ic']:.3f}")
    print(f"OOS R2: {accuracy['oos_r2']:.2%}")
    print(f"Periods: {summary['periods']}")
    print(f"Avg 20d ret: {summary['avg_20d_ret']:.2%}")
    print(f"Total return: {summary['strategy_total_return']:.1%}")
