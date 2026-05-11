from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nordic_backtest import monthly_rebalance_dates, save_plot, summarize
from value_backtest import enrich_scores_for_date, latest_fundamentals, load_currency_map, load_fx

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

TOP_N = 5
LAG_DAYS = 90
COST_BPS = 20.0
MIN_TRAIN_MONTHS = 60
TRAIN_WINDOW_MONTHS = 120

PRICE_FEATURES = [
    "mom_12_1",
    "mom_6m",
    "mom_3m",
    "ret_1m",
    "vol_3m",
    "vol_12m",
    "max_drawdown_12m",
    "above_sma50",
    "above_sma200",
    "dist_sma200",
    "volume_ratio_20_252",
    "log_dollar_volume_20",
]

FUNDAMENTAL_FEATURES = [
    "log_market_cap",
    "book_to_market",
    "earnings_yield",
    "fcf_yield",
    "ebitda_yield_ev",
    "roe_ttm",
    "fcf_to_assets_ttm",
    "debt_to_equity",
    "net_debt_to_ebitda_ttm",
    "eps_growth_yoy",
    "momentum_6m",
    "momentum_12m",
]

FEATURES = PRICE_FEATURES + FUNDAMENTAL_FEATURES


def max_drawdown(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    running_max = series.cummax()
    return float((series / running_max - 1.0).min())


def price_features_at(close_adj: pd.DataFrame, volume: pd.DataFrame, date: pd.Timestamp, ticker: str) -> dict[str, float]:
    if ticker not in close_adj.columns:
        return {}
    loc = close_adj.index.get_indexer([date], method="ffill")[0]
    if loc < 252:
        return {}

    px = close_adj[ticker]
    vol = volume[ticker] if ticker in volume.columns else pd.Series(index=close_adj.index, dtype="float64")
    current = px.iloc[loc]
    if pd.isna(current) or current <= 0:
        return {}

    returns = px.pct_change(fill_method=None)
    sma50 = px.iloc[loc - 49 : loc + 1].mean()
    sma200 = px.iloc[loc - 199 : loc + 1].mean()
    vol20 = vol.iloc[loc - 19 : loc + 1].mean()
    vol252 = vol.iloc[loc - 251 : loc + 1].mean()
    dollar_volume = (px * vol).iloc[loc - 19 : loc + 1].mean()

    def ret(days: int) -> float:
        start = px.iloc[loc - days]
        if pd.isna(start) or start <= 0:
            return np.nan
        return float(current / start - 1.0)

    p_21 = px.iloc[loc - 21]
    p_252 = px.iloc[loc - 252]
    mom_12_1 = np.nan if pd.isna(p_21) or pd.isna(p_252) or p_252 <= 0 else float(p_21 / p_252 - 1.0)

    return {
        "mom_12_1": mom_12_1,
        "mom_6m": ret(126),
        "mom_3m": ret(63),
        "ret_1m": ret(21),
        "vol_3m": float(returns.iloc[loc - 62 : loc + 1].std() * math.sqrt(252)),
        "vol_12m": float(returns.iloc[loc - 251 : loc + 1].std() * math.sqrt(252)),
        "max_drawdown_12m": max_drawdown(px.iloc[loc - 251 : loc + 1]),
        "above_sma50": float(current > sma50) if pd.notna(sma50) else np.nan,
        "above_sma200": float(current > sma200) if pd.notna(sma200) else np.nan,
        "dist_sma200": float(current / sma200 - 1.0) if pd.notna(sma200) and sma200 > 0 else np.nan,
        "volume_ratio_20_252": float(vol20 / vol252) if pd.notna(vol20) and pd.notna(vol252) and vol252 > 0 else np.nan,
        "log_dollar_volume_20": float(np.log1p(dollar_volume)) if pd.notna(dollar_volume) and dollar_volume > 0 else np.nan,
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    close_adj = pd.read_csv(OUTPUT_DIR / "adjusted_close_yahoo.csv", parse_dates=["date"]).set_index("date").sort_index().ffill(limit=5)
    close_raw = pd.read_csv(OUTPUT_DIR / "close_yahoo.csv", parse_dates=["date"]).set_index("date").sort_index().ffill(limit=5)
    volume = pd.read_csv(OUTPUT_DIR / "volume_yahoo.csv", parse_dates=["date"]).set_index("date").sort_index().ffill(limit=5)
    fx = load_fx()
    currency_map = load_currency_map()
    return close_adj, close_raw, volume, fx, currency_map


def build_monthly_panel() -> pd.DataFrame:
    close_adj, close_raw, volume, fx, currency_map = load_inputs()
    factors = latest_fundamentals()
    rebal_dates = monthly_rebalance_dates(close_adj.index)
    rows = []

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]
        fundamental_frame = enrich_scores_for_date(factors, close_raw, close_adj, fx, currency_map, date, LAG_DAYS)
        if fundamental_frame.empty:
            continue
        fundamental_frame = fundamental_frame.set_index("ticker")

        start_prices = close_adj.reindex(close_adj.index.union([date])).sort_index().ffill().loc[date]
        end_prices = close_adj.reindex(close_adj.index.union([next_date])).sort_index().ffill().loc[next_date]
        next_returns = (end_prices / start_prices - 1.0).dropna()
        cross_mean = next_returns.mean(skipna=True)

        for ticker, frow in fundamental_frame.iterrows():
            if ticker not in next_returns.index or pd.isna(next_returns[ticker]):
                continue
            pfeat = price_features_at(close_adj, volume, date, ticker)
            if not pfeat:
                continue
            market_cap = frow.get("market_cap_nok_m")
            row = {
                "date": date,
                "next_date": next_date,
                "ticker": ticker,
                "target_abs_return": next_returns[ticker],
                "target_rel_return": next_returns[ticker] - cross_mean,
                **pfeat,
                "log_market_cap": np.log1p(market_cap) if pd.notna(market_cap) and market_cap > 0 else np.nan,
                "book_to_market": frow.get("book_to_market"),
                "earnings_yield": frow.get("earnings_yield"),
                "fcf_yield": frow.get("fcf_yield"),
                "ebitda_yield_ev": frow.get("ebitda_yield_ev"),
                "roe_ttm": frow.get("roe_ttm"),
                "fcf_to_assets_ttm": frow.get("fcf_to_assets_ttm"),
                "debt_to_equity": frow.get("debt_to_equity"),
                "net_debt_to_ebitda_ttm": frow.get("net_debt_to_ebitda_ttm"),
                "eps_growth_yoy": frow.get("eps_growth_yoy"),
                "momentum_6m": frow.get("momentum_6m"),
                "momentum_12m": frow.get("momentum_12m"),
            }
            rows.append(row)

    panel = pd.DataFrame(rows).sort_values(["date", "ticker"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def rank_score_baseline(panel: pd.DataFrame) -> pd.Series:
    def z(frame: pd.DataFrame, col: str, sign: float = 1.0) -> pd.Series:
        s = pd.to_numeric(frame[col], errors="coerce")
        std = s.std(skipna=True)
        if pd.isna(std) or std == 0:
            return s * 0
        return sign * (s - s.mean(skipna=True)) / std

    scores = []
    for _, frame in panel.groupby("date"):
        score = (
            0.30 * z(frame, "mom_12_1")
            + 0.15 * z(frame, "mom_6m")
            + 0.15 * z(frame, "earnings_yield")
            + 0.15 * z(frame, "fcf_yield")
            + 0.10 * z(frame, "book_to_market")
            + 0.10 * z(frame, "roe_ttm")
            + 0.05 * z(frame, "debt_to_equity", sign=-1.0)
        )
        scores.append(score)
    return pd.concat(scores).sort_index()


def model_specs() -> dict[str, object]:
    return {
        "factor_score": None,
        "momentum_12_1": None,
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.03,
                max_leaf_nodes=8,
                l2_regularization=1.0,
                min_samples_leaf=20,
                random_state=42,
            ),
        ),
    }


def walk_forward_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy().reset_index(drop=True)
    panel["factor_score"] = rank_score_baseline(panel).values
    dates = sorted(panel["date"].unique())
    predictions = []

    for model_name, model in model_specs().items():
        for test_date in dates:
            train_dates = [d for d in dates if d < test_date]
            if len(train_dates) < MIN_TRAIN_MONTHS and model_name != "factor_score":
                continue
            test = panel[panel["date"] == test_date].copy()
            if test.empty:
                continue

            if model_name == "factor_score":
                test["prediction"] = test["factor_score"]
            elif model_name == "momentum_12_1":
                test["prediction"] = test["mom_12_1"]
            else:
                window_dates = train_dates[-TRAIN_WINDOW_MONTHS:]
                train = panel[panel["date"].isin(window_dates)].copy()
                train = train.dropna(subset=["target_rel_return"])
                usable_features = [c for c in FEATURES if c in train.columns]
                if train.empty or len(train) < 200:
                    continue
                x_train = train[usable_features].apply(pd.to_numeric, errors="coerce")
                y_train = pd.to_numeric(train["target_rel_return"], errors="coerce")
                x_test = test[usable_features].apply(pd.to_numeric, errors="coerce")
                model.fit(x_train, y_train)
                test["prediction"] = model.predict(x_test)

            test["model"] = model_name
            predictions.append(test[["model", "date", "next_date", "ticker", "prediction", "target_abs_return", "target_rel_return"]])

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def backtest_predictions(predictions: pd.DataFrame, close_adj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_returns = close_adj.pct_change(fill_method=None)
    daily_returns = raw_returns.fillna(0.0)
    rows = []
    all_curves = []
    holdings = []

    for model_name, pred in predictions.groupby("model"):
        dates = sorted(pred["date"].unique())
        weights = pd.DataFrame(0.0, index=close_adj.index, columns=close_adj.columns)
        costs = pd.Series(0.0, index=close_adj.index)
        current_weights = pd.Series(0.0, index=close_adj.columns)
        monthly_rows = []

        for date in dates:
            frame = pred[pred["date"] == date].dropna(subset=["prediction"]).sort_values("prediction", ascending=False)
            selected = frame.head(TOP_N)["ticker"].tolist()
            new_weights = pd.Series(0.0, index=close_adj.columns)
            if selected:
                new_weights.loc[selected] = 1.0 / len(selected)
            turnover = float((new_weights - current_weights).abs().sum())
            costs.loc[date] = turnover * COST_BPS / 10_000.0
            weights.loc[date] = new_weights
            current_weights = new_weights
            month_return = frame[frame["ticker"].isin(selected)]["target_abs_return"].mean() if selected else 0.0
            hit_rate = (frame[frame["ticker"].isin(selected)]["target_rel_return"] > 0).mean() if selected else np.nan
            monthly_rows.append({"model": model_name, "date": date, "selected": ",".join(selected), "turnover": turnover, "next_month_return": month_return, "hit_rate": hit_rate})
            holdings.extend(frame.assign(selected=frame["ticker"].isin(selected)).to_dict("records"))

        weights = weights.ffill().fillna(0.0)
        strategy_returns = (weights.shift(1).fillna(0.0) * daily_returns).sum(axis=1) - costs
        benchmark_returns = raw_returns.mean(axis=1, skipna=True).fillna(0.0)
        curve = pd.DataFrame(
            {
                "strategy": (1.0 + strategy_returns).cumprod(),
                "equal_weight_benchmark": (1.0 + benchmark_returns).cumprod(),
                "strategy_daily_return": strategy_returns,
                "benchmark_daily_return": benchmark_returns,
            },
            index=close_adj.index,
        )
        first_signal_date = min(dates)
        curve = curve[curve.index >= first_signal_date].copy()
        for equity_col in ["strategy", "equal_weight_benchmark"]:
            if curve[equity_col].iloc[0] != 0:
                curve[equity_col] = curve[equity_col] / curve[equity_col].iloc[0]
        if not curve.empty:
            curve.iloc[0, curve.columns.get_loc("strategy_daily_return")] = 0.0
            curve.iloc[0, curve.columns.get_loc("benchmark_daily_return")] = 0.0
        summary = summarize(curve)
        strat = summary[summary["portfolio"] == "strategy"].iloc[0].to_dict()
        bench = summary[summary["portfolio"] == "equal_weight_benchmark"].iloc[0].to_dict()
        month_df = pd.DataFrame(monthly_rows)
        rows.append(
            {
                "model": model_name,
                "start": strat["start"],
                "end": strat["end"],
                "total_return": strat["total_return"],
                "annualized_return": strat["annualized_return"],
                "max_drawdown": strat["max_drawdown"],
                "sharpe": strat["sharpe"],
                "benchmark_total_return": bench["total_return"],
                "benchmark_annualized_return": bench["annualized_return"],
                "excess_total_return": strat["total_return"] - bench["total_return"],
                "avg_monthly_turnover": month_df["turnover"].mean(),
                "avg_hit_rate": month_df["hit_rate"].mean(),
                "months": len(month_df),
            }
        )
        curve_out = curve[["strategy", "equal_weight_benchmark"]].copy()
        curve_out.columns = [f"{model_name}_strategy", f"{model_name}_equal_weight_benchmark"]
        all_curves.append(curve_out)

    summary = pd.DataFrame(rows).sort_values("annualized_return", ascending=False)
    curves = pd.concat(all_curves, axis=1).sort_index() if all_curves else pd.DataFrame()
    return summary, curves, pd.DataFrame(holdings)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    close_adj, _, _, _, _ = load_inputs()
    panel_path = OUTPUT_DIR / "ml_factor_panel.csv"
    panel = build_monthly_panel()
    panel.to_csv(panel_path, index=False)

    predictions = walk_forward_predictions(panel)
    predictions.to_csv(OUTPUT_DIR / "ml_factor_predictions.csv", index=False)
    summary, curves, holdings = backtest_predictions(predictions, close_adj)
    summary.to_csv(OUTPUT_DIR / "ml_factor_backtest_summary.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "ml_factor_equity_curves.csv")
    holdings.to_csv(OUTPUT_DIR / "ml_factor_holdings.csv", index=False)

    best_model = summary.iloc[0]["model"] if not summary.empty else None
    if best_model and f"{best_model}_strategy" in curves.columns:
        plot_curve = pd.DataFrame(
            {
                "strategy": curves[f"{best_model}_strategy"],
                "equal_weight_benchmark": curves[f"{best_model}_equal_weight_benchmark"],
            }
        ).dropna()
        plot_curve["strategy_daily_return"] = plot_curve["strategy"].pct_change(fill_method=None).fillna(0.0)
        plot_curve["benchmark_daily_return"] = plot_curve["equal_weight_benchmark"].pct_change(fill_method=None).fillna(0.0)
        save_plot(plot_curve, OUTPUT_DIR / "ml_factor_best_equity_curve.png", title=f"ML factor strategy: {best_model}")

    print("Panel:", len(panel), "rader,", panel["date"].nunique(), "måneder,", panel["ticker"].nunique(), "aksjer")
    print("\nBacktest summary:")
    print(summary.to_string(index=False))
    if not holdings.empty:
        latest_date = holdings["date"].max()
        latest = holdings[(holdings["date"] == latest_date) & (holdings["selected"])]
        print("\nSiste valgte aksjer per modell:")
        print(latest[["model", "date", "ticker", "prediction"]].sort_values(["model", "prediction"], ascending=[True, False]).to_string(index=False))
    print("\nSkrev output/ml_factor_backtest_summary.csv, output/ml_factor_panel.csv og output/ml_factor_holdings.csv")


if __name__ == "__main__":
    main()
