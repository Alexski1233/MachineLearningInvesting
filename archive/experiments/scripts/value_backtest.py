from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from nordic_backtest import monthly_rebalance_dates, save_plot, summarize
from parse_fundamentals import build_quarterly_factors, parse_all

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
RAW_DIR = ROOT / "data" / "NorskeBørs"
FX_FILE = OUTPUT_DIR / "fx_yahoo.csv"


def unix_date(year: int, month: int, day: int) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp())


def fetch_yahoo_series(symbol: str, value_name: str) -> pd.Series:
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "period1": unix_date(2000, 1, 1),
            "period2": int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)).timestamp()),
            "interval": "1d",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    return pd.Series(
        quote["close"],
        index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
        name=value_name,
    ).dropna()


def load_fx() -> pd.DataFrame:
    if FX_FILE.exists():
        return pd.read_csv(FX_FILE, parse_dates=["date"]).set_index("date").sort_index().ffill()
    usd = fetch_yahoo_series("USDNOK=X", "USD")
    eur = fetch_yahoo_series("EURNOK=X", "EUR")
    fx = pd.concat([usd, eur], axis=1).sort_index().ffill()
    fx["NOK"] = 1.0
    fx.to_csv(FX_FILE, index_label="date")
    return fx


def load_currency_map() -> dict[str, str]:
    currency = {}
    for path in RAW_DIR.glob("*.xlsx"):
        raw = pd.read_excel(path, sheet_name="Balance Sheet", header=None, nrows=16)
        company = str(raw.loc[raw.iloc[:, 0].astype(str).eq("Company Name"), 1].iloc[0])
        curr = str(raw.loc[raw.iloc[:, 0].astype(str).eq("Standardized Currency"), 1].iloc[0])
        ticker = company.split("(")[-1].split(")")[0].replace(".OL", "").upper()
        ticker = {"AFGA": "AFG", "BRGB": "BRG", "GJFG": "GJF"}.get(ticker, ticker)
        currency[ticker] = curr
    return currency


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(skipna=True)
    if pd.isna(std) or std == 0:
        return s * 0
    return (s - s.mean(skipna=True)) / std


def latest_fundamentals() -> pd.DataFrame:
    selected, _ = parse_all()
    return build_quarterly_factors(selected)


def enrich_scores_for_date(
    factors: pd.DataFrame,
    close_raw: pd.DataFrame,
    close_adj: pd.DataFrame,
    fx: pd.DataFrame,
    currency_map: dict[str, str],
    date: pd.Timestamp,
    lag_days: int,
) -> pd.DataFrame:
    asof = date - pd.Timedelta(days=lag_days)
    usable = factors[factors["period_end_date"] <= asof]
    if usable.empty:
        return pd.DataFrame()
    latest = usable.sort_values("period_end_date").groupby("ticker", as_index=False).tail(1).copy()
    latest = latest[latest["ticker"].isin(close_raw.columns)]

    raw_prices = close_raw.reindex(close_raw.index.union([date])).sort_index().ffill().loc[date]
    adj_prices = close_adj.reindex(close_adj.index.union([date])).sort_index().ffill().loc[date]
    fx_today = fx.reindex(fx.index.union([date])).sort_index().ffill().loc[date]

    rows = []
    for _, row in latest.iterrows():
        ticker = row["ticker"]
        price = raw_prices.get(ticker)
        adj_price = adj_prices.get(ticker)
        shares = row.get("shares_outstanding")
        if pd.isna(price) or pd.isna(adj_price) or pd.isna(shares) or shares <= 0:
            continue
        curr = currency_map.get(ticker, "NOK")
        fx_rate = fx_today.get(curr)
        if pd.isna(fx_rate):
            continue

        market_cap_nok = price * shares
        equity_nok = row.get("equity_used") * fx_rate
        earnings_nok = row.get("net_income_ttm_used") * fx_rate
        fcf_nok = row.get("free_cash_flow_ttm") * fx_rate
        ebitda_nok = row.get("ebitda_ttm") * fx_rate
        net_debt_nok = row.get("net_debt") * fx_rate
        enterprise_value_nok = market_cap_nok + (0 if pd.isna(net_debt_nok) else net_debt_nok)

        try:
            loc = close_adj.index.get_indexer([date], method="ffill")[0]
        except Exception:
            loc = -1
        momentum_6m = pd.NA
        momentum_12m = pd.NA
        if loc >= 126 and ticker in close_adj.columns:
            then = close_adj.iloc[loc - 126][ticker]
            if pd.notna(then) and then > 0:
                momentum_6m = adj_price / then - 1
        if loc >= 252 and ticker in close_adj.columns:
            then = close_adj.iloc[loc - 252][ticker]
            if pd.notna(then) and then > 0:
                momentum_12m = adj_price / then - 1

        out = row.to_dict()
        out.update(
            {
                "rebalance_date": date,
                "currency": curr,
                "fx_to_nok": fx_rate,
                "price_nok": price,
                "market_cap_nok_m": market_cap_nok,
                "enterprise_value_nok_m": enterprise_value_nok,
                "book_to_market": equity_nok / market_cap_nok if market_cap_nok else pd.NA,
                "earnings_yield": earnings_nok / market_cap_nok if market_cap_nok else pd.NA,
                "fcf_yield": fcf_nok / market_cap_nok if market_cap_nok else pd.NA,
                "ebitda_yield_ev": ebitda_nok / enterprise_value_nok if enterprise_value_nok else pd.NA,
                "momentum_6m": momentum_6m,
                "momentum_12m": momentum_12m,
            }
        )
        rows.append(out)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    value_parts = pd.DataFrame(
        {
            "book_to_market": zscore(frame["book_to_market"]),
            "earnings_yield": zscore(frame["earnings_yield"]),
            "fcf_yield": zscore(frame["fcf_yield"]),
            "ebitda_yield_ev": zscore(frame["ebitda_yield_ev"]),
        }
    )
    quality_parts = pd.DataFrame(
        {
            "roe": zscore(frame["roe_ttm"]),
            "fcf_assets": zscore(frame["fcf_to_assets_ttm"]),
            "debt_equity": -zscore(frame["debt_to_equity"]),
            "net_debt_ebitda": -zscore(frame["net_debt_to_ebitda_ttm"]),
        }
    )
    momentum_parts = pd.DataFrame({"mom6": zscore(frame["momentum_6m"]), "mom12": zscore(frame["momentum_12m"])})

    frame["value_score"] = value_parts.mean(axis=1, skipna=True)
    frame["quality_score_cross"] = quality_parts.mean(axis=1, skipna=True)
    frame["momentum_score"] = momentum_parts.mean(axis=1, skipna=True)
    frame["value_quality_momentum_score"] = (
        0.45 * frame["value_score"]
        + 0.35 * frame["quality_score_cross"]
        + 0.20 * frame["momentum_score"].fillna(0)
    )
    return frame.sort_values("value_quality_momentum_score", ascending=False)


def run_backtest(top_n: int = 5, lag_days: int = 90, cost_bps: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_adj = pd.read_csv(OUTPUT_DIR / "adjusted_close_yahoo.csv", parse_dates=["date"]).set_index("date").sort_index().ffill(limit=5)
    close_raw = pd.read_csv(OUTPUT_DIR / "close_yahoo.csv", parse_dates=["date"]).set_index("date").sort_index().ffill(limit=5)
    fx = load_fx()
    factors = latest_fundamentals()
    currency_map = load_currency_map()

    raw_returns = close_adj.pct_change(fill_method=None)
    returns = raw_returns.fillna(0.0)
    weights = pd.DataFrame(0.0, index=close_adj.index, columns=close_adj.columns)
    costs = pd.Series(0.0, index=close_adj.index)
    current_weights = pd.Series(0.0, index=close_adj.columns)
    rankings = []

    for date in monthly_rebalance_dates(close_adj.index):
        scores = enrich_scores_for_date(factors, close_raw, close_adj, fx, currency_map, date, lag_days)
        if scores.empty:
            weights.loc[date] = current_weights
            continue
        selected = scores.head(top_n)["ticker"].tolist()
        new_weights = pd.Series(0.0, index=close_adj.columns)
        if selected:
            new_weights.loc[selected] = 1.0 / len(selected)
        costs.loc[date] = float((new_weights - current_weights).abs().sum()) * cost_bps / 10_000.0
        weights.loc[date] = new_weights
        current_weights = new_weights
        rankings.append(scores.assign(selected=scores["ticker"].isin(selected)))

    weights = weights.ffill().fillna(0.0)
    strategy_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1) - costs
    benchmark_returns = raw_returns.mean(axis=1, skipna=True).fillna(0.0)
    curve = pd.DataFrame(
        {
            "strategy": (1 + strategy_returns).cumprod(),
            "equal_weight_benchmark": (1 + benchmark_returns).cumprod(),
            "strategy_daily_return": strategy_returns,
            "benchmark_daily_return": benchmark_returns,
        }
    )
    ranking = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    return curve, ranking


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    curve, ranking = run_backtest()
    summary = summarize(curve)
    curve.to_csv(OUTPUT_DIR / "value_equity_curve.csv")
    ranking.to_csv(OUTPUT_DIR / "value_backtest_ranking.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "value_backtest_summary.csv", index=False)
    save_plot(curve, OUTPUT_DIR / "value_equity_curve.png", title="Value + quality + momentum vs equal-weight")
    print(summary.to_string(index=False))
    if not ranking.empty:
        latest = ranking[ranking["rebalance_date"] == ranking["rebalance_date"].max()]
        print("\nSiste valgte aksjer:")
        print(latest[latest["selected"]][["rebalance_date", "ticker", "value_quality_momentum_score", "value_score", "quality_score_cross", "momentum_score", "earnings_yield", "fcf_yield", "book_to_market"]].to_string(index=False))
    print("\nSkrev output/value_backtest_summary.csv og output/value_equity_curve.png")


if __name__ == "__main__":
    main()
