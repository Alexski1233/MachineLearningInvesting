from __future__ import annotations

import argparse
import io
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


BASE_URL = "https://live.euronext.com"
SEARCH_URL = f"{BASE_URL}/en/instrumentSearch/searchJSON"
DOWNLOAD_URL = f"{BASE_URL}/en/ajax/AwlHistoricalPrice/getFullDownloadAjax"

ROOT = Path(__file__).resolve().parent
UNIVERSE_FILE = ROOT / "data" / "universe_oslo.csv"
PRICE_DIR = ROOT / "data" / "prices"
OUTPUT_DIR = ROOT / "output"


@dataclass(frozen=True)
class Instrument:
    query: str
    name: str
    symbol: str
    isin: str
    mic: str

    @property
    def instrument_id(self) -> str:
        return f"{self.isin}-{self.mic}"


OSEBX_INSTRUMENT = Instrument(
    query="OSEBX",
    name="OSEBX GR",
    symbol="OSEBX",
    isin="NO0007035327",
    mic="XOSL",
)


def clean_label_field(label: str, class_name: str) -> str:
    pattern = rf"<span class=['\"]{class_name}['\"]>(.*?)</span>"
    match = re.search(pattern, label, flags=re.IGNORECASE)
    if not match:
        return ""
    text = re.sub(r"<.*?>", "", match.group(1))
    return text.strip()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
            "Accept": "application/json, text/csv, text/plain, */*",
        }
    )
    return session


def load_queries(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Fant ikke universfil: {path}")
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("Universe CSV må ha kolonnen 'query'.")
    return [str(q).strip() for q in df["query"].dropna() if str(q).strip()]


def resolve_instrument(session: requests.Session, query: str, mic: str = "XOSL") -> Instrument:
    response = session.get(SEARCH_URL, params={"q": query}, timeout=30)
    response.raise_for_status()
    results = response.json()

    candidates = []
    for item in results:
        link = item.get("link", "")
        if item.get("mic") == mic and "/product/equities/" in link:
            label = item.get("label", "")
            symbol = clean_label_field(label, "symbol")
            name = item.get("name") or clean_label_field(label, "name") or query
            candidates.append(
                Instrument(
                    query=query,
                    name=name,
                    symbol=symbol or query,
                    isin=item["isin"],
                    mic=item["mic"],
                )
            )

    if not candidates:
        raise LookupError(f"Fant ikke Oslo Bors equity for '{query}'.")

    query_norm = query.replace(" ", "").lower()
    for candidate in candidates:
        if candidate.name.replace(" ", "").lower() == query_norm:
            return candidate
    return candidates[0]


def parse_euronext_csv(text: str) -> pd.DataFrame:
    lines = text.replace("\ufeff", "").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.startswith("Date;Open;High;Low;")),
        None,
    )
    if header_index is None:
        raise ValueError("Kunne ikke finne pris-header i Euronext CSV.")

    csv_text = "\n".join(line.rstrip(";") for line in lines[header_index:])
    df = pd.read_csv(io.StringIO(csv_text), sep=";")
    df.columns = [str(col).strip().strip('"').lower().replace(" ", "_") for col in df.columns]

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mangler kolonner i Euronext CSV: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    numeric_cols = [col for col in df.columns if col != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    df = df.sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)


def fetch_history(
    session: requests.Session,
    instrument: Instrument,
    refresh: bool = False,
    pause_seconds: float = 0.25,
) -> pd.DataFrame:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PRICE_DIR / f"{instrument.instrument_id}.csv"

    if cache_file.exists() and not refresh:
        df = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        url = f"{DOWNLOAD_URL}/{instrument.instrument_id}"
        params = {
            "format": "csv",
            "decimal_separator": ".",
            "date_form": "d/m/Y",
        }
        headers = {"Referer": f"{BASE_URL}/en/popout-page/getHistoricalPrice/{instrument.instrument_id}"}
        response = session.get(url, params=params, headers=headers, timeout=45)
        response.raise_for_status()
        if "Date;Open;High;Low;" not in response.text:
            raise ValueError(f"Uventet Euronext-svar for {instrument.instrument_id}")
        df = parse_euronext_csv(response.text)
        df.to_csv(cache_file, index=False)
        time.sleep(pause_seconds)

    df["query"] = instrument.query
    df["name"] = instrument.name
    df["symbol"] = instrument.symbol
    df["isin"] = instrument.isin
    df["mic"] = instrument.mic
    return df


def combine_close_prices(frames: Iterable[pd.DataFrame], min_history_days: int) -> pd.DataFrame:
    return combine_price_field(frames, "close", min_history_days)


def combine_price_field(frames: Iterable[pd.DataFrame], field: str, min_history_days: int) -> pd.DataFrame:
    series = []
    for df in frames:
        if len(df) < min_history_days or field not in df.columns:
            continue
        symbol = df["symbol"].iloc[0]
        s = df.set_index("date")[field].rename(symbol)
        series.append(s)

    if not series:
        raise ValueError(f"Ingen instrumenter hadde nok historikk for '{field}'.")

    values = pd.concat(series, axis=1).sort_index()
    values = values.ffill(limit=5)
    values = values.dropna(axis=1, thresh=min_history_days)
    values = values.dropna(how="all")
    return values


def build_market_data(frames: Iterable[pd.DataFrame], min_history_days: int) -> dict[str, pd.DataFrame]:
    frame_list = list(frames)
    close = combine_price_field(frame_list, "close", min_history_days)
    high = combine_price_field(frame_list, "high", min_history_days).reindex_like(close).ffill(limit=5)

    if frame_list and "number_of_shares" in frame_list[0].columns:
        volume = combine_price_field(frame_list, "number_of_shares", min_history_days).reindex_like(close).fillna(0.0)
    else:
        volume = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    if frame_list and "turnover" in frame_list[0].columns:
        turnover = combine_price_field(frame_list, "turnover", min_history_days).reindex_like(close)
    else:
        turnover = close * volume
    turnover = turnover.fillna(close * volume).fillna(0.0)

    return {
        "close": close,
        "high": high,
        "volume": volume,
        "turnover": turnover,
    }


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    monthly = pd.Series(index=index, data=index).groupby(index.to_period("M")).max()
    return list(monthly.values)


def run_momentum_backtest(
    close: pd.DataFrame,
    lookback_days: int,
    top_n: int,
    cost_bps: float,
    benchmark_close: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change().fillna(0.0)
    rebalance_dates = monthly_rebalance_dates(close.index)
    cost_rate = cost_bps / 10_000.0

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    costs = pd.Series(0.0, index=close.index)
    ranking_rows = []
    current_weights = pd.Series(0.0, index=close.columns)

    for date in rebalance_dates:
        loc = close.index.get_loc(date)
        if loc < lookback_days:
            weights.loc[date] = current_weights
            continue

        momentum = close.iloc[loc] / close.iloc[loc - lookback_days] - 1.0
        momentum = momentum.dropna().sort_values(ascending=False)
        selected = momentum[momentum > 0].head(top_n).index

        new_weights = pd.Series(0.0, index=close.columns)
        if len(selected) > 0:
            new_weights.loc[selected] = 1.0 / len(selected)

        turnover = float((new_weights - current_weights).abs().sum())
        ranking_rows.extend(
            {
                "rebalance_date": date,
                "symbol": symbol,
                "momentum": value,
                "selected": symbol in set(selected),
            }
            for symbol, value in momentum.items()
        )

        weights.loc[date] = new_weights
        current_weights = new_weights
        costs.loc[date] = turnover * cost_rate

    weights = weights.ffill().fillna(0.0)
    strategy_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    strategy_returns = strategy_returns - costs

    benchmark_returns = returns.mean(axis=1)

    curve = pd.DataFrame(
        {
            "strategy": (1.0 + strategy_returns).cumprod(),
            "equal_weight_benchmark": (1.0 + benchmark_returns).cumprod(),
            "strategy_daily_return": strategy_returns,
            "benchmark_daily_return": benchmark_returns,
        }
    )
    if benchmark_close is not None:
        aligned_benchmark = benchmark_close.reindex(close.index).ffill()
        osebx_returns = aligned_benchmark.pct_change(fill_method=None).fillna(0.0)
        curve["osebx"] = (1.0 + osebx_returns).cumprod()
        curve["osebx_daily_return"] = osebx_returns
    ranking = pd.DataFrame(ranking_rows)
    return curve, ranking


TECHNICAL_STRATEGIES = [
    "volume_surge",
    "momentum_volume",
    "breakout_volume",
    "obv_accumulation",
    "liquidity_momentum",
    "risk_adjusted_momentum",
    "volume_reversal",
    "volume_trend_confirmation",
    "quiet_accumulation",
    "breakout_liquidity",
]


def zscore(values: pd.Series) -> pd.Series:
    values = values.replace([math.inf, -math.inf], pd.NA).astype("float64")
    std = values.std(skipna=True)
    if pd.isna(std) or std == 0:
        return values * 0.0
    return (values - values.mean(skipna=True)) / std


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return (numerator / denominator).replace([math.inf, -math.inf], pd.NA).astype("float64")


def technical_min_history(params: dict[str, int | float | str]) -> int:
    keys = [
        "price_lookback",
        "volume_slow",
        "breakout_lookback",
        "ma_days",
        "risk_lookback",
        "accumulation_lookback",
        "short_lookback",
        "liquidity_lookback",
    ]
    return max(int(params.get(key, 1)) for key in keys)


def technical_scores_at(
    market_data: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    loc: int,
    strategy: str,
    params: dict[str, int | float | str],
) -> pd.Series:
    close = market_data["close"]
    high = market_data["high"]
    volume = market_data["volume"]
    turnover = market_data["turnover"]

    price_lookback = int(params.get("price_lookback", 63))
    short_lookback = int(params.get("short_lookback", 21))
    volume_fast = int(params.get("volume_fast", 10))
    volume_slow = int(params.get("volume_slow", 50))
    breakout_lookback = int(params.get("breakout_lookback", 63))
    ma_days = int(params.get("ma_days", 50))
    risk_lookback = int(params.get("risk_lookback", 63))
    accumulation_lookback = int(params.get("accumulation_lookback", 63))
    liquidity_lookback = int(params.get("liquidity_lookback", 20))
    min_liquidity_quantile = float(params.get("min_liquidity_quantile", 0.25))

    current_close = close.iloc[loc]
    momentum = current_close / close.iloc[loc - price_lookback] - 1.0
    short_momentum = current_close / close.iloc[loc - short_lookback] - 1.0
    volume_ratio = safe_ratio(
        volume.iloc[loc - volume_fast + 1 : loc + 1].mean(),
        volume.iloc[loc - volume_slow + 1 : loc + 1].mean(),
    )
    rolling_high = high.iloc[loc - breakout_lookback + 1 : loc + 1].max()
    high_position = safe_ratio(current_close, rolling_high)
    moving_average = close.iloc[loc - ma_days + 1 : loc + 1].mean()
    distance_sma = safe_ratio(current_close, moving_average) - 1.0
    realized_vol = returns.iloc[loc - risk_lookback + 1 : loc + 1].std()
    risk_adjusted_momentum = safe_ratio(momentum, realized_vol)
    liquidity = turnover.iloc[loc - liquidity_lookback + 1 : loc + 1].mean()

    signed_volume = volume.iloc[loc - accumulation_lookback + 1 : loc + 1] * returns.iloc[
        loc - accumulation_lookback + 1 : loc + 1
    ].apply(lambda col: col.map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0)))
    accumulation = safe_ratio(
        signed_volume.sum(),
        volume.iloc[loc - accumulation_lookback + 1 : loc + 1].sum(),
    )

    positive_volume = volume.iloc[loc - accumulation_lookback + 1 : loc + 1].where(
        returns.iloc[loc - accumulation_lookback + 1 : loc + 1] > 0
    ).sum()
    negative_volume = volume.iloc[loc - accumulation_lookback + 1 : loc + 1].where(
        returns.iloc[loc - accumulation_lookback + 1 : loc + 1] < 0
    ).sum()
    up_down_volume_ratio = safe_ratio(positive_volume, negative_volume)

    liquid = liquidity >= liquidity.quantile(min_liquidity_quantile)
    trend_ok = current_close > moving_average
    positive_momentum = momentum > 0

    if strategy == "volume_surge":
        score = zscore(volume_ratio) + 0.25 * zscore(momentum)
        valid = liquid & positive_momentum & (volume_ratio > 1.0)
    elif strategy == "momentum_volume":
        score = zscore(momentum) + zscore(volume_ratio)
        valid = liquid & positive_momentum & trend_ok
    elif strategy == "breakout_volume":
        score = zscore(high_position) + zscore(volume_ratio) + 0.50 * zscore(momentum)
        valid = liquid & positive_momentum & (high_position > 0.95) & (volume_ratio > 0.90)
    elif strategy == "obv_accumulation":
        score = zscore(accumulation) + 0.50 * zscore(momentum) + 0.25 * zscore(volume_ratio)
        valid = liquid & positive_momentum & (accumulation > 0)
    elif strategy == "liquidity_momentum":
        score = zscore(momentum) + 0.25 * zscore(liquidity)
        valid = liquid & positive_momentum
    elif strategy == "risk_adjusted_momentum":
        score = zscore(risk_adjusted_momentum) + 0.25 * zscore(volume_ratio)
        valid = liquid & positive_momentum & trend_ok
    elif strategy == "volume_reversal":
        score = zscore(-short_momentum) + zscore(volume_ratio) + 0.25 * zscore(accumulation)
        valid = liquid & (short_momentum < 0) & (momentum > -0.20) & (volume_ratio > 0.85)
    elif strategy == "volume_trend_confirmation":
        score = zscore(distance_sma) + zscore(volume_ratio) + zscore(up_down_volume_ratio)
        valid = liquid & trend_ok & positive_momentum & (up_down_volume_ratio > 1.0)
    elif strategy == "quiet_accumulation":
        score = zscore(accumulation) - zscore(realized_vol) + 0.50 * zscore(volume_ratio)
        valid = liquid & positive_momentum & (accumulation > 0) & (volume_ratio > 0.80)
    elif strategy == "breakout_liquidity":
        score = zscore(high_position) + zscore(momentum) + 0.25 * zscore(liquidity)
        valid = liquid & positive_momentum & (high_position > 0.95)
    else:
        raise ValueError(f"Ukjent teknisk strategi: {strategy}")

    return score.where(valid).dropna().sort_values(ascending=False)


def run_ranked_technical_backtest(
    market_data: dict[str, pd.DataFrame],
    strategy: str,
    params: dict[str, int | float | str],
    top_n: int,
    cost_bps: float,
    benchmark_close: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = market_data["close"]
    returns = close.pct_change(fill_method=None).fillna(0.0)
    rebalance_dates = monthly_rebalance_dates(close.index)
    cost_rate = cost_bps / 10_000.0
    min_history = technical_min_history(params)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    costs = pd.Series(0.0, index=close.index)
    ranking_rows = []
    current_weights = pd.Series(0.0, index=close.columns)

    for date in rebalance_dates:
        loc = close.index.get_loc(date)
        if loc < min_history:
            weights.loc[date] = current_weights
            continue

        scores = technical_scores_at(market_data, returns, loc, strategy, params)
        selected = scores.head(top_n).index

        new_weights = pd.Series(0.0, index=close.columns)
        if len(selected) > 0:
            new_weights.loc[selected] = 1.0 / len(selected)

        turnover = float((new_weights - current_weights).abs().sum())
        ranking_rows.extend(
            {
                "strategy": strategy,
                "rebalance_date": date,
                "symbol": symbol,
                "score": value,
                "selected": symbol in set(selected),
            }
            for symbol, value in scores.items()
        )

        weights.loc[date] = new_weights
        current_weights = new_weights
        costs.loc[date] = turnover * cost_rate

    weights = weights.ffill().fillna(0.0)
    strategy_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1) - costs
    benchmark_returns = returns.mean(axis=1)

    curve = pd.DataFrame(
        {
            "strategy": (1.0 + strategy_returns).cumprod(),
            "equal_weight_benchmark": (1.0 + benchmark_returns).cumprod(),
            "strategy_daily_return": strategy_returns,
            "benchmark_daily_return": benchmark_returns,
        }
    )
    if benchmark_close is not None:
        aligned_benchmark = benchmark_close.reindex(close.index).ffill()
        osebx_returns = aligned_benchmark.pct_change(fill_method=None).fillna(0.0)
        curve["osebx"] = (1.0 + osebx_returns).cumprod()
        curve["osebx_daily_return"] = osebx_returns

    return curve, pd.DataFrame(ranking_rows)


def technical_strategy_specs() -> list[dict[str, int | float | str]]:
    specs = []
    volume_strategies = {
        "volume_surge",
        "momentum_volume",
        "breakout_volume",
        "obv_accumulation",
        "risk_adjusted_momentum",
        "volume_reversal",
        "volume_trend_confirmation",
        "quiet_accumulation",
    }
    for strategy in TECHNICAL_STRATEGIES:
        volume_windows = [(5, 20), (10, 50), (20, 100)] if strategy in volume_strategies else [(10, 50)]
        breakout_windows = [63, 126] if "breakout" in strategy else [63]
        for price_lookback in [21, 42, 63, 126, 189]:
            for volume_fast, volume_slow in volume_windows:
                for breakout_lookback in breakout_windows:
                    specs.append(
                        {
                            "strategy": strategy,
                            "price_lookback": price_lookback,
                            "short_lookback": 21,
                            "volume_fast": volume_fast,
                            "volume_slow": volume_slow,
                            "breakout_lookback": breakout_lookback,
                            "ma_days": 50,
                            "risk_lookback": 63,
                            "accumulation_lookback": 63,
                            "liquidity_lookback": 20,
                            "min_liquidity_quantile": 0.25,
                        }
                    )
    return specs


def run_technical_sweep(
    market_data: dict[str, pd.DataFrame],
    benchmark_close: pd.Series | None,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    best_curve = pd.DataFrame()
    best_ranking = pd.DataFrame()
    best_return = -math.inf

    for spec in technical_strategy_specs():
        strategy = str(spec["strategy"])
        params = {key: value for key, value in spec.items() if key != "strategy"}
        if technical_min_history(params) >= len(market_data["close"]) - 30:
            continue
        for top_n in [3, 5, 8, 10]:
            curve, ranking = run_ranked_technical_backtest(
                market_data=market_data,
                strategy=strategy,
                params=params,
                top_n=top_n,
                cost_bps=cost_bps,
                benchmark_close=benchmark_close,
            )
            summary = summarize(curve).set_index("portfolio")
            strategy_summary = summary.loc["strategy"]
            row = {
                **spec,
                "top_n": top_n,
                "strategy_total_return": strategy_summary["total_return"],
                "strategy_annualized_return": strategy_summary["annualized_return"],
                "strategy_max_drawdown": strategy_summary["max_drawdown"],
                "strategy_sharpe": strategy_summary["sharpe"],
                "equal_weight_total_return": summary.loc["equal_weight_benchmark", "total_return"],
            }
            if "osebx" in summary.index:
                row["osebx_total_return"] = summary.loc["osebx", "total_return"]
                row["excess_vs_osebx"] = row["strategy_total_return"] - row["osebx_total_return"]
            rows.append(row)

            if row["strategy_total_return"] > best_return:
                best_return = float(row["strategy_total_return"])
                best_curve = curve.copy()
                best_ranking = ranking.copy()
                best_curve.attrs["best_spec"] = row

    results = pd.DataFrame(rows).sort_values("strategy_total_return", ascending=False)
    return results, best_curve, best_ranking


def best_result_per_strategy(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    return (
        results.sort_values("strategy_total_return", ascending=False)
        .groupby("strategy", as_index=False)
        .head(1)
        .sort_values("strategy_total_return", ascending=False)
    )


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return math.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return math.nan
    return float(equity.iloc[-1] ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def sharpe_ratio(daily_returns: pd.Series) -> float:
    std = daily_returns.std()
    if std == 0 or pd.isna(std):
        return math.nan
    return float((daily_returns.mean() / std) * math.sqrt(252))


def summarize(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    portfolios = [
        ("strategy", "strategy", "strategy_daily_return"),
        ("equal_weight_benchmark", "equal_weight_benchmark", "benchmark_daily_return"),
    ]
    if "osebx" in curve.columns:
        portfolios.append(("osebx", "osebx", "osebx_daily_return"))

    for label, equity_col, return_col in portfolios:
        equity = curve[equity_col]
        daily_returns = curve[return_col]
        rows.append(
            {
                "portfolio": label,
                "start": equity.index[0].date(),
                "end": equity.index[-1].date(),
                "days": len(equity),
                "total_return": equity.iloc[-1] - 1.0,
                "annualized_return": annualized_return(equity),
                "max_drawdown": max_drawdown(equity),
                "sharpe": sharpe_ratio(daily_returns),
            }
        )
    return pd.DataFrame(rows)


def run_parameter_sweep(
    close: pd.DataFrame,
    benchmark_close: pd.Series | None,
    cost_bps: float,
) -> pd.DataFrame:
    rows = []
    lookbacks = [21, 42, 63, 126, 189, 252]
    top_ns = [3, 5, 8, 10]

    for lookback_days in lookbacks:
        if lookback_days >= len(close) - 30:
            continue
        for top_n in top_ns:
            curve, _ = run_momentum_backtest(
                close=close,
                lookback_days=lookback_days,
                top_n=top_n,
                cost_bps=cost_bps,
                benchmark_close=benchmark_close,
            )
            summary = summarize(curve).set_index("portfolio")
            strategy = summary.loc["strategy"]
            row = {
                "lookback_days": lookback_days,
                "top_n": top_n,
                "strategy_total_return": strategy["total_return"],
                "strategy_annualized_return": strategy["annualized_return"],
                "strategy_max_drawdown": strategy["max_drawdown"],
                "strategy_sharpe": strategy["sharpe"],
            }
            if "equal_weight_benchmark" in summary.index:
                row["equal_weight_total_return"] = summary.loc["equal_weight_benchmark", "total_return"]
            if "osebx" in summary.index:
                row["osebx_total_return"] = summary.loc["osebx", "total_return"]
            rows.append(row)

    return pd.DataFrame(rows).sort_values("strategy_total_return", ascending=False)


def save_plot(curve: pd.DataFrame, output_file: Path, title: str = "Momentum strategy vs equal-weight benchmark") -> bool:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plot_cols = [col for col in ["strategy", "equal_weight_benchmark", "osebx"] if col in curve.columns]
    ax = curve[plot_cols].plot(figsize=(10, 5), grid=True)
    ax.set_title(title)
    ax.set_ylabel("Growth of 1.0")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    return True


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gratis Oslo Bors data + momentum-backtest.")
    parser.add_argument("--refresh", action="store_true", help="Last ned data pa nytt selv om cache finnes.")
    parser.add_argument("--lookback-days", type=int, default=63, help="Momentum lookback i handelsdager.")
    parser.add_argument("--top-n", type=int, default=3, help="Antall aksjer i portefoljen.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Transaksjonskostnad i basispunkter per turnover.")
    parser.add_argument("--min-history-days", type=int, default=250, help="Minimum antall prisdager per aksje.")
    parser.add_argument("--sweep", action="store_true", help="Test flere lookback/top-n-kombinasjoner.")
    parser.add_argument("--technical-sweep", action="store_true", help="Test volum og andre tekniske strategier.")
    args = parser.parse_args()

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()
    queries = load_queries(UNIVERSE_FILE)

    instruments = []
    failures = []
    for query in queries:
        try:
            instruments.append(resolve_instrument(session, query))
            time.sleep(0.1)
        except Exception as exc:
            failures.append({"query": query, "stage": "resolve", "error": str(exc)})

    frames = []
    for instrument in instruments:
        try:
            frames.append(fetch_history(session, instrument, refresh=args.refresh))
        except Exception as exc:
            failures.append(
                {
                    "query": instrument.query,
                    "stage": "download",
                    "error": str(exc),
                    "instrument_id": instrument.instrument_id,
                }
            )

    try:
        osebx_frame = fetch_history(session, OSEBX_INSTRUMENT, refresh=args.refresh)
        benchmark_close = osebx_frame.set_index("date")["close"].rename("OSEBX")
    except Exception as exc:
        benchmark_close = None
        failures.append({"query": "OSEBX", "stage": "download", "error": str(exc), "instrument_id": "NO0007035327-XOSL"})

    failures_file = OUTPUT_DIR / "failures.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failures_file, index=False)
    elif failures_file.exists():
        failures_file.unlink()

    market_data = build_market_data(frames, min_history_days=args.min_history_days)
    close = market_data["close"]
    close.to_csv(OUTPUT_DIR / "close_prices.csv")

    curve, ranking = run_momentum_backtest(
        close=close,
        lookback_days=args.lookback_days,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        benchmark_close=benchmark_close,
    )
    summary = summarize(curve)

    curve.to_csv(OUTPUT_DIR / "equity_curve.csv")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    if not ranking.empty:
        latest_date = ranking["rebalance_date"].max()
        latest = ranking[ranking["rebalance_date"] == latest_date].sort_values("momentum", ascending=False)
        latest.to_csv(OUTPUT_DIR / "latest_ranking.csv", index=False)

    plot_created = save_plot(curve, OUTPUT_DIR / "equity_curve.png")

    print(f"Instrumenter lost: {len(instruments)}")
    print(f"Instrumenter med nok prisdata: {close.shape[1]}")
    print(f"Prisrad fra {close.index.min().date()} til {close.index.max().date()}")
    if failures:
        print(f"Feil: {len(failures)} - se output/failures.csv")
    print("")
    for _, row in summary.iterrows():
        print(
            f"{row['portfolio']}: total {format_pct(row['total_return'])}, "
            f"annualisert {format_pct(row['annualized_return'])}, "
            f"max drawdown {format_pct(row['max_drawdown'])}, "
            f"Sharpe {row['sharpe']:.2f}"
        )
    print("")
    print("Skrev output/summary.csv, output/equity_curve.csv og output/latest_ranking.csv")
    if plot_created:
        print("Skrev output/equity_curve.png")

    if args.sweep:
        sweep = run_parameter_sweep(close, benchmark_close, args.cost_bps)
        sweep.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
        print("")
        print("Beste parameterkombinasjoner i denne korte perioden:")
        for _, row in sweep.head(5).iterrows():
            print(
                f"lookback {int(row['lookback_days'])}, top {int(row['top_n'])}: "
                f"total {format_pct(row['strategy_total_return'])}, "
                f"drawdown {format_pct(row['strategy_max_drawdown'])}, "
                f"Sharpe {row['strategy_sharpe']:.2f}"
            )
        print("Skrev output/sweep_results.csv")

    if args.technical_sweep:
        technical_results, best_curve, best_ranking = run_technical_sweep(market_data, benchmark_close, args.cost_bps)
        technical_results.to_csv(OUTPUT_DIR / "technical_sweep_results.csv", index=False)
        best_by_strategy = best_result_per_strategy(technical_results)
        best_by_strategy.to_csv(OUTPUT_DIR / "technical_best_by_strategy.csv", index=False)
        if not best_curve.empty:
            best_curve.to_csv(OUTPUT_DIR / "best_technical_equity_curve.csv")
            save_plot(
                best_curve,
                OUTPUT_DIR / "best_technical_equity_curve.png",
                title="Best technical strategy vs benchmarks",
            )
        if not best_ranking.empty:
            latest_date = best_ranking["rebalance_date"].max()
            best_ranking[best_ranking["rebalance_date"] == latest_date].to_csv(
                OUTPUT_DIR / "best_technical_latest_ranking.csv", index=False
            )

        print("")
        print("Beste volum/tekniske strategier i denne korte perioden:")
        for _, row in technical_results.head(10).iterrows():
            print(
                f"{row['strategy']}, lookback {int(row['price_lookback'])}, "
                f"vol {int(row['volume_fast'])}/{int(row['volume_slow'])}, "
                f"top {int(row['top_n'])}: total {format_pct(row['strategy_total_return'])}, "
                f"drawdown {format_pct(row['strategy_max_drawdown'])}, "
                f"Sharpe {row['strategy_sharpe']:.2f}"
            )
        print("Skrev output/technical_sweep_results.csv og output/technical_best_by_strategy.csv")
        if not best_curve.empty:
            print("Skrev output/best_technical_equity_curve.csv og output/best_technical_equity_curve.png")


if __name__ == "__main__":
    main()
