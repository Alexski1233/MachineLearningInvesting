"""Point-in-time portfolio backtesting with explicit execution and costs."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final, cast

import numpy as np
import pandas as pd

from .config import BacktestConfig


TRADING_DAYS_PER_YEAR: Final[int] = 252
_EPS: Final[float] = 1e-10


@dataclass(frozen=True)
class BacktestMetrics:
    """Headline net performance, risk, turnover, and cost statistics."""

    net_total_return: float
    net_cagr: float
    daily_excess_sharpe: float
    max_drawdown: float
    turnover: float
    annualized_turnover: float
    total_costs: float
    total_cost_rate: float
    ending_equity: float
    trading_days: int

    def as_dict(self) -> dict[str, float | int]:
        """Return metrics in a serialization-friendly representation."""
        return {
            "net_total_return": self.net_total_return,
            "net_cagr": self.net_cagr,
            "daily_excess_sharpe": self.daily_excess_sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "annualized_turnover": self.annualized_turnover,
            "total_costs": self.total_costs,
            "total_cost_rate": self.total_cost_rate,
            "ending_equity": self.ending_equity,
            "trading_days": self.trading_days,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Complete auditable result of :func:`run_backtest`."""

    daily_equity: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    selections: pd.DataFrame
    metrics: BacktestMetrics

    @property
    def equity(self) -> pd.DataFrame:
        """Backward-friendly alias for ``daily_equity``."""
        return self.daily_equity

    @property
    def signal_selections(self) -> pd.DataFrame:
        """Descriptive alias for ``selections``."""
        return self.selections


@dataclass
class _Position:
    units: float
    last_mark: float
    stale_days: int = 0


@dataclass(frozen=True)
class _Bar:
    open: float
    close: float
    median_dollar_volume: float
    delisting_return: float


_DAILY_COLUMNS = [
    "date",
    "cash",
    "holdings_value",
    "equity",
    "daily_return",
    "risk_free_daily_return",
    "excess_return",
    "cumulative_return",
    "traded_notional",
    "costs",
    "recovery_loss",
    "one_way_turnover",
]
_TRADE_COLUMNS = [
    "signal_date",
    "date",
    "ticker",
    "side",
    "units",
    "price",
    "notional",
    "median_dollar_volume",
    "participation",
    "cost_bps",
    "cost",
    "desired_notional",
    "capacity_limited",
    "cash_after",
]
_HOLDING_COLUMNS = [
    "date",
    "ticker",
    "units",
    "mark_price",
    "market_value",
    "weight",
    "stale_days",
    "mark_status",
]
_SELECTION_COLUMNS = [
    "signal_date",
    "execution_date",
    "ticker",
    "rank",
    "signal",
    "expected_return",
    "volatility",
    "median_dollar_volume",
    "threshold_eligible",
    "retention_eligible",
    "entry_threshold",
    "retention_threshold",
    "incumbent",
    "selected",
    "retained_by_buffer",
    "target_weight",
    "execution_price_available",
]


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a net, long-only backtest without using future returns.

    A signal stamped on date ``t`` is treated as known after that close and is
    executed only at adjusted open on the first market date strictly after
    ``t``. Required signal columns are ``date``, ``ticker``, and one of
    ``score``, ``expected_return``, ``horizon_sessions``, and
    ``model_refit_date``. The score is used only for ranking; expected return
    must be an absolute return forecast in holding-period units.
    Signal volatility and trailing median dollar volume may be supplied; when
    absent, they are derived solely from prices available through the signal
    close. Price data require ``date``, ``ticker``, volume, adjusted close, and
    adjusted open (or open/close plus adjusted close, from which adjusted open
    is derived).
    """
    px = _prepare_prices(prices)
    sig = _prepare_signals(signals, px)
    if px.empty or sig.empty:
        return _empty_result(config.initial_capital)

    market_dates = pd.DatetimeIndex(px["date"].drop_duplicates().sort_values())
    schedule: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in sig["date"].drop_duplicates().sort_values():
        location = market_dates.searchsorted(signal_date, side="right")
        if location < len(market_dates):
            execution_date = market_dates[location]
            previous = schedule.get(execution_date)
            if previous is None or signal_date > previous:
                schedule[execution_date] = signal_date
    if not schedule:
        return _empty_result(config.initial_capital)

    bars = {
        (row["date"], row["ticker"]): _Bar(
            open=float(row["_adj_open"]),
            close=float(row["_adj_close"]),
            median_dollar_volume=float(row["_median_dollar_volume"]),
            delisting_return=float(row["delisting_return"]),
        )
        for _, row in px.iterrows()
    }
    by_ticker = {ticker: frame for ticker, frame in px.groupby("ticker", sort=False)}
    simulation_dates = market_dates[market_dates >= min(schedule)]
    engine = _Engine(config, sig, bars, by_ticker)
    return engine.run(simulation_dates, schedule)


def expected_return_hurdle(
    horizon_sessions: int,
    config: BacktestConfig,
) -> float:
    """Return the minimum absolute forecast needed for a new position."""
    if (
        isinstance(horizon_sessions, bool)
        or not isinstance(horizon_sessions, (int, np.integer))
        or horizon_sessions < 1
    ):
        raise ValueError("horizon_sessions must be a positive integer.")
    one_way_cost = (
        config.commission_bps + config.half_spread_bps + config.impact_bps * sqrt(config.max_participation_rate)
    ) / 10_000
    risk_free_horizon = (1 + config.annual_risk_free_rate) ** (horizon_sessions / TRADING_DAYS_PER_YEAR) - 1
    return risk_free_horizon + 2 * one_way_cost + config.minimum_expected_edge


class _Engine:
    def __init__(
        self,
        config: BacktestConfig,
        signals: pd.DataFrame,
        bars: dict[tuple[pd.Timestamp, str], _Bar],
        prices_by_ticker: dict[str, pd.DataFrame],
    ) -> None:
        self.config = config
        self.signals = signals
        self.bars = bars
        self.prices_by_ticker = prices_by_ticker
        self.cash = float(config.initial_capital)
        self.positions: dict[str, _Position] = {}
        self.trade_rows: list[dict[str, object]] = []
        self.selection_rows: list[dict[str, object]] = []
        self.holding_rows: list[dict[str, object]] = []
        self.daily_rows: list[dict[str, object]] = []

    def run(
        self,
        dates: pd.DatetimeIndex,
        schedule: dict[pd.Timestamp, pd.Timestamp],
    ) -> BacktestResult:
        daily_rf = (1 + self.config.annual_risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        previous_equity = float(self.config.initial_capital)
        for date in dates:
            self.cash *= 1 + daily_rf
            pretrade_equity = previous_equity
            if date in schedule:
                pretrade_equity = self._rebalance(schedule[date], date)

            recovery_loss = self._mark_and_resolve(date)
            marked = [row for row in self.holding_rows if row["date"] == date]
            holdings_value = sum(cast(float, row["market_value"]) for row in marked)
            equity = self.cash + holdings_value
            for row in marked:
                row["weight"] = cast(float, row["market_value"]) / equity if equity > 0 else np.nan

            regular = [row for row in self.trade_rows if row["date"] == date and row["side"] in {"buy", "sell"}]
            buys = sum(cast(float, row["notional"]) for row in regular if row["side"] == "buy")
            sells = sum(cast(float, row["notional"]) for row in regular if row["side"] == "sell")
            traded = buys + sells
            costs = sum(cast(float, row["cost"]) for row in regular)
            daily_return = equity / previous_equity - 1 if previous_equity else np.nan
            one_way_turnover = max(buys, sells) / pretrade_equity if pretrade_equity > 0 else np.nan
            self.daily_rows.append(
                {
                    "date": date,
                    "cash": self.cash,
                    "holdings_value": holdings_value,
                    "equity": equity,
                    "daily_return": daily_return,
                    "risk_free_daily_return": daily_rf,
                    "excess_return": daily_return - daily_rf,
                    "cumulative_return": equity / self.config.initial_capital - 1,
                    "traded_notional": traded,
                    "costs": costs,
                    "recovery_loss": recovery_loss,
                    "one_way_turnover": one_way_turnover,
                }
            )
            previous_equity = equity

        daily = pd.DataFrame(self.daily_rows, columns=_DAILY_COLUMNS)
        trades = pd.DataFrame(self.trade_rows, columns=_TRADE_COLUMNS)
        holdings = pd.DataFrame(self.holding_rows, columns=_HOLDING_COLUMNS)
        selections = pd.DataFrame(self.selection_rows, columns=_SELECTION_COLUMNS)
        return BacktestResult(daily, trades, holdings, selections, _metrics(daily, trades, self.config))

    def _rebalance(self, signal_date: pd.Timestamp, execution_date: pd.Timestamp) -> float:
        snapshot = self.signals[self.signals["date"] == signal_date].copy()
        snapshot = snapshot.sort_values(["_signal", "ticker"], ascending=[False, True])
        snapshot["_rank"] = np.arange(1, len(snapshot) + 1)
        risk_free_horizon = (1 + self.config.annual_risk_free_rate) ** (
            snapshot["horizon_sessions"] / TRADING_DAYS_PER_YEAR
        ) - 1
        threshold = snapshot["horizon_sessions"].map(lambda horizon: expected_return_hurdle(int(horizon), self.config))
        snapshot["_entry_threshold"] = threshold
        snapshot["_retention_threshold"] = risk_free_horizon
        snapshot["_entry_eligible"] = (
            (snapshot["_expected_return"] > snapshot["_entry_threshold"])
            & np.isfinite(snapshot["_volatility"])
            & (snapshot["_volatility"] > 0)
        )
        snapshot["_retention_eligible"] = (
            (snapshot["_expected_return"] > snapshot["_retention_threshold"])
            & np.isfinite(snapshot["_volatility"])
            & (snapshot["_volatility"] > 0)
        )
        incumbents = set(self.positions)
        retained = snapshot[
            snapshot["_retention_eligible"]
            & snapshot["ticker"].isin(incumbents)
            & (snapshot["_rank"] <= self.config.buffer_n)
        ]["ticker"].tolist()[: self.config.top_n]
        selected = list(retained)
        for ticker in snapshot.loc[snapshot["_entry_eligible"], "ticker"]:
            if len(selected) >= self.config.top_n:
                break
            if ticker not in selected:
                selected.append(ticker)

        selected_frame = snapshot[snapshot["ticker"].isin(selected)]
        volatilities = dict(zip(selected_frame["ticker"], selected_frame["_volatility"], strict=True))
        if self.config.weighting_method == "equal":
            weights = _equal_weights(tuple(volatilities), self.config.max_weight)
        else:
            weights = _capped_inverse_volatility(volatilities, self.config.max_weight)
        open_values = {
            ticker: position.units * self._execution_price(execution_date, ticker)
            for ticker, position in self.positions.items()
            if np.isfinite(self._execution_price(execution_date, ticker))
        }
        pretrade_equity = self.cash + sum(
            open_values.get(ticker, position.units * position.last_mark) for ticker, position in self.positions.items()
        )

        for _, row in snapshot.iterrows():
            ticker = str(row["ticker"])
            rank = int(row["_rank"])
            is_selected = ticker in selected
            self.selection_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "ticker": ticker,
                    "rank": rank,
                    "signal": float(row["_signal"]),
                    "expected_return": float(row["_expected_return"]),
                    "volatility": float(row["_volatility"]),
                    "median_dollar_volume": float(row["_median_dollar_volume"]),
                    "threshold_eligible": bool(row["_entry_eligible"]),
                    "retention_eligible": bool(row["_retention_eligible"]),
                    "entry_threshold": float(row["_entry_threshold"]),
                    "retention_threshold": float(row["_retention_threshold"]),
                    "incumbent": ticker in incumbents,
                    "selected": is_selected,
                    "retained_by_buffer": is_selected and ticker in incumbents and rank > self.config.top_n,
                    "target_weight": weights.get(ticker, 0.0),
                    "execution_price_available": np.isfinite(self._execution_price(execution_date, ticker)),
                }
            )

        target_values = {ticker: weight * pretrade_equity for ticker, weight in weights.items()}
        orders: list[tuple[str, float, float, float]] = []
        all_tickers = set(self.positions) | set(target_values)
        liquidity = dict(zip(snapshot["ticker"], snapshot["_median_dollar_volume"], strict=True))
        for ticker in sorted(all_tickers):
            price = self._execution_price(execution_date, ticker)
            if not np.isfinite(price):
                continue
            current = self.positions.get(ticker)
            current_value = 0.0 if current is None else current.units * price
            delta = target_values.get(ticker, 0.0) - current_value
            if abs(delta) < self.config.rebalance_tolerance * pretrade_equity:
                continue
            median = float(liquidity.get(ticker, self._asof_liquidity(ticker, signal_date)))
            orders.append((ticker, delta, price, median))

        for ticker, delta, price, median in sorted(orders, key=lambda item: item[1]):
            if delta < -_EPS:
                self._trade(signal_date, execution_date, ticker, delta, price, median)
        for ticker, delta, price, median in sorted(orders, key=lambda item: item[1], reverse=True):
            if delta > _EPS:
                self._trade(signal_date, execution_date, ticker, delta, price, median)
        return float(pretrade_equity)

    def _trade(
        self,
        signal_date: pd.Timestamp,
        date: pd.Timestamp,
        ticker: str,
        desired_delta: float,
        price: float,
        median_dollar_volume: float,
    ) -> None:
        if not np.isfinite(median_dollar_volume) or median_dollar_volume <= 0:
            return
        capacity = self.config.max_participation_rate * median_dollar_volume
        notional = min(abs(desired_delta), capacity)
        side = "buy" if desired_delta > 0 else "sell"
        if side == "sell":
            position = self.positions.get(ticker)
            if position is None:
                return
            notional = min(notional, position.units * price)
        else:
            notional = self._affordable_notional(notional, median_dollar_volume)
        if notional <= _EPS:
            return

        participation = notional / median_dollar_volume
        cost_bps = self._cost_bps(participation)
        cost = notional * cost_bps / 10_000
        units = notional / price
        if side == "buy":
            self.cash -= notional + cost
            position = self.positions.get(ticker)
            if position is None:
                self.positions[ticker] = _Position(units=units, last_mark=price)
            else:
                position.units += units
                position.last_mark = price
                position.stale_days = 0
        else:
            self.cash += notional - cost
            position = self.positions[ticker]
            position.units -= units
            if position.units <= _EPS:
                del self.positions[ticker]
        if abs(self.cash) < _EPS:
            self.cash = 0.0
        self.trade_rows.append(
            {
                "signal_date": signal_date,
                "date": date,
                "ticker": ticker,
                "side": side,
                "units": units,
                "price": price,
                "notional": notional,
                "median_dollar_volume": median_dollar_volume,
                "participation": participation,
                "cost_bps": cost_bps,
                "cost": cost,
                "desired_notional": abs(desired_delta),
                "capacity_limited": notional + _EPS < abs(desired_delta),
                "cash_after": self.cash,
            }
        )

    def _mark_and_resolve(self, date: pd.Timestamp) -> float:
        recovery_loss = 0.0
        for ticker in sorted(list(self.positions)):
            position = self.positions[ticker]
            bar = self.bars.get((date, ticker))
            mark_status = "carried"
            if bar is not None and np.isfinite(bar.close):
                position.last_mark = bar.close
                position.stale_days = 0
                mark_status = "close"
            elif bar is not None and np.isfinite(bar.open):
                position.last_mark = bar.open
                position.stale_days = 0
                mark_status = "open"
            else:
                position.stale_days += 1

            if bar is not None and np.isfinite(bar.delisting_return):
                value = position.units * position.last_mark
                proceeds = max(0.0, value * (1 + bar.delisting_return))
                recovery_loss += value - proceeds
                self.cash += proceeds
                self._forced_exit(date, ticker, position, proceeds, "delisting")
                del self.positions[ticker]
                continue
            if position.stale_days > self.config.max_stale_days:
                value = position.units * position.last_mark
                proceeds = value * self.config.stale_position_recovery
                recovery_loss += value - proceeds
                self.cash += proceeds
                self._forced_exit(date, ticker, position, proceeds, "stale_liquidation")
                del self.positions[ticker]
                continue

            market_value = position.units * position.last_mark
            self.holding_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "units": position.units,
                    "mark_price": position.last_mark,
                    "market_value": market_value,
                    "weight": np.nan,
                    "stale_days": position.stale_days,
                    "mark_status": mark_status,
                }
            )
        return recovery_loss

    def _forced_exit(
        self,
        date: pd.Timestamp,
        ticker: str,
        position: _Position,
        proceeds: float,
        side: str,
    ) -> None:
        price = proceeds / position.units if position.units > 0 else 0.0
        self.trade_rows.append(
            {
                "signal_date": pd.NaT,
                "date": date,
                "ticker": ticker,
                "side": side,
                "units": position.units,
                "price": price,
                "notional": proceeds,
                "median_dollar_volume": np.nan,
                "participation": np.nan,
                "cost_bps": 0.0,
                "cost": 0.0,
                "desired_notional": proceeds,
                "capacity_limited": False,
                "cash_after": self.cash,
            }
        )

    def _execution_price(self, date: pd.Timestamp, ticker: str) -> float:
        bar = self.bars.get((date, ticker))
        return np.nan if bar is None else bar.open

    def _asof_liquidity(self, ticker: str, date: pd.Timestamp) -> float:
        frame = self.prices_by_ticker.get(ticker)
        if frame is None:
            return np.nan
        known = frame[frame["date"] <= date]
        return np.nan if known.empty else float(known.iloc[-1]["_median_dollar_volume"])

    def _cost_bps(self, participation: float) -> float:
        return (
            self.config.commission_bps
            + self.config.half_spread_bps
            + self.config.impact_bps * sqrt(max(participation, 0.0))
        )

    def _affordable_notional(self, requested: float, median_dollar_volume: float) -> float:
        if requested <= 0 or self.cash <= 0:
            return 0.0
        low, high = 0.0, min(requested, self.cash)
        for _ in range(40):
            mid = (low + high) / 2
            rate = self._cost_bps(mid / median_dollar_volume) / 10_000
            if mid * (1 + rate) <= self.cash:
                low = mid
            else:
                high = mid
        return low


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")
    out = prices.copy()
    out["date"] = _normalise_dates(out["date"])
    out["ticker"] = _normalise_tickers(out["ticker"], "prices")
    if out.duplicated(["ticker", "date"]).any():
        raise ValueError("prices must contain at most one row per ticker/date.")
    if "exchange" in out.columns:
        exchanges = out["exchange"].astype("string").str.strip()
        if exchanges.isna().any() or exchanges.eq("").any():
            raise ValueError("prices exchange values cannot be missing or blank.")
        if exchanges.nunique() > 1:
            raise ValueError("run_backtest supports one exchange calendar per run.")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    close_col = _first_column(out, ["adj_close", "adjusted_close", "close"])
    if close_col is None:
        raise ValueError("prices require adjusted close or close.")
    open_col = _first_column(out, ["adj_open", "adjusted_open"])
    out["_adj_close"] = pd.to_numeric(out[close_col], errors="coerce")
    if open_col is not None:
        out["_adj_open"] = pd.to_numeric(out[open_col], errors="coerce")
    elif {"open", "close", close_col}.issubset(out.columns):
        raw_open = pd.to_numeric(out["open"], errors="coerce")
        raw_close = pd.to_numeric(out["close"], errors="coerce")
        out["_adj_open"] = raw_open * out["_adj_close"] / raw_close.replace(0, np.nan)
    elif "open" in out:
        out["_adj_open"] = pd.to_numeric(out["open"], errors="coerce")
    else:
        raise ValueError("prices require adjusted open, or open/close plus adjusted close.")
    out.loc[out["_adj_open"] <= 0, "_adj_open"] = np.nan
    out.loc[out["_adj_close"] <= 0, "_adj_close"] = np.nan

    liquidity_col = _first_column(
        out,
        ["median_dollar_volume", "trailing_median_dollar_volume", "turnover_60d_median", "dollar_volume_60d_median"],
    )
    if liquidity_col is not None:
        out["_median_dollar_volume"] = pd.to_numeric(out[liquidity_col], errors="coerce")
    else:
        if "volume" not in out:
            raise ValueError("prices require volume or trailing median dollar volume.")
        volume = pd.to_numeric(out["volume"], errors="coerce")
        raw_price = pd.to_numeric(out["close"], errors="coerce") if "close" in out.columns else out["_adj_close"]
        dollar_volume = raw_price * volume
        out["_median_dollar_volume"] = dollar_volume.groupby(out["ticker"]).transform(
            lambda series: series.rolling(60, min_periods=1).median()
        )
    if "delisting_return" not in out:
        out["delisting_return"] = np.nan
    raw_delisting = out["delisting_return"]
    missing_delisting = raw_delisting.isna() | raw_delisting.astype("string").str.strip().eq("")
    out["delisting_return"] = pd.to_numeric(raw_delisting, errors="coerce")
    invalid_delisting = (~missing_delisting & out["delisting_return"].isna()) | (
        out["delisting_return"].notna() & ((out["delisting_return"] < -1) | ~np.isfinite(out["delisting_return"]))
    )
    if invalid_delisting.any():
        raise ValueError("delisting_return must be finite and no smaller than -1.")
    returns = out.groupby("ticker")["_adj_close"].pct_change(fill_method=None)
    out["_derived_volatility"] = returns.groupby(out["ticker"]).transform(
        lambda series: series.rolling(20, min_periods=2).std()
    )
    return out


def _prepare_signals(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "score",
        "expected_return",
        "horizon_sessions",
        "model_refit_date",
    }
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")
    out = signals.copy()
    out["date"] = _normalise_dates(out["date"])
    out["ticker"] = _normalise_tickers(out["ticker"], "signals")
    if out.duplicated(["ticker", "date"]).any():
        raise ValueError("signals must contain at most one row per ticker/date.")
    out["_signal"] = pd.to_numeric(out["score"], errors="coerce")
    out["_expected_return"] = pd.to_numeric(out["expected_return"], errors="coerce")
    horizon = pd.to_numeric(out["horizon_sessions"], errors="coerce")
    valid_horizon = np.isfinite(horizon) & horizon.gt(0) & horizon.eq(np.floor(horizon))
    if not valid_horizon.all():
        raise ValueError("horizon_sessions must contain positive integers.")
    out["horizon_sessions"] = horizon.astype(int)
    out["model_refit_date"] = _normalise_dates(out["model_refit_date"])
    if (out["model_refit_date"] > out["date"]).any():
        raise ValueError("model_refit_date cannot be later than the signal date.")
    volatility_col = _first_column(out, ["volatility", "vol_20d", "vol_60d", "realized_volatility"])
    liquidity_col = _first_column(
        out,
        ["median_dollar_volume", "trailing_median_dollar_volume", "turnover_60d_median", "dollar_volume_60d_median"],
    )
    if volatility_col is not None:
        out["_volatility"] = pd.to_numeric(out[volatility_col], errors="coerce")
    else:
        out["_volatility"] = _point_in_time_lookup(out, prices, "_derived_volatility")
    if liquidity_col is not None:
        out["_median_dollar_volume"] = pd.to_numeric(out[liquidity_col], errors="coerce")
    else:
        out["_median_dollar_volume"] = _point_in_time_lookup(out, prices, "_median_dollar_volume")
    finite = np.isfinite(out["_signal"]) & np.isfinite(out["_expected_return"])
    return out[finite].sort_values(["date", "ticker"]).reset_index(drop=True)


def _point_in_time_lookup(signals: pd.DataFrame, prices: pd.DataFrame, column: str) -> pd.Series:
    values: list[float] = []
    groups = {ticker: frame for ticker, frame in prices.groupby("ticker", sort=False)}
    for row in signals[["ticker", "date"]].itertuples(index=False):
        frame = groups.get(row.ticker)
        known = None if frame is None else frame[frame["date"] <= row.date]
        values.append(np.nan if known is None or known.empty else float(known.iloc[-1][column]))
    return pd.Series(values, index=signals.index, dtype=float)


def _capped_inverse_volatility(volatilities: dict[str, float], cap: float) -> dict[str, float]:
    if not volatilities:
        return {}
    inverse = {ticker: 1 / value for ticker, value in volatilities.items() if np.isfinite(value) and value > 0}
    weights = {ticker: 0.0 for ticker in inverse}
    remaining = set(inverse)
    remaining_mass = min(1.0, len(remaining) * cap)
    while remaining and remaining_mass > _EPS:
        total_inverse = sum(inverse[ticker] for ticker in remaining)
        proposed = {ticker: remaining_mass * inverse[ticker] / total_inverse for ticker in remaining}
        capped = {ticker for ticker, weight in proposed.items() if weight > cap + _EPS}
        if not capped:
            weights.update(proposed)
            break
        for ticker in capped:
            weights[ticker] = cap
            remaining_mass -= cap
            remaining.remove(ticker)
    return weights


def _equal_weights(tickers: tuple[str, ...], cap: float) -> dict[str, float]:
    if not tickers:
        return {}
    weight = min(1 / len(tickers), cap)
    return {ticker: weight for ticker in tickers}


def _metrics(daily: pd.DataFrame, trades: pd.DataFrame, config: BacktestConfig) -> BacktestMetrics:
    if daily.empty:
        return _empty_metrics(config.initial_capital)
    ending = float(daily["equity"].iloc[-1])
    total_return = ending / config.initial_capital - 1
    elapsed_days = max((daily["date"].iloc[-1] - daily["date"].iloc[0]).days + 1, 1)
    years = elapsed_days / 365.2425
    cagr = (ending / config.initial_capital) ** (1 / years) - 1 if ending > 0 else -1.0
    excess = daily["excess_return"].dropna()
    std = excess.std(ddof=1)
    sharpe = float(excess.mean() / std * sqrt(TRADING_DAYS_PER_YEAR)) if len(excess) > 1 and std > 0 else np.nan
    curve = pd.Series([config.initial_capital, *daily["equity"].tolist()], dtype=float)
    max_drawdown = float((curve / curve.cummax() - 1).min())
    turnover = float(daily["one_way_turnover"].fillna(0).sum())
    costs = float(trades.loc[trades["side"].isin(["buy", "sell"]), "cost"].sum()) if not trades.empty else 0.0
    return BacktestMetrics(
        net_total_return=float(total_return),
        net_cagr=float(cagr),
        daily_excess_sharpe=sharpe,
        max_drawdown=max_drawdown,
        turnover=turnover,
        annualized_turnover=turnover / years,
        total_costs=costs,
        total_cost_rate=costs / config.initial_capital,
        ending_equity=ending,
        trading_days=len(daily),
    )


def _empty_metrics(initial_capital: float) -> BacktestMetrics:
    return BacktestMetrics(0.0, np.nan, np.nan, 0.0, 0.0, np.nan, 0.0, 0.0, initial_capital, 0)


def _empty_result(initial_capital: float) -> BacktestResult:
    return BacktestResult(
        pd.DataFrame(columns=_DAILY_COLUMNS),
        pd.DataFrame(columns=_TRADE_COLUMNS),
        pd.DataFrame(columns=_HOLDING_COLUMNS),
        pd.DataFrame(columns=_SELECTION_COLUMNS),
        _empty_metrics(initial_capital),
    )


def _first_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _normalise_dates(values: pd.Series) -> pd.Series:
    normalized: list[pd.Timestamp] = []
    for value in values:
        if pd.isna(value):
            raise ValueError("Dates cannot be missing.")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid date value: {value!r}.") from error
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)
        normalized.append(timestamp.normalize())
    return pd.Series(normalized, index=values.index, dtype="datetime64[ns]")


def _normalise_tickers(values: pd.Series, source: str) -> pd.Series:
    tickers = values.astype("string").str.strip()
    if tickers.isna().any() or tickers.eq("").any():
        raise ValueError(f"{source} ticker values cannot be missing or blank.")
    return tickers.astype(str)


__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "expected_return_hurdle",
    "run_backtest",
]
