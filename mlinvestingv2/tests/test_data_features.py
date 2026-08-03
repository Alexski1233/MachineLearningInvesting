from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlinvesting.config import FeatureConfig
from mlinvesting.data import attach_universe_membership, drop_invalid_price_rows, validate_prices
from mlinvesting.features import (
    LABEL_DATE_COLUMN,
    MARKET_FEATURE_COLUMNS,
    RAW_RETURN_COLUMN,
    TARGET_COLUMN,
    build_feature_panel,
)


def _prices(days: int = 90, tickers: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers):
        for date_index, date in enumerate(dates):
            close = 10 + ticker_index + 0.05 * date_index
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": close - 0.02,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000 + 10_000 * ticker_index,
                }
            )
    return pd.DataFrame(rows)


def test_duplicate_prices_are_rejected() -> None:
    prices = _prices()
    duplicate = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per"):
        validate_prices(duplicate)


def test_invalid_exchange_and_ohlc_values_are_rejected() -> None:
    prices = _prices(days=2, tickers=("A",))
    prices["exchange"] = ["OSE", None]
    with pytest.raises(ValueError, match="exchange values"):
        validate_prices(prices)

    prices = _prices(days=2, tickers=("A",))
    prices.loc[0, "high"] = prices.loc[0, "close"] - 1
    with pytest.raises(ValueError, match="high cannot"):
        validate_prices(prices)

    prices = _prices(days=2, tickers=("A",))
    prices["delisting_return"] = [None, "not-a-return"]
    with pytest.raises(ValueError, match="delisting_return"):
        validate_prices(prices)


def test_invalid_vendor_rows_can_be_dropped_before_validation() -> None:
    prices = _prices(days=3, tickers=("A",))
    prices.loc[1, "adj_close"] = 0.0

    cleaned, dropped = drop_invalid_price_rows(prices)

    assert len(cleaned) == 2
    assert len(dropped) == 1
    assert dropped.iloc[0]["date"] == prices.loc[1, "date"]
    assert len(validate_prices(cleaned)) == 2


def test_timezone_normalization_preserves_local_exchange_date() -> None:
    prices = _prices(days=1, tickers=("A",))
    prices["date"] = prices["date"].astype(object)
    prices.loc[0, "date"] = "2020-01-02 00:00:00+01:00"
    validated = validate_prices(prices)
    assert validated.loc[0, "date"] == pd.Timestamp("2020-01-02")


def test_universe_membership_is_point_in_time() -> None:
    prices = _prices(days=10, tickers=("A",))
    membership = pd.DataFrame(
        {
            "ticker": ["A"],
            "listed_from": [prices["date"].iloc[2]],
            "listed_to": [prices["date"].iloc[6]],
        }
    )
    attached = attach_universe_membership(prices, membership)
    assert attached["in_universe"].sum() == 5
    assert not attached.loc[attached["date"] < membership["listed_from"].iloc[0], "in_universe"].any()


def test_invalid_membership_end_date_is_rejected() -> None:
    membership = pd.DataFrame({"ticker": ["A"], "listed_from": ["2020-01-01"], "listed_to": ["not-a-date"]})
    with pytest.raises(ValueError, match="invalid dates"):
        attach_universe_membership(_prices(days=2, tickers=("A",)), membership)


def test_labels_use_next_open_and_retain_unlabeled_rows() -> None:
    prices = _prices(days=90)
    config = FeatureConfig(
        holding_days=5,
        min_history_days=20,
        liquidity_lookback_days=10,
        min_median_dollar_volume=1,
    )
    panel = build_feature_panel(prices, config)
    ticker = panel[panel["ticker"] == "A"].sort_values("date").reset_index(drop=True)
    signal_index = 40
    expected = ticker.loc[signal_index + 6, "adj_open"] / ticker.loc[signal_index + 1, "adj_open"] - 1
    assert ticker.loc[signal_index, RAW_RETURN_COLUMN] == pytest.approx(expected)
    assert ticker.loc[signal_index, LABEL_DATE_COLUMN] == ticker.loc[signal_index + 6, "date"]
    assert ticker.tail(6)[RAW_RETURN_COLUMN].isna().all()
    assert len(panel) == len(prices)


def test_delisting_payoff_is_included_in_label_before_planned_exit() -> None:
    prices = _prices(days=90)
    dates = pd.bdate_range("2020-01-01", periods=90)
    delisting_date = dates[43]
    prices = prices[~((prices["ticker"] == "A") & (prices["date"] > delisting_date))].copy()
    prices["delisting_return"] = np.nan
    event = (prices["ticker"] == "A") & (prices["date"] == delisting_date)
    prices.loc[event, "delisting_return"] = -0.5
    panel = build_feature_panel(
        prices,
        FeatureConfig(
            holding_days=5,
            min_history_days=20,
            liquidity_lookback_days=10,
            min_median_dollar_volume=1,
        ),
    )
    ticker = panel[panel["ticker"] == "A"].sort_values("date").reset_index(drop=True)
    signal_index = 40
    expected_payoff = ticker.loc[43, "adj_close"] * 0.5
    expected_return = expected_payoff / ticker.loc[41, "adj_open"] - 1
    assert ticker.loc[signal_index, RAW_RETURN_COLUMN] == pytest.approx(expected_return)
    assert ticker.loc[signal_index, LABEL_DATE_COLUMN] == delisting_date


def test_future_label_availability_does_not_change_eligibility() -> None:
    prices = _prices(days=320)
    config = FeatureConfig(
        holding_days=20,
        min_history_days=252,
        liquidity_lookback_days=20,
        min_median_dollar_volume=1,
    )
    panel = build_feature_panel(prices, config)
    latest = panel[panel["date"] == panel["date"].max()]
    assert latest[RAW_RETURN_COLUMN].isna().all()
    assert latest["eligible"].all()


def test_adjusted_open_uses_corporate_action_factor() -> None:
    prices = _prices(days=10, tickers=("A",))
    prices.loc[0, "close"] = 20.0
    prices.loc[0, "open"] = 19.0
    prices.loc[0, "adj_close"] = 10.0
    prices.loc[0, "high"] = 20.1
    prices.loc[0, "low"] = 18.9
    validated = validate_prices(prices)
    assert validated.loc[0, "adj_open"] == pytest.approx(9.5)


def test_target_is_absolute_winsorized_forward_return() -> None:
    prices = _prices(days=320)
    config = FeatureConfig(
        holding_days=5,
        min_history_days=252,
        liquidity_lookback_days=20,
        min_median_dollar_volume=1,
        target_winsor_quantile=0,
    )
    panel = build_feature_panel(prices, config)
    usable = panel[TARGET_COLUMN].notna()
    np.testing.assert_allclose(
        panel.loc[usable, TARGET_COLUMN],
        panel.loc[usable, RAW_RETURN_COLUMN],
    )


def test_market_features_ignore_names_outside_point_in_time_universe() -> None:
    base = _prices(days=320, tickers=("A", "B"))
    base["in_universe"] = True
    extra = _prices(days=320, tickers=("C",))
    growth = np.exp(np.arange(len(extra)) * 0.01)
    for column in ("open", "high", "low", "close", "adj_close"):
        extra[column] = extra[column].to_numpy() * growth
    extra["in_universe"] = False
    extended = pd.concat([base, extra], ignore_index=True)
    config = FeatureConfig(min_median_dollar_volume=1)

    base_panel = build_feature_panel(base, config)
    extended_panel = build_feature_panel(extended, config)
    base_features = base_panel.sort_values(["date", "ticker"])[list(MARKET_FEATURE_COLUMNS)]
    extended_features = extended_panel[extended_panel["ticker"].isin(["A", "B"])]
    extended_features = extended_features.sort_values(["date", "ticker"])[list(MARKET_FEATURE_COLUMNS)]
    np.testing.assert_allclose(base_features, extended_features, equal_nan=True)
