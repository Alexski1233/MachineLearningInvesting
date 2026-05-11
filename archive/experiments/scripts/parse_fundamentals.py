from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "NorskeBørs"
OUTPUT_DIR = ROOT / "output"

FIELD_MAP = {
    "Balance Sheet": {
        "total_assets": ["Total Assets", "Total Assets - Reported"],
        "total_liabilities": ["Total Liabilities"],
        "shareholders_equity": ["Total Shareholders' Equity", "Total Shareholders' Equity - including Minority Interest & Hybrid Debt"],
        "common_equity": ["Common Equity - Total", "Common Equity Attributable to Parent Shareholders"],
        "tangible_book_value": ["Tangible Book Value"],
        "net_debt": ["Net Debt"],
        "debt_total": ["Debt - Total"],
        "cash": ["Cash & Cash Equivalents", "Cash & Cash Equivalents - Total"],
        "shares_outstanding": ["Common Shares - Outstanding - Total"],
        "working_capital": ["Working Capital"],
    },
    "Income Statement": {
        "revenue": ["Revenues", "Revenue from Goods & Services", "Revenue from Business Activities - Total", "Revenue from Business-Related Activities - Other - Total"],
        "ebit": ["Earnings before Interest & Taxes (EBIT)", "Operating Profit"],
        "ebitda": ["Earnings before Interest, Taxes, Depreciation & Amortization (EBITDA)"],
        "net_income": ["Net Income", "Net Income after Minority Interest", "Net Income After Tax", "Net Income after Tax", "Net Income before Minority Interest"],
        "eps": ["EPS - Basic - excluding Extraordinary Items Applicable to Common - Total", "EPS - Basic - including Extraordinary Items Applicable to Common - Total"],
        "dividend_per_share": ["Dividends per Share - Common Stock Primary Issue", "Dividend per Share"],
    },
    "Cash Flow": {
        "operating_cash_flow": ["Net Cash Flow from Operating Activities", "Operating Cash Flow - Indirect"],
        "capex": ["Capital Expenditures - Net - Cash Flow", "Capital Expenditures - Total"],
        "free_cash_flow": ["Free Cash Flow"],
        "dividends_paid": ["Dividends Paid - Cash - Total - Cash Flow", "Dividends - Common - Cash Paid"],
    },
}

TICKER_ALIASES = {
    "AFGA": "AFG",
    "BRGB": "BRG",
    "GJFG": "GJF",
}


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    ticker = ticker.replace(".OL", "")
    ticker = ticker.replace(" ", "")
    return TICKER_ALIASES.get(ticker, ticker)


def extract_company_and_ticker(raw: pd.DataFrame, fallback: str) -> tuple[str, str]:
    row = raw.loc[raw.iloc[:, 0].astype(str).eq("Company Name")]
    if row.empty:
        return fallback, normalize_ticker(fallback)
    company = str(row.iloc[0, 1])
    match = re.search(r"\(([^)]+)\)", company)
    ticker = normalize_ticker(match.group(1) if match else fallback)
    return company, ticker


def first_matching_field(frame: pd.DataFrame, field_names: list[str]) -> pd.Series | None:
    lowered = pd.Series([str(idx).strip().lower() for idx in frame.index], index=frame.index)
    for field in field_names:
        key = field.strip().lower()
        matches = frame.loc[lowered == key]
        if not matches.empty:
            if isinstance(matches, pd.DataFrame):
                non_empty = matches[matches.notna().any(axis=1)]
                return (non_empty if not non_empty.empty else matches).iloc[0]
            return matches
    return None


def parse_sheet(path: Path, sheet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    company, ticker = extract_company_and_ticker(raw, path.stem)

    field_rows = raw.index[raw.iloc[:, 0].astype(str).eq("Field Name")]
    if field_rows.empty:
        raise ValueError(f"Fant ikke Field Name i {path.name} / {sheet}")
    field_row = int(field_rows[0])

    period_dates = pd.to_datetime(raw.iloc[field_row, 1:], dayfirst=True, errors="coerce")
    fields = raw.iloc[field_row + 1 :, 0].astype(str).str.strip()
    values = raw.iloc[field_row + 1 :, 1:].copy()
    values.index = fields
    values.columns = period_dates
    values = values.loc[:, values.columns.notna()]
    values = values.apply(pd.to_numeric, errors="coerce")

    records = []
    selected = pd.DataFrame(index=values.columns)
    for metric, candidates in FIELD_MAP.get(sheet, {}).items():
        series = first_matching_field(values, candidates)
        if series is not None:
            selected[metric] = series.values

    selected = selected.reset_index(names="period_end_date")
    selected.insert(0, "statement", sheet)
    selected.insert(0, "ticker", ticker)
    selected.insert(1, "company", company)
    selected.insert(3, "source_file", path.name)

    long_rows = []
    for field_name, row in values.iterrows():
        for period_end_date, value in row.dropna().items():
            long_rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "source_file": path.name,
                    "statement": sheet,
                    "period_end_date": period_end_date,
                    "field": field_name,
                    "value": value,
                }
            )
    return selected, pd.DataFrame(long_rows)


def parse_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    long_frames = []
    failures = []

    for path in sorted(RAW_DIR.glob("*.xlsx")):
        for sheet in FIELD_MAP:
            try:
                selected, long = parse_sheet(path, sheet)
                selected_frames.append(selected)
                long_frames.append(long)
            except Exception as exc:
                failures.append({"file": path.name, "sheet": sheet, "error": str(exc)})

    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "fundamental_parse_failures.csv", index=False)

    selected_all = pd.concat(selected_frames, ignore_index=True)
    long_all = pd.concat(long_frames, ignore_index=True)
    return selected_all, long_all


def build_quarterly_factors(selected: pd.DataFrame) -> pd.DataFrame:
    sheets = []
    for statement, frame in selected.groupby("statement"):
        possible_metrics = [c for c in frame.columns if c not in {"ticker", "company", "statement", "source_file", "period_end_date"}]
        metric_cols = [c for c in possible_metrics if frame[c].notna().any()]
        slim = frame[["ticker", "company", "period_end_date", *metric_cols]].copy()
        slim = slim.sort_values(["ticker", "period_end_date"])
        sheets.append(slim)

    wide = sheets[0]
    for frame in sheets[1:]:
        wide = wide.merge(frame, on=["ticker", "company", "period_end_date"], how="outer")

    wide = wide.sort_values(["ticker", "period_end_date"])

    def col(name: str) -> pd.Series:
        if name in wide.columns:
            return pd.to_numeric(wide[name], errors="coerce")
        return pd.Series(float("nan"), index=wide.index, dtype="float64")

    for metric in ["revenue", "ebit", "ebitda", "net_income", "eps", "operating_cash_flow", "capex", "free_cash_flow"]:
        if metric in wide.columns:
            wide[f"{metric}_ttm"] = wide.groupby("ticker")[metric].transform(lambda s: s.rolling(4, min_periods=3).sum())

    g = wide.groupby("ticker")
    equity = col("shareholders_equity").fillna(col("common_equity"))
    net_income_ttm = col("net_income_ttm")
    if net_income_ttm.isna().all():
        net_income_ttm = col("eps_ttm") * col("shares_outstanding")

    wide["equity_used"] = equity
    wide["net_income_ttm_used"] = net_income_ttm
    wide["roe_ttm"] = net_income_ttm / equity
    wide["ebit_margin_ttm"] = col("ebit_ttm") / col("revenue_ttm")
    wide["fcf_to_assets_ttm"] = col("free_cash_flow_ttm") / col("total_assets")
    wide["debt_to_equity"] = col("debt_total") / equity
    wide["net_debt_to_ebitda_ttm"] = col("net_debt") / col("ebitda_ttm")
    wide["asset_turnover_ttm"] = col("revenue_ttm") / col("total_assets")
    wide["revenue_growth_yoy"] = g["revenue_ttm"].pct_change(4, fill_method=None) if "revenue_ttm" in wide.columns else pd.NA
    wide["eps_growth_yoy"] = g["eps_ttm"].pct_change(4, fill_method=None) if "eps_ttm" in wide.columns else pd.NA

    return wide.replace([float("inf"), float("-inf")], pd.NA)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(skipna=True)
    if pd.isna(std) or std == 0:
        return s * 0
    return (s - s.mean(skipna=True)) / std


def latest_ranking(factors: pd.DataFrame) -> pd.DataFrame:
    latest = factors.sort_values("period_end_date").groupby("ticker", as_index=False).tail(1).copy()

    quality_parts = pd.DataFrame(
        {
            "roe": zscore(latest.get("roe_ttm")),
            "fcf_assets": zscore(latest.get("fcf_to_assets_ttm")),
            "ebit_margin": zscore(latest.get("ebit_margin_ttm")),
            "debt_equity": -zscore(latest.get("debt_to_equity")),
            "net_debt_ebitda": -zscore(latest.get("net_debt_to_ebitda_ttm")),
        }
    )
    growth_parts = pd.DataFrame(
        {
            "revenue_growth": zscore(latest.get("revenue_growth_yoy")),
            "eps_growth": zscore(latest.get("eps_growth_yoy")),
        }
    )
    latest["quality_score"] = quality_parts.mean(axis=1, skipna=True)
    latest["growth_score"] = growth_parts.mean(axis=1, skipna=True)
    latest["fundamental_score"] = 0.75 * latest["quality_score"] + 0.25 * latest["growth_score"].fillna(0)
    return latest.sort_values("fundamental_score", ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, long = parse_all()
    factors = build_quarterly_factors(selected)
    ranking = latest_ranking(factors)

    selected.to_csv(OUTPUT_DIR / "fundamentals_selected_raw.csv", index=False)
    long.to_csv(OUTPUT_DIR / "fundamentals_long_raw.csv", index=False)
    factors.to_csv(OUTPUT_DIR / "fundamental_factors_quarterly.csv", index=False)
    ranking.to_csv(OUTPUT_DIR / "fundamental_latest_ranking.csv", index=False)

    print(f"Leste {selected['ticker'].nunique()} tickere fra {RAW_DIR}")
    print(f"Valgte felt-rader: {len(selected):,}")
    print(f"Long raw rows: {len(long):,}")
    print(f"Faktor-rader: {len(factors):,}")
    print("\nTopp 10 siste fundamental ranking:")
    cols = [
        "ticker",
        "period_end_date",
        "fundamental_score",
        "quality_score",
        "growth_score",
        "roe_ttm",
        "fcf_to_assets_ttm",
        "debt_to_equity",
        "revenue_growth_yoy",
    ]
    print(ranking[cols].head(10).to_string(index=False))
    print("\nSkrev output/fundamental_factors_quarterly.csv og output/fundamental_latest_ranking.csv")


if __name__ == "__main__":
    main()
