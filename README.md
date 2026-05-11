# Machine Learning Investing

Private research project for testing Nordic equity data workflows.

This repo is intentionally a clean project shell. It does not include proprietary market data, LSEG/Refinitiv exports, generated outputs, or local strategy experiments. Put local data into the folders below and build experiments from there.

## Folder Structure

```text
data/
  universe_oslo.csv          # ticker/universe list
  raw_fundamentals/          # local LSEG/Refinitiv Excel exports, not committed
  raw_prices/                # local price exports, not committed
  processed/                 # cleaned datasets, not committed
notebooks/                   # local notebooks, add as needed
scripts/                     # local scripts, add as needed
docs/                        # notes about data format and workflow
```

## Local Data

Do not commit licensed data files. Keep them local in:

```text
data/raw_fundamentals/
data/raw_prices/
data/processed/
```

Suggested LSEG fundamental export per stock:

```text
Balance Sheet
Income Statement
Cash Flow
Quarterly
Longest history available
Standardized currency
```

Suggested price/market export per stock:

```text
Adjusted close or total return index
Close price
Volume
Shares outstanding
Market cap
Enterprise value
P/E
P/B
EV/EBITDA
Dividend yield
```

## Universe

The initial universe is 25 Norwegian stocks in `data/universe_oslo.csv`. This is only a starting point; adjust it locally if needed.

## Notes

This is not investment advice and not a portfolio recommendation. The point is to keep a reproducible local research setup where different people can test their own ideas without publishing paid/proprietary data.
