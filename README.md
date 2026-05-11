# Machine Learning Investing

Private research project for testing Nordic equity data workflows. The goal is to use machine learning to pick stocks on a trading platform.

## Folder Structure

```text
data/
  universe_oslo.csv          # ticker/universe list
  raw_fundamentals/          # local LSEG/Refinitiv Excel exports, not committed
  raw_prices/                # local price CSV exports, not committed
  processed/                 # cleaned datasets, not committed
src/                         # pipeline code (load, features, model, run)
environment.yml              # conda environment definition
```

## Install & run

Create the conda environment (one-time):

```bash
conda env create -f environment.yml
```

Activate it and run the end-to-end pipeline:

```bash
conda activate mlinvest
python src/main.py
```

This loads every CSV in `data/raw_prices/`, builds features, trains a gradient-boosted model on pre-2020 data, backtests a long-top-5 rebalancing rule, and prints today's BUY picks to paste into the paper-trading site.

To update an existing environment after editing `environment.yml`:

```bash
conda env update -f environment.yml --prune
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
