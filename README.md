# Machine Learning Investing

This repository contains two stock selection methods for Nordic equities. The holdout strategy uses one fixed test period. The walk forward strategy refits through time and models trading costs and next open execution.

## Setup

Create the conda environment and install the local package.

```bash
conda env create -f environment.yml
conda activate mlinvest
python -m pip install -e ".[dev]"
```

Use this command to update an existing environment.

```bash
conda env update -f environment.yml --prune
```

## Price data

Price files belong in `data/raw_prices`. Refresh the Yahoo Finance files when needed.

```bash
python src/fetch_prices.py
```

Each CSV needs `ticker`, `date`, `adj_close`, and `volume`. The walk forward strategy also needs `open` and `close`. It accepts `high`, `low`, `exchange`, `in_universe`, and `delisting_return` when those fields are available.

A walk forward run must contain one exchange calendar. Prices, volume, and portfolio capital must also use a consistent currency.

## Holdout strategy

```bash
python src/run_holdout.py
```

The holdout strategy trains and selects models on data before 2020. It evaluates the selected model from 2020 onward. Its features cover momentum, volatility, trend, drawdown, and liquidity. The prediction target is the next 20 trading day return.

The backtest buys the five highest ranked stocks every 20 trading days and compares the result with an equal weighted universe. It does not model transaction costs, next open execution, or historical index membership.

Limit the run to a CSV universe when required. The file must contain a `ticker` column.

```bash
python src/run_holdout.py --universe data/universe_fondsfinans.csv
```

## Walk forward strategy

```bash
python src/run_walk_forward.py
```

Signals are calculated after the close and executed at the next adjusted open. Each refit can only use outcomes known before its signal date. Candidate models are selected with historical cross sectional rank IC.

The preset combines 25 percent machine learning with 75 percent 12 minus 1 momentum. It holds at most five stocks. Position sizing uses inverse volatility, a maximum weight, a ranking buffer, and a no trade band.

The backtest accounts for commission, spread, market impact, volume capacity, cash, stale prices, and delisting returns when the required data are present. Results are written to `output/walk_forward`.

## Configurable walk forward runs

Use the command line interface to change the start date, portfolio size, capital, costs, or output location.

```bash
walk-forward research --prices-dir data/raw_prices --start 2020-01-02 --capital 1000000 --top-n 10 --output-dir output/walk_forward
```

Generate a candidate list for the latest eligible date.

```bash
walk-forward latest --prices-dir data/raw_prices --top-n 10 --output-dir output/walk_forward
```

The optional `--universe` argument accepts a historical membership file with this structure.

```text
ticker,listed_from,listed_to
AAA,2004-05-01,2021-09-30
BBB,2010-03-15,
```

Without historical membership data, a result may contain survivorship bias. The price data must also include delisted companies and their payouts. A membership file cannot restore missing price histories.

## Tests

```bash
python -m pytest -q
```

Backtest results are research estimates. A test period becomes development data if the method is changed after its results have been reviewed.
