# Machine Learning Investing

Research code for testing a machine learning stock picker on Nordic equity data.
The script trains on local price files, evaluates on unseen data, backtests a simple top 5 rule, and prints the latest paper trading candidates.

## Install

Create the conda environment once.

```bash
conda env create -f environment.yml
```

Activate it.

```bash
conda activate mlinvest
```

Update an existing environment after dependency changes.

```bash
conda env update -f environment.yml --prune
```

## Run

```bash
python src/main.py
```

Input price files live in `data/raw_prices/`. Each CSV should have at least `ticker`, `date`, `adj_close`, and `volume`.

To refresh the Nordic Yahoo Finance price files, run this before the model.

```bash
python src/fetch_prices.py
```

## What It Does

The pipeline loads all price CSVs, builds price based features, trains several candidate models, and selects the best model on validation data.
The 2020 onward period is kept for unseen testing.

Features include momentum, volatility, trend, drawdown, liquidity, and same day cross sectional ranks. The label is the next 20 trading day return for the same stock.

The candidate models are ridge regression, random forest, extra trees, histogram gradient boosting, and Huber loss gradient boosting.
This model set follows common empirical asset pricing practice.
Gu, Kelly, and Xiu compare similar machine learning models in [The Review of Financial Studies](https://academic.oup.com/rfs/article/33/5/2223/5758276).

## Output

The script prints progress while it loads data and trains models.

`Data` shows the number of rows, number of tickers, date range, and whether the latest local price file is fresh.

`Model Selection` shows validation results from the pre 2020 period. The selected model is trained again on all eligible pre 2020 data.

`Unseen Test Accuracy` reports accuracy from 2020 onward.
Direction accuracy measures whether the model got the return sign right.
Top 5 hit rate measures how often selected stocks had positive forward returns.
Rank IC measures whether the model ranked better stocks higher.
OOS R squared tests exact return prediction against the training mean.

`Backtest` simulates buying the top 5 names every 20 trading days. It compares the strategy with an equal weighted universe benchmark.

`Top 5 BUY Picks` is the latest ranked list for paper trading. Treat `Pred 20d` as a ranking score, not a precise return forecast.
