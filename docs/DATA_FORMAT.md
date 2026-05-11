# Data Format Notes

## Fundamentals

Expected local folder:

```text
data/raw_fundamentals/
```

Recommended file naming:

```text
EQNR.xlsx
DNB.xlsx
NHY.xlsx
```

Each workbook should ideally contain:

```text
Balance Sheet
Income Statement
Cash Flow
```

Useful fields:

```text
Period End Date
Standardized Currency
Revenue
EBIT
EBITDA
Net Income
EPS
Total Assets
Total Liabilities
Equity
Net Debt
Debt Total
Cash
Operating Cash Flow
Capex
Free Cash Flow
Shares Outstanding
```

## Prices and Market Data

Expected local folder:

```text
data/raw_prices/
```

Useful fields:

```text
Date
Ticker
Close
Adjusted Close or Total Return Index
Volume
Market Cap
Enterprise Value
P/E
P/B
EV/EBITDA
Dividend Yield
```

## Licensing

Do not push LSEG/Refinitiv, exchange, or other licensed data to GitHub unless the license explicitly allows redistribution.
