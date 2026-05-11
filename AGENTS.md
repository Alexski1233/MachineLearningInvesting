# Agent Coding Style

These instructions apply to this repository.

## Main Rule

Keep the code compact and human written.

Do not run an auto formatter that rewrites wrapping across the project. Make focused manual edits.

## Imports

Keep imports on one line when practical.

Use this style.

```python
from model import TOP_N, backtest_top_n, build_dataset, evaluate_predictions, summarize_backtest, train_and_predict
```

Do not use parenthesized import blocks unless the line becomes genuinely hard to read.

Do not use `try` and `except ImportError` around imports. If a dependency is required, import it directly.

## Line Wrapping

Prefer one line for short calls, function signatures, and constructor calls.

This kind of line is acceptable.

```python
"hist_gradient_boosting": HistGradientBoostingRegressor(learning_rate=0.03, max_iter=300, max_leaf_nodes=15, l2_regularization=0.01, early_stopping=False, random_state=random_state),
```

If a line becomes much longer than that, split it.

Long metric rows and table data should be split into readable lists.

Use this style.

```python
rows = [
    ("Labeled rows", f"{metrics['rows']:,}"),
    ("Direction accuracy", format_pct(metrics["direction_accuracy"])),
    (f"Top {TOP_N} hit rate", format_pct(metrics["top_n_hit_rate"])),
]
```

Do not force a huge list into one line.

## Output Code

Keep terminal output code simple.

Progress prints may stay as one line when they are short enough.

Use `PrettyTable` directly for tables. Do not add fallback table renderers unless the user asks.

## Scope

Do not edit unrelated files.

Do not rewrap existing code just because a formatter would do it.

When touching a file, preserve the current compact style unless a line is clearly too long to read.
