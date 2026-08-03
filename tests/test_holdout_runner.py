import pandas as pd

import run_holdout


def test_load_prices_filters_to_requested_universe(tmp_path, monkeypatch) -> None:
    prices = pd.DataFrame({"ticker": ["A", "B", "C"], "value": [1, 2, 3]})
    universe_path = tmp_path / "universe.csv"
    pd.DataFrame({"ticker": ["B", "C", "MISSING"]}).to_csv(universe_path, index=False)
    monkeypatch.setattr(run_holdout, "load_all_prices", lambda: prices)

    selected = run_holdout.load_prices(universe_path)

    assert selected["ticker"].tolist() == ["B", "C"]
