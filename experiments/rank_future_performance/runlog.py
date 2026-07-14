"""Persist recorded experiment runs: a runs.csv comparison log plus per-run model and results artifacts."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb


def save_run(
    params: dict[str, object], metrics: dict[str, object], model: xgb.XGBRanker, results: pd.DataFrame,
    base_dir: str | Path = 'runs'
) -> str:
    """Record one experiment run under base_dir and return its run id (a timestamp).

    Appends one row (params + metrics) to base_dir/runs.csv and writes the trained model and the
    per-ticker results table to base_dir/<run_id>/. The csv is re-read and rewritten on append, so
    runs recorded with different parameter sets line up column-wise (missing values stay empty).
    """
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True)  # a run-id collision (two saves in the same second) fails loudly instead of overwriting

    model.save_model(run_dir / 'model.ubj')
    results.to_csv(run_dir / 'results.csv', index=False)

    log = pd.DataFrame([{'run': run_id, **params, **metrics}])
    csv_path = Path(base_dir) / 'runs.csv'
    if csv_path.exists():
        log = pd.concat([pd.read_csv(csv_path), log], ignore_index=True)
    log.to_csv(csv_path, index=False)
    return run_id
