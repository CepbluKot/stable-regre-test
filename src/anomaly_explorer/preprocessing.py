from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Prepared:
    original: pd.DataFrame
    sorted_points: pd.DataFrame
    grid_times: np.ndarray
    grid_values: np.ndarray
    step_ms: int
    interpolation_count: int
    warnings: list[str]


def prepare(frame: pd.DataFrame, step_override: int | None = None) -> Prepared:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("input must be a pandas DataFrame")
    if len(frame) < 3:
        raise ValueError("at least 3 input rows are required")
    if len(frame) > 100_000:
        raise ValueError("source row limit is 100,000")
    required = {"timestamp", "value_0"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    warnings = []
    extra = sorted(set(frame.columns) - required - {"label"})
    if extra:
        warnings.append("Ignored extra columns: " + ", ".join(extra))
    out = frame[[c for c in ("timestamp", "value_0", "label") if c in frame]].copy()
    raw_ts = out.timestamp
    # Check exact integer Unix milliseconds, never infer seconds from magnitude.
    numeric_ts = pd.to_numeric(raw_ts, errors="coerce")
    if (
        numeric_ts.isna().any()
        or not np.isfinite(numeric_ts.to_numpy(dtype=float)).all()
    ):
        raise ValueError("timestamp must contain finite Unix milliseconds")
    ts_float = numeric_ts.to_numpy(dtype=float)
    if np.any(ts_float != np.floor(ts_float)) or np.any(
        np.abs(ts_float) > np.iinfo(np.int64).max
    ):
        raise ValueError("timestamp must contain int64 Unix milliseconds")
    ts = numeric_ts.astype("int64")
    if ts.duplicated().any():
        raise ValueError("duplicate timestamps are not allowed")
    values = pd.to_numeric(out.value_0, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "value_0 must contain finite numbers; missing numeric rows are not interpolated"
        )
    if "label" in out:
        labels = pd.to_numeric(out.label, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("label must contain only 0 or 1")
        out["label"] = labels.astype(int)
    out["row_id"] = np.arange(len(out), dtype=np.int64)
    out["timestamp"] = ts.to_numpy()
    out["value_0"] = values
    chron = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = chron.timestamp.to_numpy(dtype=np.int64)
    diffs = np.diff(times)
    if np.any(diffs <= 0):
        raise ValueError("timestamps must be distinct")
    if step_override is None:
        steps, counts = np.unique(diffs, return_counts=True)
        step_ms = int(steps[np.flatnonzero(counts == counts.max())[0]])
    else:
        step_ms = step_override
    if int(diffs.max()) > 12 * step_ms:
        raise ValueError(
            f"timestamp gap exceeds 12 steps ({step_ms} ms); split the series or choose a valid step"
        )
    count = int((int(times[-1]) - int(times[0]) + step_ms - 1) // step_ms) + 1
    if count > 200_000:
        raise ValueError("regularized grid limit is 200,000 points")
    grid = int(times[0]) + np.arange(count, dtype=np.int64) * step_ms
    regular_values = np.interp(
        grid.astype(float) - float(times[0]),
        times.astype(float) - float(times[0]),
        chron.value_0.to_numpy(),
    )
    matched = np.isin(grid, times)
    interpolation_count = int((~matched).sum())
    if interpolation_count:
        warnings.append(
            f"Interpolated {interpolation_count} grid points; these are excluded from alert percentages"
        )
    if grid[-1] > times[-1]:
        warnings.append(
            "Final grid point extends less than one step past the final observation using its last value"
        )
    return Prepared(
        out, chron, grid, regular_values, step_ms, interpolation_count, warnings
    )
