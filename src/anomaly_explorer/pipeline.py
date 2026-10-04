from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from . import __version__
from .adapter import epsilon, score_points
from .alerts import anomaly_intervals, evaluate_alert
from .evaluation import evaluate_labels
from .models.ar import ordinary_ar
from .models.seasonal import seasonal
from .models.stable_ar import stable_ar
from .preprocessing import prepare
from .schemas import Config, DetectionResult


def detect(frame: pd.DataFrame, config: Config | None = None) -> DetectionResult:
    t0 = perf_counter()
    c = config or Config()
    prep = prepare(frame, c.step_ms)
    x = prep.grid_values
    if c.model == "ar":
        model = ordinary_ar(x, c.order)
    elif c.model == "stable_ar":
        model = stable_ar(x, c.order, c.gamma, c.alpha)
    else:
        model = seasonal(
            x,
            prep.step_ms,
            c.season_duration_ms,
            c.season_neighborhood,
            c.scale_neighborhood,
        )
    # Map forecast and uncertainty to observed timestamps, then score ORIGINAL observations.
    original = prep.original.copy()
    xt = original.timestamp.to_numpy(dtype=np.int64).astype(float) - float(
        prep.grid_times[0]
    )
    gt = prep.grid_times.astype(float) - float(prep.grid_times[0])
    expected = np.interp(xt, gt, model.expected)
    sigma = np.maximum(np.interp(xt, gt, model.sigma), epsilon(x))
    valid = (
        original.timestamp.to_numpy(dtype=np.int64)
        >= int(prep.grid_times[model.warmup])
        if model.warmup
        else np.ones(len(original), dtype=bool)
    )
    # Warmup values have a neutral score and an observed expected value.
    expected[~valid] = original.value_0.to_numpy()[~valid]
    computed = score_points(
        original.value_0.to_numpy(), expected, sigma, c.threshold, c.direction, valid
    )
    original["expected_value"] = expected
    original["sigma"] = sigma
    for key, val in computed.items():
        original[key] = val
    original["is_evaluable"] = valid
    columns = [
        "row_id",
        "timestamp",
        "value_0",
        "expected_value",
        "sigma",
        "lower_bound",
        "upper_bound",
        "signed_z",
        "anomaly_score",
        "is_anomaly",
        "is_evaluable",
    ]
    if "label" in original:
        columns.append("label")
    points = original[columns]
    alert = evaluate_alert(
        points, window_duration_ms=c.window_duration_ms, percentage=c.percentage
    )
    intervals = anomaly_intervals(points, prep.step_ms)
    evaluation = (
        evaluate_labels(points, c.threshold, c.direction, prep.step_ms)
        if "label" in points
        else None
    )
    meta = {
        "model": c.model,
        "configuration": c.to_dict(),
        "source_count": len(points),
        "grid_count": len(x),
        "step_ms": prep.step_ms,
        "warmup_count": int((~valid).sum()),
        "interpolation_count": prep.interpolation_count,
        "warnings": prep.warnings,
        "elapsed_ms": (perf_counter() - t0) * 1000,
        "package_version": __version__,
        "model_diagnostics": model.diagnostics,
    }
    return DetectionResult(points, meta, alert, intervals, evaluation)
