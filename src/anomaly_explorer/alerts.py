from __future__ import annotations

import pandas as pd


def evaluate_alert(
    points: pd.DataFrame,
    as_of: int | None = None,
    window_duration_ms: int = 1_800_000,
    percentage: float = 50.0,
) -> dict:
    if window_duration_ms <= 0 or not 0 <= percentage <= 100:
        raise ValueError(
            "window duration must be positive and percentage must be 0..100"
        )
    if points.empty:
        raise ValueError("alert requires original points")
    as_of = int(points.timestamp.max()) if as_of is None else int(as_of)
    start = as_of - int(window_duration_ms)
    selected = points.loc[
        (points.timestamp > start) & (points.timestamp <= as_of) & points.is_evaluable
    ]
    total = len(selected)
    numerator = int(selected.is_anomaly.sum())
    fraction = numerator / total if total else 0.0
    enough = int(points.timestamp.min()) <= start and total >= 3
    return {
        "as_of": as_of,
        "window_start": start,
        "window_end": as_of,
        "numerator": numerator,
        "denominator": total,
        "fraction": fraction,
        "status": "ACTIVE"
        if enough and 100 * fraction > percentage
        else "OK"
        if enough
        else "INSUFFICIENT_DATA",
        "active": bool(enough and 100 * fraction > percentage),
        "reason": None
        if enough
        else "Full window context and at least 3 evaluable original observations are required",
        "config": {
            "window_duration_ms": int(window_duration_ms),
            "percentage": float(percentage),
        },
    }


def anomaly_intervals(points: pd.DataFrame, step_ms: int) -> list[dict]:
    ordered = points.sort_values("timestamp", kind="stable")
    result = []
    group = []

    def flush():
        if not group:
            return
        times = [int(x.timestamp) for x in group]
        signs = {"above" if float(x.signed_z) > 0 else "below" for x in group}
        result.append(
            {
                "start": times[0],
                "end": times[-1],
                "duration_ms": times[-1] - times[0],
                "point_count": len(group),
                "max_score": max(float(x.anomaly_score) for x in group),
                "direction": next(iter(signs)) if len(signs) == 1 else "mixed",
            }
        )

    for row in ordered.itertuples(index=False):
        active = bool(row.is_evaluable and row.is_anomaly)
        if not active:
            flush()
            group = []
            continue
        if group and int(row.timestamp) - int(group[-1].timestamp) > 1.5 * step_ms:
            flush()
            group = []
        group.append(row)
    flush()
    return result
