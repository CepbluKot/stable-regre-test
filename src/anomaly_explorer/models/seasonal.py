from __future__ import annotations

import numpy as np

from anomaly_explorer.adapter import epsilon, robust_zero_scale

from .ar import ModelOutput


def phase_indices(
    length: int, period: int, phase: int, season: int, radius: int
) -> np.ndarray:
    indices = {
        season * period + ((phase + j) % period) for j in range(-radius, radius + 1)
    }
    return np.array(sorted(k for k in indices if k < length), dtype=np.int64)


def seasonal(
    values: np.ndarray,
    step_ms: int,
    season_duration_ms: int,
    season_neighborhood: int | None,
    scale_neighborhood: int | None,
) -> ModelOutput:
    n = len(values)
    if season_duration_ms % step_ms:
        raise ValueError("season duration must be an integral number of grid steps")
    period = season_duration_ms // step_ms
    if period < 3 or n < 3 * period:
        raise ValueError(
            f"seasonal model needs period >=3 and three complete seasons; period={period}, grid={n}"
        )
    ws = (
        min(2, (period - 1) // 2)
        if season_neighborhood is None
        else season_neighborhood
    )
    w = min(2, (period - 1) // 2) if scale_neighborhood is None else scale_neighborhood
    if min(ws, w) < 0 or 2 * max(ws, w) + 1 > period:
        raise ValueError("season neighborhood must fit in one cycle")
    t = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n), t])
    try:
        coef = np.linalg.lstsq(design, values, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise ValueError(f"seasonal trend fit failed: {e}") from e
    trend = design @ coef
    detrended = values - trend
    seasons = (n + period - 1) // period
    profile = np.zeros(period)
    for i in range(period):
        ids = np.concatenate(
            [phase_indices(n, period, i, q, ws) for q in range(seasons)]
        )
        profile[i] = np.median(detrended[ids])
    expected = trend + profile[np.arange(n) % period]
    residuals = values - expected
    fallback = max(robust_zero_scale(residuals), epsilon(values))
    spreads = np.full(period, fallback)
    for i in range(period):
        estimates = []
        for q in range(seasons):
            ids = phase_indices(n, period, i, q, w)
            if len(ids) >= 2:
                estimates.append(float(np.std(residuals[ids], ddof=0)))
        if estimates:
            cutoff = float(np.quantile(estimates, 0.8, method="linear"))
            spreads[i] = max(
                float(np.median([v for v in estimates if v <= cutoff])), epsilon(values)
            )
    return ModelOutput(
        expected,
        spreads[np.arange(n) % period],
        0,
        {
            "period_steps": period,
            "season_neighborhood": ws,
            "scale_neighborhood": w,
            "trend_intercept": float(coef[0]),
            "trend_slope": float(coef[1]),
        },
    )
