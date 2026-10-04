from dataclasses import dataclass

import numpy as np

from anomaly_explorer.adapter import epsilon, robust_zero_scale


@dataclass
class ModelOutput:
    expected: np.ndarray
    sigma: np.ndarray
    warmup: int
    diagnostics: dict


def fit_ar(values: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if n < 3 * order + 1:
        raise ValueError(
            f"AR order {order} requires at least {3 * order + 1} grid points; got {n}"
        )
    if order < 1:
        raise ValueError("AR order must be positive")
    # Column i contains lag i+1, earliest usable target is at index p.
    design = np.column_stack(
        [np.ones(n - order)] + [values[order - i : n - i] for i in range(1, order + 1)]
    )
    target = values[order:]
    try:
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        raise ValueError(f"AR least-squares fit failed: {e}") from e
    pred = np.array(values, copy=True)
    pred[order:] = design @ coef
    return coef, pred


def ordinary_ar(values: np.ndarray, order: int) -> ModelOutput:
    n = len(values)
    if n < 3 * order + 1:
        raise ValueError(
            f"AR order {order} requires at least {3 * order + 1} grid points; got {n}"
        )
    if np.all(values == values[0]):
        return ModelOutput(
            np.array(values, copy=True),
            np.full(n, epsilon(values)),
            order,
            {"constant": True},
        )
    coef, pred = fit_ar(values, order)
    s = max(robust_zero_scale(values[order:] - pred[order:]), epsilon(values))
    return ModelOutput(
        pred, np.full(n, s), order, {"constant": False, "coefficients": coef.tolist()}
    )
