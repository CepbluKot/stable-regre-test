from __future__ import annotations

from math import log, pi, sqrt

import numpy as np

from anomaly_explorer.adapter import epsilon

from .ar import ModelOutput, fit_ar


def stable_step(
    observation: float,
    prediction: float,
    previous_prediction: float,
    sigma_current: float,
    sigma0: float,
    gamma: float,
    alpha: float,
) -> tuple[float, float, float, float, float, float]:
    z = (observation - prediction) / sigma_current
    s = sigma_current / sigma0
    r = (0.5 * z * z + log(sqrt(2 * pi))) / (s * gamma)
    w = 1 / (1 + r)
    hidden = prediction * (1 - w) + observation * w
    next_scale = alpha * sigma_current + (1 - alpha) * (
        sigma0 + abs(prediction - previous_prediction) / 3 * (1 - w)
    )
    return prediction, w, hidden, next_scale, z, r


def stable_ar(
    values: np.ndarray, order: int, gamma: float, alpha: float
) -> ModelOutput:
    n = len(values)
    if n < 3 * order + 1:
        raise ValueError(
            f"Stable AR order {order} requires at least {3 * order + 1} grid points; got {n}"
        )
    if np.all(values == values[0]):
        return ModelOutput(
            np.array(values, copy=True),
            np.full(n, epsilon(values)),
            order,
            {"constant": True},
        )
    mu = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    u = (values - mu) / scale
    coef, baseline = fit_ar(u, order)
    eps_u = epsilon(values) / scale
    sigma0 = max(float(np.std(u[order:] - baseline[order:], ddof=0)), eps_u)
    sigma = sigma0
    hidden = np.array(u, copy=True)
    expected = np.array(u, copy=True)
    spreads = np.full(n, sigma0)
    previous_prediction = float(u[order - 1])
    for t in range(order, n):
        prediction = float(coef[0] + np.dot(coef[1:], hidden[t - order : t][::-1]))
        spreads[t] = sigma
        _, _, h, next_sigma, _, _ = stable_step(
            float(u[t]), prediction, previous_prediction, sigma, sigma0, gamma, alpha
        )
        hidden[t] = h
        expected[t] = prediction
        previous_prediction = prediction
        sigma = max(float(next_sigma), eps_u)
    return ModelOutput(
        expected * scale + mu,
        spreads * scale,
        order,
        {"constant": False, "sigma0_normalized": sigma0, "coefficients": coef.tolist()},
    )
