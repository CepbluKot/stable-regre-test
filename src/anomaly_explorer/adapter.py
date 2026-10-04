import numpy as np

NORMAL_Q75 = 0.6744897501960817


def robust_zero_scale(residuals) -> float:
    x = np.asarray(residuals, dtype=float)
    if len(x) == 0:
        raise ValueError("no evaluable residuals")
    return float(np.median(np.abs(x)) / NORMAL_Q75)


def epsilon(values) -> float:
    return 1e-8 * max(1.0, float(np.std(np.asarray(values, dtype=float), ddof=0)))


def score_points(
    values, expected, sigma, threshold: float, direction: str, evaluable
) -> dict:
    v = np.asarray(values, dtype=float)
    e = np.asarray(expected, dtype=float)
    s = np.asarray(sigma, dtype=float)
    mask = np.asarray(evaluable, dtype=bool)
    if v.shape != e.shape or v.shape != s.shape or v.shape != mask.shape:
        raise ValueError("point arrays must have equal shape")
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("sigma must be finite and positive")
    signed = (v - e) / s
    signed = np.where(mask, signed, 0.0)
    score = np.abs(signed)
    if direction == "both":
        flags = score > threshold
    elif direction == "above":
        flags = signed > threshold
    elif direction == "below":
        flags = signed < -threshold
    else:
        raise ValueError("unknown direction")
    return {
        "signed_z": signed,
        "anomaly_score": score,
        "lower_bound": e - threshold * s,
        "upper_bound": e + threshold * s,
        "is_anomaly": flags & mask,
    }
