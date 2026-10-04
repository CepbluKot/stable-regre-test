"""Display-only interval helpers; these do not change detection or alerts."""

from statistics import NormalDist

import numpy as np


def gaussian_reference_interval(expected, sigma, confidence: float = 0.95):
    """Return a normal-reference interval, without claiming calibrated coverage."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    center = np.asarray(expected, dtype=float)
    scale = np.asarray(sigma, dtype=float)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    return center - z * scale, center + z * scale
