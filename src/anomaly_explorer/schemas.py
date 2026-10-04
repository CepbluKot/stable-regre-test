from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

ModelName = Literal["ar", "stable_ar", "seasonal"]
Direction = Literal["both", "above", "below"]


@dataclass(frozen=True)
class Config:
    model: ModelName = "stable_ar"
    threshold: float = 3.0
    percentage: float = 50.0
    window_duration_ms: int = 1_800_000
    direction: Direction = "both"
    order: int = 20
    gamma: float = 1.0
    alpha: float = 0.9
    season_duration_ms: int = 86_400_000
    season_neighborhood: int | None = None
    scale_neighborhood: int | None = None
    step_ms: int | None = None

    def __post_init__(self):
        if self.model not in ("ar", "stable_ar", "seasonal"):
            raise ValueError("model must be ar, stable_ar, or seasonal")
        if self.direction not in ("both", "above", "below"):
            raise ValueError("direction must be both, above, or below")
        for name in ("threshold", "percentage", "gamma", "alpha"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.threshold <= 0 or not 0 <= self.percentage <= 100:
            raise ValueError("threshold must be >0 and percentage must be 0..100")
        if self.gamma <= 0 or not 0 <= self.alpha < 1:
            raise ValueError("gamma must be >0 and alpha must be in [0,1)")
        for name in ("window_duration_ms", "order", "season_duration_ms"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("step_ms", "season_neighborhood", "scale_neighborhood"):
            v = getattr(self, name)
            if v is not None and (
                not isinstance(v, int)
                or isinstance(v, bool)
                or v < (1 if name == "step_ms" else 0)
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer"
                    if name != "step_ms"
                    else "step_ms must be positive"
                )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    points: pd.DataFrame
    metadata: dict
    alert: dict
    intervals: list[dict]
    evaluation: dict | None = None

    @property
    def expected_values(self):
        return self.points.expected_value.to_numpy(copy=True)

    @property
    def expected_bounds(self):
        return self.points[["lower_bound", "upper_bound"]].to_numpy(copy=True)

    @property
    def anomaly_scores(self):
        return self.points.anomaly_score.to_numpy(copy=True)

    @property
    def is_anomaly(self):
        return self.points.is_anomaly.to_numpy(copy=True)

    def to_dict(self) -> dict:
        rows = self.points.to_dict(orient="records")
        result = {
            "points": rows,
            "metadata": self.metadata,
            "alert": self.alert,
            "intervals": self.intervals,
            "evaluation": self.evaluation,
        }
        # Round-trip through strict JSON normalizes NumPy scalars and checks for non-finite output.
        return json.loads(
            json.dumps(
                result,
                allow_nan=False,
                default=lambda v: v.item() if hasattr(v, "item") else str(v),
            )
        )
