"""Stateless, local anomaly detection for univariate telemetry."""

__version__ = "0.1.0"
from .pipeline import detect
from .schemas import Config, DetectionResult

__all__ = ["Config", "DetectionResult", "detect"]
