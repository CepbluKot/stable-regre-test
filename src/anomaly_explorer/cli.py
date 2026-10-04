"""Command line access to the same detector used by the UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import detect
from .schemas import Config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run local anomaly detection on a CSV")
    parser.add_argument("action", choices=["detect"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", choices=["ar", "stable_ar", "seasonal"], default="stable_ar"
    )
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--percentage", type=float, default=50.0)
    parser.add_argument("--window-ms", type=int, default=1_800_000)
    parser.add_argument(
        "--direction", choices=["both", "above", "below"], default="both"
    )
    parser.add_argument("--order", type=int, default=20)
    parser.add_argument("--season-ms", type=int, default=86_400_000)
    parser.add_argument("--step-ms", type=int)
    args = parser.parse_args(argv)
    config = Config(
        model=args.model,
        threshold=args.threshold,
        percentage=args.percentage,
        window_duration_ms=args.window_ms,
        direction=args.direction,
        order=args.order,
        season_duration_ms=args.season_ms,
        step_ms=args.step_ms,
    )
    result = detect(pd.read_csv(args.input), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"{args.output}: {len(result.points)} points, {len(result.intervals)} intervals, alert={result.alert['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
