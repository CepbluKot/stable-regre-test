"""Uncached, repeatable benchmark over the four original unlabeled CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import sklearn
import statsmodels
import streamlit

from anomaly_explorer import Config, detect

ROOT = Path(__file__).resolve().parents[1]


def cold_run(file: Path, model: str) -> dict:
    script = """
import json
import time
import pandas as pd
from anomaly_explorer import Config, detect
frame = pd.read_csv(__import__('sys').argv[1])
start = time.perf_counter()
result = detect(frame, Config(model=__import__('sys').argv[2]))
print(json.dumps({'detect_ms': (time.perf_counter() - start) * 1000,
                  'rows': len(result.points), 'grid': result.metadata['grid_count']}))
"""
    start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", script, str(file), model],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    wall_ms = (time.perf_counter() - start) * 1000
    return {"process_wall_ms": wall_ms, **json.loads(completed.stdout.strip())}


def cpu_identity() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/benchmark.json"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repeats are required")
    data = []
    for file in sorted((ROOT / "data/examples").glob("*.csv")):
        frame = pd.read_csv(file)
        for model in ("ar", "stable_ar", "seasonal"):
            config = Config(model=model)
            cold = cold_run(file, model)
            times = []
            result = None
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = detect(frame, config)
                times.append((time.perf_counter() - start) * 1000)
                p = result.points
                assert (
                    len(p) == len(frame)
                    and np.isfinite(
                        p[
                            [
                                "expected_value",
                                "sigma",
                                "anomaly_score",
                                "lower_bound",
                                "upper_bound",
                            ]
                        ].to_numpy()
                    ).all()
                )
            data.append(
                {
                    "file": file.name,
                    "model": model,
                    "rows": len(frame),
                    "grid": result.metadata["grid_count"],
                    "configuration": config.to_dict(),
                    "times_ms": times,
                    "first_call_ms": times[0],
                    "cold_process": cold,
                    "repeat_median_ms": statistics.median(times[1:]),
                    "all_median_ms": statistics.median(times),
                    "intervals": len(result.intervals),
                }
            )
            print(f"{file.name:20} {model:10} {statistics.median(times):9.1f} ms")
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": cpu_identity(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "streamlit": streamlit.__version__,
            "plotly": plotly.__version__,
            "thread_settings": {
                k: os.environ.get(k)
                for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
            },
        },
        "cases": data,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            lineterminator="\n",
            fieldnames=[
                "file", "model", "rows", "grid", "cold_process_wall_ms",
                "cold_detect_ms", "first_call_ms", "repeat_median_ms", "all_median_ms", "intervals",
            ],
        )
        writer.writeheader()
        for case in data:
            writer.writerow({
                "file": case["file"], "model": case["model"],
                "rows": case["rows"], "grid": case["grid"],
                "cold_process_wall_ms": case["cold_process"]["process_wall_ms"],
                "cold_detect_ms": case["cold_process"]["detect_ms"],
                "first_call_ms": case["first_call_ms"],
                "repeat_median_ms": case["repeat_median_ms"],
                "all_median_ms": case["all_median_ms"], "intervals": case["intervals"],
            })
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
