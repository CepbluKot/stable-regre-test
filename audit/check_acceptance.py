"""Independent output invariants and sample matrix for the local self-audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_explorer import Config, detect

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = [4600, 4020, 6336, 9216]


def main() -> None:
    cases = []
    for i, file in enumerate(sorted((ROOT / "data/examples").glob("*.csv"))):
        frame = pd.read_csv(file)
        assert len(frame) == EXPECTED_ROWS[i]
        assert list(frame.columns) == ["timestamp", "value_0"]
        sha256 = hashlib.sha256(file.read_bytes()).hexdigest()
        for model in ("ar", "stable_ar", "seasonal"):
            cfg = Config(model=model)
            result = detect(frame, cfg)
            points = result.points
            assert len(points) == len(frame)
            assert points.row_id.tolist() == list(range(len(frame)))
            assert points.timestamp.tolist() == frame.timestamp.tolist()
            assert np.all(np.isfinite(points.select_dtypes(include="number")))
            assert (points.sigma > 0).all()
            assert (points.lower_bound <= points.expected_value).all()
            assert (points.expected_value <= points.upper_bound).all()
            np.testing.assert_allclose(
                points.signed_z,
                (points.value_0 - points.expected_value) / points.sigma,
                rtol=1e-6,
                atol=1e-8,
            )
            np.testing.assert_allclose(points.anomaly_score, points.signed_z.abs())
            np.testing.assert_array_equal(
                points.is_anomaly,
                (points.anomaly_score > cfg.threshold) & points.is_evaluable,
            )
            exported = result.to_dict()
            assert len(exported["points"]) == len(frame)
            json.dumps(exported, allow_nan=False)
            assert exported["alert"] == result.alert
            assert exported["intervals"] == result.intervals
            cases.append({
                "file": file.name,
                "sha256": sha256,
                "model": model,
                "config": cfg.to_dict(),
                "rows": len(points),
                "grid": result.metadata["grid_count"],
                "finite": True,
                "anomalies": int(points.is_anomaly.sum()),
                "intervals": len(result.intervals),
                "alert": result.alert["status"],
            })
    labeled = pd.read_csv(ROOT / "tests/fixtures/labeled-shift.csv")
    labeled_result = detect(labeled, Config(model="stable_ar"))
    assert labeled_result.evaluation is not None
    assert labeled_result.evaluation["revised"]["tp"] == 1
    assert labeled_result.evaluation["revised"]["fp"] == 0
    assert labeled_result.evaluation["revised"]["fn"] == 0
    report = {"source_revision": "48400cd4a180adad2f294a9030214b0e6347ac54", "cases": cases,
              "labeled_fixture": {"rows": len(labeled), "evaluation": labeled_result.evaluation}}
    target = ROOT / "artifacts/audit/evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"{len(cases)} cases passed; saved {target}")


if __name__ == "__main__":
    main()
