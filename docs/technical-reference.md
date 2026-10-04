# Technical reference

This document describes the implemented contract, including local choices where the [source report and notebook](source-mapping.md) differ. The implementation is an educational, stateless reconstruction; it is not the reference author's production service.

## Data flow and modules

```text
CSV / pandas DataFrame
  → preprocessing.prepare (validate, sort, regularize)
  → models.ar / stable_ar / seasonal (expected value and sigma on grid)
  → pipeline.detect (map to original timestamps, score)
  → alerts + optional evaluation
  → DetectionResult → Streamlit UI, Python caller, or CLI JSON
```

`src/anomaly_explorer/schemas.py` defines `Config` and `DetectionResult`. `preprocessing.py` owns input validation and grid construction. The three modules under `models/` supply an expected series and positive uncertainty. `adapter.py` applies the shared scoring rule. `alerts.py` creates latest-window status and anomaly intervals; `evaluation.py` calculates optional label metrics. `visualization.py` supplies the display-only green band. `app.py` is the local Streamlit UI, and `cli.py` uses the same `detect()` function. There is no background worker, persisted model state, or external API.

## Python and CLI interface

```python
import pandas as pd
from anomaly_explorer import Config, detect

frame = pd.read_csv("data/examples/time_series_1.csv")
result = detect(frame, Config(model="stable_ar", threshold=3.0))
print(result.points[["timestamp", "value_0", "anomaly_score", "is_anomaly"]].head())
print(result.alert["status"])
payload = result.to_dict()  # strict JSON-compatible values
```

`detect()` does not mutate its input DataFrame. Repeated calls do not share fitted state. The effective configuration is recorded under `metadata.configuration`.

```sh
uv run python -m anomaly_explorer.cli detect \
  --input data/examples/time_series_1.csv \
  --model stable_ar --threshold 3 --percentage 50 \
  --output artifacts/example-result.json
uv run python -m anomaly_explorer.cli --help
```

The CLI supports `--model`, `--threshold`, `--percentage`, `--window-ms`, `--direction`, `--order`, `--season-ms`, and `--step-ms`; `gamma`, `alpha`, and seasonal radii currently use `Config` defaults in CLI mode. The Python API can set every `Config` field. The CLI creates the output directory and writes UTF-8 JSON. Invalid input raises an error rather than producing a partial result.

### Configuration defaults

| Field | Default | Validity / role |
|---|---:|---|
| `model` | `stable_ar` | `ar`, `stable_ar`, `seasonal` |
| `threshold` | 3.0 | Finite, positive z-score cutoff |
| `percentage` | 50.0 | Finite, 0–100 inclusive; alert percentage |
| `window_duration_ms` | 1,800,000 | Positive integer |
| `direction` | `both` | `both`, `above`, `below` |
| `order` | 20 | Positive integer, used by AR models |
| `gamma` | 1.0 | Finite, positive; Stable AR weight control |
| `alpha` | 0.9 | Finite, `[0, 1)`; Stable AR scale smoothing |
| `season_duration_ms` | 86,400,000 | Positive integer, seasonal cycle duration |
| `season_neighborhood` / `scale_neighborhood` | `None` | Nonnegative integer or `None`; `None` selects radius 2, capped to fit the cycle |
| `step_ms` | `None` | Positive integer override; `None` infers cadence |

The UI restricts controls to practical ranges described in the [user guide](user-guide.md#controls). `Config` validation is the library contract and is wider than some UI slider ranges.

## Preprocessing and alignment

`prepare()` validates at least three and at most 100,000 rows; requires `timestamp` and `value_0`; rejects duplicate, missing, non-finite, non-integer, or out-of-int64 timestamps; rejects missing/non-finite values; and accepts only 0/1 labels when present. It warns about and drops extra columns. It sorts a copy by timestamp for computation and gives each source row a `row_id` before sorting.

The default step is the modal positive sorted timestamp difference, with the smallest value breaking ties. An override replaces that inference. A gap greater than `12 × step_ms` fails. The regular grid starts at the earliest timestamp and extends through the last original timestamp, possibly with one final position less than a step beyond it; grid size cannot exceed 200,000. Linear interpolation supplies internal grid values. A final grid position after the last observation uses the last observed value. Interpolated grid positions are reported but never become output observations.

The model returns `expected` and `sigma` on the regular grid. Both are linearly interpolated to each original timestamp, after which the original `value_0` is scored. This prevents an interpolated score from disagreeing with the original point's threshold crossing. The result DataFrame retains input row order; charts, alert windows, and intervals use chronological order. For AR models, the first `order` grid positions are warmup. Original points before the first usable grid timestamp receive a neutral score, have `is_evaluable=False`, and cannot be anomalies.

## Models and scoring

All models return an expected value and strictly positive `sigma` in original units. Constant series receive a small finite scale floor. Ordinary AR fits an order-`p` linear regression with an intercept on lagged grid values, using all usable grid positions; it uses `median(abs(residual)) / 0.6744897501960817` as a zero-centered robust scale. Both AR models need at least `3p + 1` grid values. Stable AR fits the initial regression on normalized values, then recursively downweights large deviations when updating hidden state and smooths dynamic sigma; see `models/stable_ar.py` and [source mapping](source-mapping.md) for the exact recurrence and choices.

Seasonal decomposition fits a linear trend, calculates a cyclic median profile for each phase, then estimates a trimmed phase-local residual spread. It needs a season duration exactly divisible by `step_ms`, at least three grid positions per cycle, and at least three complete cycles. A weak or mismatched season can produce a very high flag rate. Fitting is retrospective over the supplied request; it is not a leakage-free online forecast.

For an evaluable original observation `x`, expected value `e`, and scale `s > 0`:

```text
signed_z       = (x − e) / s
anomaly_score  = abs(signed_z)
lower_bound    = e − threshold × s
upper_bound    = e + threshold × s
```

`both` flags `anomaly_score > threshold`; `above` flags `signed_z > threshold`; `below` flags `signed_z < −threshold`. Exact-boundary points are normal. The output `anomaly_score` is absolute even when a one-sided direction is selected. The green chart band uses `e ± 1.95996 × s` independently of the threshold. It is only a normal-reference visualization and has no calibrated coverage claim.

## Result contract

`DetectionResult` has `points` (pandas DataFrame), `metadata` (dict), `alert` (dict), `intervals` (list of dicts), and `evaluation` (dict or `None`). `to_dict()` converts this structure to strict JSON-compatible values and rejects non-finite serialization.

| Point field | Meaning |
|---|---|
| `row_id` | Zero-based original CSV row position |
| `timestamp` | Original Unix milliseconds |
| `value_0` | Original observation |
| `expected_value` | Model expectation at original timestamp |
| `sigma` | Positive model uncertainty at original timestamp |
| `lower_bound`, `upper_bound` | Blue anomaly threshold bounds in original units |
| `signed_z` | Signed standardized deviation; 0 during AR warmup |
| `anomaly_score` | Absolute standardized deviation |
| `is_anomaly` | Boolean flag after direction and strict threshold; false during warmup |
| `is_evaluable` | Whether the point participates in detection, alerts, and metrics |
| `label` | Original 0/1 ground truth, only when supplied |

`metadata` includes `model`, `configuration`, `source_count`, `grid_count`, `step_ms`, `warmup_count`, `interpolation_count`, `warnings`, `elapsed_ms`, `package_version`, and model-specific `model_diagnostics`. `elapsed_ms` is in-process computation time, not full page load. The JSON and CSV downloads do not include the display-only 95% reference band.

## Alerts and intervals

The latest alert's `as_of` is the maximum source timestamp. Its window is `(as_of − window_duration_ms, as_of]` over evaluable original observations. `numerator` counts flagged observations; `denominator` counts eligible observations; `fraction` is their ratio. A full window of source history and at least three eligible observations are required. Otherwise status is `INSUFFICIENT_DATA` with an explanatory `reason`. Given enough data, `ACTIVE` requires `100 × fraction > percentage`; equality is `OK`. The object also contains `active`, `window_start`, `window_end`, and the alert `config`. This is a preview only; no notification is delivered.

`intervals` groups consecutive flagged original observations in chronological order. A normal or unevaluable point, or a gap greater than `1.5 × step_ms`, ends an interval. Each item has `start`, `end`, `duration_ms`, `point_count`, `max_score`, and `direction` (`above`, `below`, or `mixed`). A single-point interval has zero duration. Intervals are historical groups and do not imply alert delivery.

## Evaluation

Labels are optional and are never read during fitting or scoring. Point-wise precision, recall, and F1 use evaluable original points, the current threshold, and selected direction. Revised segment adjustment collapses each contiguous ground-truth positive segment into one positive item carrying its maximum score; each normal point remains an item. It splits segments at unevaluable positions and timestamp gaps above `1.5 × step_ms`. The revised threshold metrics use direction-adjusted scores. AP is step-integral Average Precision using absolute z-scores; it and revised `F1-best` are retrospective ranking diagnostics across both directions, regardless of the selected flag direction. `F1-best` chooses an oracle threshold from the labeled result and never changes the detector or alert. Undefined metrics are `null` in JSON, with a `reason` when no evaluable positive labels exist.

## Operational boundaries

The app has no polling, streaming ingestion, persistent alert state, notification delivery, authentication, or remote API. It analyzes an uploaded or bundled snapshot when requested. The four sample CSVs are unlabeled; no detection-quality claim can be inferred from them. Performance measurements are machine-specific, and acceptance gaps are recorded in the [verification guide](verification.md).
