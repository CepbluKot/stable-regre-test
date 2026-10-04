# Signal Lab: local anomaly detection explorer

A CPU-only, stateless Python implementation of the detector concepts in [Uspenskii's Anomaly-Detection-Demo](https://github.com/wwwwwert/Anomaly-Detection-Demo/tree/48400cd4a180adad2f294a9030214b0e6347ac54). It includes ordinary autoregression, stable autoregression, median seasonal decomposition, a recent-window alert preview, optional-label evaluation, and a Streamlit results page. The UI runs locally; no monitoring service or notification destination is connected.

## Documentation

- [User guide](docs/user-guide.md): setup, CSV format, automatic time-grid regularization, controls, charts, exports, and common errors.
- [Technical reference](docs/technical-reference.md): data flow, configuration, model and scoring rules, output fields, alerts, and label metrics.
- [Verification guide](docs/verification.md): tests, manual acceptance checks, benchmark reproduction, and outstanding review gaps.
- [Source mapping](docs/source-mapping.md): reference material and explicit implementation choices.

## Install and run

Python 3.11+ is required. From this directory:

```sh
uv sync --extra dev --locked
uv run streamlit run app.py --server.address 127.0.0.1
```

Open the printed localhost address. The four original sample CSVs are ready in the sidebar. You can also upload a CSV with `timestamp,value_0` and an optional `label` (0/1). Timestamps are Unix milliseconds. Use **Run analysis** after choosing settings. Changing any setting hides the previous result until rerun. Charts update only when you run analysis; there is no timed replay. The Results tab has the full observed values, expected range, anomaly points, scores, alert preview and anomaly intervals. Evaluation appears only if labels are supplied. Data and export has the original-row table and CSV/JSON downloads.

The chart also shows a green **approximate 95% normal-reference interval**, computed as `expected ± 1.95996 × model sigma`. It is a visual guide, not a calibrated statistical confidence or prediction interval; no empirical coverage is claimed. The blue anomaly threshold band remains `expected ± threshold × sigma` and continues to determine flags and alerts. The reference interval is display-only and is not added to exports.

You may also install with pip: `python -m pip install -e '.[dev]'`. `uv.lock` pins the tested dependency resolution for the uv path.

## CLI and library

```sh
uv run python -m anomaly_explorer.cli detect --input data/examples/time_series_1.csv --model stable_ar --output artifacts/example-result.json
uv run python scripts/benchmark.py --output artifacts/benchmark.json
uv run pytest -q
uv run ruff check src app.py scripts tests
```

```python
import pandas as pd
from anomaly_explorer import Config, detect

source = pd.read_csv("data/examples/time_series_1.csv")
result = detect(source, Config(model="stable_ar", threshold=3, percentage=50))
print(result.points.head())
print(result.alert)
```

`result.points` has one row per original row in original order. It includes timestamp, value, expected value, positive sigma, lower/upper bounds, signed and absolute z-score, anomaly and evaluability flags. Metadata includes the effective configuration and preprocessing diagnostics. `result.to_dict()` is strict JSON-compatible. `result.intervals` are contiguous anomalous observations; `result.alert` is the latest-window percentage decision. They are distinct concepts.

## Input and mathematical conventions

The detector fits unsupervised statistics only within the supplied request. No per-series model state persists. Evaluation is retrospective full-context analysis: fitting and detrending can use all observations supplied in that call. This is not a future-forecast or strictly causal streaming validation.

The regular grid uses the modal positive timestamp difference, with the smallest step on a tie. A gap over 12 steps is rejected. Internal missing grid values are interpolated linearly. A final grid point less than one step after the last original observation uses that observation's value. Predictions and sigma are interpolated back to original times, then scores and flags are recomputed against the actual original values. The first p grid points are AR warmup and excluded from alert/evaluation. The request must have at most 100,000 rows and at most 200,000 grid points. A row with a missing or non-finite value is rejected.

The expected range is `expected ± threshold × sigma` in original units. Its Gaussian intuition does not establish calibrated coverage on arbitrary telemetry. The shared fallback scale uses the report's zero-centered `median(abs(residuals))/0.6744897501960817`. Stable AR initializes from its normalized residual standard deviation. Seasonal uses separate, trimmed uncertainty estimates for each phase. See [source mapping](docs/source-mapping.md) for the formulas and assignment choices.

The latest alert uses evaluable original observations in `(as_of-window, as_of]`. It is active only when the anomalous fraction is **strictly greater** than percentage, with a full window and at least three eligible observations. It sends no notifications. Anomaly intervals split on normal points and timestamp gaps. A one-point interval has duration zero.

Optional `label` is used only after detection. Point metrics use the current threshold and selected direction. Revised segment metrics collapse each contiguous ground-truth positive interval into one positive item with its maximum score, while retaining every normal point. AP uses step-integral Average Precision on absolute scores. F1-best chooses an oracle threshold retrospectively and is never applied to normal alerting. The source CSVs are unlabeled, so no accuracy claim is made from them.

## Limits

Stable AR's initial OLS fit may already contain anomalies. Seasonal decomposition can flag a large fraction of points when a data set lacks a stable seasonal pattern or when local phase spread is very small. Choose the model based on series structure and inspect the chart. A production alert requires additional validation on labeled operational data. The local benchmark records CPU times on this machine and is not comparable directly to the authors' published capacity figures. There is no external API, gRPC, MCP, scheduler, alert delivery or real monitoring connection.

Source CSVs are redistributed under the author's [MIT license](LICENSE). The implementation is an independent educational reconstruction of the public report plus explicit choices in `docs/source-mapping.md`; it is not the authors' private production code.

## Project materials

The original [implementation assignment](docs/assignment/01-implementation-prompt.md), [independent review assignment](docs/assignment/02-independent-review-prompt.md), and [context handoff](docs/assignment/00-context-and-handoff.md) are included alongside the supplied [slides](docs/reference/uspensky-anomaly-detection-slides.pdf) and [transcript](docs/reference/uspensky-anomaly-detection-subtitles.txt). Paths inside the two assignment prompts were adapted to this repository. The [self-audit report](artifacts/audit/acceptance-report.md) records completed checks and the remaining independent verification gaps. `dist/` contains handoff archives; the repository root is the editable source of truth.
