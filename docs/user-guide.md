# User guide

Signal Lab is a local, retrospective anomaly explorer. It analyzes the CSV currently selected in the sidebar when you click **Run analysis**. It does not watch a data source or update charts on a timer.

## Start the app

From the repository root, with Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --extra dev --locked
uv run streamlit run app.py --server.address 127.0.0.1
```

Open the address printed by Streamlit. `127.0.0.1` makes the app available on this machine only. Select one of the four bundled examples, or upload your own CSV. An uploaded file takes precedence over the example selector. Click **Run analysis** after changing the file or settings. The old result is hidden when its settings no longer match the sidebar; it is not silently reused.

## CSV input

Use a header row and these columns:

| Column | Required | Meaning |
|---|---|---|
| `timestamp` | Yes | Distinct Unix time in **milliseconds**, as an integer. UTC is used for chart labels. Seconds or formatted dates are not inferred. |
| `value_0` | Yes | A finite numeric observation. Missing values, `NaN`, and infinity are rejected. |
| `label` | No | Ground truth for retrospective evaluation: exactly `0` (normal) or `1` (anomaly). Labels do not affect detection. |

For example, these timestamps are five minutes apart:

```csv
timestamp,value_0,label
1700000000000,10.0,0
1700000300000,10.2,0
1700000600000,14.5,1
```

The CSV may be in any row order. Output rows retain that original order and receive a zero-based `row_id`; charts are sorted by time. Duplicate timestamps are rejected. Extra columns are ignored with a warning. At least three rows are required; the limit is 100,000 input rows and 200,000 points on the internal grid. Some models need more history than the three-row minimum.

### Do timestamps need equal spacing?

Usually no. Signal Lab infers a step from the most common positive gap between sorted timestamps; if several gaps tie, it chooses the smallest. You can instead set **Step override (ms; 0 = infer)** in Advanced model settings. It builds an evenly spaced internal grid and linearly interpolates values at missing grid positions. A final grid point less than one step beyond the last observation uses the last value. Gaps longer than 12 steps are rejected, so split such a series or choose a valid step. The UI reports how many grid points were interpolated.

Interpolation is for model fitting only. The result has one row per **actual input observation**, including off-grid timestamps. Expected value and uncertainty are mapped back to those timestamps, and scores are recomputed against their actual values. Artificial grid points do not count toward alert percentages, anomaly intervals, or label metrics. A missing `value_0` in an existing row is an error; it is not treated as a missing timestamp. If sampling intervals vary substantially without a meaningful base cadence, inspect the inferred `step_ms` and warnings before trusting the model.

## Controls

| Control | Effect |
|---|---|
| Model | `stable_ar` (default), `ar`, or `seasonal`. See [technical reference](technical-reference.md) for how they differ. |
| Threshold | Width of the blue anomaly band in multiples of model uncertainty; default 3. A point is flagged only when its score is **strictly greater** than the threshold. |
| Anomalous points for alert | Percentage required in the recent window; default 50%. The fraction must be **strictly greater** than this setting. |
| Recent window | Duration of the latest alert preview; default 30 minutes. It does not limit the fitted history. |
| Direction | Flag both sides, only above expected, or only below expected. |
| Season duration | Cycle length for the seasonal model; default 24 hours. It must be an integer multiple of the effective grid step and have at least three steps. |
| AR order | Number of lags for both AR models; default 20. At least `3 × order + 1` grid points are needed. |
| Stable AR gamma / alpha | Trust-weight and scale-smoothing parameters; defaults 1 and 0.9. |
| Seasonal profile / scale radius | Number of neighboring phase steps on each side, within a cycle; defaults 2. |
| Step override | Fixed sampling step in milliseconds. Zero means infer it from timestamps. |

The advanced fields for other models can stay at their defaults; the detector uses them only for their corresponding model.

## Reading the results

**Results** contains counts, elapsed computation time, the latest alert preview, any data/model warnings, and two charts. The first chart shows observed and expected values, red anomaly markers, a blue anomaly threshold band, a green approximate 95% normal-reference interval, and a shaded latest alert window. The **Focus** selector zooms to an anomaly interval. The second chart shows absolute z-scores and the selected threshold. The interval table describes contiguous flagged observations; the JSON below it describes the single latest-window alert decision. These are different outputs. Plotly zoom/pan controls change only the view.

The green band is `expected ± 1.95996 × sigma`. It is a visual normal-reference guide, **not** a calibrated confidence or prediction interval. It does not decide flags, alerts, or exports. The blue band is `expected ± threshold × sigma` and does decide flags. A `>30%` flag rate adds a warning because it often indicates a poor model fit; it is not proof that a third of the data is truly anomalous.

The alert preview uses only evaluable original observations in `(latest timestamp − window duration, latest timestamp]`. It needs a full window of source history and at least three eligible observations. Otherwise its status is `INSUFFICIENT_DATA`. When there is enough context, status is `ACTIVE` only if the anomalous fraction is strictly above the selected percentage; otherwise it is `OK`. No notification is sent.

**Evaluation** appears when the CSV has `label`. It shows point and revised-segment precision/recall/F1 for the selected threshold and direction, plus retrospective ranking metrics. Without labels, quality metrics are unavailable. The four bundled example series have no labels; their flag counts cannot establish accuracy. See [technical reference](technical-reference.md#evaluation) for metric definitions.

**Data and export** shows metadata, alert details, and the full original-row result table. **Download point CSV** exports that table. **Download full JSON** includes points, metadata, alert, intervals, and evaluation. The green reference interval is display-only and is not an export field.

## Common errors

| Message or symptom | What to check |
|---|---|
| Missing `timestamp` or `value_0` | Use the exact column names and a header row. |
| Non-finite timestamp/value or invalid label | Remove blank/non-numeric values; use integer milliseconds and labels 0/1. |
| Duplicate timestamps | Keep one observation per timestamp, or aggregate duplicates before upload. |
| Gap exceeds 12 steps | Check the inferred cadence; split disconnected series or use a justified step override. |
| AR needs more grid points | Reduce AR order or provide more history. |
| Seasonal needs three complete seasons / integral period | Provide enough cycles and choose a season duration divisible by the grid step. |
| Settings changed warning | Click **Run analysis** to compute with the current sidebar settings. |

This is exploratory software, not an operational alerting service. Models fit the supplied dataset with full retrospective context, so scores are not a causal future-forecast evaluation.
