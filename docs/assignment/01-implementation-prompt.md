# Implementation task: stateless anomaly detection with a local results UI

You are the implementation agent. Build the working application described below, including tests, a local UI, and measured validation. Do not stop at a plan or a notebook. Work in the project directory provided with this prompt; if none is provided, create `anomaly-explorer` under the current workspace. Inspect applicable instructions and existing files first; preserve unrelated work. No deployment or external notifications are requested.

## 1. Goal and scope

Build a CPU-only, stateless Python library that detects anomalies in univariate time series and explains them through expected values, a band of normal values, and anomaly scores. Add a simple Streamlit + Plotly UI for loading data, choosing a model, inspecting results, and previewing alert rules.

Required models:

1. Ordinary autoregression (`ar`) as a comparison baseline.
2. Stable autoregression (`stable_ar`) with a weighted hidden state and adaptive residual scale.
3. Median seasonal decomposition (`seasonal`) with phase-dependent residual scale.

Required supporting features: input validation, regularization and mapping back to original timestamps, alert preview, incident summaries, optional-label evaluation, CSV/JSON exports, reproducible tests and a small benchmark.

This is a local research/inspection application. Do not build a production monitoring platform. gRPC, MCP, a scheduler, external metric providers, notification delivery, authentication, databases, Kubernetes, GPU models, and a full reproduction of the authors' public benchmark are deferred. Keep the core library independent of Streamlit so later integration is possible.

## 2. Sources and precedence

Reference repository: https://github.com/wwwwwert/Anomaly-Detection-Demo
Pinned revision inspected for this assignment: `48400cd4a180adad2f294a9030214b0e6347ac54`.

Read these at that revision:

- `anomaly_detection.ipynb`: runnable baseline and original result contract.
- `tech_report/paragraphs/4_algorithms.tex`: mathematical target for Stable AR and seasonal scale.
- `tech_report/paragraphs/3_metrics.tex` and `extras/point_adjustment_scheme.tex`: evaluation semantics.
- `tech_report/paragraphs/5_experiments.tex` and `7_integration.tex`: execution and alerting context.
- `time_series/*.csv`, `plots_plotly.py`, `LICENSE`.

Additional original materials on the user's machine:

- `docs/reference/uspensky-anomaly-detection-slides.pdf`
- `docs/reference/uspensky-anomaly-detection-subtitles.txt`

The PDF has 61 image-only pages. The notebook is incomplete relative to the talk: it does not implement Stable AR and uses simpler uncertainty estimates. The technical report provides the missing formulas. This assignment resolves implementation ambiguities explicitly; those choices are engineering defaults, not claims about the authors' private production code. Precedence: explicit user corrections, this assignment, report formulas, slides/transcript, notebook. Record deviations rather than quietly changing the target.

A neighboring `outputs/anomaly-demo` browser demo may exist. It is illustrative, not a numerical oracle. Do not copy its exponential trust-weight formula into Stable AR. Do not treat existing screenshots as proof of your implementation.

Preserve MIT attribution for reused source/data. Do not copy incidental links or unrelated bibliography into the product.

## 3. Execution semantics

- Each detector invocation uses only the supplied series and configuration. No fitted parameters or hidden state survive between calls or series.
- Unsupervised fitting inside one request is allowed. Here “zero-shot” means no previous training, labels, or per-series saved model; it does not forbid fitting AR coefficients to the request's context.
- Analyze the entire supplied context. Do not silently crop to two weeks. Two weeks is an example of a caller-provided context, not a fixed requirement.
- This is full-context retrospective analysis. Fitting/detrending may use the entire supplied context. Do not claim strictly causal streaming behavior or require prefix invariance.
- Alert preview is computed on a recent time window, not on the full historical context.
- Labels never enter fitting, model selection, normalization, scale estimation, or alert configuration. Evaluation receives labels separately.
- Given identical data/configuration, numerical results must be deterministic. Timings need not be.

## 4. Suggested structure and commands

Use Python 3.11+, NumPy, pandas, statsmodels, Streamlit, Plotly, pytest; scikit-learn may be used for evaluation. Resolve and lock a tested dependency set; do not blindly install the notebook's entire Jupyter export.

Suggested structure:

```text
app.py
src/anomaly_explorer/
  schemas.py
  preprocessing.py
  adapter.py
  models/ar.py
  models/stable_ar.py
  models/seasonal.py
  pipeline.py
  alerts.py
  evaluation.py
  cli.py
data/examples/                 # the four original CSVs, unchanged
scripts/benchmark.py
tests/
docs/source-mapping.md
README.md
pyproject.toml
```

Provide working commands equivalent to:

```sh
python -m pip install -e '.[dev]'
python -m pytest
python -m streamlit run app.py --server.address 127.0.0.1
python -m anomaly_explorer.cli detect --input data/examples/time_series_1.csv --model stable_ar --output result.json
python scripts/benchmark.py --output artifacts/benchmark.json
```

One Python process for the UI is sufficient. Do not add a separate web frontend/API service.

## 5. Data and result contract (R01)

CSV input: required `timestamp` (Unix milliseconds, int64) and `value_0` (finite float). Optional `label` must be exactly 0 or 1 for every input row. Ignore extra columns with a visible warning. Do not guess seconds versus milliseconds. ISO strings and alternate column names are optional, not required.

Validation:

- Reject empty data, missing columns, invalid/missing timestamps, non-finite/missing values, duplicate timestamps, invalid labels, and fewer than 3 points with specific errors.
- Unsorted timestamps are allowed: sort internally and restore original input row order in point exports/results. Preserve `row_id` for verification.
- Missing timestamps (gaps) are different from a row with a missing numeric value: interpolate the former; reject the latter in this version.
- Enforce limits before allocating a grid: at most 100,000 source rows and 200,000 grid points. Reject excess; never truncate silently.

Expose a typed `detect(frame, config)` entry point and `DetectionResult`. Input frame is not mutated. The point result has exactly one row per original input row:

```text
row_id, timestamp, value_0,
expected_value, sigma, lower_bound, upper_bound,
signed_z, anomaly_score, is_anomaly, is_evaluable
```

All numeric fields on successful results are finite. `sigma > 0`; `lower_bound <= expected_value <= upper_bound`. Also expose the notebook-compatible arrays `expected_values`, `expected_bounds` with shape `(N,2)` ordered `[lower, upper]`, `anomaly_scores`, `is_anomaly`.

Metadata: model name, full effective configuration, source count, grid count, effective step, warmup count, interpolation count, warnings, pipeline elapsed milliseconds, package version. Do not include labels in the model input. JSON uses ordinary numeric/boolean values; no NaN/Infinity.

## 6. Preprocessing and output alignment (R02)

Infer the step as the most frequent positive difference between sorted timestamps; on ties choose the smallest difference. Allow an explicit positive integer step in milliseconds in advanced settings.

Reject a gap larger than 12 effective steps with a readable message. This bounded-gap policy is a deliberate local-version safeguard, not part of the research formula. Do not manufacture hours of normal-looking observations across an outage.

Build grid `t0 + k*step`, from the earliest timestamp through `ceil((t_last-t0)/step)`. Linearly interpolate values in time onto this grid. If the final grid point is just beyond the last observation, use the last value there; this endpoint extension is less than one step and must be documented. Do not interpolate labels. Track synthetic grid points in diagnostics.

Models operate on the grid. Linearly interpolate their expected values and positive sigma back to original timestamps. Then recompute residuals, scores, bounds and flags from the ORIGINAL observed values. This deliberately improves on the notebook's interpolation of scores: interpolated scores can disagree with an actual observation's band crossing. Preserve original units. Synthetic grid points are not observations in alert percentages or evaluation.

For AR models the first `p` grid points are warmup. At those points use observation as expected value, positive base sigma, score zero, false flag, `is_evaluable=false`. An original point is evaluable only when its timestamp is at or after grid point `p`. Seasonal points are evaluable once its context-length check passes.

No optional smoothing by default: it can erase anomalies. Do not add normalization controls to the basic UI; internal normalization must be reversed correctly.

## 7. Shared adapter (R03)

The model returns expected values and either a positive sigma vector or no sigma. The shared adapter owns scoring and band construction:

```text
residual_t = observed_t - expected_t
signed_z_t = residual_t / sigma_t
anomaly_score_t = abs(signed_z_t)
lower_t = expected_t - threshold * sigma_t
upper_t = expected_t + threshold * sigma_t
```

If the model supplies no sigma, use the report's zero-centered robust estimate:

```text
sigma = median(abs(residuals_on_evaluable_points)) / 0.6744897501960817
```

This is intentionally not `median(abs(r-median(r)))`; the report assumes residuals are centered at zero. Distinguish centered MAD when discussing alternative statistics.

Numerical floor in original units: `eps_x = 1e-8 * max(1, std(values, ddof=0))`. Floor sigma to `eps_x`. Do not replace ordinary small values with arbitrary percentages of series magnitude.

Flags use STRICT comparisons and an evaluability mask:

- both: `abs(signed_z) > threshold`;
- above: `signed_z > threshold`;
- below: `signed_z < -threshold`.

`anomaly_score` remains absolute z for all directions. Threshold must be finite and >0. Equality to a boundary is not anomalous. Increasing threshold cannot change expected values or sigma; it only widens the band and reduces or preserves flags. Neither percentage nor alert-window length changes the detector output.

Call the displayed band “Expected range” or “Normal-value band.” Do not promise calibrated 99.7% coverage or interpret z-scores as probabilities on arbitrary telemetry.

## 8. Ordinary AR baseline (R04)

Default order `p=20`, configurable only under advanced settings. Require integer `p>=1` and `N_grid >= 3*p+1`; reject insufficient data rather than silently changing order. Fit OLS AR(p) with intercept, lags 1 through p and no added trend, on the full supplied regularized context, equivalent to statsmodels AutoReg. No ridge penalty by default. Use a stable least-squares solver, not an explicit inverse of the normal-equation matrix.

For `t>=p`, predict from actual previous observations. Use the shared robust scale estimate on non-warmup residuals. Thus the target baseline retains the notebook's AR backbone but intentionally replaces its RMS residual scale with the report's robust adapter. Explain this difference in source mapping.

For an exactly constant series, bypass the singular fit: expected equals the constant, sigma equals the floor, no anomalies; retain the configured AR warmup policy. Other failures return a named model error, never an unexplained fallback to a different algorithm.

## 9. Stable AR (R05)

Implement the technical report, not a rolling mean, clipping filter, EWMA, or ordinary AR with a new label.

Use the same `p` and context-length check as baseline. Let `mu=mean(x)` and `scale=std(x, ddof=0)`. Handle exactly constant input as above. Normalize `u=(x-mu)/scale`; fit OLS AR(p) with intercept directly in normalized space. This avoids manual conversion of the intercept. Let the ordinary normalized AR predictions be `base_pred`.

Use only `t>=p` residuals for initialization (explicit warmup convention):

```text
eps_u = eps_x / scale
sigma0 = max(std(u[p:] - base_pred[p:], ddof=0), eps_u)
sigma_current = sigma0
h[:p] = u[:p]
previous_prediction = u[p-1]
```

Fixed defaults: `gamma=1.0` (report), `alpha=0.9` (assignment choice because the report gives no default). Advanced validation: gamma>0, 0<=alpha<1.

For each `t=p,...,N-1`, in this order:

```text
prediction = intercept + sum(phi[i-1] * h[t-i] for i in 1..p)
sigma_used[t] = sigma_current
z = (u[t] - prediction) / sigma_current
S = sigma_current / sigma0
r = (0.5*z*z + log(sqrt(2*pi))) / (S*gamma)
w = 1 / (1+r)
h[t] = prediction*(1-w) + u[t]*w
expected_u[t] = prediction
sigma_next = alpha*sigma_current + (1-alpha) * (
    sigma0 + abs(prediction-previous_prediction)/3 * (1-w)
)
previous_prediction = prediction
sigma_current = max(sigma_next, eps_u)
```

The uncertainty used for point t is the pre-update value; the new one applies to t+1. Update timing is an explicit assignment convention. Return `expected_x=expected_u*scale+mu` and `sigma_x=sigma_used*scale`. Scale has units: never compare normalized sigma to raw observations.

Do not substitute `exp(z²/2)` for the quadratic expression above. Expected value is the prediction, NOT h. The hidden state is private to the request. User threshold does not participate in hidden-state recursion. Make the recurrence separately testable, with optional diagnostic traces of predictions, weights and scales for tests.

Do not claim this guarantees detection of every sustained anomaly: the initial OLS fit itself can be contaminated. Measure behavior and disclose limitations instead of tuning against a test label.

## 10. Seasonal detector (R06)

Require an explicit season duration; default one day, visible when seasonal is selected. Convert it to an integer number of grid steps L. Reject non-integral multiples, L<3, and contexts with fewer than three complete seasons (`N_grid < 3*L`). Do not silently invent a shorter season. Automatic period selection is deferred.

1. Fit an OLS linear trend `a+b*t`, using sample index t, and subtract it.
2. Default neighborhood radii `ws=w=min(2, floor((L-1)/2))`; allow valid nonnegative integer overrides with `2*radius+1<=L` in advanced settings. These defaults are assignment choices.
3. For phase i, collect unique valid indices `q*L + ((i+j) mod L)` for every available season q and j in [-ws,ws]. Include a partial last season only at indices that exist. Seasonal profile at i is the median of these detrended observations. Repeat by phase. This fixes the report's ambiguous `k ± iL + j` indexing using its explicit cyclic phase convention.
4. Expected value is trend plus seasonal profile; residual is x minus expected.
5. For each phase i and each season q, collect residuals at unique valid indices `q*L+((i+j) mod L)`, j in [-w,w]. Ignore sets with fewer than 2 observations. Compute their `std(ddof=0)`.
6. For the available per-season standard deviations of phase i, compute Q0.8 with linear quantile interpolation; retain values <=Q0.8 and take their median. If no estimate exists, use global `median(abs(residual))/0.6744897501960817`. Floor the result at eps_x.
7. Repeat the phase scales, truncate to N_grid, and call the shared adapter.

Do not replace this with statsmodels STL/LOESS or a single global sigma. The same numerical scale must drive the band and z-score.

## 11. Alert preview and incidents (R07)

The UI has two main sensitivity controls: threshold=3 and percentage=50. Also show model choice, direction (both/above/below), and a time window (default 30 minutes); these are distinct settings, not hidden model hyperparameters.

`evaluate_alert(points, as_of, window_duration, percentage)` uses original, evaluable observations whose timestamps satisfy `(as_of-window_duration, as_of]`. Reject percentage outside [0,100] and nonpositive windows. `as_of` defaults to the last ORIGINAL timestamp, never wall-clock time for a historical CSV.

- Require context start <= as_of-window_duration and at least 3 evaluable observations in the window. Otherwise status=`INSUFFICIENT_DATA`, active=false, with a reason.
- fraction = anomalous eligible observed points / eligible observed points.
- active = `fraction*100 > percentage` (strict, matching the report's “exceeds”; exactly 50% does not trigger at percentage=50).
- Old anomalies outside the window do not count. Synthetic interpolated points never enter numerator or denominator.
- Export `as_of`, window start/end, numerator, denominator, fraction, status and effective config.

Build anomaly intervals from consecutive anomalous original observations in chronological order, splitting whenever a non-anomalous/unevaluable point intervenes or a timestamp gap exceeds 1.5*step. Do not merge across missing observations. Record start, end, duration_ms=end-start, point_count, max_score and direction (above/below/mixed). A one-point interval has duration zero and point_count one. These are anomaly intervals, not delivered alerts. Latest-window alert status is a separate object. Historical notification replay and deduplication are out of scope.

## 12. Evaluation (R08)

The bundled four CSVs have NO ground-truth labels. Never display accuracy/F1 for them. With optional labels, evaluate only original evaluable points sorted chronologically. Excluded points break ground-truth segments; timestamp gaps >1.5*step also break them.

Report point-wise precision/recall/F1 at the current threshold and direction. For ranking metrics use absolute scores (both-direction evaluation), and label that convention even when alert direction is above/below.

Implement Revised Point Adjustment explicitly as segment collapse:

- Each contiguous ground-truth positive segment becomes ONE positive item whose score is the maximum detector score in that segment.
- Every ground-truth normal point remains one individual negative item with its own score.
- Thresholding collapsed scores yields one TP per detected incident, one FN per missed incident, and one FP per false-positive normal point.
- Do not fill the entire segment with TP points and call that the same metric.

Report revised precision/recall/F1, revised F1-best and best_threshold, and revised PR average precision (AP). Use step-integral Average Precision, e.g. `average_precision_score`, rather than silently switching to trapezoidal PR area. Also show point-wise AP. State the metric definition in docs/UI.

For F1-best evaluate thresholds consisting of all unique collapsed scores plus `nextafter(min_score, -infinity)`. These cover all attainable partitions, including all-negative and all-positive cases, respecting strict `score>threshold` and ties. Label this as a retrospective, label-assisted oracle diagnostic. Never feed best_threshold into normal detection or auto-select it for a user. Ties for F1-best choose the largest threshold among evaluated candidates. It is acceptable for the all-positive evaluation candidate to be negative; that is not a permitted production threshold.

If there are no positives, show recall/F1/AP/F1-best as N/A with a reason and report false positives explicitly. Precision is N/A when TP+FP=0. With positives and no predictions, recall=0, F1=0; do not allow undefined precision to hide a missed incident. No-label input shows “No labels: quality metrics unavailable.”

## 13. Simple UI (R09)

One page with sidebar and three tabs is sufficient:

- Sidebar: four example datasets or CSV upload; model; threshold; percentage; window duration; direction; seasonal duration when relevant; Run analysis. Advanced expander holds p, alpha, gamma, neighborhoods and step override.
- Results tab: observed series, expected curve, translucent normal-value band and distinct anomaly markers in original units; score chart with threshold; latest-window shading; zoom/pan/hover and readable timestamps. Sort charts chronologically even when exports preserve input order.
- Summary above charts: point count, anomaly count, interval count, elapsed time, latest alert state and fraction. Warn about warmup/interpolation. Selecting an interval in a table should focus that time range, or provide a time-range selector if simpler.
- Evaluation tab: metrics only if labels exist, definitions and an explicit distinction between current-threshold and oracle metrics.
- Data/export tab: point table, effective configuration, CSV point download, JSON points + incidents + alert + metadata download.

Changing inputs marks previous results stale until rerun; the UI must never present old results as matching new controls. An error replaces active results with an actionable message, not a stale success chart. UI, CLI and library share the same pipeline. Cache raw example files if useful; do not cache fitted models across calls. Avoid a cache that hides benchmark execution.

No login, chat assistant, dashboards with invented stats, external fonts requirement, or fake monitoring integrations. Make it usable at 1280x800 and on a narrower viewport without overlapping controls.

## 14. Tests and acceptance evidence (R10)

Implement meaningful tests, including independent expected-value examples:

- Adapter: observations [10,16,4], expected [10,10,10], sigma [2,2,2], threshold=3 => scores [0,3,3], bounds [4,16], no anomalies; threshold=2 => flags [false,true,true]. Above and below select opposite single points.
- Zero-centered scale: residuals [1,1,1,10] => sigma=1/0.6744897501960817, not zero.
- Stable recurrence: u=2, prediction=0, previous_prediction=0, sigma_current=sigma0=0.2, gamma=1, alpha=0.9 => z=10, r≈50.918938533204674, w≈0.01926079438932388, h≈0.03852158877864776, next sigma=0.2. Separately use changing predictions to exercise nonconstant sigma.
- Check lag ordering, normalized-to-original conversion, warmup, partial last season, cyclic phase neighbors and Q0.8 trimming against a small independently computed fixture.
- Sorted vs permuted input yields the same timestamp-associated results, restored to original row order. Missing timestamps, irregular timestamps and non-grid-aligned final timestamps preserve count and score/band consistency.
- Constant, all-zero, short, invalid, duplicate, huge-gap and excessive-grid inputs produce finite results or the specified explicit error.
- Threshold monotonicity and invariance of expected/sigma; changing percentage/window cannot alter detector scores. A -> B -> A calls return identical numerical A results. Input mutation and labels have no influence.
- Alert window with four eligible points, two anomalies, percentage=50 is inactive; three anomalies is active. Also test exact left boundary, no data, warmup, old anomalies and fewer than 3 points.
- RPA fixture: labels [0,0,1,1,1,0,0,1,1,1], scores [.2,.3,.5,.9,.8,.7,.8,.1,.1,.1], threshold=.6 => collapsed labels [0,0,1,0,0,1], scores [.2,.3,.9,.7,.8,.1], TP=1, FP=2, FN=1, precision=1/3, recall=1/2, F1=.4. Revised AP=2/3 and F1-best=2/3 under the prescribed conventions.

Create reproducible synthetic examples with fixed seeds: clean noisy baseline; isolated positive/negative spikes; a temporary level shift lasting >p; seasonal data with a missing expected peak; seasonal data with higher daytime than nighttime noise. Compare AR and Stable AR and show seasonal scale varies with phase. Freeze fixtures/configurations before comparing outputs; do not optimize seeds or thresholds until a favored model wins. Formula conformance is mandatory; if a behavioral advantage is absent on a fixture, report it honestly and investigate rather than faking acceptance.

Run all three models on all four original CSVs (using seasonal duration compatible with their history). Original row counts are 4600, 4020, 6336, 9216 for files 0,1,2,3 respectively. Verify finite, aligned output and inspect real charts. No external data download is required beyond the reference repo.

Benchmark at least three fresh calls per case, without result cache; record each timing and median, row/grid counts, parameters, Python/library versions, CPU identity and thread settings. Record cold process/first-call timing separately from repeated-call timing. Aim for interactive execution (roughly <=2 seconds for ~4,032 points on the test machine); this is an engineering target, not a universal SLO or a reason to falsify results. Report slower cases and resource risks explicitly.

Run the actual UI and exercise sample selection, upload, all three models, controls, invalid input, zoom and both downloads. Save screenshots and evidence. Unit tests or an HTTP 200 alone do not establish UI acceptance. If browser interaction is unavailable, report that part as unverified.

## 15. Implementation order and final handoff (R11)

Complete in small verifiable stages:

1. Read sources; create source mapping and lock the contracts/defaults above.
2. Build schemas, validation, regularization and adapter; test boundaries.
3. Implement baseline AR, Stable AR, seasonal model; test numerical formulas.
4. Add alignment, alerts, intervals and exports; test time semantics.
5. Add independent evaluation; test the worked metric fixture.
6. Add UI using the same pipeline; exercise the actual workflow.
7. Run all sample cases, benchmark and document evidence.

Deliver runnable source, locked dependencies, tests, unchanged sample CSVs with attribution, setup/run instructions, `docs/source-mapping.md`, known limitations, benchmark JSON/CSV and a concise completion report. Source mapping must distinguish report formulas, notebook behavior and assignment choices. Report passed/failed/unverified requirements by R01-R11. Never claim reproduction of the authors' benchmark quality or production capacity from four unlabeled examples. Do not stop with stubs, TODO implementations, or charts made from precomputed fake results.
