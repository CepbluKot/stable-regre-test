# Independent acceptance task: anomaly detector and results UI

You are an independent verification agent. Audit the completed implementation against the target below and the accompanying `01-implementation-prompt.md`. You are not the implementer. Do not accept its completion report, screenshots, test names or README as proof. Inspect code, independently calculate expected outputs, execute tests, and interact with the running UI.

Your deliverable is a reproducible acceptance report and a prioritized repair task. Do not modify production code or weaken acceptance requirements. You may add isolated audit scripts/fixtures under `audit/` and write reports under `artifacts/audit/`. Preserve user changes. Do not deploy, send notifications, alter source datasets, or update dependency versions to conceal a failure.

## 1. Establish the target before looking at implementation claims

Expected product: a local CPU-only Python library and a simple Streamlit/Plotly UI for univariate time-series anomaly detection, using ordinary AR, Stable AR and median seasonal decomposition. Required pipeline: validate -> regularize -> model -> expected value and uncertainty -> original-timestamp alignment -> shared z-score and bounds -> alert preview / incident summaries / optional-label evaluation -> UI and exports.

Sources:

- Repository https://github.com/wwwwwert/Anomaly-Detection-Demo at `48400cd4a180adad2f294a9030214b0e6347ac54`.
- Numerical target: `tech_report/paragraphs/4_algorithms.tex`.
- Metric target: `tech_report/paragraphs/3_metrics.tex`, `extras/point_adjustment_scheme.tex`.
- Integration context: `tech_report/paragraphs/7_integration.tex`.
- Demonstration, not complete target: `anomaly_detection.ipynb`.
- Talk PDF: `docs/reference/uspensky-anomaly-detection-slides.pdf`.
- Transcript: `docs/reference/uspensky-anomaly-detection-subtitles.txt`.

Read the accompanying implementation assignment as the normative acceptance contract. It contains explicit resolutions for ambiguities in the research material. If it is not accessible, use this prompt's checks, but mark conformance to its remaining details UNVERIFIED; do not invent a missing specification. An implementer's own rewritten spec is not a substitute.

Scope excludes gRPC/MCP servers, external monitoring integration, production deployment, notification transport, GPU models, and a full 631-series research benchmark. Do not fail the local application for lacking those deferred features. Conversely, a nice UI with only the notebook's ordinary AR is insufficient.

## 2. Inspection and execution procedure

1. Record implementation path, Git revision/dirty status where applicable, runtime versions, machine information and source revision. Read local instructions before execution.
2. Build a requirement matrix for R01-R11 from the implementation assignment. Do not let the implementer's checklist define your test coverage.
3. Follow README installation/start commands in an isolated environment. Record actual commands, exit codes, durations and relevant output. Distinguish environment/network blockers from code defects.
4. Inspect numerical implementation and data flow. Locate actual code for recurrence, scale estimation, original-timestamp mapping, segment collapse and alert-window selection.
5. Run repository tests, then add independent checks below. Independent expected values must not call the same production helper being tested.
6. Exercise all bundled CSV/model combinations and synthetic cases. Compute output invariants independently.
7. Start the UI, inspect real rendered charts and exercise controls/uploads/downloads. Compare exported results to direct library/CLI calls with identical settings.
8. Write the acceptance report, evidence manifest and repair task. Do not repair while auditing or certify work you have not rerun.

Use a known configuration throughout; save it with every artifact. Numerical comparisons: normally rtol=1e-6, atol=1e-8 for nondegenerate fixtures. Justify different tolerances; do not demand bit-identical OLS coefficients for ill-conditioned data.

## 3. Non-negotiable mathematical checks

### Shared adapter

For evaluable original observations:

```text
z_signed = (value - expected)/sigma
score = abs(z_signed)
lower = expected - threshold*sigma
upper = expected + threshold*sigma
```

Sigma must be positive, in the same units as values. Both-direction flag is `score>threshold`; above is `z_signed>threshold`; below is `z_signed<-threshold`. Equality is normal. The expected range is not a calibrated probability guarantee.

The report's fallback scale is `median(abs(residual))/0.6744897501960817` under its zero-centered assumption, not centered MAD or RMS. Distinguish this from Stable AR's specific initialization with standard deviation and seasonal local scale estimation.

Independent fixture: x=[10,16,4], expected=[10,10,10], sigma=[2,2,2]. At threshold 3: scores=[0,3,3], lower=[4,4,4], upper=[16,16,16], flags all false. At threshold 2: flags=[false,true,true]. Check above/below separately. Residuals=[1,1,1,10] must yield fallback sigma≈1.4826022185.

### Ordinary AR

OLS AR(p) with intercept and actual previous observations; default p=20, no silent ridge or order reduction. First p grid points are explicitly unevaluable. Baseline uses shared robust scale rather than notebook RMS; that difference is prescribed, not a failure.

### Stable AR

Read its loop carefully. It must fit once in normalized space and then predict from the evolving hidden state, not the raw lag buffer. For each point, using PRE-UPDATE scale:

```text
prediction = c + sum(phi_i*h[t-i])
z = (u[t]-prediction)/sigma
S = sigma/sigma0
r = (0.5*z*z + log(sqrt(2*pi)))/(S*gamma)
w = 1/(1+r)
h[t] = prediction*(1-w)+u[t]*w
sigma_next = alpha*sigma + (1-alpha) * (
    sigma0 + abs(prediction-previous_prediction)/3*(1-w)
)
```

Gamma defaults to 1, alpha to 0.9. Base sigma is std of normalized baseline residuals excluding warmup, ddof=0. The returned expectation is the prediction, not h. Convert both expectation and sigma back to original units. There is no exponential in r. User threshold must not affect weights or forecasts. All state resets per invocation.

Golden single-step fixture: u=2, prediction=previous_prediction=0, sigma=sigma0=.2, gamma=1, alpha=.9 gives z=10, r=50.918938533204674, w=.01926079438932388, h=.03852158877864776, next sigma=.2. Also verify a nonzero prediction change to expose an implementation with constant uncertainty. Check lag coefficient ordering against an independent short recurrence with supplied coefficients.

Check constant/zero data, small nonzero residual scales, normalization at very large value offsets and affine transformations. Do not apply an exact scale-invariance assertion where the specified numerical floor dominates.

### Seasonal decomposition

Must contain all of: OLS linear detrending; a phase median over cycles and neighboring phases; separate per-phase uncertainty; restoration of original trend/units.

For phase i, season q and local offset j, valid unique indices are `q*L+((i+j) mod L)`. Profile uses median detrended observations over ws-neighborhoods. For scale, compute per-season std(ddof=0) of residuals in w-neighborhoods; skip neighborhoods with <2 observations; discard estimates above the linearly interpolated Q0.8; median the remainder. Empty estimate falls back to global zero-centered robust scale; apply the prescribed epsilon floor.

Verify phase wrap at 0/L-1, partial final season and no duplicate counting. Period must be an integral number of grid steps, L>=3 and at least three complete seasons must exist. No silent replacement with a shorter period. A global sigma or ordinary STL is not this target.

## 4. Input, alignment, statelessness and leakage checks

Prepare inputs with unsorted rows, a missing timestamp, irregular timestamps, a final off-grid timestamp, duplicate timestamp, missing value, NaN, Infinity, malformed label, too-short history, a huge gap and a grid exceeding its cap.

Expected policies from the assignment:

- Timestamp is Unix milliseconds; do not infer seconds by magnitude.
- Reject invalid/missing numeric data and duplicates. Accept unsorted rows, returning exactly one result per original row in original order; charts sort in time.
- Mode positive step, ties choose smallest; optional explicit step. Gap >12 steps is rejected; excessive row/grid count is rejected before allocation, not silently truncated.
- Linear interpolation builds the internal grid; labels are never interpolated.
- Interpolate expected values AND sigma back to original timestamps, then recompute scores/flags from ORIGINAL observations. Interpolating z-scores or flags alone is a defect.
- Every returned point is finite; positive sigma; bounds ordered; array lengths and row IDs match. Warmup is flagged unevaluable and excluded from metrics/alerts.

Check threshold 2/3/5 on the same series: expected/sigma remain identical, bounds widen, anomaly set cannot grow. Changing percentage or window cannot alter detector scores. Analyze A, then B, then A: numerical A results match; input frames remain unchanged.

Change label values only: detector results must remain unchanged. Inspect for label-assisted threshold/model selection, seed cherry-picking, fitted-model caches and persistent hidden state.

Full-context fitting is intended. Do not flag use of later observations within a supplied retrospective context as leakage by itself. Do flag claims of strictly causal streaming behavior or future predictive performance that the implementation has not demonstrated.

## 5. Alert semantics and intervals

Use `as_of=last_original_timestamp` for a historical CSV. The time interval is `(as_of-window, as_of]`. Count only original evaluable observations. Synthetic points do not dilute or increase the fraction.

Require full window context and at least 3 eligible observations, otherwise INSUFFICIENT_DATA and active=false. Active iff `100*count_anomalous/count_eligible > percentage`, strictly. Check 2/4 at 50% is inactive, 3/4 is active; check exact left boundary, old anomalies outside the window, warmup, percentage 0/100 and empty windows. Ensure changing input labels cannot affect the alert.

Anomaly intervals are contiguous anomalous original observations, splitting on a normal/unevaluable point or gap >1.5*step. A single point has duration_ms=0 and point_count=1. Do not confuse an anomalous point/interval with an active rolling-window alert or a delivered notification.

## 6. Independent metric oracle

Revised Point Adjustment here collapses each ground-truth positive segment into ONE item with its maximum score. Normal points stay separate. Gaps >1.5*step and unevaluable points break segments. Check first/last segments, one-point segments, multiple incidents, ties, no predictions and no positives.

Use this fixed fixture, without importing production metric helpers:

```text
labels: [0,0,1,1,1,0,0,1,1,1]
scores: [.2,.3,.5,.9,.8,.7,.8,.1,.1,.1]
threshold: .6
collapsed labels: [0,0,1,0,0,1]
collapsed scores: [.2,.3,.9,.7,.8,.1]
TP=1, FP=2, FN=1
precision=1/3, recall=1/2, F1=.4
revised AP=2/3
revised F1-best=2/3
```

AP is step-integral Average Precision, not trapezoidal PR area. Check F1-best across all attainable partitions with strict comparisons and tied scores. Do not accept standard point-adjustment that fills all anomalous points as TP and overweights long segments. Do not accept a formula that collapses only missed incidents.

Current-threshold metrics and oracle F1-best must be separated. Best threshold must never automatically become the default alert threshold. Ranking metrics use absolute scores and explicitly say both-direction evaluation; threshold metrics honor the selected direction. No positives => N/A for recall/F1/AP/F1-best and explicit FP count. No labels => no quality claims.

The four source CSVs have only timestamp/value_0, with row counts 4600/4020/6336/9216. They cannot substantiate detection accuracy. Published Yahoo F1=.95 and production resource numbers are not acceptance thresholds for this local implementation.

## 7. Behavioral and performance evidence

Use fixed-seed clean noise, signed spikes, a temporary shift longer than p, seasonal missing peak and day/night heteroscedasticity. Inspect raw scores, forecasts, scales and interval coverage, not just a handpicked screenshot. Compare baseline vs Stable AR after the start of a sustained deviation. Check daytime/nighttime median widths for the seasonal model, away from phase-transition neighborhoods.

Separate formula conformance from empirical usefulness. A failed behavior test requires evidence and explanation; never force a favorable threshold or claim that one fixture proves universal superiority. Run an additional seed not used by the implementer as a diagnostic, without redefining the contractual pass criteria afterward.

Run all 12 bundled model/data combinations. Record actual settings, row/grid counts, finite-output checks and fresh uncached timings. Measure at least 3 calls per case and retain individual values and median. Record first/cold execution separately. Compare the stated interactive target (~2 seconds at ~4,032 points) to actual hardware; slower timings are a documented performance limitation unless the user has imposed a hard SLO. Do not compare unrelated hardware to the authors' table as if equivalent.

## 8. Actual UI acceptance

Start using the documented command and open the page in a real browser if available. Test:

1. Choose each original dataset and all three models; obtain real results.
2. Upload a new valid CSV and a malformed CSV; verify useful errors and no stale success state.
3. Change threshold, percentage, direction, window and seasonal duration; rerun and validate the resulting numbers independently.
4. Inspect expected curve, ordered filled band, actual values, anomaly markers, score/threshold chart, recent-window marking and hover units/timestamps.
5. Zoom or select a time interval. Check readable controls at 1280x800 and a narrower width.
6. Check no-label vs labeled evaluation, including clear current-threshold/oracle labels.
7. Download CSV and JSON. Parse them independently, verify counts/order/finite values/configuration/alert summary and compare to library/CLI results.
8. Ensure edits mark displayed results stale until rerun; invalid new runs cannot leave old output labeled current.

A successful server health check is not UI acceptance. Save screenshots linked to exact data/configuration. If browser tools are unavailable, mark UI interaction UNVERIFIED and explain what was checked instead. Do not report screenshots you did not capture.

## 9. Verdict and report format

Create `artifacts/audit/acceptance-report.md`, `artifacts/audit/evidence.json`, and `artifacts/audit/repair-task.md`.

Report structure:

- Verdict: PASS / FAIL / BLOCKED.
- Assessed implementation revision and environment.
- Requirement table: ID, target, PASS/FAIL/UNVERIFIED, evidence path, command or reproduction step.
- Numerical conformance findings, including expected versus actual values.
- UI and export findings with screenshots where actually captured.
- Behavioral results, actual timings and limitations.
- Defects ordered by impact: P0 unusable/data loss; P1 wrong algorithm/metrics/alignment/alert or missing required model; P2 incomplete workflow/diagnostics/robustness; P3 polish. Assign severity based on impact, not just a keyword.
- Explicit unverified items and environmental blockers.

PASS requires every mandatory R01-R11 behavior to have evidence, no outstanding substantive defects, and actual UI verification. FAIL applies when an observed mandatory requirement fails, even if other checks are blocked. BLOCKED applies when missing prerequisites prevent a verdict and no definite mandatory failure has been established. Performance aspirations and deferred integrations are recorded separately and do not silently become mandatory gates.

For each defect provide: requirement ID, file/line, minimal input/configuration, exact reproduction command or UI steps, expected result, actual result, consequence, and the smallest repair direction. `repair-task.md` must be directly usable by a weaker implementation model and request a regression test for each substantive bug. Do not prescribe a wholesale rewrite when a local fix suffices.

End with a concise statement of what was proven, what failed, and what remains unverified. Do not say “everything works” based only on the implementer's tests.
