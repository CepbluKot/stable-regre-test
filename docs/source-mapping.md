# Source mapping and explicit choices

Pinned reference: `wwwwwert/Anomaly-Detection-Demo` commit `48400cd4a180adad2f294a9030214b0e6347ac54`. Additional context: supplied 61-page slide PDF and 609-line subtitles. The report's LaTeX source is the numerical reference; the notebook is a demonstration baseline.

| Implemented feature | Source | Local choice / difference |
|---|---|---|
| Structured result and interpolation | notebook cells 14, 20-26; report §7 | Interpolate expected value and sigma, then recompute score/flag from original observation. The notebook interpolates scores, which can disagree with band crossing. Bounds are `[lower, upper]`; plotting helpers reverse these names. |
| Shared z adapter | report §4, z-test | Zero-centered median absolute residual scale divided by 0.6744897501960817; epsilon `1e-8*max(1,std(grid))`. No calibrated coverage claim. |
| AR baseline | notebook cell 16; report §4 | Same OLS AR backbone; robust residual scale instead of notebook RMS. First p points are masked warmup. |
| Stable AR | report §4, Stable Autoregression | Exact quadratic trust ratio, normalized hidden state and dynamic sigma. `gamma=1` from report; `alpha=.9`, excluding warmup from sigma0, using pre-update sigma and starting previous prediction at u[p-1] are assignment choices. |
| Seasonal | report §4, Seasonal Decomposition | Linear detrend; cyclic phase median; phase-local std, Q0.8 trim. Default profile/scale radii 2 (or max fitting); one-day period required by default. These defaults and the explicit cyclic indexing resolve underspecified report details. |
| Evaluation | report §3 and §5, revised point-adjustment diagram | One item per positive ground-truth segment, maximum score; normal points separate. AP is step-integral Average Precision. Current-direction threshold metric is distinct from absolute-score ranking diagnostics. |
| Alerting | report §7; slides 45-48 | Threshold and percentage are main controls. Alert window defaults to 30 minutes; strict `>` percentage. Latest window uses original observations and last timestamp as as_of. No notification replay. |
| Local UI | talk slides 40-42; report §7 | Streamlit plus Plotly, four original CSVs, upload and exports. No monitoring API, gRPC, MCP or production claims. |

The report's published CPU/RAM figures disagree with the slide excerpt, and dataset counts differ between narrative and table. They are not accepted as this implementation's measurements. The notebook has no Stable AR and uses a simplified global seasonal scale. The neighboring browser-only illustration uses a different exponential trust weight; this implementation follows the report formula instead.

Other policy choices: sorted internal computation with original-row output, modal positive step with smallest tie, a 12-step maximum gap, 100k input/200k grid caps, reject missing numeric rows, no label interpolation, full window plus three eligible observations for alert preview. Those are requirements of the accompanying local implementation assignment, not claims about private Monium behavior.
