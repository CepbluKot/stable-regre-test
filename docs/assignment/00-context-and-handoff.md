# Anomaly detection: context review and model handoff

Prepared 2026-10-04. These documents specify a future implementation; no implementation or acceptance run was performed as part of this handoff.

## Use the two assignments

1. Give `01-implementation-prompt.md` to the implementation model in its intended project directory. It contains the complete target, formulas, defaults, data contracts, UI requirements, implementation stages and acceptance examples.
2. After implementation, give `02-independent-review-prompt.md` plus `01-implementation-prompt.md` to a fresh verification model with access to that project. The second task produces an evidence-based verdict and a concrete repair assignment. It does not trust the implementer's claimed success.
3. If repairs are needed, return the generated repair task to the implementer and then repeat the affected audit checks and relevant regressions.

The prompts are in English, following the workspace's language default. They do not require previous conversation history. The source paths are supplied for this machine, and a pinned GitHub revision is supplied for portability.

## What was studied

- All 61 pages of the supplied image-only slide PDF, rendered and visually inspected. It is an excerpt assembled from talk frames, with duplicate/transition pages; PDF page numbers differ from the visible slide numbers.
- The full 609-line subtitle file. Automated transcription contains errors, so mathematical details were cross-checked against the report.
- The reference repository at commit `48400cd4a180adad2f294a9030214b0e6347ac54`: README, all 30 notebook cells, both plotting helpers, dependency manifests, all four CSV schemas/counts, license, and the technical report's substantive LaTeX sections and diagrams/tables.
- The README and detector code of the neighboring browser demo, solely to distinguish an existing illustration from the requested target.

Pinned source: [Anomaly-Detection-Demo](https://github.com/wwwwwert/Anomaly-Detection-Demo/tree/48400cd4a180adad2f294a9030214b0e6347ac54).

## Intended solution, distilled

The central idea is a shared, interpretable output contract. A forecasting backbone estimates normal behavior; the adapter converts residuals and uncertainty into z-scores, expected bounds and anomaly flags. Stable AR prevents strong outliers from fully entering its lag context. The seasonal model compares phases across cycles and estimates uncertainty by phase. A separate recent-window percentage rule determines whether the anomalous observations justify an alert.

The target is stateless across calls, zero-shot in the sense of no previously fitted per-series model, and CPU-only. It may fit unsupervised statistics on the context inside each request. Evaluation in the source is retrospective full-context evaluation, not a conventional held-out forecast experiment. Those distinctions are essential for an accurate implementation and review.

A simple local Streamlit application is also explicitly consistent with the talk's original experimental workflow. The proposed first version includes a library, UI, exports, alert preview and evaluation. Production transport, scheduling, notifications and MCP are later extensions, since no deployment environment or metric-provider contract was supplied.

## Source map

| Topic | Supplied PDF pages | Precise repository source |
|---|---:|---|
| Monitoring purpose, stateless/CPU constraints | 1-18 | `tech_report/paragraphs/2_introduction.tex` |
| Shared adapter and expected range | 19-20, 27-28 | `4_algorithms.tex`, subsection z-test |
| Ordinary versus Stable AR | 21-24 | `4_algorithms.tex`, Stable AR subsection; notebook cell 16 is only baseline |
| Median seasonality and phase scale | 25-26 | `4_algorithms.tex`, seasonal subsection; notebook cell 29 is simplified |
| Evaluation and revised adjustment | 29-39 | `3_metrics.tex`, `5_experiments.tex`, `extras/point_adjustment_scheme.tex` |
| Local UI and library pipeline | 40-44 | `7_integration.tex`; notebook cells 8, 22, 26 |
| Recent-window alert settings | 45-48 | `7_integration.tex`, alert configuration |
| Deployment observations and limitations | 49-53 | `6_results.tex`, `7_integration.tex` |
| Agent/MCP usage and conclusions | 54-61 | Primarily talk material; future integration scope |

Paths in the last column are beneath `tech_report/paragraphs/` unless stated otherwise.

## Discrepancies resolved in the assignments

1. **The notebook is not the full target.** It supplies ordinary AR and a simplified seasonal method, uses global RMS scales and lacks Stable AR. The report provides the target recurrence and phase-scale algorithm.
2. **Do not inherit plotting mistakes.** Both plotting helpers assign array column 0 to “upper” although detector bounds are constructed `[lower, upper]`. The new contract explicitly fixes order and tests it.
3. **Do not inherit notebook global-variable behavior.** The pipeline's final thresholding references global `THRESHOLD` rather than a consistently passed config. Every layer in the assignment uses explicit effective configuration.
4. **Stable AR has a precise trust formula.** The report uses `(0.5*z² + log(sqrt(2π)))/(S*gamma)`. The neighboring illustrative JavaScript demo uses an exponential expression and ridge fitting; it is not a reference oracle.
5. **MAD terminology is loose.** The report explicitly gives `median(abs(r))/Phi^-1(.75)` under zero-centered residuals. The prompt freezes that interpretation for the shared adapter, while retaining std initialization in Stable AR and trimmed phase estimates in the seasonal detector.
6. **Interpolation needs consistency.** The notebook interpolates scores back to original timestamps. The assignment explicitly chooses to interpolate expected value/sigma and then recompute scores against original observations, ensuring flags correspond to actual band crossings.
7. **Segment metrics need an exact definition.** The textual explanation is less precise than the report's diagram. The assignment collapses every positive ground-truth segment into one maximum-score item, retains normal points individually, and supplies a hand-computable oracle. AP and F1-best are defined separately from current-threshold metrics.
8. **Some defaults are not published.** Alpha, neighborhood radii, update timing, epsilon, warmup treatment, validation policies and gap limits are labeled assignment choices. They are not represented as the private production implementation.
9. **Published statistics conflict across versions.** PDF page 36 lists dataset counts 29/10/15/210/367 (total 631), while the report describes some counts differently. PDF page 49 shows 5,550 alerts, 35 CPU and 220 GB; the report says 5,650 alerts, 3.5 CPU and 12.5 GB. These are source-reported observations, not verified capacity facts or local acceptance requirements. Likewise, Stable AR does not dominate baseline on every row of the authors' own benchmark table.
10. **Stateless does not mean no in-request fitting.** Copying that shorthand literally would forbid the very AR algorithm described. The prompts make the intended distinction explicit.

## Why these scope choices

Three plausible approaches were considered: extending the existing browser-only illustration; extracting Python algorithms into a local library with Streamlit; building a full service plus integrations. The Python/Streamlit option best preserves proximity to the numerical reference, keeps UI work small for a weaker model, and allows direct formula and data-contract tests. The browser-only option would require maintaining a separate numerical port; a production service would require additional infrastructure requirements.

The two prompts therefore specify a concrete local first version, with clearly documented deviations and deferred work. They deliberately do not promise exact reproduction of the unpublished production code, the full benchmark, or the operational improvements reported in the talk.
