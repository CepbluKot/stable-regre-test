# Verification and reproducibility

Run these commands from the repository root after `uv sync --extra dev --locked`:

```sh
uv run pytest -q
uv run ruff check src app.py scripts tests audit
uv run ruff format --check app.py tests/test_ui.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python audit/check_acceptance.py
```

The unit and Streamlit AppTest suite currently has 15 tests. It covers input validation, time-grid alignment and row-order preservation, scoring boundaries and direction, model invariants, alert and interval boundaries, label metrics, stateless repeat calls, the display-only 95% interval, and basic valid/invalid UI upload behavior. The independent acceptance script exercises 12 model/source cases, checks finite outputs and the shared score invariants, and writes `artifacts/audit/evidence.json`. These checks are meaningful regression evidence, not proof of detection accuracy on the unlabeled examples.

For a manual UI check, start `uv run streamlit run app.py --server.address 127.0.0.1`, click **Run analysis**, and confirm that Results shows the full series, blue threshold band, green approximate 95% band, red markers, score chart, alert preview, and intervals. The page has three tabs: Results, Evaluation, and Data and export. It has no timed chart replay; charts remain on the analyzed snapshot until **Run analysis** is clicked again. Change a setting and confirm the old result is hidden. Upload `tests/fixtures/labeled-shift.csv`, rerun, and confirm 120 observations plus evaluation metrics. Upload a malformed numeric CSV and confirm an error with no stale metrics. Download both CSV and JSON and compare row count, row order, metadata, and alert against a direct `detect()` call.

For timing data, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/benchmark.py --output artifacts/benchmark.json
```

The script covers four bundled source CSVs under all three models. It records one cold subprocess, at least three fresh in-process calls per case, dependency and CPU identity, and writes JSON plus CSV. Times vary by machine and include different work in the cold-process and `detect()` measurements. A benchmark is not a production throughput guarantee.

The [implementer self-audit](../artifacts/audit/acceptance-report.md) records prior evidence and explicitly marks strict independent acceptance **blocked**. The [remaining acceptance task](../artifacts/audit/repair-task.md) specifies browser upload, downloaded-byte parsing, and deeper independent edge-case checks. An independent reviewer can use the [review assignment](assignment/02-independent-review-prompt.md). Do not relabel the self-audit as an independent pass merely because local tests pass.

The code, assignment, source slides, transcript, and source mapping are versioned in this repository. The reference repository is pinned in [source mapping](source-mapping.md), so reconstruction choices are auditable. `dist/anomaly-explorer-source.zip` is a packaged source copy; edit the repository tree, not the archive.
