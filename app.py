"""Streamlit results explorer for the stateless anomaly detector."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly_explorer import Config, detect
from anomaly_explorer.visualization import gaussian_reference_interval

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted((ROOT / "data/examples").glob("*.csv"))

st.set_page_config(page_title="Signal Lab", page_icon="📈", layout="wide")
st.title("Signal Lab")
st.caption(
    "Explore unexpected changes in a time series. The blue band sets the anomaly threshold; the green 95% interval is a normal-reference guide, not calibrated coverage."
)

with st.sidebar:
    st.header("Analysis")
    sample = st.selectbox(
        "Example series",
        EXAMPLES,
        format_func=lambda p: p.stem.replace("_", " ").title(),
        index=1,
    )
    upload = st.file_uploader(
        "Or upload CSV",
        type=["csv"],
        help="Columns: timestamp (Unix milliseconds), value_0, optional label (0/1). Uploaded data takes precedence.",
    )
    model = st.selectbox(
        "Model",
        ["stable_ar", "seasonal", "ar"],
        format_func=lambda v: {
            "stable_ar": "Stable autoregression",
            "seasonal": "Seasonal decomposition",
            "ar": "Ordinary autoregression",
        }[v],
    )
    threshold = st.slider("Threshold (range width)", 0.5, 8.0, 3.0, 0.1)
    percentage = st.slider("Anomalous points for alert (%)", 0, 100, 50)
    window_minutes = st.number_input(
        "Recent window (minutes)", min_value=1, max_value=10080, value=30, step=5
    )
    direction = st.selectbox(
        "Direction",
        ["both", "above", "below"],
        format_func=lambda v: {
            "both": "Both",
            "above": "Above expected",
            "below": "Below expected",
        }[v],
    )
    season_hours = st.number_input(
        "Season duration (hours)",
        min_value=1,
        max_value=336,
        value=24,
        step=1,
        help="Used only for the seasonal model",
    )
    with st.expander("Advanced model settings"):
        order = st.number_input(
            "AR order", min_value=1, max_value=500, value=20, help="Used for AR models"
        )
        gamma = st.number_input(
            "Stable AR gamma",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            help="Used only for Stable AR",
        )
        alpha = st.number_input(
            "Stable AR alpha",
            min_value=0.0,
            max_value=0.999,
            value=0.9,
            step=0.05,
            help="Used only for Stable AR",
        )
        phase_radius = st.number_input(
            "Seasonal profile radius",
            min_value=0,
            max_value=48,
            value=2,
            help="Used only for the seasonal model",
        )
        scale_radius = st.number_input(
            "Seasonal scale radius",
            min_value=0,
            max_value=48,
            value=2,
            help="Used only for the seasonal model",
        )
        step_override = st.number_input(
            "Step override (ms; 0 = infer)",
            min_value=0,
            max_value=86_400_000,
            value=0,
            step=1000,
        )
    run = st.button("Run analysis", type="primary", width="stretch")

signature = (
    sample.name,
    upload.name if upload is not None else None,
    sha256(upload.getvalue()).hexdigest() if upload is not None else None,
    model,
    float(threshold),
    int(percentage),
    int(window_minutes),
    direction,
    int(season_hours),
    int(order),
    float(gamma),
    float(alpha),
    int(phase_radius),
    int(scale_radius),
    int(step_override),
)

if run:
    st.session_state.pop("analysis", None)
    try:
        raw = pd.read_csv(upload if upload is not None else sample)
        cfg = Config(
            model=model,
            threshold=float(threshold),
            percentage=float(percentage),
            window_duration_ms=int(window_minutes) * 60_000,
            direction=direction,
            order=int(order),
            gamma=float(gamma),
            alpha=float(alpha),
            season_duration_ms=int(season_hours) * 3_600_000,
            season_neighborhood=int(phase_radius) if model == "seasonal" else None,
            scale_neighborhood=int(scale_radius) if model == "seasonal" else None,
            step_ms=int(step_override) if step_override else None,
        )
        result = detect(raw, cfg)
        st.session_state.analysis = (
            result,
            upload.name if upload is not None else sample.name,
            signature,
        )
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not analyze this CSV: {exc}")

if "analysis" not in st.session_state:
    st.info("Choose an example or upload a CSV, then select Run analysis.")
    st.stop()

result, source_name, previous_signature = st.session_state.analysis
if signature != previous_signature:
    st.warning("Analysis settings changed. Select Run analysis to update the results.")
    st.stop()
p = result.points.sort_values("timestamp", kind="stable").reset_index(drop=True)
meta = result.metadata
st.subheader(source_name)
a, b, c, d, e = st.columns(5)
a.metric("Observed points", f"{len(p):,}")
b.metric("Anomalous points", f"{int(p.is_anomaly.sum()):,}")
c.metric("Anomaly intervals", f"{len(result.intervals):,}")
d.metric("Elapsed", f"{meta['elapsed_ms']:.0f} ms")
e.metric(
    "Latest alert",
    result.alert["status"],
    f"{result.alert['fraction'] * 100:.1f}% of points",
)
for warning in meta["warnings"]:
    st.warning(warning)
if int(p.is_anomaly.sum()) / len(p) > 0.3:
    st.warning(
        "More than 30% of observations are flagged. The selected model or season may not fit this series; inspect the band before using its alert preview."
    )
if meta["warmup_count"]:
    st.caption(
        f"{meta['warmup_count']} original points are AR warmup and are excluded from flags, alerts and metrics."
    )

results_tab, evaluation_tab, data_tab = st.tabs(
    ["Results", "Evaluation", "Data and export"]
)
with results_tab:
    choices = {"Full series": None}
    for i, event in enumerate(result.intervals, 1):
        start = datetime.fromtimestamp(event["start"] / 1000, UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        choices[f"Interval {i}: {start}, {event['point_count']} points"] = event
    selected = st.selectbox("Focus", list(choices), key="interval_focus")
    t = pd.to_datetime(p.timestamp, unit="ms", utc=True)
    reference_low, reference_high = gaussian_reference_interval(
        p.expected_value, p.sigma
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=p.upper_bound,
            name="Upper",
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=p.lower_bound,
            name="Anomaly threshold band",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(40,120,220,.20)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=reference_high,
            name="95% reference upper",
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=reference_low,
            name="Approx. 95% interval",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(33,161,121,.25)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=p.expected_value,
            name="Expected",
            line={"color": "#3366cc", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=p.value_0, name="Observed", line={"color": "#313848", "width": 1.5}
        )
    )
    abnormal = p.loc[p.is_anomaly]
    if len(abnormal):
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(abnormal.timestamp, unit="ms", utc=True),
                y=abnormal.value_0,
                name="Anomalies",
                mode="markers",
                marker={"color": "#e45756", "size": 7},
            )
        )
    window_start = pd.to_datetime(result.alert["window_start"], unit="ms", utc=True)
    window_end = pd.to_datetime(result.alert["window_end"], unit="ms", utc=True)
    fig.add_vrect(
        x0=window_start,
        x1=window_end,
        fillcolor="rgba(250,184,49,.12)",
        line_width=0,
        annotation_text="Alert window",
        annotation_position="top left",
    )
    if choices[selected] is not None:
        item = choices[selected]
        pad = max(int(meta["step_ms"]) * 6, int(item["duration_ms"]) * 0.15)
        fig.update_xaxes(
            range=[
                pd.to_datetime(item["start"] - pad, unit="ms", utc=True),
                pd.to_datetime(item["end"] + pad, unit="ms", utc=True),
            ]
        )
    fig.update_layout(
        height=440,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "y": 1.08},
        xaxis_title="Time (UTC)",
        yaxis_title="Original value",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Green: expected ± 1.96 × model sigma, shown as an approximate 95% normal-reference interval. "
        "It is not a calibrated confidence or prediction interval; the blue anomaly band uses the selected threshold."
    )
    sf = go.Figure()
    sf.add_trace(
        go.Scatter(
            x=t, y=p.anomaly_score, name="Absolute z-score", line={"color": "#21a179"}
        )
    )
    sf.add_hline(
        y=meta["configuration"]["threshold"],
        line_color="#e45756",
        line_dash="dash",
        annotation_text="Threshold",
    )
    sf.update_layout(
        height=240,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Time (UTC)",
        yaxis_title="Absolute z-score",
    )
    st.plotly_chart(sf, width="stretch")
    st.caption(
        "The highlighted recent window controls the alert preview. Anomaly intervals below describe individual deviations; no notifications are sent."
    )
    st.dataframe(pd.DataFrame(result.intervals), width="stretch", hide_index=True)
    st.json(result.alert)
with evaluation_tab:
    if result.evaluation is None:
        st.info(
            "No labels: quality metrics unavailable. The bundled examples do not include ground truth."
        )
    else:
        ev = result.evaluation
        st.write("Current threshold, selected direction")
        st.dataframe(
            pd.DataFrame(
                [
                    {"method": "Point-wise", **ev["point"]},
                    {"method": "Revised segment adjustment", **ev["revised"]},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.write(
            "Retrospective ranking diagnostics (absolute z-score, both directions)"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Point AP": ev["point_ap"],
                        "Revised AP": ev["revised_ap"],
                        "Oracle revised F1-best": ev["revised_f1_best"],
                        "Oracle threshold": ev["best_threshold"],
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if ev["reason"]:
            st.warning(ev["reason"])
        st.caption(
            "Revised adjustment collapses each true anomaly segment into one positive item at its maximum score, while normal points remain separate. AP is step-integral average precision. Oracle F1-best uses labels retrospectively and never changes the alert threshold."
        )
with data_tab:
    st.json({"metadata": meta, "alert": result.alert})
    st.dataframe(result.points, width="stretch", hide_index=True)
    csv = result.points.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download point CSV", csv, file_name="anomaly-points.csv", mime="text/csv"
    )
    data = json.dumps(
        result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    st.download_button(
        "Download full JSON",
        data,
        file_name="anomaly-results.json",
        mime="application/json",
    )
