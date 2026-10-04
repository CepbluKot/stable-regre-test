import numpy as np
import pandas as pd
import pytest

from anomaly_explorer.adapter import robust_zero_scale, score_points
from anomaly_explorer.alerts import anomaly_intervals, evaluate_alert
from anomaly_explorer.evaluation import evaluate_labels
from anomaly_explorer.models.stable_ar import stable_step
from anomaly_explorer.pipeline import detect
from anomaly_explorer.schemas import Config
from anomaly_explorer.visualization import gaussian_reference_interval


def test_gaussian_reference_interval_is_independent_of_anomaly_threshold():
    low, high = gaussian_reference_interval([10.0], [2.0])
    assert low[0] == pytest.approx(6.080072031)
    assert high[0] == pytest.approx(13.919927969)
    with pytest.raises(ValueError):
        gaussian_reference_interval([10.0], [2.0], confidence=1.0)


def frame(values, step=300000, timestamps=None, labels=None):
    d = {
        "timestamp": timestamps
        if timestamps is not None
        else np.arange(len(values)) * step + 1_700_000_000_000,
        "value_0": values,
    }
    if labels is not None:
        d["label"] = labels
    return pd.DataFrame(d)


def test_adapter_exact_boundary_and_directions():
    x = np.array([10.0, 16.0, 4.0])
    e = np.array([10.0, 10.0, 10.0])
    s = np.array([2.0, 2.0, 2.0])
    both = score_points(x, e, s, 3, "both", np.ones(3, bool))
    assert np.allclose(both["anomaly_score"], [0, 3, 3])
    assert np.allclose(both["lower_bound"], [4, 4, 4])
    assert np.allclose(both["upper_bound"], [16, 16, 16])
    assert not any(both["is_anomaly"])
    assert score_points(x, e, s, 2, "above", np.ones(3, bool))[
        "is_anomaly"
    ].tolist() == [False, True, False]
    assert score_points(x, e, s, 2, "below", np.ones(3, bool))[
        "is_anomaly"
    ].tolist() == [False, False, True]
    assert robust_zero_scale([1, 1, 1, 10]) == pytest.approx(1 / 0.6744897501960817)


def test_stable_step_golden():
    _pred, weight, hidden, next_scale, z, r = stable_step(2, 0, 0, 0.2, 0.2, 1, 0.9)
    assert z == pytest.approx(10)
    assert r == pytest.approx(50.918938533204674)
    assert weight == pytest.approx(0.01926079438932388)
    assert hidden == pytest.approx(0.03852158877864776)
    assert next_scale == pytest.approx(0.2)
    assert stable_step(2, 1, 0, 0.2, 0.2, 1, 0.9)[3] > 0.2


def test_alignment_unsorted_and_recomputed_scores():
    t0 = 1_700_000_000_000
    ts = [t0 + i * 300000 for i in range(100)]
    ts[35] += 30000
    source = frame(np.sin(np.arange(100) / 7) + np.arange(100) * 0.002, timestamps=ts)
    a = detect(source, Config(model="ar", order=5))
    b = detect(source.iloc[::-1].reset_index(drop=True), Config(model="ar", order=5))
    assert len(a.points) == len(source)
    assert b.points.timestamp.tolist() == source.timestamp.iloc[::-1].tolist()
    aa = a.points.sort_values("timestamp")
    bb = b.points.sort_values("timestamp")
    assert np.allclose(aa.expected_value, bb.expected_value)
    assert np.allclose(
        aa.anomaly_score, (aa.value_0 - aa.expected_value).abs() / aa.sigma
    )
    assert not a.points.iloc[:5].is_evaluable.any()


def test_label_does_not_change_detection_or_threshold_forecast():
    x = frame(np.sin(np.arange(120) / 6) + np.arange(120) * 0.01)
    c = Config(model="stable_ar", order=5)
    a = detect(x, c)
    b = detect(x.assign(label=np.arange(120) % 2), c)
    assert np.allclose(a.points.anomaly_score, b.points.anomaly_score)
    wide = detect(x, Config(model="stable_ar", order=5, threshold=5))
    assert np.allclose(a.points.expected_value, wide.points.expected_value)
    assert np.allclose(a.points.sigma, wide.points.sigma)
    assert wide.points.is_anomaly.sum() <= a.points.is_anomaly.sum()


def test_alert_strict_fraction_and_boundary():
    t0 = 1_700_000_000_000
    p = frame([0] * 5, step=60000)
    p["is_evaluable"] = True
    p["is_anomaly"] = [False, False, True, True, False]
    a = evaluate_alert(
        p, as_of=t0 + 4 * 60000, window_duration_ms=3 * 60000, percentage=50
    )
    assert a["denominator"] == 3 and a["numerator"] == 2 and a["active"]
    p["is_anomaly"] = [False, True, True, False, False]
    a = evaluate_alert(
        p, as_of=t0 + 4 * 60000, window_duration_ms=4 * 60000, percentage=50
    )
    assert a["denominator"] == 4 and a["numerator"] == 2 and not a["active"]


def test_revised_point_adjustment_oracle():
    labels = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    scores = [0.2, 0.3, 0.5, 0.9, 0.8, 0.7, 0.8, 0.1, 0.1, 0.1]
    p = frame([0] * 10, step=1000, labels=labels)
    p["anomaly_score"] = scores
    p["is_evaluable"] = True
    p["is_anomaly"] = np.array(scores) > 0.6
    p["signed_z"] = scores
    result = evaluate_labels(p, 0.6, "both", 1000)
    assert result["revised"]["tp"] == 1
    assert result["revised"]["fp"] == 2
    assert result["revised"]["fn"] == 1
    assert result["revised"]["f1"] == pytest.approx(0.4)
    assert result["revised_ap"] == pytest.approx(2 / 3)
    assert result["revised_f1_best"] == pytest.approx(2 / 3)


def test_validation_and_constant_series():
    base = frame(np.ones(100))
    for model in ["ar", "stable_ar", "seasonal"]:
        cfg = Config(model=model, order=5, season_duration_ms=10 * 300000)
        result = detect(base, cfg)
        assert np.isfinite(result.points.sigma).all()
        assert not result.points.is_anomaly.any()
    for broken in [
        base.assign(value_0=np.nan),
        base.assign(timestamp=1),
        pd.concat([base, base.iloc[:1]]),
    ]:
        with pytest.raises(ValueError):
            detect(broken, Config(model="ar", order=5))


def test_missing_timestamp_and_oversized_gap_policy():
    x = frame(np.sin(np.arange(90) / 8), step=300000)
    missing = x.drop(index=[40]).reset_index(drop=True)
    result = detect(missing, Config(model="ar", order=5))
    assert len(result.points) == 89
    assert result.metadata["interpolation_count"] == 1
    x.loc[40:, "timestamp"] += 20 * 300000
    with pytest.raises(ValueError, match="gap exceeds 12 steps"):
        detect(x, Config(model="ar", order=5))


def test_metrics_no_positive_labels_and_isolated_interval():
    x = frame(np.zeros(6), step=1000, labels=[0] * 6)
    x["is_evaluable"] = True
    x["is_anomaly"] = [False, False, True, False, False, False]
    x["signed_z"] = [0, 0, 5, 0, 0, 0]
    x["anomaly_score"] = [0, 0, 5, 0, 0, 0]
    metrics = evaluate_labels(x, 3, "both", 1000)
    assert metrics["revised"]["fp"] == 1
    assert metrics["revised_ap"] is None
    intervals = anomaly_intervals(x, 1000)
    assert intervals[0]["duration_ms"] == 0 and intervals[0]["point_count"] == 1


def test_seasonal_scale_varies_by_phase():
    rng = np.random.default_rng(2026)
    t = np.arange(24 * 12)
    phase = t % 24
    x = (
        100
        + 20 * np.sin(2 * np.pi * phase / 24)
        + rng.normal(0, np.where(phase < 12, 0.2, 2), len(t))
    )
    x[24 * 8 + 5] -= 20
    result = detect(
        frame(x, step=3600000), Config(model="seasonal", season_duration_ms=86400000)
    )
    spread = result.points.sigma.to_numpy()
    assert np.median(spread[phase >= 12]) > 2 * np.median(spread[phase < 12])
    assert result.points.iloc[24 * 8 + 5].is_anomaly


def test_stateless_repeated_calls_and_input_unchanged():
    rng = np.random.default_rng(13)
    a = frame(rng.normal(size=120))
    b = frame(rng.normal(size=90))
    original = a.copy(deep=True)
    c = Config(model="stable_ar", order=5)
    first = detect(a, c)
    detect(b, c)
    last = detect(a, c)
    np.testing.assert_allclose(first.anomaly_scores, last.anomaly_scores)
    pd.testing.assert_frame_equal(a, original)
