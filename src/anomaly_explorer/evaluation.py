from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _counts(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _collapsed(
    points: pd.DataFrame, step_ms: int, key: str
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    positive = []
    previous = None

    def flush():
        if positive:
            labels.append(True)
            scores.append(max(positive))
            positive.clear()

    for row in points.itertuples(index=False):
        valid = bool(row.is_evaluable)
        if not valid:
            flush()
            previous = None
            continue
        ts = int(row.timestamp)
        if previous is not None and ts - previous > 1.5 * step_ms:
            flush()
        if bool(row.label):
            positive.append(float(getattr(row, key)))
        else:
            flush()
            labels.append(False)
            scores.append(float(getattr(row, key)))
        previous = ts
    flush()
    return np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)


def evaluate_labels(
    points: pd.DataFrame, threshold: float, direction: str, step_ms: int
) -> dict:
    if "label" not in points:
        raise ValueError("evaluation requires a label column")
    p = points.sort_values("timestamp", kind="stable").copy()
    signed = p.signed_z.to_numpy()
    if direction == "above":
        directional = signed
    elif direction == "below":
        directional = -signed
    elif direction == "both":
        directional = np.abs(signed)
    else:
        raise ValueError("invalid direction")
    p["directional_score"] = directional
    valid = p.loc[p.is_evaluable]
    y = valid.label.to_numpy(dtype=bool)
    predictions = valid.is_anomaly.to_numpy(dtype=bool)
    point = _counts(y, predictions)
    revised_y, revised_score = _collapsed(p, step_ms, "directional_score")
    revised = _counts(revised_y, revised_score > threshold)
    rank_y, rank_scores = _collapsed(p, step_ms, "anomaly_score")
    positives = int(rank_y.sum())
    output = {
        "point": point,
        "revised": revised,
        "point_ap": None,
        "revised_ap": None,
        "revised_f1_best": None,
        "best_threshold": None,
        "positive_segments": positives,
        "ranking_direction": "both (absolute z-score)",
        "reason": None,
    }
    if positives == 0:
        output["reason"] = (
            "No positive labels in evaluable observations; recall, F1 and AP are undefined"
        )
        return output
    output["point_ap"] = float(
        average_precision_score(y, valid.anomaly_score.to_numpy())
    )
    output["revised_ap"] = float(average_precision_score(rank_y, rank_scores))
    candidates = np.r_[
        np.unique(rank_scores), np.nextafter(float(np.min(rank_scores)), -np.inf)
    ]
    best = max(
        (
            (float(_counts(rank_y, rank_scores > v)["f1"] or 0.0), float(v))
            for v in candidates
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
    output["revised_f1_best"], output["best_threshold"] = best
    return output
