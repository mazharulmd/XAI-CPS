from __future__ import annotations

import numpy as np
from .data import contiguous_events


def binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pred = scores > threshold
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y)); tn = int(np.sum(~pred & ~y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    positives = int(y.sum())
    if positives:
        cum_tp = np.cumsum(sorted_y)
        rank = np.arange(1, len(y) + 1)
        ap = float(np.sum((cum_tp / rank) * sorted_y) / positives)
    else:
        ap = float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / max(1, fp + tn), "specificity": tn / max(1, fp + tn),
            "average_precision": ap, "predicted_samples": int(pred.sum()), "threshold": float(threshold)}


def event_metrics(y: np.ndarray, pred: np.ndarray, timestamps: np.ndarray) -> dict:
    y = np.asarray(y, dtype=bool); pred = np.asarray(pred, dtype=bool)
    events = contiguous_events(y)
    delays = []
    detected = 0
    for start, end in events:
        where = np.flatnonzero(pred[start:end])
        if where.size:
            detected += 1
            delays.append(float(where[0]))
    pred_events = contiguous_events(pred)
    false_event_count = sum(not y[a:b].any() for a, b in pred_events)
    dt = np.diff(timestamps.astype("datetime64[s]").astype(np.int64))
    seconds_per_sample = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0
    normal_hours = max(1e-9, float((~y).sum()) * seconds_per_sample / 3600.0)
    return {"attack_events": len(events), "detected_attack_events": detected,
            "event_recall": detected / max(1, len(events)),
            "mean_detection_delay_seconds": float(np.mean(delays) * seconds_per_sample) if delays else None,
            "median_detection_delay_seconds": float(np.median(delays) * seconds_per_sample) if delays else None,
            "false_alarm_events": int(false_event_count),
            "false_alarm_events_per_normal_hour": float(false_event_count / normal_hours)}


def bootstrap_ci(values, seed: int = 42, draws: int = 2000) -> list[float] | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x): return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_bootstrap_difference(a, b, seed: int = 42, draws: int = 5000) -> dict:
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    ok=np.isfinite(a)&np.isfinite(b); delta=a[ok]-b[ok]
    if not len(delta): return {"n":0,"mean_difference":None,"ci95":None,"two_sided_p":None}
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(delta),size=(draws,len(delta)))
    means=delta[ix].mean(axis=1)
    # Centered paired bootstrap null; finite-sample plus-one correction.
    null=means-delta.mean()
    p=(np.sum(np.abs(null)>=abs(delta.mean()))+1)/(draws+1)
    return {"n":int(len(delta)),"mean_difference":float(delta.mean()),
            "ci95":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],"two_sided_p":float(p)}
