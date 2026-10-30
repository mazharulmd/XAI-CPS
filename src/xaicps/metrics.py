"""Detection, localisation, stability, faithfulness and deadline metrics."""
from __future__ import annotations
import numpy as np


def alarm_runs(alarm):
    a = np.asarray(alarm, int)
    idx = np.flatnonzero(np.diff(np.r_[0, a, 0]))
    return list(zip(idx[::2], idx[1::2]))


def detection_metrics(scores, labels, events, theta, grace=60, fs=1.0):
    """Point-wise P/R/F1 + event recall, detection delay, false alarms per hour.

    An event is detected if an alarm starts within [t_on, t_off + grace). False alarms
    are alarm runs that do not overlap any [t_on, t_off + grace) interval."""
    tp = fp = fn = 0
    det, delays, fa_runs, normal_s = {}, [], 0, 0.0
    by_seg = {}
    for ev in events:
        by_seg.setdefault(ev.seg, []).append(ev)
    for si, (s, y) in enumerate(zip(scores, labels)):
        ok = np.isfinite(s)
        al = np.zeros(len(s), bool); al[ok] = s[ok] > theta
        tp += int((al & (y == 1)).sum()); fp += int((al & (y == 0)).sum()); fn += int((~al & (y == 1)).sum())
        evs = by_seg.get(si, [])
        mask = np.zeros(len(s), bool)
        for ev in evs:
            mask[ev.t_on:min(len(s), ev.t_off + grace)] = True
            hit = np.flatnonzero(al[ev.t_on:min(len(s), ev.t_off + grace)])
            det[ev.eid] = int(ev.t_on + hit[0]) if len(hit) else None
            if len(hit):
                delays.append(float(hit[0]) / fs)
        for a, b in alarm_runs(al):
            if not mask[a:b].any():
                fa_runs += 1
        normal_s += float((~mask & ok).sum()) / fs
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-12),
                event_recall=float(np.mean([v is not None for v in det.values()])),
                delay_median_s=float(np.median(delays)) if delays else np.nan,
                fa_per_hour=fa_runs / max(normal_s / 3600, 1e-9)), det


def target_rank(scores, target_idx):
    """1-based rank of the best-ranked target among L groups (ties -> pessimistic)."""
    scores = np.asarray(scores, float)
    best = max(scores[target_idx])
    return int((scores > best).sum() + (scores[[i for i in range(len(scores)) if i not in target_idx]] == best).sum() + 1)


def loc_scores(scores, target_idx, L):
    r = target_rank(scores, target_idx)
    return dict(hr1=float(r <= 1), hr2=float(r <= 2), hr3=float(r <= 3), nrank=(r - 1) / max(L - 1, 1), rank=r)


def topk(scores, k):
    return set(np.argsort(-np.asarray(scores), kind="stable")[:k].tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)
