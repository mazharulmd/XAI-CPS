"""H6: how much does a random window split inflate supervised detection on ICS data?

Protocol (HAI 21.03 attack files only, as supervised papers do):
  * features: last value, mean and std of each tag over a 32 s window, stride 5 s
  * random   : 70/30 random split of windows (overlapping windows can fall on both sides)
  * blocked  : train on test1-3 (Jul 7-14), evaluate on test4-5 (Jul 28-31): disjoint
               days and disjoint attack instances
Classifier: histogram gradient boosting (sklearn), fixed hyper-parameters, 5 seeds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xaicps import data  # noqa: E402


def featurise(X, y, w=32, stride=5):
    F, Y = [], []
    for t in range(w, len(X), stride):
        W = X[t - w:t]
        F.append(np.concatenate([W[-1], W.mean(0), W.std(0)])); Y.append(y[t - 1])
    return np.array(F, np.float32), np.array(Y)


def main():
    D = data.load_hai()
    per = [featurise(X, y) for X, y in zip(D.test, D.test_label)]
    out = {"random": [], "blocked": []}
    for seed in range(5):
        F = np.concatenate([p[0] for p in per]); Y = np.concatenate([p[1] for p in per])
        Ftr, Fte, ytr, yte = train_test_split(F, Y, test_size=0.3, random_state=seed, stratify=Y)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Ftr, ytr)
        p = clf.predict_proba(Fte)[:, 1]
        out["random"].append(dict(f1=f1_score(yte, p > 0.5), auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p)))
        Ftr = np.concatenate([p[0] for p in per[:3]]); ytr = np.concatenate([p[1] for p in per[:3]])
        Fte = np.concatenate([p[0] for p in per[3:]]); yte = np.concatenate([p[1] for p in per[3:]])
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Ftr, ytr)
        p = clf.predict_proba(Fte)[:, 1]
        out["blocked"].append(dict(f1=f1_score(yte, p > 0.5), auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p)))
    summ = {k: {m: (float(np.mean([r[m] for r in v])), float(np.std([r[m] for r in v]))) for m in v[0]} for k, v in out.items()}
    print(json.dumps(summ, indent=1))
    p = ROOT / "results" / "tables"; p.mkdir(parents=True, exist_ok=True)
    json.dump(dict(runs=out, summary=summ), open(p / "leakage_hai.json", "w"), indent=1)


if __name__ == "__main__":
    main()
