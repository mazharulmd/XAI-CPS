"""Phase-2 analyses: reference-gap ablation, sensor-noise robustness (RQ4), decoy (RQ5)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
from xaicps import explain, stats  # noqa: E402

METHODS = ["RE", "IG", "KS-1024", "LIME-1024", "Fung-ens", "LX-bg", "LX"]


def loc_at(runs, offsets=(0, 60)):
    rows = []
    for r in runs:
        L, theta, step = len(r["loops"]), r["theta"], r["cfg"]["grid_step"]
        for e in r["events"]:
            if e["t_alarm"] is None or not e["targets"]:
                continue
            for off in offsets:
                k = off // step
                if k >= len(e["grid"]):
                    continue
                for m in METHODS:
                    for var in ("single", "tea"):
                        v = A.loop_vector_at(e, m, k, var == "tea", theta)
                        top = int(np.argmax(v))
                        dec = e["meta"].get("decoy_loop")
                        rows.append(dict(detector=r["detector"], seed=r["seed"], eid=e["eid"], method=m, variant=var, offset=off,
                                         hr1=float(top in e["targets"]),
                                         decoy_top1=float(dec is not None and r["loops"][top] == dec)))
    return pd.DataFrame(rows)


def summary(df, cond):
    out = []
    for keys, sub in df.groupby(["offset", "method", "variant"]):
        g = sub.groupby(["detector", "eid"])[["hr1", "decoy_top1"]].mean()
        for v in ("hr1", "decoy_top1"):
            mu, lo, hi = stats.bootstrap_ci(g[v].values)
            out.append(dict(condition=cond, offset=keys[0], method=keys[1], variant=keys[2], metric=v, mean=mu, ci_lo=lo, ci_hi=hi, n=len(g)))
    return out


def paired(df, cond, ds, metric, offset, higher=True):
    """LX+TEA vs other methods (+TEA) on (detector, attack) units, Holm-corrected."""
    sub = df[(df.offset == offset) & (df.variant == "tea")]
    g = sub.groupby(["method", "detector", "eid"])[metric].mean().unstack("method").dropna()
    res = stats.compare_to_reference({m: g[m].values for m in METHODS if m in g}, "LX", higher_is_better=higher)
    return [dict(dataset=ds, condition=cond, metric=metric, offset=offset, n=res["n"], **r) for r in res["pairs"]]


def main():
    rows, tests = [], []
    for ds in ("sim", "hai"):
        base = [r for r in A.load_runs(ds) if r["detector"] == "cnn" and r["seed"] == 0]
        conds = [("ref60 (default)", ""), ("ref32", "_ref32"), ("ref120", "_ref120"), ("ref300", "_ref300"),
                 ("noise0.1 clean-theta", "_noise0.1"), ("noise0.3 clean-theta", "_noise0.3"),
                 ("noise0.1 recal", "_noise0.1_recal"), ("noise0.3 recal", "_noise0.3_recal")]
        for cond, suf in conds:
            runs = base if suf == "" else A.load_runs(ds, suf)
            if not runs:
                continue
            df = loc_at(runs)
            rows += [dict(dataset=ds, **x) for x in summary(df, cond)]
            if "recal" in cond:
                tests += paired(df, cond, ds, "hr1", 60)
            d = runs[0]["detection"]
            for k in ("event_recall", "precision", "fa_per_hour", "delay_median_s"):
                if k in d:
                    rows.append(dict(dataset=ds, condition=cond, metric="det_" + k, mean=d[k], method="-", variant="-", offset=-1))
    # decoy vs no-decoy (same detectors linear+cnn, seeds 0-2)
    for cond, ds in (("no decoy", "sim"), ("decoy", "sim-decoy")):
        runs = [r for r in A.load_runs(ds) if r["detector"] in ("linear", "cnn")]
        if runs:
            df = loc_at(runs)
            rows += [dict(dataset="sim", **x) for x in summary(df, cond)]
            if cond == "decoy":
                for off in (0, 60):
                    tests += paired(df, cond, "sim", "decoy_top1", off, higher=False)
                    tests += paired(df, cond, "sim", "hr1", off)
    pd.DataFrame(tests).to_csv(A.TAB / "phase2_tests.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(A.TAB / "phase2.csv", index=False)
    print(df[df.metric.isin(["hr1", "decoy_top1"]) & (df.method.isin(["LX", "IG", "KS-1024", "RE"]))]
          .pivot_table(index=["dataset", "condition"], columns=["metric", "method", "variant", "offset"], values="mean").round(2).T.to_string())


if __name__ == "__main__":
    main()
