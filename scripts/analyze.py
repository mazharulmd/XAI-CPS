"""Aggregate raw runs into tables (results/tables) used by the paper.

Every number in the manuscript is produced here from results/raw/**.pkl.
"""
from __future__ import annotations

import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xaicps import explain, metrics, stats  # noqa: E402

TAB = ROOT / "results" / "tables"
OFFSETS = (0, 10, 30, 60, 120, 180)          # delivery time after the alarm [s]
DEADLINES = (10, 30, 60, 120, 300)            # deadline after attack onset [s]
MAIN_METHODS = ["RE", "SM", "IG", "KS-eq", "KS-1024", "LIME-eq", "LIME-1024", "Fung-ens", "PI-CAX", "LX-bg", "CLX", "RPX", "LX"]
PROPOSED = "LX"


def load_runs(dataset, suffix=""):
    runs = []
    for p in sorted((ROOT / "results" / "raw" / dataset).glob("*.pkl")):
        stem = p.stem
        det, rest = stem.split("_s", 1)
        seed, _, suf = rest.partition("_")
        if (("_" + suf) if suf else "") != suffix:
            continue
        r = pickle.load(open(p, "rb")); r["detector"], r["seed"] = det, int(seed)
        merge_picax(r, ROOT / "results" / "raw_picax" / dataset / p.name)
        runs.append(r)
    return runs


def merge_picax(run, path):
    """Attach PI-CAX sequences (scripts/run_picax.py) computed on the same grid as this run."""
    if not path.exists():
        return
    pc = pickle.load(open(path, "rb"))["events"]
    for e in run["events"]:
        if e["eid"] in pc and "seq" in e:
            o = pc[e["eid"]]
            assert len(o["seq"]) == len(e["grid"])
            e["seq"]["PI-CAX"] = o["seq"]; e["wall"]["PI-CAX"] = o["wall"]; e["evals"]["PI-CAX"] = o["evals"]
            for key in ("tags_at", "fid_at", "stab_at"):
                for off, val in o.get(key, {}).items():
                    if off in e.get(key, {}):
                        e[key][off]["PI-CAX"] = val


def loop_vector_at(e, m, k, use_tea, theta, stride=1):
    """Loop relevance delivered after the k-th grid explanation (sub-sampled by stride)."""
    seq = e["seq"][m][: k + 1: stride] if stride > 1 else e["seq"][m][: k + 1]
    if not use_tea:
        return seq[-1]
    sc = e["score"][: k + 1: stride] if stride > 1 else e["score"][: k + 1]
    A, _ = explain.tea(seq, sc, theta)
    return A[-1]


# ----------------------------------------------------------------------------- per-event records
def event_table(runs, methods, latency=None, slowdown=1.0, t_act=0.0):
    """Long table: one row per (detector, seed, event, method, variant, quantity)."""
    rows = []
    for r in runs:
        L, theta, step = len(r["loops"]), r["theta"], r["cfg"]["grid_step"]
        lat = latency or {}
        for e in r["events"]:
            base = dict(detector=r["detector"], seed=r["seed"], eid=e["eid"], primitive=e["meta"].get("primitive", ""),
                        n_targets=e["meta"].get("n_targets", 1), detected=e["t_alarm"] is not None)
            tg = e["targets"]
            if not tg:
                continue
            # best-guess single shot
            for m in methods:
                if "bestguess" in e and m in e["bestguess"]:
                    rows.append({**base, "method": m, "variant": "single", "timing": "bestguess",
                                 **metrics.loc_scores(e["bestguess"][m], tg, L)})
            # practical timing, by delivery offset after the alarm (compute latency ignored)
            if e["t_alarm"] is not None:
                K = len(e["grid"])
                for off in OFFSETS:
                    k = off // step
                    if k >= K:
                        continue
                    for m in methods:
                        for var in ("single", "tea"):
                            v = loop_vector_at(e, m, k, var == "tea", theta)
                            rows.append({**base, "method": m, "variant": var, "timing": f"alarm+{off}",
                                         **metrics.loc_scores(v, tg, L)})
            # deadline-conditioned localisation, with latency
            for m in methods:
                ell = slowdown * lat.get(m, np.median([e2["wall"][m] for e2 in r["events"] if "wall" in e2]))
                for var in ("single", "tea"):
                    dl = list(DEADLINES)
                    hz = e["meta"].get("hazard_D", np.nan)
                    if np.isfinite(hz):
                        dl.append(("haz", hz - t_act))
                    for D in dl:
                        tag, Dv = (D if isinstance(D, tuple) else ("fixed", D))
                        ok, hr1, hr3 = _dcl(e, m, var, Dv, ell, step, theta, L)
                        rows.append({**base, "method": m, "variant": var, "timing": f"dcl:{tag}:{Dv if tag == 'fixed' else 'hz'}",
                                     "hr1": hr1, "hr3": hr3, "delivered": ok, "latency_s": ell})
    return pd.DataFrame(rows)


def _dcl(e, m, var, D, ell, step, theta, L):
    if e["t_alarm"] is None:
        return False, 0.0, 0.0
    g_eff = step * int(np.ceil(max(step, ell) / step))
    stride = g_eff // step
    j_last = None
    for j in range(len(e["grid"][::stride])):
        if e["t_alarm"] + j * g_eff + ell <= e["t_on"] + D:
            j_last = j
    if j_last is None:
        return False, 0.0, 0.0
    v = loop_vector_at(e, m, j_last * stride, var == "tea", theta, stride=stride)
    s = metrics.loc_scores(v, e["targets"], L)
    return True, s["hr1"], s["hr3"]


# ----------------------------------------------------------------------------- stability, fidelity, tags
def stability_table(runs, methods, horizon=120):
    rows = []
    for r in runs:
        theta, step = r["theta"], r["cfg"]["grid_step"]
        for e in r["events"]:
            if e["t_alarm"] is None or not e["targets"]:
                continue
            K = min(len(e["grid"]), horizon // step + 1)
            if K < 3:
                continue
            for m in methods:
                for var in ("single", "tea"):
                    seq = e["seq"][m][:K]
                    V = explain.tea(seq, e["score"][:K], theta)[0] if var == "tea" else seq
                    for k in (1, 3):
                        j = [metrics.jaccard(metrics.topk(V[i], k), metrics.topk(V[i + 1], k)) for i in range(K - 1)]
                        rows.append(dict(detector=r["detector"], seed=r["seed"], eid=e["eid"], method=m, variant=var,
                                         k=k, temporal_jaccard=float(np.mean(j))))
            for off, d in e.get("stab_at", {}).items():
                for m in methods:
                    a, b = d[m]
                    from scipy.stats import spearmanr
                    rho = spearmanr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else np.nan
                    rows.append(dict(detector=r["detector"], seed=r["seed"], eid=e["eid"], method=m, variant="noise",
                                     k=3, offset=off, noise_jaccard=metrics.jaccard(metrics.topk(a, 3), metrics.topk(b, 3)),
                                     noise_spearman=rho))
    return pd.DataFrame(rows)


def fidelity_table(runs, methods):
    rows = []
    for r in runs:
        for e in r["events"]:
            for off, d in e.get("fid_at", {}).items():
                for m in methods:
                    rows.append(dict(detector=r["detector"], seed=r["seed"], eid=e["eid"], method=m, offset=off,
                                     del_lkg=d[m][0], del_mean=d[m][1]))
    return pd.DataFrame(rows)


def tag_table(runs):
    rows = []
    for r in runs:
        tags = r["tags"]
        for e in r["events"]:
            pts = e["points"]
            if "primitive" in e["meta"] and e["meta"]["primitive"]:          # PCL-Sim: derive from primitive
                from xaicps.data import sim_target_tags
                pts = sim_target_tags(e["loops"][0], e["meta"]["primitive"])
            vis = [tags.index(p) for p in pts if p in tags]
            if not vis:
                continue
            for off, d in e.get("tags_at", {}).items():
                for m, v in d.items():
                    if v is None:
                        continue
                    rk = metrics.target_rank(v, vis)
                    rows.append(dict(detector=r["detector"], seed=r["seed"], eid=e["eid"], method=m, offset=off,
                                     hr1=float(rk <= 1), hr3=float(rk <= 3), hr5=float(rk <= 5), nrank=(rk - 1) / (len(tags) - 1)))
    return pd.DataFrame(rows)


def cost_table(runs, methods):
    rows = []
    for r in runs:
        for m in methods:
            w = [e["wall"][m] for e in r["events"] if "wall" in e]
            ev = [e["evals"][m] for e in r["events"] if "evals" in e]
            if w:
                rows.append(dict(detector=r["detector"], seed=r["seed"], method=m, wall_ms_median=1e3 * np.median(w),
                                 wall_ms_p95=1e3 * np.quantile(w, 0.95), evals=int(np.median(ev))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- summaries + tests
def summarise(df, value, by, ref=PROPOSED, ref_variant=None, higher=True):
    """Mean and bootstrap CI per method/variant; Wilcoxon vs ref on (event, detector) units."""
    out = []
    g = df.groupby(["detector", "eid"] + by)[value].mean().reset_index()      # average seeds
    for keys, sub in g.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        mu, lo, hi = stats.bootstrap_ci(sub[value].values)
        out.append(dict(zip(by, keys), mean=mu, ci_lo=lo, ci_hi=hi, n=len(sub)))
    return pd.DataFrame(out), g


def paired_tests(g, value, method_col="method", ref=PROPOSED, ref_variant="tea", variant=None, higher=True):
    sub = g if variant is None else g[g["variant"] == variant]
    key = ["detector", "eid"]
    piv = sub.pivot_table(index=key, columns=[method_col, "variant"] if "variant" in sub else method_col, values=value)
    table = {}
    for c in piv.columns:
        name = f"{c[0]}|{c[1]}" if isinstance(c, tuple) else c
        table[name] = piv[c].values
    refname = f"{ref}|{ref_variant}" if ref_variant else ref
    return stats.compare_to_reference(table, refname, higher_is_better=higher)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["sim", "hai"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--slowdown", type=float, default=1.0)
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds in args.datasets:
        runs = load_runs(ds, args.suffix)
        if not runs:
            continue
        tag = f"{ds}{args.suffix}" + (f"_x{args.slowdown:g}" if args.slowdown != 1 else "")
        methods = [m for m in MAIN_METHODS if m in runs[0]["events"][0].get("seq", runs[0]["events"][-1].get("seq", {}))
                   or any(m in e.get("seq", {}) for e in runs[0]["events"])]
        det = pd.DataFrame([dict(detector=r["detector"], seed=r["seed"], **r["detection"]) for r in runs])
        det.to_csv(TAB / f"detection_{tag}.csv", index=False)
        ev = event_table(runs, methods, slowdown=args.slowdown, t_act=30.0 if ds.startswith("sim") else 0.0)
        ev.to_csv(TAB / f"events_{tag}.csv.gz", index=False)
        rep = {"detection": det.groupby("detector").mean(numeric_only=True).to_dict("index")}
        loc = ev[~ev["timing"].str.startswith("dcl")]
        for metric in ("hr1", "hr3", "nrank"):
            s, g = summarise(loc, metric, ["timing", "method", "variant"])
            s.to_csv(TAB / f"loc_{metric}_{tag}.csv", index=False)
            for timing in ("alarm+0", "alarm+60", "bestguess"):
                gg = g[g.timing == timing]
                if len(gg) and gg.variant.nunique() >= 1:
                    rv = "tea" if timing != "bestguess" else "single"
                    rep[f"test_{metric}_{timing}"] = paired_tests(gg, metric, ref_variant=rv, higher=(metric != "nrank"))
        dcl = ev[ev["timing"].str.startswith("dcl")]
        for metric in ("hr1", "hr3"):
            s, g = summarise(dcl, metric, ["timing", "method", "variant"])
            s.to_csv(TAB / f"dcl_{metric}_{tag}.csv", index=False)
            hz = g[g.timing == "dcl:haz:hz"]
            if len(hz):
                rep[f"test_dcl_haz_{metric}"] = paired_tests(hz, metric, ref_variant="tea")
            g60 = g[g.timing == "dcl:fixed:60"]
            if len(g60):
                rep[f"test_dcl60_{metric}"] = paired_tests(g60, metric, ref_variant="tea")
        st = stability_table(runs, methods); st.to_csv(TAB / f"stability_{tag}.csv.gz", index=False)
        if len(st):
            tj = st[st.variant.isin(["single", "tea"])]
            s, g = summarise(tj, "temporal_jaccard", ["k", "method", "variant"]); s.to_csv(TAB / f"stab_temporal_{tag}.csv", index=False)
            nz = st[st.variant == "noise"]
            if len(nz):
                for v in ("noise_jaccard", "noise_spearman"):
                    s, _ = summarise(nz, v, ["method"]); s.to_csv(TAB / f"stab_{v}_{tag}.csv", index=False)
        fd = fidelity_table(runs, methods)
        if len(fd):
            for v in ("del_lkg", "del_mean"):
                s, g = summarise(fd, v, ["offset", "method"]); s.to_csv(TAB / f"fid_{v}_{tag}.csv", index=False)
        tt = tag_table(runs)
        if len(tt):
            for v in ("hr1", "hr3", "hr5"):
                s, g = summarise(tt, v, ["offset", "method"]); s.to_csv(TAB / f"tag_{v}_{tag}.csv", index=False)
        ct = cost_table(runs, methods); ct.to_csv(TAB / f"cost_{tag}.csv", index=False)
        report[tag] = rep
    json.dump(report, open(TAB / f"report{args.suffix}{'_x%g' % args.slowdown if args.slowdown != 1 else ''}.json", "w"),
              indent=1, default=float)
    print("wrote", TAB)


if __name__ == "__main__":
    main()
