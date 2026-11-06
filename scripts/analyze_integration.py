"""Analyses added when integrating the review of the parallel research branches.

1. Ground-truth cross-check: our HAI 21.03 loop/tag ground truth against an independent
   transcription of the HAI technical manual (configs/hai/independent/, from the
   codex/xai-cps-research-full-20261031 branch), and a sensitivity check for the one
   disagreement (A401).
2. Chance reference: expected HR@1/HR@3 of a uniformly random loop ranking on the same
   (attack, detector) units as the localization tables.
3. Action-type strata on HAI: HR@1 by manipulated variable type (SP, CV, +PV spoof), read
   from the target-controller code of the official attack table (e.g. P1-PC-SP1PV1).
4. PI-CAX component ablation (results/raw_picax_ablation, seed 0, all four detectors).

Writes results/tables/integration_*.csv and integration.json.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
L = {"hai": 9, "sim": 6}  # ranked groups (PCL-Sim: five loops + disturbance group)
FOCUS = ["RE", "IG", "KS-1024", "LX-bg", "PI-CAX", "LX"]
ALIAS = {"P2_SCO": "P2_CO_rpm", "P2_AutoSD": "P2_ASD", "P2_LCV01D": "P3_LCV01D"}  # manual name -> CSV column


def seed_avg(ev):
    loc = ev[~ev["timing"].str.startswith("dcl")]
    return loc.groupby(["detector", "eid", "timing", "method", "variant"], as_index=False)[["hr1", "hr3", "n_targets"]].mean()


def crosscheck():
    ours = pd.read_csv(ROOT / "configs/hai/hai2103_attacks.csv")
    ind = pd.read_csv(ROOT / "configs/hai/independent/hai21_event_target_map.csv")
    m = ours.merge(ind, on="attack_id")
    rows = []
    for _, r in m.iterrows():
        il = sorted({"-".join(x.split("-")[:2]) for x in r.target_controllers.split("|")})
        ol = sorted(set(r.target_loops.split(";")))
        ip = sorted({ALIAS.get(x, x) for x in r.documented_target_points.split("|")})
        op = sorted(set(r.target_points.split(";")))
        rows.append(dict(attack_id=r.attack_id, loops_ours=";".join(ol), loops_indep=";".join(il),
                         points_ours=";".join(op), points_indep=";".join(ip), loop_agree=ol == il, point_agree=op == ip))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "integration_gt_crosscheck.csv", index=False)
    return dict(n=len(df), loop_agree=int(df.loop_agree.sum()), point_agree=int(df.point_agree.sum()),
                disagree=df.loc[~(df.loop_agree & df.point_agree), "attack_id"].tolist())


def chance(ds):
    g = seed_avg(pd.read_csv(TAB / f"events_{ds}.csv.gz"))
    u = g[(g.timing == "alarm+0") & (g.method == "LX") & (g.variant == "single")].dropna(subset=["hr1"])
    n = u["n_targets"].round().astype(int)
    h1 = (n / L[ds]).mean()
    h3 = (1 - n.map(lambda k: comb(L[ds] - k, 3) / comb(L[ds], 3))).mean()
    return dict(units=len(u), hr1=float(h1), hr3=float(h3))


def action_type(code):
    return "+".join(t for t, k in (("SP", "SP"), ("CV", "CO"), ("PV", "PV")) if k in code)


def strata():
    ind = pd.read_csv(ROOT / "configs/hai/independent/hai21_event_target_map.csv")
    cls = {}
    for _, r in ind.iterrows():
        ctl = r.target_controllers.split("|")
        cls[r.attack_id] = action_type(ctl[0].split("-")[2]) if len(ctl) == 1 else "combined"
    g = seed_avg(pd.read_csv(TAB / "events_hai.csv.gz"))
    g["atype"] = g["eid"].map(cls)
    out = []
    for (t, v) in (("alarm+0", "single"), ("alarm+60", "tea")):
        sub = g[(g.timing == t) & (g.variant == v) & g.method.isin(FOCUS)].dropna(subset=["hr1"])
        for (a, meth), s in sub.groupby(["atype", "method"]):
            out.append(dict(timing=t, variant=v, atype=a, method=meth, hr1=s.hr1.mean(), n_units=len(s),
                            n_attacks=s.eid.nunique()))
    df = pd.DataFrame(out)
    df.to_csv(TAB / "integration_action_strata.csv", index=False)
    return df


def a401_sensitivity():
    g = seed_avg(pd.read_csv(TAB / "events_hai.csv.gz"))
    res = {}
    for (t, v) in (("alarm+0", "single"), ("alarm+60", "tea")):
        for meth in FOCUS:
            s = g[(g.timing == t) & (g.variant == v) & (g.method == meth)].dropna(subset=["hr1"])
            res[f"{meth}|{t}|{v}"] = dict(all=float(s.hr1.mean()), without_A401=float(s[s.eid != "A401"].hr1.mean()))
    return res


def setpoint_visibility(pre_s=600):
    """Does a targeted setpoint tag change during the attack beyond its pre-attack range?"""
    ind = pd.read_csv(ROOT / "configs/hai/independent/hai21_event_target_map.csv")
    raw = ROOT / "data/raw/hai-21.03"
    if not raw.exists():
        return None
    cache, rows = {}, []
    for _, r in ind.iterrows():
        f = r.dataset_file
        if f not in cache:
            cache[f] = pd.read_csv(raw / f)
        d, a, b = cache[f], int(r.start_row), int(r.end_row_exclusive)
        for p in sorted({ALIAS.get(p, p) for p in r.documented_target_points.split("|")}):
            if p not in d or not p.startswith(("P1_B", "P2_VTR", "P2_RTR", "P2_ASD")):   # setpoint tags only
                continue
            pre, att = d[p].iloc[max(0, a - pre_s):a], d[p].iloc[a:b]
            rows.append(dict(attack_id=r.attack_id, tag=p, pre_range=pre.max() - pre.min(), att_range=att.max() - att.min(),
                             visible=bool(att.max() - att.min() > 1e-6 + (pre.max() - pre.min()))))
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "integration_setpoint_visibility.csv", index=False)
    return dict(n=len(df), visible=int(df.visible.sum()), constant=int((df.att_range == 0).sum()))


def picax_ablation():
    import pickle, sys
    sys.path.insert(0, str(ROOT / "src"))
    from xaicps import explain, metrics
    rows = []
    for ds in ("hai", "sim"):
        for p in sorted((ROOT / "results" / "raw_picax_ablation" / ds).glob("*.pkl")):
            run = pickle.load(open(ROOT / "results" / "raw" / ds / p.name, "rb"))
            ab = pickle.load(open(p, "rb"))["events"]
            Lg, step = len(run["loops"]), run["cfg"]["grid_step"]
            for e in run["events"]:
                if e["eid"] not in ab or not e["targets"]:
                    continue
                for var, seq in ab[e["eid"]].items():
                    for off, tea in ((0, False), (60, False), (60, True)):
                        k = off // step
                        if k >= len(seq):
                            continue
                        v = explain.tea(seq[:k + 1], e["score"][:k + 1], run["theta"])[0][-1] if tea else seq[k]
                        rows.append(dict(dataset=ds, detector=p.stem.split("_s")[0], eid=e["eid"], variant=var,
                                         timing=f"alarm+{off}" + ("+tea" if tea else ""),
                                         hr1=metrics.loc_scores(v, e["targets"], Lg)["hr1"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    out = df.groupby(["dataset", "variant", "timing"]).hr1.agg(["mean", "size"]).reset_index()
    out.to_csv(TAB / "integration_picax_ablation.csv", index=False)
    return {f"{r.dataset}|{r.variant}|{r.timing}": float(r["mean"]) for _, r in out.iterrows()}


def picax_frozen():
    """HR@1 of PI-CAX under recalibrated-detector noise when its own statistics are NOT refitted."""
    import pickle, sys
    sys.path.insert(0, str(ROOT / "src"))
    from xaicps import explain, metrics
    out = {}
    for p in sorted((ROOT / "results" / "raw_picax_frozen").glob("*/*.pkl")):
        ds = p.parent.name
        run = pickle.load(open(ROOT / "results" / "raw" / ds / p.name, "rb"))
        pc = pickle.load(open(p, "rb"))["events"]
        h0, h60 = [], []
        for e in run["events"]:
            if e["eid"] not in pc or not e["targets"]:
                continue
            seq, Lg = pc[e["eid"]]["seq"], len(run["loops"])
            h0.append(metrics.loc_scores(seq[0], e["targets"], Lg)["hr1"])
            if len(seq) > 12:
                A = explain.tea(seq[:13], e["score"][:13], run["theta"])[0][-1]
                h60.append(metrics.loc_scores(A, e["targets"], Lg)["hr1"])
        out[f"{ds}|{p.stem}"] = dict(alarm=float(np.mean(h0)), tea60=float(np.mean(h60)), n=len(h0))
    return out


def picax_tests():
    """Holm-corrected Wilcoxon tests with PI-CAX+TEA as the reference (post hoc comparison)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from xaicps import stats
    out = {}
    for ds in ("hai", "sim"):
        ev = pd.read_csv(TAB / f"events_{ds}.csv.gz")
        for tm in ("alarm+0", "alarm+60", "dcl:fixed:60", "dcl:haz"):
            sub = ev[ev.timing == tm]
            if sub.empty:
                continue
            g = sub.groupby(["detector", "eid", "method", "variant"]).hr1.mean().reset_index()
            g["key"] = g.method + "|" + g.variant
            piv = g.pivot_table(index=["detector", "eid"], columns="key", values="hr1")
            cols = [c for c in piv.columns if c.endswith("|tea") or c == "PI-CAX|single"]
            res = stats.compare_to_reference({c: piv[c].values for c in cols}, "PI-CAX|tea")
            out[f"{ds}|{tm}"] = dict(n=res["n"], pairs={r["method"]: dict(p_holm=r["p_holm"], ref=r["ref_mean"],
                                                                     other=r["other_mean"], rb=r["effect_rb"]) for r in res["pairs"]})
    return out


def hai_delivery_ceiling():
    """Share of HAI units with any explanation delivered by D=60 s, and accuracy given delivery."""
    e = pd.read_csv(TAB / "events_hai.csv.gz")
    x = e[(e.timing == "dcl:fixed:60") & (e.variant == "tea")]
    out = dict(delivered=float(x[x.method == "LX"].delivered.mean()))
    for m in ("PI-CAX", "LX", "IG"):
        out[m] = float(x[(x.method == m) & (x.delivered == 1)].hr1.mean())
    return out


def main():
    rep = dict(crosscheck=crosscheck(), chance={ds: chance(ds) for ds in ("hai", "sim")},
               a401=a401_sensitivity(), setpoints=setpoint_visibility(), picax_ablation=picax_ablation(), picax_frozen=picax_frozen(), picax_tests=picax_tests(), hai_ceiling=hai_delivery_ceiling())
    st = strata()
    rep["strata_counts"] = st[(st.method == "LX") & (st.timing == "alarm+0")].set_index("atype").n_attacks.to_dict()
    (TAB / "integration.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "a401"}, indent=1, default=float))
    print(pd.DataFrame(rep["a401"]).T.round(3))
    print(st.pivot_table(index=["timing", "atype"], columns="method", values="hr1").round(2))


if __name__ == "__main__":
    main()
