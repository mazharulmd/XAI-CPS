"""Generate paper/generated/*.tex (macros + tables) from results/tables. No number in the
manuscript's Results section is typed by hand."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB, GEN = ROOT / "results" / "tables", ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

ORDER = ["RE", "SM", "IG", "KS-eq", "KS-1024", "LIME-eq", "LIME-1024", "Fung-ens", "CLX", "RPX", "LX-bg", "LX"]
NAME = {"RE": "RE", "SM": "SM", "IG": "IG", "KS-eq": "KernelSHAP-eq", "KS-1024": "KernelSHAP-1024",
        "LIME-eq": "LIME-eq", "LIME-1024": "LIME-1024", "Fung-ens": "Fung ens.", "CLX": "CLX (clusters)",
        "RPX": "RPX (random)", "LX-bg": "\\LX-bg", "LX": "\\textbf{\\LX{} (ours)}"}
macros = {}


def mac(name, val, fmt="{:.2f}"):
    macros[name] = fmt.format(val) if isinstance(val, (float, np.floating)) else str(val)


def first(a):
    return float(a[0]) if len(a) else np.nan


def f2(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def load(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() else None


def bold_best(vals, higher=True):
    v = np.array([round(x, 2) if np.isfinite(x) else (-np.inf if higher else np.inf) for x in vals])
    best = v.max() if higher else v.min()
    return [abs(x - best) < 1e-9 for x in v]


def table(path, header, rows, colspec, caption, label, note=None, size="\\scriptsize"):
    lines = ["\\begin{table}[t]", "\\centering" + size, f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\resizebox{\\linewidth}{!}{%", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + "\\\\", "\\midrule"]
    lines += [r + "\\\\" if not r.startswith("\\midrule") else r for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}}"]
    if note:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {note}}}")
    lines.append("\\end{table}")
    (GEN / path).write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- datasets / setup macros
def setup_macros():
    from xaicps import data as D
    S = D.load_sim()
    hz = np.array([e.meta["hazard_D"] for e in S.events])
    fin = hz[np.isfinite(hz)]
    mac("SimNattacks", len(S.events), "{}"); mac("SimNhaz", len(fin), "{}")
    mac("SimHazMedian", float(np.median(fin)), "{:.0f}"); mac("SimHazMin", float(fin.min()), "{:.0f}")
    mac("SimHazMax", float(fin.max()), "{:.0f}")
    H = D.load_hai()
    mac("HAId", len(H.tags), "{}"); mac("HAIL", len(H.loops), "{}")
    env = TAB / "benchmark_env.json"
    cpu = json.load(open(env)).get("cpu_model", "x86-64 CPU") if env.exists() else "x86-64 CPU"
    mac("CPUmodel", cpu.replace("(R)", "").replace("@", "at"), "{}")


# ----------------------------------------------------------------------------- detection
def detection_table():
    rows = []
    for ds, dn in (("sim", "PCL-Sim"), ("hai", "HAI 21.03")):
        d = load(f"detection_{ds}.csv")
        if d is None:
            continue
        g = d.groupby("detector").agg(["mean", "std"])
        for det in ["linear", "mlp", "cnn", "gru"]:
            if det not in g.index:
                continue
            r = g.loc[det]
            cell = lambda k, dd=2: f"{r[(k, 'mean')]:.{dd}f}" + (f"$\\pm${r[(k, 'std')]:.{dd}f}" if r[(k, 'std')] > 1e-9 else "")
            rows.append(f"{dn if det == 'linear' else ''} & {det.upper() if det != 'linear' else 'Linear'} & {cell('precision')} & {cell('recall')} & {cell('f1')} & {cell('event_recall')} & {cell('delay_median_s', 0)} & {cell('fa_per_hour')}")
            mac(f"det{ds.capitalize()}{det.capitalize()}ER", r[("event_recall", "mean")])
            mac(f"det{ds.capitalize()}{det.capitalize()}FA", r[("fa_per_hour", "mean")])
        rows.append("\\midrule")
    table("tab_detection.tex", "Data & Detector & P & R & F1 & Event rec. & Delay [s] & FA/h", rows[:-1],
          "@{}llcccccc@{}", "Detection performance (mean$\\pm$sd over 3 seeds; no point adjustment). Event recall: alarm within the attack or 60\\,s after it. FA/h: false-alarm runs per hour of attack-free data.",
          "tab:detection")


# ----------------------------------------------------------------------------- localization
def loc_table():
    for ds, dn in (("sim", "PCL-Sim"), ("hai", "HAI 21.03")):
        h1, h3, nr = load(f"loc_hr1_{ds}.csv"), load(f"loc_hr3_{ds}.csv"), load(f"loc_nrank_{ds}.csv")
        if h1 is None:
            continue
        get = lambda d, t, m, v: first(d[(d.timing == t) & (d.method == m) & (d.variant == v)]["mean"].values)
        cols = [("bestguess", "single", h1), ("alarm+0", "single", h1), ("alarm+60", "single", h1), ("alarm+60", "tea", h1),
                ("alarm+180", "single", h1), ("alarm+180", "tea", h1), ("alarm+60", "tea", h3), ("alarm+60", "tea", nr)]
        M = [m for m in ORDER if m in set(h1.method)]
        vals = np.array([[get(d, t, m, v) for (t, v, d) in cols] for m in M])
        bold = np.array([bold_best(vals[:, j], higher=(j != len(cols) - 1)) for j in range(len(cols))]).T
        rows = []
        for i, m in enumerate(M):
            cells = [("\\textbf{%s}" if bold[i, j] else "%s") % f2(vals[i, j]) for j in range(len(cols))]
            rows.append(NAME[m] + " & " + " & ".join(cells))
            for j, (t, v, d) in enumerate(cols):
                key = f"{ds}{m}{t}{v}{'hr3' if d is h3 else ('nr' if d is nr else 'hr1')}"
                macros["loc" + "".join(ch for ch in key if ch.isalnum())] = f2(vals[i, j])
        table(f"tab_loc_{ds}.tex",
              "& \\multicolumn{6}{c}{HR@1} & HR@3 & nRank\\\\\\cmidrule(lr){2-7}\\cmidrule(lr){8-8}\\cmidrule(lr){9-9} Method & best-guess & alarm & +60\\,s & +60\\,s\\,\\TEA & +180\\,s & +180\\,s\\,\\TEA & +60\\,s\\,\\TEA & +60\\,s\\,\\TEA",
              rows, "@{}lcccccccc@{}",
              f"Loop localization on {dn} ({'pooled over 4 detectors' if True else ''}; mean over attacks, seeds averaged). Bold: best in column. Single-shot unless marked \\TEA.",
              f"tab:loc_{ds}")


def stats_macros():
    for rep_f in ["report.json"]:
        p = TAB / rep_f
        if not p.exists():
            continue
        rep = json.load(open(p))
        for ds in ("sim", "hai"):
            if ds not in rep:
                continue
            for test in ("test_hr1_alarm+60", "test_hr1_alarm+0", "test_dcl_haz_hr1", "test_dcl60_hr1", "test_hr1_bestguess"):
                t = rep[ds].get(test)
                if not t:
                    continue
                for row in t["pairs"]:
                    key = f"p{ds}{test}{row['method']}"
                    macros["".join(ch for ch in key if ch.isalnum())] = (f"{row['p_holm']:.1e}" if row["p_holm"] < 0.001 else f"{row['p_holm']:.3f}")
                    macros["".join(ch for ch in "r" + key if ch.isalnum())] = f"{row['effect_rb']:.2f}"
                macros["".join(ch for ch in f"n{ds}{test}" if ch.isalnum())] = str(t["n"])


# ----------------------------------------------------------------------------- deadlines + cost
def deadline_table():
    for ds, dn in (("sim", "PCL-Sim"), ("hai", "HAI 21.03")):
        d1, d3, ct = load(f"dcl_hr1_{ds}.csv"), load(f"dcl_hr3_{ds}.csv"), load(f"cost_{ds}.csv")
        if d1 is None:
            continue
        g = lambda d, t, m, v: first(d[(d.timing == t) & (d.method == m) & (d.variant == v)]["mean"].values)
        cols = [(d1, "dcl:fixed:30", "tea"), (d1, "dcl:fixed:60", "single"), (d1, "dcl:fixed:60", "tea"),
                (d1, "dcl:fixed:120", "tea"), (d3, "dcl:fixed:60", "tea")]
        if ds == "sim":
            cols += [(d1, "dcl:haz:hz", "single"), (d1, "dcl:haz:hz", "tea"), (d3, "dcl:haz:hz", "tea")]
        M = [m for m in ORDER if m in set(d1.method)]
        vals = np.array([[g(d, t, m, v) for d, t, v in cols] for m in M])
        bold = np.array([bold_best(vals[:, j]) for j in range(len(cols))]).T
        cost = ct.groupby("method")[["wall_ms_median", "evals"]].median() if ct is not None else None
        rows = []
        for i, m in enumerate(M):
            cells = [("\\textbf{%s}" if bold[i, j] else "%s") % f2(vals[i, j]) for j in range(len(cols))]
            c = [f"{cost.loc[m, 'wall_ms_median']:.1f}", f"{int(cost.loc[m, 'evals'])}"] if cost is not None and m in cost.index else ["--", "--"]
            rows.append(NAME[m] + " & " + " & ".join(cells + c))
            for j, (d, t, v) in enumerate(cols):
                key = f"dcl{ds}{m}{t}{v}{'hrthree' if d is d3 else ''}"
                macros["".join(ch for ch in key if ch.isalnum())] = f2(vals[i, j])
        hdr = "Method & \\multicolumn{4}{c}{DCL@1($D$)} & DCL@3"
        hdr += (" & \\multicolumn{2}{c}{In-time@1} & In-time@3" if ds == "sim" else "") + " & Lat.\\,[ms] & Evals\\\\"
        hdr += " & 30\\,s\\,T & 60\\,s & 60\\,s\\,T & 120\\,s\\,T & 60\\,s\\,T" + (" & single & T & T" if ds == "sim" else "") + " & & "
        table(f"tab_deadline_{ds}.tex", hdr, rows, "@{}l" + "c" * (len(cols) + 2) + "@{}",
              f"Deadline-conditioned localization on {dn} (T = with \\TEA). Latency: median measured per explanation during the runs (1 thread, pooled over detectors); Evals: detector evaluations per explanation."
              + (" In-time: hazard deadline minus a 30\\,s operator reserve, hazardous attacks only." if ds == "sim" else ""),
              f"tab:deadline_{ds}")


def quality_table():
    rows = []
    for ds, dn in (("sim", "PCL-Sim"), ("hai", "HAI 21.03")):
        st, nj, fm, t1, t3 = (load(f"stab_temporal_{ds}.csv"), load(f"stab_noise_jaccard_{ds}.csv"),
                              load(f"fid_del_mean_{ds}.csv"), load(f"tag_hr1_{ds}.csv"), load(f"tag_hr3_{ds}.csv"))
        if st is None:
            continue
        ev = pd.read_csv(TAB / f"stability_{ds}.csv.gz") if (TAB / f"stability_{ds}.csv.gz").exists() else None
        M = [m for m in ORDER if m in set(st.method)]
        g = lambda d, q: first(d.query(q)["mean"].values)
        vals = []
        for m in M:
            vals.append([g(st, f"k==1 and method=='{m}' and variant=='single'"), g(st, f"k==1 and method=='{m}' and variant=='tea'"),
                         g(nj, f"method=='{m}'"), g(fm, f"offset==0 and method=='{m}'"),
                         g(t1, f"offset==0 and method=='{'H-LX' if m == 'LX' else m}'") if t1 is not None else np.nan,
                         g(t3, f"offset==0 and method=='{'H-LX' if m == 'LX' else m}'") if t3 is not None else np.nan])
        vals = np.array(vals)
        hib = [True, True, True, False, True, True]
        bold = np.array([bold_best(vals[:, j], hib[j]) for j in range(vals.shape[1])]).T
        for i, m in enumerate(M):
            cells = [("\\textbf{%s}" if bold[i, j] else "%s") % f2(vals[i, j]) for j in range(vals.shape[1])]
            rows.append(f"{dn if i == 0 else ''} & {NAME[m]}{' (H-\\LX)' if m == 'LX' else ''} & " + " & ".join(cells))
            for j, nm in enumerate(["tsingle", "ttea", "noise", "del", "tagone", "tagthree"]):
                macros["".join(ch for ch in f"q{ds}{m}{nm}" if ch.isalpha())] = f2(vals[i, j])
        rows.append("\\midrule")
    table("tab_quality.tex", "Data & Method & \\multicolumn{2}{c}{Temporal top-1 agr.} & Noise Jacc.@3 & Deletion$\\downarrow$ & Tag HR@1 & Tag HR@3\\\\ & & single & \\TEA & & & & ",
          rows[:-1], "@{}llcccccc@{}",
          "Explanation quality. Temporal agreement: share of consecutive explanations (5\\,s apart, first 120\\,s) with the same top-1 loop. Noise Jaccard: top-3 overlap under Gaussian input noise ($\\sigma=0.1$) at alarm and +60\\,s. Deletion: area under the normalised score curve when loops are restored to the training mean in ranked order (lower is better), at the alarm. Tag HR@$k$: rank of a manipulated tag among all tags at the alarm (for \\LX: H-\\LX{} refinement); HAI excludes P2 setpoint targets that are invisible in the data.",
          "tab:quality")


def tests_table():
    p = TAB / "report.json"
    if not p.exists():
        return
    rep = json.load(open(p))
    rows = []
    for ds, dn in (("sim", "PCL-Sim"), ("hai", "HAI 21.03")):
        if ds not in rep:
            continue
        for test, lab in (("test_hr1_alarm+0", "HR@1, alarm"), ("test_hr1_alarm+60", "HR@1, +60\\,s"),
                          ("test_dcl60_hr1", "DCL@1(60\\,s)"), ("test_dcl_haz_hr1", "In-time@1")):
            t = rep[ds].get(test)
            if not t:
                continue
            first = True
            for row in t["pairs"]:
                m, v = row["method"].split("|")
                if not ((v == "tea" and m in ("IG", "KS-1024", "LIME-1024", "Fung-ens", "LX-bg", "RE", "CLX", "RPX")) or (m == "LX" and v == "single")):
                    continue
                name = (NAME[m].replace("\\textbf{", "").replace(" (ours)}", "") + (" + \\TEA" if v == "tea" else " single-shot"))
                pv = row["p_holm"]; pstr = "$<$0.001" if pv < 0.001 else f"{pv:.3f}"
                rows.append(f"{dn if first else ''} & {lab if first else ''} & {name} & {row['ref_mean']:.3f} & {row['other_mean']:.3f} & {pstr} & {row['effect_rb']:+.2f}")
                first = False
            rows.append("\\midrule")
    table("tab_tests.tex", "Data & Metric & Comparator & \\LX+\\TEA & Comp. & $p_{\\mathrm{Holm}}$ & $r_{\\mathrm{rb}}$", rows[:-1],
          "@{}lllcccc@{}", "Paired comparisons of \\LX+\\TEA{} against selected configurations (two-sided Wilcoxon signed-rank on (attack, detector) units, Holm correction over all 23 comparisons per row group; $r_{\\mathrm{rb}}>0$ favours \\LX+\\TEA). At the alarm \\TEA{} has no history, so ``\\LX+\\TEA'' equals \\LX{} single-shot.",
          "tab:tests", size="\\tiny")


def phase2_table():
    d = load("phase2.csv")
    if d is None:
        return
    rows = []
    for ds, dn in (("hai", "HAI"), ("sim", "PCL-Sim")):
        for cond in ["ref32", "ref60 (default)", "ref120", "ref300", "noise0.1", "noise0.3"]:
            sub = d[(d.dataset == ds) & (d.condition == cond) & (d.metric == "hr1")]
            if not len(sub):
                continue
            cell = lambda m, v, o: f2(first(sub[(sub.method == m) & (sub.variant == v) & (sub.offset == o)]["mean"].values))
            rows.append(f"{dn} & {cond.replace('ref', '$\\Delta_{\\mathrm{ref}}$=').replace(' (default)', '\\,s (default)').replace('noise', '$\\sigma_n$=')} & "
                        + " & ".join(cell(m, v, o) for m in ("LX", "IG", "KS-1024", "RE") for v, o in (("single", 0), ("tea", 60))))
        rows.append("\\midrule")
    for cond in ("no decoy", "decoy"):
        sub = d[(d.dataset == "sim") & (d.condition == cond)]
        if not len(sub):
            continue
        for metric, lab in (("hr1", "HR@1"), ("decoy_top1", "decoy@1")):
            s2 = sub[sub.metric == metric]
            cell = lambda m, v, o: f2(first(s2[(s2.method == m) & (s2.variant == v) & (s2.offset == o)]["mean"].values))
            rows.append(f"PCL-Sim & {cond}: {lab} & " + " & ".join(cell(m, v, o) for m in ("LX", "IG", "KS-1024", "RE") for v, o in (("single", 0), ("tea", 60))))
    table("tab_phase2.tex", "Data & Condition & \\multicolumn{2}{c}{\\LX} & \\multicolumn{2}{c}{IG} & \\multicolumn{2}{c}{KS-1024} & \\multicolumn{2}{c}{RE}\\\\ & & alarm & +60\\,T & alarm & +60\\,T & alarm & +60\\,T & alarm & +60\\,T",
          rows, "@{}llcccccccc@{}",
          "Ablations and robustness (HR@1 unless noted; CNN detector, seed 0 for the reference and noise rows; linear and CNN detectors, 3 seeds, for the decoy rows). decoy@1: share of attacks in which the innocent decoy loop is ranked first. Noise is added to the test data only, so detection also changes (see text).",
          "tab:phase2", size="\\tiny")


def leakage():
    p = TAB / "leakage_hai.json"
    if not p.exists():
        return
    s = json.load(open(p))["summary"]
    for split in ("random", "blocked"):
        for m in ("f1", "auc", "ap"):
            mac(f"leak{split.capitalize()}{m.upper()}", s[split][m][0], "{:.3f}")
            mac(f"leak{split.capitalize()}{m.upper()}sd", s[split][m][1], "{:.3f}")


def latency_table():
    rows = []
    for ds, dn in (("sim", "PCL-Sim"), ("hai", "HAI 21.03")):
        d = load(f"latency_{ds}.csv")
        if d is None:
            continue
        g = d.groupby(["method", "detector"])[["wall_ms", "evals"]].median().unstack("detector")
        order = [m for m in ["RE", "SM", "IG", "KS-eq", "KS-1024", "LIME-eq", "LIME-1024", "Fung-ens", "LX-bg", "ShaTS-sensor", "LX", "H-LX (refine top loop)"] if m in g.index]
        for i, m in enumerate(order):
            r = g.loc[m]
            cells = [f"{r[('wall_ms', k)]:.1f}" if ("wall_ms", k) in r.index and np.isfinite(r[("wall_ms", k)]) else "--" for k in ("linear", "mlp", "cnn", "gru")]
            ev = int(np.nanmax([r[("evals", k)] for k in ("linear", "mlp", "cnn", "gru") if ("evals", k) in r.index]))
            nm = NAME.get(m, m.replace("H-LX (refine top loop)", "H-\\LX{} (tags)"))
            rows.append(f"{dn if i == 0 else ''} & {nm} & {ev} & " + " & ".join(cells))
            if m == "LX":
                for k in ("linear", "mlp", "cnn", "gru"):
                    if ("wall_ms", k) in r.index:
                        mac(f"lat{ds}LX{k}", r[("wall_ms", k)], "{:.1f}")
            if m == "ShaTS-sensor":
                mac(f"evals{ds}ShaTS", ev, "{}")
                for k in ("linear", "cnn"):
                    if ("wall_ms", k) in r.index and np.isfinite(r[("wall_ms", k)]):
                        mac(f"lat{ds}ShaTS{k}", r[("wall_ms", k)] / 1000, "{:.1f}")
        rows.append("\\midrule")
    table("tab_latency.tex", "Data & Method & Evals & Linear & MLP & CNN & GRU", rows[:-1], "@{}llrrrrr@{}",
          "Explanation latency on one CPU thread of an otherwise idle machine (median over 10 incidents $\\times$ 5 repetitions, ms; ShaTS: 2 incidents, 1 repetition). Evals: detector evaluations per explanation.",
          "tab:latency", size="\\tiny")


def shats():
    import pickle
    for ds in ("sim", "hai"):
        p = ROOT / "results" / "raw" / f"shats_{ds}.pkl"
        if not p.exists():
            continue
        rec = pickle.load(open(p, "rb"))
        for det, rr in rec.items():
            if rr:
                mac(f"shats{ds}{det}HR", float(np.mean([r["shats_hr1"] for r in rr])))
                mac(f"shats{ds}{det}LXHR", float(np.mean([r["lx_hr1"] for r in rr])))
                mac(f"shats{ds}{det}N", len(rr), "{}")


def intime_decomposition():
    p = TAB / "events_sim.csv.gz"
    if not p.exists():
        return
    ev = pd.read_csv(p)
    h = ev[(ev.timing == "dcl:haz:hz") & (ev.variant == "tea")]
    lx = h[h.method == "LX"]
    fail = lx[lx.hr1 == 0]
    mac("intimeFail", len(fail), "{}")
    mac("intimeFailUndelivered", int((~fail.delivered.astype(bool)).sum()), "{}")
    mac("intimeFailUndeliveredPct", 100 * float((~fail.delivered.astype(bool)).mean()), "{:.0f}")
    for m in ("LX", "IG", "LX-bg", "RE", "KS-1024", "Fung-ens"):
        x = h[(h.method == m) & h.delivered.astype(bool)]
        mac("intimeDelivered" + "".join(ch for ch in m if ch.isalnum()), float(x.hr1.mean()))


if __name__ == "__main__":
    setup_macros(); detection_table(); loc_table(); stats_macros(); deadline_table(); quality_table(); tests_table(); phase2_table(); leakage(); latency_table(); shats(); intime_decomposition()
    for name in ["evalshaiShaTS", "latshaiShaTScnn", "shatshaicnnHR", "shatshaicnnLXHR"]:
        if name not in macros:
            macros[name] = "\\textbf{??}"; print("WARNING: placeholder for", name)
    for f in ["phase2_text.tex", "findings_intro.tex", "appendix_tables.tex", "tab_phase2.tex"]:
        if not (GEN / f).exists():
            (GEN / f).write_text("\\textbf{[pending]}\n"); print("WARNING: stub", f)
    out = ["% generated by scripts/make_numbers.py -- do not edit"]
    words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
    clean = {}
    for k, v in macros.items():
        clean["".join(words.get(ch, ch) for ch in k if ch.isalnum())] = v
    for k, v in sorted(clean.items()):
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (GEN / "numbers.tex").write_text("\n".join(out) + "\n")
    print(len(macros), "macros ->", GEN)
