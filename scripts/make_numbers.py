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


def f2(x, d=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def load(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() else None


def bold_best(vals, higher=True):
    v = np.array([x if np.isfinite(x) else (-np.inf if higher else np.inf) for x in vals])
    best = v.max() if higher else v.min()
    return [abs(x - best) < 1e-9 for x in v]


def table(path, header, rows, colspec, caption, label, note=None, size="\\scriptsize"):
    lines = ["\\begin{table}[t]", "\\centering" + size, f"\\caption{{{caption}}}\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + "\\\\", "\\midrule"]
    lines += [r + "\\\\" if not r.startswith("\\midrule") else r for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
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
        get = lambda d, t, m, v: (d[(d.timing == t) & (d.method == m) & (d.variant == v)]["mean"].values[:1] or [np.nan])[0]
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


if __name__ == "__main__":
    setup_macros(); detection_table(); loc_table(); stats_macros()
    out = ["% generated by scripts/make_numbers.py -- do not edit"]
    for k, v in sorted(macros.items()):
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (GEN / "numbers.tex").write_text("\n".join(out) + "\n")
    print(len(macros), "macros ->", GEN)
