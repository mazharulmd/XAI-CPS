"""Figures for the paper (PDF, vector). Reads results/tables/*.csv only."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "results" / "tables", ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# reference categorical palette (fixed order) + neutral for context series
C = {"LX": "#2a78d6", "KS-1024": "#eb6834", "RE": "#1baf7a", "IG": "#eda100", "other": "#9a9890"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 1.6, "pdf.fonttype": 42})
DSN = {"sim": "PCL-Sim (280 attacks)", "hai": "HAI 21.03 (50 attacks)"}
LABEL = {"LX": "LX (ours)", "KS-1024": "KernelSHAP-1024", "RE": "Raw error (RE)", "IG": "Integrated gradients"}


def fig_over_time():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, ds in zip(axes, ["sim", "hai"]):
        p = TAB / f"loc_hr1_{ds}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d = d[d.timing.str.startswith("alarm+")]
        d["off"] = d.timing.str[6:].astype(int)
        for m in ["LX", "KS-1024", "RE", "IG"]:
            for var, ls in (("tea", "-"), ("single", (0, (3, 2)))):
                s = d[(d.method == m) & (d.variant == var)].sort_values("off")
                ax.plot(s.off, s["mean"], color=C[m], ls=ls, marker="o", ms=3,
                        label=f"{LABEL[m]} + TEA" if var == "tea" else f"{LABEL[m]} (single-shot)")
                if m == "LX" and var == "tea":
                    ax.fill_between(s.off, s.ci_lo, s.ci_hi, color=C[m], alpha=0.12, lw=0)
        ax.set_title(DSN[ds], color=INK, fontsize=8.5); ax.set_xlabel("delivery time after alarm [s]")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("HR@1 (attacked loop ranked first)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "hr1_over_time.pdf", bbox_inches="tight"); plt.close(fig)


def fig_dcl():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, ds in zip(axes, ["sim", "hai"]):
        p = TAB / f"dcl_hr1_{ds}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d = d[d.timing.str.startswith("dcl:fixed")]
        d["D"] = d.timing.str.split(":").str[2].astype(int)
        for m in ["LX", "KS-1024", "RE", "IG"]:
            for var, ls in (("tea", "-"), ("single", (0, (3, 2)))):
                s = d[(d.method == m) & (d.variant == var)].sort_values("D")
                ax.plot(s.D, s["mean"], color=C[m], ls=ls, marker="o", ms=3)
        ax.set_xscale("log"); ax.set_xticks([10, 30, 60, 120, 300]); ax.set_xticklabels(["10", "30", "60", "120", "300"])
        ax.set_title(DSN[ds], color=INK, fontsize=8.5); ax.set_xlabel("deadline after attack onset D [s]"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("DCL@1(D)")
    fig.tight_layout(); fig.savefig(FIG / "dcl.pdf", bbox_inches="tight"); plt.close(fig)


def fig_cost():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, ds in zip(axes, ["sim", "hai"]):
        pc, pl = TAB / f"cost_{ds}.csv", TAB / f"loc_hr1_{ds}.csv"
        if not (pc.exists() and pl.exists()):
            continue
        cost = pd.read_csv(pc).groupby("method")[["wall_ms_median", "evals"]].median()
        loc = pd.read_csv(pl); loc = loc[(loc.timing == "alarm+60") & (loc.variant == "tea")].set_index("method")["mean"]
        for m in cost.index:
            if m not in loc:
                continue
            col = C.get(m, C["other"])
            ax.scatter(cost.loc[m, "wall_ms_median"], loc[m], s=26 if m == "LX" else 16, color=col,
                       edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(m, (cost.loc[m, "wall_ms_median"], loc[m]), xytext=(4, 2), textcoords="offset points",
                        fontsize=6.5, color=INK if m == "LX" else INK2)
        ax.set_xscale("log"); ax.set_xlabel("median latency per explanation [ms, 1 CPU thread]")
        ax.set_title(DSN[ds], color=INK, fontsize=8.5); ax.set_ylim(0, 1)
    axes[0].set_ylabel("HR@1 at alarm+60 s (with TEA)")
    fig.tight_layout(); fig.savefig(FIG / "cost_accuracy.pdf", bbox_inches="tight"); plt.close(fig)


def fig_stability():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, ds in zip(axes, ["sim", "hai"]):
        p = TAB / f"stab_temporal_{ds}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d = d[d.k == 1]
        piv = d.pivot_table(index="method", columns="variant", values="mean").sort_values("tea")
        y = np.arange(len(piv))
        for i, (m, r) in enumerate(piv.iterrows()):
            ax.plot([r.single, r.tea], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter(piv.single, y, color=C["other"], s=14, zorder=2, label="single-shot")
        ax.scatter(piv.tea, y, color=C["LX"], s=14, zorder=3, label="with TEA")
        ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=6.5)
        ax.set_xlabel("top-1 agreement between consecutive explanations"); ax.set_title(DSN[ds], color=INK, fontsize=8.5)
        ax.set_xlim(0, 1.02)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "temporal_stability.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_over_time(); fig_dcl(); fig_cost(); fig_stability()
    print("figures ->", FIG)
