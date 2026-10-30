"""Train a detector, evaluate detection, and record per-time explanations for every attack.

Usage: python scripts/run_localization.py --dataset hai --detector cnn --seed 0

Outputs (results/raw/<dataset>/<detector>_s<seed><suffix>.pkl): detection metrics, and for
each event the loop attributions of every method on a grid of explanation times starting
at the alarm (practical timing), plus a best-guess single shot, tag-level outputs,
deletion fidelity and perturbation stability at selected delivery times.
"""
from __future__ import annotations

import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xaicps import data, detectors, explain, metrics  # noqa: E402

CFG = dict(w=32, alpha=0.1, q=0.999, grid_step=5, horizon=180, ref_gap=60, k_bg=8,
           tag_eval_offsets=(0, 30, 60, 120), fid_offsets=(0, 60), noise_stab=0.1,
           budget_large=1024, max_windows={"hai": 120_000, "sim": 60_000, "sim-decoy": 60_000})


def get_dataset(name, noise=0.0):
    D = data.load_hai() if name == "hai" else data.load_sim(decoy=(name == "sim-decoy"))
    if noise > 0:                                     # RQ4: sensor noise on TEST data only
        rng = np.random.default_rng(12345)
        D.test = [(a + noise * rng.standard_normal(a.shape)).astype(np.float32) for a in D.test]
    return D


def get_detector(D, dsname, kind, seed):
    base = "sim" if dsname.startswith("sim") else "hai"     # decoy episodes reuse sim detectors
    p = ROOT / "results" / "models" / f"{base}_{kind}_s{seed}.pt"
    if p.exists():
        return detectors.Detector.load(p)
    if dsname == "sim-decoy":
        raise SystemExit("train the sim detector first")
    p.parent.mkdir(parents=True, exist_ok=True)
    det = detectors.Detector(kind, len(D.tags), w=CFG["w"], seed=seed, alpha=CFG["alpha"], q=CFG["q"])
    det.fit(D.train, D.val, max_windows=CFG["max_windows"][base])
    det.save(p)
    return det


def method_table(ctx, D, partitions, L, rng_seed):
    rng = lambda: np.random.default_rng(rng_seed)
    eq = 1 << L
    B = CFG["budget_large"]
    return {
        "RE": lambda t: explain.residual_error(ctx, t),
        "SM": lambda t: explain.saliency(ctx, t),
        "IG": lambda t: explain.integrated_gradients(ctx, t),
        "KS-eq": lambda t: explain.kernel_shap(ctx, t, eq, rng()),
        f"KS-{B}": lambda t: explain.kernel_shap(ctx, t, B, rng()),
        "LIME-eq": lambda t: explain.lime(ctx, t, eq, rng()),
        f"LIME-{B}": lambda t: explain.lime(ctx, t, B, rng()),
        "Fung-ens": None,                                     # composed from RE, SM, LIME-eq
        "LX": lambda t: explain.lx(ctx, t),
        "LX-bg": lambda t: explain.lx_background(ctx, t),
        "CLX": lambda t: explain.partition_exact(ctx, t, partitions["cluster"]),
        "RPX": lambda t: explain.partition_exact(ctx, t, partitions["random"]),
    }


def run_methods(table, ctx, t):
    out, wall = {}, {}
    for name, fn in table.items():
        if fn is None:
            continue
        t0 = time.perf_counter(); out[name] = fn(t); wall[name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["Fung-ens"] = explain.fung_ensemble(ctx, t, parts=dict(RE=out["RE"], SM=out["SM"], LIME=out["LIME-eq"]))
    wall["Fung-ens"] = wall["RE"] + wall["SM"] + wall["LIME-eq"] + time.perf_counter() - t0
    return out, wall


def deletion_auc(ctx, t, loop_scores, ref=None):
    """Restore the top-ranked loops one by one; area under the normalised score curve."""
    cur = ctx.seg(t); ref = ctx.ref() if ref is None else ref
    order = np.argsort(-loop_scores, kind="stable")
    masks = np.ones((ctx.L + 1, ctx.d), bool)
    for k in range(1, ctx.L + 1):
        masks[k] = masks[k - 1]
        masks[k, ctx.groups[order[k - 1]]] = False
    e = explain._eval_masks(ctx, cur, ref, masks)
    f = (e - e[-1]) / max(e[0] - e[-1], 1e-12)
    return float(np.trapezoid(f, dx=1.0 / ctx.L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["hai", "sim", "sim-decoy"])
    ap.add_argument("--detector", default="cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--ref-gap", type=int, default=CFG["ref_gap"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--extras", action="store_true", help="tag refinement, fidelity, stability")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    suffix = (f"_noise{args.noise}" if args.noise else "") + (f"_ref{args.ref_gap}" if args.ref_gap != CFG["ref_gap"] else "")
    out_p = ROOT / "results" / "raw" / args.dataset / f"{args.detector}_s{args.seed}{suffix}.pkl"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    D = get_dataset(args.dataset, args.noise)
    det = get_detector(D, args.dataset, args.detector, args.seed)
    groups = [np.array(v) for v in D.loops.values()]
    L, w = len(groups), det.w
    partitions = dict(cluster=explain.correlation_clusters(D.train, L), random=explain.random_partition(groups, len(D.tags), args.seed))

    t0 = time.time()
    scores = [det.score_series(X) for X in D.test]
    det_m, det_t = metrics.detection_metrics(scores, D.test_label, D.events, det.theta)
    print(f"[{args.dataset}/{args.detector}/s{args.seed}] detection {json.dumps({k: round(v, 3) for k, v in det_m.items()})} ({time.time() - t0:.0f}s)", flush=True)

    events = D.events[:args.max_events] if args.max_events else D.events
    rec = dict(cfg=CFG, args=vars(args), loops=list(D.loops), tags=D.tags, theta=det.theta,
               detection=det_m, train_info=det.train_info, events=[])
    for ei, ev in enumerate(events):
        X = D.test[ev.seg]; s = scores[ev.seg]
        tgt = [list(D.loops).index(l) for l in ev.loops if l in D.loops]
        er = dict(eid=ev.eid, loops=ev.loops, targets=tgt, points=ev.points, meta=ev.meta,
                  t_on=ev.t_on, t_off=ev.t_off, t_alarm=det_t[ev.eid])
        bgi = np.random.default_rng(ei).choice(np.arange(w, len(D.val[0]) - 1), CFG["k_bg"], replace=False)
        background = torch.stack([torch.from_numpy(D.val[0][i - w:i + 1]) for i in bgi])
        # --- best-guess timing (Fung et al.): window starts at the attack onset; oracle reference
        t_bg = ev.t_on + w
        if ev.t_on - 1 - w >= 0 and t_bg < len(X):
            ctx = explain.Context(det, X, groups, D.actuator, t_ref=ev.t_on - 1, background=background)
            table = method_table(ctx, D, partitions, L, args.seed)
            o, _ = run_methods(table, ctx, t_bg)
            er["bestguess"] = {m: v["loop"] for m, v in o.items()}
        # --- practical timing grid
        ta = det_t[ev.eid]
        if ta is not None:
            t_ref = max(ta - args.ref_gap, w)
            ctx = explain.Context(det, X, groups, D.actuator, t_ref=t_ref, background=background)
            table = method_table(ctx, D, partitions, L, args.seed)
            grid = [t for t in range(ta, min(len(X), ta + CFG["horizon"] + 1), CFG["grid_step"])]
            seq, walls, evals = {}, {}, {}
            for t in grid:
                o, wl = run_methods(table, ctx, t)
                for m in o:
                    seq.setdefault(m, []).append(o[m]["loop"]); walls.setdefault(m, []).append(wl[m])
                    evals[m] = o[m]["evals"]
                if args.extras and (t - ta) in CFG["tag_eval_offsets"]:
                    tg = {m: o[m]["tag"] for m in o if o[m]["tag"] is not None}
                    top = int(np.argmax(o["LX"]["loop"]))
                    tg["H-LX"], _ = explain.refine_tags(ctx, t, top)
                    er.setdefault("tags_at", {})[t - ta] = tg
                if args.extras and (t - ta) in CFG["fid_offsets"]:
                    zero = torch.zeros_like(ctx.ref())
                    er.setdefault("fid_at", {})[t - ta] = {
                        m: (deletion_auc(ctx, t, o[m]["loop"]), deletion_auc(ctx, t, o[m]["loop"], ref=zero)) for m in o}
                    Xn = X.copy(); rng = np.random.default_rng(1000 + ei)
                    Xn[t - w:t + 1] += CFG["noise_stab"] * rng.standard_normal((w + 1, X.shape[1])).astype(np.float32)
                    ctxn = explain.Context(det, Xn, groups, D.actuator, t_ref=t_ref); ctxn.background = ctx.background
                    on, _ = run_methods(method_table(ctxn, D, partitions, L, args.seed), ctxn, t)
                    er.setdefault("stab_at", {})[t - ta] = {m: (o[m]["loop"], on[m]["loop"]) for m in o}
            er.update(grid=np.array(grid), score=s[grid], seq={m: np.array(v) for m, v in seq.items()},
                      wall={m: float(np.median(v)) for m, v in walls.items()}, evals=evals)
        rec["events"].append(er)
        if (ei + 1) % 10 == 0:
            print(f"  {ei + 1}/{len(events)} events ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(rec, open(out_p, "wb"))
    print(f"saved {out_p} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
