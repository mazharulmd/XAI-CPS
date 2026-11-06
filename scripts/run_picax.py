"""Evaluate the ported PI-CAX explainer on the same incidents, alarms and grid as a main run.

Reads results/raw/<dataset>/<detector>_s<seed>.pkl (written by run_localization.py) and the
cached detector, computes PI-CAX loop attributions at every grid time, and writes
results/raw_picax/<dataset>/<detector>_s<seed>.pkl. analyze.py merges these into the main runs.
"""
import argparse, pickle, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_localization as RL  # noqa: E402
from xaicps import explain  # noqa: E402
from xaicps.picax import PICAX  # noqa: E402


VARIANTS = {"full": {}, "-causal": dict(causal=False), "-manifold": dict(manifold=False),
            "-envelope": dict(envelope=False), "-tier2": dict(tier2=False),
            "none": dict(causal=False, manifold=False, envelope=False, tier2=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["hai", "sim", "sim-decoy"])
    ap.add_argument("--detector", default="cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--recal", action="store_true")
    ap.add_argument("--ref-gap", type=int, default=RL.CFG["ref_gap"])
    ap.add_argument("--frozen-stats", action="store_true", help="with --recal: do NOT recalibrate PI-CAX (sensitivity check)")
    ap.add_argument("--ablation", action="store_true", help="also run the component ablations (seq per variant)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    suffix = (f"_noise{args.noise}" if args.noise else "") + ("_recal" if args.recal else "") + (f"_ref{args.ref_gap}" if args.ref_gap != RL.CFG["ref_gap"] else "")
    name = f"{args.detector}_s{args.seed}{suffix}.pkl"          # same naming as run_localization.py
    src = ROOT / "results" / "raw" / args.dataset / name
    sub = "raw_picax_ablation" if args.ablation else ("raw_picax_frozen" if args.frozen_stats else "raw_picax")
    out = ROOT / "results" / sub / args.dataset / name
    out.parent.mkdir(parents=True, exist_ok=True)
    run = pickle.load(open(src, "rb"))
    D = RL.get_dataset(args.dataset, args.noise, noisy_val=args.recal)
    det = RL.get_detector(D, args.dataset, args.detector, args.seed)
    if args.recal and args.noise > 0:
        det.calibrate(D.val_noisy)
    assert abs(det.theta - run["theta"]) < 1e-6, "detector differs from the one used for the main run"
    groups = [np.array(v) for v in D.loops.values()]
    t0 = time.time()
    pc = PICAX(det, D.train, groups, D.actuator, seed=args.seed)
    if args.recal and args.noise > 0 and not args.frozen_stats:
        pc.recalibrate(D.val_noisy, seed=args.seed)       # same noisy validation data as the detector
    fit_s = time.time() - t0
    ev_by_id = {e.eid: e for e in D.events}
    res = {}
    cfg, w = run["cfg"], det.w
    for ei, er in enumerate(run["events"]):
        if er["t_alarm"] is None or "grid" not in er:
            continue
        ev = ev_by_id[er["eid"]]
        X = D.test[ev.seg]
        ctx = explain.Context(det, X, groups, D.actuator, t_ref=max(er["t_alarm"] - args.ref_gap, w))
        if args.ablation:
            res[er["eid"]] = {v: np.array([pc.explain(ctx, int(t), **kw)["loop"] for t in er["grid"]]) for v, kw in VARIANTS.items()}
            continue
        seq, walls = [], []
        for t in er["grid"]:
            t1 = time.perf_counter(); o = pc.explain(ctx, int(t)); walls.append(time.perf_counter() - t1)
            seq.append(o["loop"])
        res[er["eid"]] = dict(seq=np.array(seq), wall=float(np.median(walls)), evals=o["evals"])
        # same extras as run_localization.py --extras (identical offsets, references and noise draws)
        ta = er["t_alarm"]
        for t in er["grid"]:
            t = int(t)
            if (t - ta) in cfg["tag_eval_offsets"]:
                res[er["eid"]].setdefault("tags_at", {})[t - ta] = pc.explain(ctx, t)["tag"]
            if (t - ta) in cfg["fid_offsets"]:
                v = pc.explain(ctx, t)["loop"]
                zero = torch.zeros_like(ctx.ref())
                res[er["eid"]].setdefault("fid_at", {})[t - ta] = (RL.deletion_auc(ctx, t, v), RL.deletion_auc(ctx, t, v, ref=zero))
                Xn = X.copy(); rng = np.random.default_rng(1000 + ei)
                Xn[t - w:t + 1] += cfg["noise_stab"] * rng.standard_normal((w + 1, X.shape[1])).astype(np.float32)
                ctxn = explain.Context(det, Xn, groups, D.actuator, t_ref=ctx.t_ref)
                res[er["eid"]].setdefault("stab_at", {})[t - ta] = (v, pc.explain(ctxn, t)["loop"])
    pickle.dump(dict(fit_s=fit_s, events=res), open(out, "wb"))
    print(f"[{args.dataset}/{args.detector}/s{args.seed}] PI-CAX on {len(res)} incidents, fit {fit_s:.0f}s, "
          f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
