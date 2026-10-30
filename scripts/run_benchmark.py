"""Latency / cost benchmark (single thread, run on an otherwise idle machine) and
ShaTS-sensor accuracy at practical timing.

Usage: python scripts/run_benchmark.py --dataset hai --detectors linear mlp cnn gru
"""
from __future__ import annotations

import argparse, json, pickle, platform, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from xaicps import explain, metrics  # noqa: E402
import run_localization as RL  # noqa: E402


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts)), float(np.max(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim")
    ap.add_argument("--detectors", nargs="+", default=["linear", "mlp", "cnn", "gru"])
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--shats-accuracy", nargs="*", default=["linear", "cnn"])
    args = ap.parse_args()
    torch.set_num_threads(1)
    D = RL.get_dataset(args.dataset)
    groups = [np.array(v) for v in D.loops.values()]
    L = len(groups)
    rows, shats_rec = [], {}
    for kind in args.detectors:
        det = RL.get_detector(D, args.dataset, kind, 0)
        scores = [det.score_series(X) for X in D.test]
        _, det_t = metrics.detection_metrics(scores, D.test_label, D.events, det.theta)
        parts = dict(cluster=explain.correlation_clusters(D.train, L), random=explain.random_partition(groups, len(D.tags), 0))
        evs = [e for e in D.events if det_t[e.eid] is not None]
        bg = torch.stack([torch.from_numpy(D.val[0][i - det.w:i + 1]) for i in range(det.w + 100, det.w + 100 + 8 * 97, 97)])
        for ev in evs[:args.n_events]:
            ta = det_t[ev.eid]
            ctx = explain.Context(det, D.test[ev.seg], groups, D.actuator, t_ref=max(ta - 60, det.w), background=bg)
            table = RL.method_table(ctx, D, parts, L, 0)
            table["Fung-ens"] = lambda: explain.fung_ensemble(ctx, ta)
            table["H-LX (refine top loop)"] = lambda: explain.refine_tags(ctx, ta, int(np.argmax(explain.lx(ctx, ta)["loop"])))
            for name, fn in table.items():
                f = (lambda fn=fn: fn(ta)) if name not in ("Fung-ens", "H-LX (refine top loop)") else fn
                out, med, mx = timeit(f, args.reps)
                ev_n = out["evals"] if isinstance(out, dict) else out[1]
                rows.append(dict(dataset=args.dataset, detector=kind, eid=ev.eid, method=name, wall_ms=1e3 * med, wall_ms_max=1e3 * mx, evals=ev_n))
            if ev is evs[0] or ev is evs[min(2, len(evs) - 1)]:
                out, med, mx = timeit(lambda: explain.shats_sensor(ctx, ta), 1)
                rows.append(dict(dataset=args.dataset, detector=kind, eid=ev.eid, method="ShaTS-sensor", wall_ms=1e3 * med, wall_ms_max=1e3 * mx, evals=out["evals"]))
        print(kind, "timed", flush=True)
        if kind in args.shats_accuracy:
            rec = []
            for ev in D.events:
                ta = det_t[ev.eid]
                tg = [list(D.loops).index(l) for l in ev.loops if l in D.loops]
                if ta is None or not tg:
                    continue
                ctx = explain.Context(det, D.test[ev.seg], groups, D.actuator, t_ref=max(ta - 60, det.w))
                o = explain.shats_sensor(ctx, ta)
                rec.append(dict(eid=ev.eid, targets=tg, loop=o["loop"], tag=o["tag"], lx=explain.lx(ctx, ta)["loop"],
                                **{f"shats_{k}": v for k, v in metrics.loc_scores(o["loop"], tg, L).items()},
                                **{f"lx_{k}": v for k, v in metrics.loc_scores(explain.lx(ctx, ta)["loop"], tg, L).items()}))
            shats_rec[kind] = rec
            print(kind, "ShaTS accuracy done", flush=True)
    import pandas as pd
    out = ROOT / "results" / "tables"; out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / f"latency_{args.dataset}.csv", index=False)
    pickle.dump(shats_rec, open(ROOT / "results" / "raw" / f"shats_{args.dataset}.pkl", "wb"))
    env = dict(python=platform.python_version(), torch=torch.__version__, cpu=platform.processor() or platform.machine(),
               threads=torch.get_num_threads())
    try:
        env["cpu_model"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    json.dump(env, open(out / "benchmark_env.json", "w"), indent=1)
    print(df.groupby(["detector", "method"])[["wall_ms", "evals"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
