"""Datasets in a common format.

A `Dataset` holds chronologically split, train-standardised arrays plus a loop map and
attack events with ground truth. Leakage rules enforced here:
  * scaling statistics come from the TRAIN portion only;
  * validation = last `val_frac` of every normal training file (chronological);
  * test files are never touched when fitting anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Event:
    eid: str
    seg: int            # index into Dataset.test
    t_on: int           # onset (row index within the segment)
    t_off: int          # end (exclusive)
    loops: list         # ground-truth target loop names
    points: list        # ground-truth target tags (may include tags dropped as constant)
    meta: dict = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    tags: list                  # model input tags (order of columns)
    loops: dict                 # loop -> list of tag indices (into tags)
    train: list                 # list of [n,d] float32 arrays (normal)
    val: list                   # list of [n,d] float32 arrays (normal)
    test: list                  # list of [n,d] float32 arrays
    test_label: list            # list of [n] int arrays (1 = attack)
    events: list                # list of Event
    actuator: np.ndarray        # [d] bool, actuator-type tags (for the Fung ensemble)
    mean: np.ndarray = None
    std: np.ndarray = None

    @property
    def loop_names(self):
        return list(self.loops)

    def loop_of_tag(self):
        m = np.full(len(self.tags), -1)
        for li, (_, idx) in enumerate(self.loops.items()):
            m[idx] = li
        return m


def _standardise(train, others):
    cat = np.concatenate(train, 0)
    mu = cat.mean(0)
    sd = cat.std(0)
    sd[sd < 1e-6] = 1.0
    f = lambda a: ((a - mu) / sd).astype(np.float32)
    return [f(a) for a in train], [[f(a) for a in grp] for grp in others], mu, sd


# --------------------------------------------------------------------------- HAI 21.03
HAI_DIR = ROOT / "data" / "raw" / "hai-21.03"


def _hai_actuator(tag):
    return (tag.endswith(("D", "Z")) and tag[3:5] in ("FC", "LC", "PC", "PP")) or tag in (
        "P1_PP01AD", "P1_PP01AR", "P1_PP01BD", "P1_PP01BR", "P1_PP02D", "P1_PP02R",
        "P2_CO_rpm", "P3_LCP01D", "P3_LCV01D", "P1_B4005", "P4_ST_GOV")


def load_hai(val_frac=0.2, cache=True):
    cache_f = ROOT / "data" / "cache" / "hai2103.npz"
    loops_cfg = yaml.safe_load(open(ROOT / "configs" / "hai" / "hai2103_loops.yaml"))
    gt = pd.read_csv(ROOT / "configs" / "hai" / "hai2103_attacks.csv")
    if cache and cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        raw_tr, raw_te, lab, times, all_tags = (list(z["raw_tr"]), list(z["raw_te"]), list(z["lab"]),
                                                list(z["times"]), list(z["all_tags"]))
    else:
        all_tags = None
        raw_tr, raw_te, lab, times = [], [], [], []
        for f in ["train1", "train2", "train3"]:
            d = pd.read_csv(HAI_DIR / f"{f}.csv.gz")
            all_tags = [c for c in d.columns if c[:2] in ("P1", "P2", "P3", "P4")]
            raw_tr.append(d[all_tags].to_numpy(np.float32))
        for f in ["test1", "test2", "test3", "test4", "test5"]:
            d = pd.read_csv(HAI_DIR / f"{f}.csv.gz")
            raw_te.append(d[all_tags].to_numpy(np.float32))
            lab.append(d[["attack", "attack_P1", "attack_P2", "attack_P3"]].to_numpy(np.int8))
            times.append(d["time"].astype(str).to_numpy())
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_f, raw_tr=np.array(raw_tr, dtype=object), raw_te=np.array(raw_te, dtype=object),
                 lab=np.array(lab, dtype=object), times=np.array(times, dtype=object),
                 all_tags=np.array(all_tags))
    all_tags = [str(t) for t in all_tags]
    # chronological split of each normal file
    tr, va = [], []
    for a in raw_tr:
        cut = int(len(a) * (1 - val_frac))
        tr.append(a[:cut]); va.append(a[cut:])
    # drop tags that are constant over the TRAIN portion (no information for a forecaster)
    cat = np.concatenate(tr, 0)
    keep = np.flatnonzero(cat.std(0) > 1e-6)
    tags = [all_tags[i] for i in keep]
    tr = [a[:, keep] for a in tr]; va = [a[:, keep] for a in va]; te = [a[:, keep] for a in raw_te]
    tr, (va, te), mu, sd = _standardise(tr, [va, te])
    te = [np.clip(a, -50, 50) for a in te]
    loops = {}
    for lp, members in loops_cfg.items():
        idx = [tags.index(t) for t in members if t in tags]
        if idx:
            loops[lp] = idx
    # events: contiguous runs of `attack`, matched in order to the documented table
    events = []
    k = 0
    gt_rows = gt.to_dict("records")
    for si, L in enumerate(lab):
        a = L[:, 0].astype(int)
        idx = np.flatnonzero(np.diff(np.r_[0, a, 0]))
        for s, e in zip(idx[::2], idx[1::2]):
            row = gt_rows[k]; k += 1
            assert row["file"] == f"test{si + 1}", (row, si)
            hh, mm = map(int, times[si][s][11:16].split(":")); dh, dm = map(int, row["doc_start"].split(":"))
            assert abs(((hh * 60 + mm) - (dh * 60 + dm) + 720) % 1440 - 720) <= 2, (times[si][s], row["doc_start"])
            ev_loops = row["target_loops"].split(";")
            events.append(Event(row["attack_id"], si, int(s), int(e), ev_loops,
                                row["target_points"].split(";"),
                                dict(scenarios=row["scenarios"], n_targets=len(ev_loops))))
    assert k == len(gt_rows) == 50
    act = np.array([_hai_actuator(t) for t in tags])
    return Dataset("HAI-21.03", tags, loops, tr, va, te, [L[:, 0].astype(int) for L in lab], events, act, mu, sd)


# --------------------------------------------------------------------------- PCL-Sim
def load_sim(train_seeds=range(10), val_seeds=(10, 11), test_seeds=(100, 101, 102, 103),
             hours=6, decoy=False, cache=True):
    from . import sim
    tag = "decoy" if decoy else "main"
    cache_f = ROOT / "data" / "cache" / f"pclsim_{tag}.npz"
    suite = sim.make_attack_suite(seeds=test_seeds, n_steps=3600, decoy=decoy, rng_seed=1 if decoy else 0)
    if cache and cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        raw_tr, raw_va, raw_te, haz = list(z["tr"]), list(z["va"]), list(z["te"]), list(z["haz"])
    else:
        raw_tr = [sim.simulate(hours * 3600, seed=int(s))[0] for s in train_seeds]
        raw_va = [sim.simulate(hours * 3600, seed=int(s))[0] for s in val_seeds]
        raw_te, haz = [], []
        for i, ep in enumerate(suite):
            X, info = sim.simulate(ep["n_steps"], seed=ep["seed"] * 1000 + i, attack=ep["attack"])
            raw_te.append(X); haz.append(info["hazard_t"])
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_f, tr=np.array(raw_tr), va=np.array(raw_va), te=np.array(raw_te), haz=np.array(haz))
    tr, (va, te), mu, sd = _standardise(list(raw_tr), [list(raw_va), list(raw_te)])
    tags = list(sim.TAGS)
    loops = {lp: [tags.index(t) for t in members] for lp, members in sim.LOOPS.items()}
    events, labels = [], []
    for i, (ep, h) in enumerate(zip(suite, haz)):
        a = ep["attack"]
        lab = np.zeros(ep["n_steps"], int); lab[a.t_on:a.t_on + a.dur] = 1
        labels.append(lab)
        events.append(Event(f"S{i:03d}", i, a.t_on, a.t_on + a.dur, [a.loop], [f"{a.loop}_PV"],
                            dict(primitive=a.primitive, level=ep["level"], seed=ep["seed"],
                                 hazard_D=float(h - a.t_on) if np.isfinite(h) else np.inf,
                                 decoy_loop=a.decoy_loop, n_targets=1)))
    act = np.array([t in sim.ACTUATOR_TAGS for t in tags])
    return Dataset("PCL-Sim" + ("-decoy" if decoy else ""), tags, loops, tr, va, te, labels, events, act, mu, sd)
