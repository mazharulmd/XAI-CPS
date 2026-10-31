from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LABEL_COLUMNS = ("attack", "attack_p1", "attack_p2", "attack_p3")

@dataclass
class Segment:
    name: str
    x: np.ndarray
    dx: np.ndarray
    attack: np.ndarray
    process_labels: np.ndarray
    timestamps: np.ndarray
    label_known: np.ndarray | None = None

@dataclass
class HAIDataset:
    feature_names: list[str]
    process_groups: list[str]
    train_x: np.ndarray
    train_dx: np.ndarray
    validation: Segment
    tests: list[Segment]
    manifest: dict


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing HAI file {path}. Run scripts/download_hai_21_03.py first.")
    frame = pd.read_csv(path, compression="infer", low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _convert(frame: pd.DataFrame, name: str, feature_names: list[str] | None = None) -> tuple[Segment, list[str]]:
    names = {str(c).lower(): c for c in frame.columns}
    time_col = names.get("time") or names.get("timestamp")
    if not time_col:
        raise ValueError(f"No timestamp column in {name}")
    label_map = {key: names[key] for key in LABEL_COLUMNS if key in names}
    if "attack" not in label_map:
        raise ValueError(f"No attack label column in {name}")
    if feature_names is None:
        excluded = {time_col, *label_map.values()}
        feature_names = [c for c in frame.columns if c not in excluded]
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature schema differs in {name}; missing {missing[:8]}")
    x = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        bad = int((~np.isfinite(x)).sum())
        raise ValueError(f"{name} has {bad} non-finite process values; no implicit imputation is allowed")
    attack = pd.to_numeric(frame[label_map["attack"]], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    process = np.zeros((len(frame), 3), dtype=np.int8)
    for j, key in enumerate(("attack_p1", "attack_p2", "attack_p3")):
        if key in label_map:
            process[:, j] = pd.to_numeric(frame[label_map[key]], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    ts = pd.to_datetime(frame[time_col], errors="coerce").to_numpy()
    if pd.isna(ts).any():
        raise ValueError(f"{name} contains invalid timestamps")
    dx = np.diff(x, axis=0, prepend=x[[0], :])
    return Segment(name, x, dx, attack, process, ts, np.ones(len(frame), dtype=bool)), feature_names


def load_hai(config_path: str | Path, max_train_rows: int = 150_000, seed: int = 42) -> HAIDataset:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = config_path.parent.parent / data_dir
    normal_files = list(config["normal_files"])
    validation_file = config["validation_file"]
    test_files = list(config["attack_files"])
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    feature_names = None
    train_parts: list[tuple[np.ndarray, np.ndarray]] = []
    for filename in normal_files:
        frame = _read(data_dir / filename)
        seg, feature_names = _convert(frame, filename, feature_names)
        if seg.attack.any():
            raise ValueError(f"Training file {filename} contains attack labels; refusing train/test contamination")
        train_parts.append((seg.x, seg.dx))
    train_x = np.concatenate([p[0] for p in train_parts], axis=0)
    train_dx = np.concatenate([p[1] for p in train_parts], axis=0)
    if len(train_x) > max_train_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(train_x), size=max_train_rows, replace=False))
        train_x, train_dx = train_x[idx], train_dx[idx]

    val_frame = _read(data_dir / validation_file)
    validation, _ = _convert(val_frame, validation_file, feature_names)
    if validation.attack.any():
        raise ValueError(f"Validation file {validation_file} contains attack labels")
    tests = []
    for filename in test_files:
        frame = _read(data_dir / filename)
        seg, _ = _convert(frame, filename, feature_names)
        tests.append(seg)
    groups = [name.split("_", 1)[0] if "_" in name else "OTHER" for name in feature_names]
    return HAIDataset(feature_names, groups, train_x, train_dx, validation, tests, manifest)


def _read_batadal(path: Path, feature_names: list[str] | None = None):
    frame=pd.read_csv(path)
    frame.columns=[str(c).strip() for c in frame.columns]
    if "DATETIME" not in frame or "ATT_FLAG" not in frame:
        raise ValueError(f"BATADAL schema missing DATETIME or ATT_FLAG in {path.name}")
    ts=pd.to_datetime(frame["DATETIME"].astype(str).str.strip(),format="%d/%m/%y %H",errors="raise")
    if not ts.is_monotonic_increasing:
        raise ValueError(f"BATADAL timestamps are not chronological in {path.name}")
    labels=pd.to_numeric(frame["ATT_FLAG"],errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError(f"BATADAL ATT_FLAG has non-numeric values in {path.name}")
    if feature_names is None:
        feature_names=[c for c in frame.columns if c not in {"DATETIME","ATT_FLAG"}]
    missing=sorted(set(feature_names)-set(frame.columns))
    if missing:
        raise ValueError(f"BATADAL schema differs in {path.name}; missing {missing[:8]}")
    x=frame[feature_names].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"BATADAL contains non-finite process values in {path.name}")
    dx=np.diff(x,axis=0,prepend=x[[0],:])
    known=labels!=-999
    attack=labels==1
    process=np.zeros((len(frame),3),dtype=np.int8)
    segment=Segment(path.name,x,dx,attack.astype(np.int8),process,ts.to_numpy(),known)
    return segment,feature_names,labels


def load_batadal(config_path: str | Path) -> HAIDataset:
    """Load official BATADAL normal training data and partially labeled development attacks.

    Unknown ATT_FLAG=-999 rows are preserved as unknown and must not be counted as negatives.
    The attack subset comes from BATADAL training dataset 2; the official test file is unlabeled.
    """
    config_path=Path(config_path)
    config=json.loads(config_path.read_text(encoding="utf-8-sig"))
    data_dir=Path(config["data_dir"])
    if not data_dir.is_absolute(): data_dir=config_path.parent.parent/data_dir
    manifest_path=data_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    normal,feature_names,normal_labels=_read_batadal(data_dir/config["normal_file"])
    if np.any(normal_labels!=0):
        raise ValueError("BATADAL normal training file contains nonzero/unknown ATT_FLAG values")
    fraction=float(config.get("train_fraction",0.75))
    if not 0.5<=fraction<=0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    split=int(len(normal.x)*fraction)
    if split<2 or split>=len(normal.x):
        raise ValueError("BATADAL chronological train/validation split is empty")
    train_x,train_dx=normal.x[:split],normal.dx[:split]
    validation=Segment(normal.name+"#validation",normal.x[split:],normal.dx[split:],
                       normal.attack[split:],normal.process_labels[split:],normal.timestamps[split:],
                       normal.label_known[split:])
    partial,_,partial_labels=_read_batadal(data_dir/config["partial_label_file"],feature_names)
    if not np.any(partial_labels==1):
        raise ValueError("BATADAL partial-label file contains no documented positive ATT_FLAG rows")
    groups=[name.split("_",1)[0] if "_" in name else "OTHER" for name in feature_names]
    manifest.setdefault("dataset","BATADAL")
    manifest.setdefault("label_note","ATT_FLAG=-999 is unknown; only ATT_FLAG=1 is treated as a labeled attack")
    return HAIDataset(feature_names,groups,train_x,train_dx,validation,[partial],manifest)


def contiguous_events(labels: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(labels, dtype=bool)
    if y.size == 0:
        return []
    starts = np.flatnonzero(y & ~np.r_[False, y[:-1]])
    ends = np.flatnonzero(y & ~np.r_[y[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))
