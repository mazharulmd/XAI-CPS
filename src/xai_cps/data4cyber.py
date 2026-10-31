from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_COLUMNS = ("attack_active", "attack_phase", "attack_phase_all")
TIME_COLUMNS = ("timestamp",)


@dataclass
class Data4CyberSplit:
    """One scenario's telemetry and post-hoc annotations, kept in separate fields."""

    scenario: str
    features: pd.DataFrame
    attack: np.ndarray
    phase: np.ndarray
    phase_all: np.ndarray
    timestamps: np.ndarray


@dataclass
class NormalMedianImputer:
    """Median imputer whose statistics come only from outer-fold normal training data."""

    feature_names: list[str]
    medians: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Feature schema missing training columns: {missing[:8]}")
        values = frame[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).copy()
        if np.isinf(values).any():
            raise ValueError("Telemetry contains infinite values")
        rows, columns = np.where(np.isnan(values))
        values[rows, columns] = self.medians[columns]
        return values


def fit_normal_median_imputer(train_normal: pd.DataFrame) -> NormalMedianImputer:
    """Fit an imputer on training-normal rows only; all-missing channels are dropped."""
    numeric = train_normal.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Training telemetry contains infinite values")
    medians = np.array([
        np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
        for column in values.T
    ])
    keep = np.isfinite(medians)
    if not keep.any():
        raise ValueError("No telemetry feature has an observed training-normal value")
    return NormalMedianImputer(
        feature_names=[str(name) for name, retained in zip(numeric.columns, keep) if retained],
        medians=medians[keep],
    )


def _read_split(path: Path, expected_attack: bool) -> Data4CyberSplit:
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"timestamp", *ANNOTATION_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path.name} has no rows")

    labels = frame["attack_active"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if labels.isna().any():
        raise ValueError(f"{path.name} contains invalid attack_active values")
    attack = labels.to_numpy(dtype=bool)
    if np.any(attack != expected_attack):
        kind = "attack" if expected_attack else "benign"
        raise ValueError(f"{path.name} {kind} file disagrees with attack_active annotation")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path.name} timestamps are not chronological")

    # Labels and every explicit clock-like field are unavailable to detector/explainer.
    # Endpoint telemetry values such as *.realtime are clocks, not process magnitudes.
    excluded = set(ANNOTATION_COLUMNS) | set(TIME_COLUMNS)
    excluded |= {
        column for column in frame.columns
        if any(token in column.lower() for token in ("timestamp", "datetime", ".realtime"))
    }
    candidates = [column for column in frame.columns if column not in excluded]
    numeric = frame[candidates].apply(pd.to_numeric, errors="coerce")
    features = numeric.loc[:, numeric.notna().any(axis=0)].copy()
    if features.shape[1] == 0:
        raise ValueError(f"{path.name} has no numeric telemetry features after leakage exclusions")
    if not np.isfinite(features.to_numpy(dtype=float, na_value=np.nan)[~pd.isna(features).to_numpy()]).all():
        raise ValueError(f"{path.name} contains infinite numeric telemetry values")

    phases = frame["attack_phase"].fillna("").astype(str).to_numpy()
    phases_all = frame["attack_phase_all"].fillna("").astype(str).to_numpy()
    return Data4CyberSplit(
        scenario=path.parent.name,
        features=features,
        attack=attack,
        phase=phases,
        phase_all=phases_all,
        timestamps=timestamps.to_numpy(),
    )


def load_data4cyber_scenario(scenario_dir: str | Path) -> tuple[Data4CyberSplit, Data4CyberSplit]:
    """Load attack and benign telemetry without allowing annotations into features.

    Missing numeric values are preserved as NaN; imputation and feature selection must
    be fit on training-normal rows within each outer scenario split.
    """
    scenario_dir = Path(scenario_dir)
    attack_path = scenario_dir / "dataset_attack.csv"
    benign_path = scenario_dir / "dataset_benign.csv"
    if not attack_path.is_file() or not benign_path.is_file():
        raise FileNotFoundError(f"Expected attack and benign CSVs under {scenario_dir}")
    attack = _read_split(attack_path, expected_attack=True)
    benign = _read_split(benign_path, expected_attack=False)
    if list(attack.features.columns) != list(benign.features.columns):
        raise ValueError("Attack and benign feature schemas differ; refusing implicit alignment")
    return attack, benign


def load_data4cyber_benign_file(path: str | Path) -> Data4CyberSplit:
    """Load one normal-only table, including S0 where the attack table is empty."""
    path = Path(path)
    if path.name != "dataset_benign.csv":
        raise ValueError("Expected a dataset_benign.csv file")
    return _read_split(path, expected_attack=False)
