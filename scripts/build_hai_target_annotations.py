from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "annotations" / "hai21_attack_scenarios.csv"
DEFAULT_OUTPUT = ROOT / "data" / "annotations" / "hai21_event_target_map.csv"
DEFAULT_AUDIT = ROOT / "data" / "annotations" / "hai21_mapping_audit.json"
SOURCE_COMMIT = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
SOURCE_MANUAL_URL = (
    "https://github.com/icsdataset/hai/blob/"
    f"{SOURCE_COMMIT}/hai_dataset_technical_details.pdf"
)
# Candidate name correspondences are retained as hypotheses, never treated as
# gold labels. The current CSV headers differ from the manual's P2 tag names.
UNVERIFIED_SCHEMA_ALIASES = {
    "P2_AutoSD": "P2_ASD",
    "P2_SCO": "P2_CO_rpm",
    # The manual says P2_LCV01D for controller P3-LC-CO2; P3_LCV01D exists in
    # this CSV release, but this apparent typo needs maintainer confirmation.
    "P2_LCV01D": "P3_LCV01D",
}


def _event_intervals(frame: pd.DataFrame) -> list[dict]:
    attack = frame["attack"].to_numpy(dtype=bool)
    starts = np.flatnonzero(attack & ~np.r_[False, attack[:-1]])
    ends = np.flatnonzero(attack & ~np.r_[attack[1:], False]) + 1
    timestamps = pd.to_datetime(frame["time"], errors="raise")
    rows = []
    for event_index, (start, end) in enumerate(zip(starts, ends)):
        stages = [
            f"P{index}"
            for index in range(1, 4)
            if bool(frame[f"attack_P{index}"].iloc[start:end].any())
        ]
        rows.append(
            {
                "start_row": int(start),
                "event_index": event_index,
                "end_row_exclusive": int(end),
                "start": timestamps.iloc[start],
                "end_exclusive": timestamps.iloc[end - 1] + pd.Timedelta(seconds=1),
                "duration_seconds": int(end - start),
                "active_process_labels": stages,
            }
        )
    return rows


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_mapping(metadata_path: Path, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if metadata["attack_id"].duplicated().any():
        raise ValueError("Scenario metadata contains duplicate attack IDs")
    if len(metadata) != 50:
        raise ValueError(f"Expected 50 documented HAI 21.03 attacks, found {len(metadata)}")

    event_cache: dict[str, tuple[pd.DataFrame, list[dict]]] = {}
    mapped_rows = []
    errors = []
    for entry in metadata.to_dict("records"):
        filename = entry["dataset_file"]
        if filename not in event_cache:
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing HAI test file: {path}")
            frame = pd.read_csv(path, compression="infer", low_memory=False)
            required = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Unexpected HAI label schema in {filename}")
            event_cache[filename] = frame, _event_intervals(frame)
        frame, events = event_cache[filename]

        expected_minute = pd.Timestamp(entry["documented_start_minute"])
        # The manual records attack starts only to minute precision, whereas the
        # CSV labels are sampled once per second. Match within that minute bin's
        # rounding envelope and require a unique interval in the same file.
        matches = [
            event
            for event in events
            if abs((event["start"] - expected_minute).total_seconds()) < 60
        ]
        if len(matches) != 1:
            errors.append(
                f"{entry['attack_id']}: expected one interval in {filename} at "
                f"{expected_minute}, found {len(matches)}"
            )
            continue
        event = matches[0]
        points = _split(entry["target_points"])
        controllers = _split(entry["target_controllers"])
        controller_stages = sorted(
            {
                match.group(1)
                for controller in controllers
                if (match := re.match(r"^(P[1-3])(?:-|$)", controller))
            }
        )
        present_points = [point for point in points if point in frame.columns]
        missing_points = [point for point in points if point not in frame.columns]
        alias_candidates = [
            f"{point}->{UNVERIFIED_SCHEMA_ALIASES[point]}"
            for point in missing_points
            if UNVERIFIED_SCHEMA_ALIASES.get(point) in frame.columns
        ]
        observed_stages = event["active_process_labels"]
        stage_match = sorted(observed_stages) == controller_stages
        duration = int(entry["duration_seconds"])
        row = {
            "attack_id": entry["attack_id"],
            "dataset_file": filename,
            "start_row": event["start_row"],
            "event_index": event["event_index"],
            "end_row_exclusive": event["end_row_exclusive"],
            "documented_start_minute": expected_minute.strftime("%Y-%m-%d %H:%M"),
            "observed_start": event["start"].strftime("%Y-%m-%d %H:%M:%S"),
            "observed_end_exclusive": event["end_exclusive"].strftime("%Y-%m-%d %H:%M:%S"),
            "documented_duration_seconds": duration,
            "observed_labeled_duration_seconds": event["duration_seconds"],
            "duration_difference_seconds": event["duration_seconds"] - duration,
            "scenario_ids": entry["scenario_ids"],
            "target_controllers": entry["target_controllers"],
            "documented_target_points": entry["target_points"],
            "target_points_present_in_csv": "|".join(present_points),
            "target_points_missing_from_csv": "|".join(missing_points),
            "unverified_schema_alias_candidates": "|".join(alias_candidates),
            "target_annotation_complete": not missing_points,
            "controller_stage_labels": "|".join(controller_stages),
            "csv_active_process_labels": "|".join(observed_stages),
            "stage_labels_match_controller_metadata": stage_match,
            "manual_printed_pages": entry["manual_printed_pages"],
        }
        mapped_rows.append(row)
        if not stage_match:
            errors.append(
                f"{entry['attack_id']}: CSV stages {observed_stages} do not match "
                f"documented controller stages {controller_stages}"
            )

    if errors:
        raise ValueError("Scenario-to-interval alignment failed:\n" + "\n".join(errors))
    if len(mapped_rows) != len(metadata):
        raise ValueError(f"Mapped {len(mapped_rows)} of {len(metadata)} attacks")

    out = pd.DataFrame(mapped_rows)
    if out["observed_start"].duplicated().any():
        raise ValueError("Documented timestamps are not unique within the mapping")
    by_file = {
        filename: int((out["dataset_file"] == filename).sum())
        for filename in sorted(out["dataset_file"].unique())
    }
    mismatch_duration = out.loc[
        out["duration_difference_seconds"] != 0,
        ["attack_id", "documented_duration_seconds", "observed_labeled_duration_seconds", "duration_difference_seconds"],
    ].to_dict("records")
    incomplete_targets = out.loc[
        ~out["target_annotation_complete"],
        ["attack_id", "documented_target_points", "target_points_missing_from_csv", "unverified_schema_alias_candidates"],
    ].to_dict("records")
    audit = {
        "dataset": "HAI 21.03",
        "source_manual_url": SOURCE_MANUAL_URL,
        "source_manual_pdf_sha256": hashlib.sha256(
            (ROOT / "tmp" / "pdfs" / "hai_dataset_technical_details.pdf").read_bytes()
        ).hexdigest()
        if (ROOT / "tmp" / "pdfs" / "hai_dataset_technical_details.pdf").exists()
        else None,
        "source_manual_pages_printed": "36-38",
        "metadata_rows": int(len(metadata)),
        "mapped_attack_intervals": int(len(out)),
        "attack_intervals_by_file": by_file,
        "all_timestamp_matches_unique": True,
        "controller_stage_labels_match_all_intervals": bool(out["stage_labels_match_controller_metadata"].all()),
        "duration_discrepancies": mismatch_duration,
        "incomplete_target_point_mappings": incomplete_targets,
        "interpretation": (
            "Attack IDs and target points are aligned by the official manual's minute-resolution "
            "start time to a unique CSV attack interval. Stage labels are cross-checked against "
            "the documented target controller. Manual-listed points absent from this CSV schema "
            "remain unresolved and are excluded from tag-level localization; plausible name "
            "aliases are emitted only as unverified candidates. Duration differences "
            "are preserved as annotation discrepancies; they do not alter timestamp matching."
        ),
    }
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Align official HAI 21.03 attack targets to CSV intervals")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "hai-21.03")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    mapping, audit = build_mapping(args.metadata, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output, index=False)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "mapped_intervals": audit["mapped_attack_intervals"],
        "by_file": audit["attack_intervals_by_file"],
        "duration_discrepancies": audit["duration_discrepancies"],
        "incomplete_targets": audit["incomplete_target_point_mappings"],
        "output": str(args.output),
        "audit": str(args.audit),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
