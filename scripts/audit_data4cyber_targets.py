from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re


POINT_RE = re.compile(r"^\d+\.(?P<unit>\d+)\.[^.]+\.(?P<address>\d+)$")
TARGET_RE = re.compile(r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3}):\d+\s+unit\s+(?P<unit>\d+)", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def epoch_utc(value: str) -> datetime:
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def norm_unit(value: str) -> str:
    return str(int(float(value)))


def audit_scenario(scenario_dir: Path) -> dict[str, object]:
    phases = json.loads((scenario_dir / "attack-phases.json").read_text(encoding="utf-8"))
    events_path = scenario_dir / "attack_events.csv"
    mapping_path = scenario_dir / "endpoint_variable_mapping.csv"
    events = read_csv(events_path) if events_path.exists() else []
    mapping = read_csv(mapping_path) if mapping_path.exists() else []
    results = []
    for phase in phases:
        start = datetime.fromisoformat(phase["start"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(phase["end"].replace("Z", "+00:00"))
        target = str(phase.get("attack_point") or phase.get("target") or "unlabeled")
        target_match = TARGET_RE.search(target)
        selected = [event for event in events if start <= epoch_utc(event["timestamp"]) < end]
        points = sorted({event.get("data_point", "") for event in selected})
        candidates: set[str] = set()
        mapped_points: set[str] = set()
        for point in points:
            match = POINT_RE.match(point)
            if not match:
                continue
            unit, address = match.group("unit"), match.group("address")
            rows = [
                row for row in mapping
                if row.get("start_address", "") == address
                and row.get("unit_id", "")
                and norm_unit(row["unit_id"]) == unit
                and row.get("dataset_variable", "").strip()
            ]
            if target_match:
                rows = [
                    row for row in rows
                    if row.get("device_ip", "") == target_match.group("ip")
                    and norm_unit(row["unit_id"]) == target_match.group("unit")
                ]
            if rows:
                mapped_points.add(point)
                candidates.update(row["dataset_variable"].strip() for row in rows)
        exact_target = bool(target_match and points and len(mapped_points) == len(points))
        results.append({
            "phase": phase.get("phase"),
            "phase_type": phase.get("type"),
            "target_annotation": target,
            "interval_start_utc": phase["start"],
            "interval_end_utc": phase["end"],
            "event_rows_in_interval": len(selected),
            "unique_event_points": len(points),
            "points_with_process_mapping": len(mapped_points),
            "candidate_process_variables": sorted(candidates),
            "explicit_ip_unit_target": bool(target_match),
            "exact_register_to_tag_join_from_phase_target": exact_target,
        })
    return {"scenario": scenario_dir.name, "phases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Data4Cyber phase/event/register target coverage")
    parser.add_argument("root", type=Path, help="Directory containing scenario folders")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    records = [audit_scenario(path) for path in sorted(args.root.iterdir())
               if path.is_dir() and (path / "attack-phases.json").exists()]
    result = {"evidence_scope": "phase intervals and event/register maps only; not model results",
              "scenarios": records}
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
