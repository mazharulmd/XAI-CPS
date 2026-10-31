from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)
    result: dict[str, object] = {"rows": len(rows), "columns": columns}
    for name in ("attack_active", "attack_phase", "attack_phase_all"):
        if name in columns:
            values: dict[str, int] = {}
            for row in rows:
                value = row.get(name, "")
                values[value] = values.get(value, 0) + 1
            result[f"{name}_counts"] = values
    if "timestamp" in columns and rows:
        result["first_timestamp"] = rows[0]["timestamp"]
        result["last_timestamp"] = rows[-1]["timestamp"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit selected Data4Cyber archive metadata and pilot tables")
    parser.add_argument("root", type=Path, help="Directory containing README.md and scenario subdirectories")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {"root": root.name, "selected_files": {}, "scenarios": {}}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.json":
            continue
        relative = path.relative_to(root).as_posix()
        expected = manifest.get(relative)
        if expected is None:
            raise ValueError(f"Selected file absent from archive SHA-256 manifest: {relative}")
        actual = sha256(path)
        if actual.lower() != str(expected).lower():
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")
        result["selected_files"][relative] = {"sha256": actual, "verified": True}

    for scenario_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        scenario: dict[str, object] = {}
        phases_path = scenario_dir / "attack-phases.json"
        if phases_path.exists():
            phases = json.loads(phases_path.read_text(encoding="utf-8"))
            scenario["phases"] = phases
        for filename in ("attack_events.csv", "dataset_attack.csv", "dataset_benign.csv"):
            path = scenario_dir / filename
            if path.exists():
                scenario[filename] = csv_summary(path)
        mapping = scenario_dir / "modbus_variable_mapping.csv"
        if mapping.exists():
            with mapping.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            scenario["modbus_mapping_rows"] = len(rows)
            scenario["modbus_mapping_columns"] = reader.fieldnames
        result["scenarios"][scenario_dir.name] = scenario

    aggregate = {"rows": 0, "attack_active_true": 0, "attack_active_false": 0, "empty_numeric_cells": 0}
    for scenario_name, scenario in result["scenarios"].items():
        for filename in ("dataset_attack.csv", "dataset_benign.csv"):
            table = scenario.get(filename)
            if not table:
                continue
            aggregate["rows"] += int(table["rows"])
            counts = table.get("attack_active_counts", {})
            aggregate["attack_active_true"] += int(counts.get("True", 0))
            aggregate["attack_active_false"] += int(counts.get("False", 0))
            if table["rows"]:
                # Count blank CSV cells across feature columns; labels/timestamps are not feature missingness.
                import pandas as pd
                frame = pd.read_csv(root / scenario_name / filename)
                feature_columns = [c for c in frame.columns if c not in {"timestamp", "attack_active", "attack_phase", "attack_phase_all"}]
                aggregate["empty_numeric_cells"] += int(frame[feature_columns].isna().sum().sum())
    result["archive_aggregate"] = aggregate
    result["zenodo_record_metadata_as_seen_2026_10_31"] = {
        "listed_benign": 5880,
        "listed_attack": 8474,
        "discrepancy_note": "The v3 archive README and CHANGES.md report the corrected split (6,768 benign / 7,586 attack-active). The 5,880 / 8,474 landing-page counts equal the earlier S3-all-attack labeling; prefer SHA-verified archive contents and document this record-metadata inconsistency."
    }

    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
