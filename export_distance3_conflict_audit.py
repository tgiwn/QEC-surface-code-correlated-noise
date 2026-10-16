"""Export machine-readable correlated-DEM logical-label conflict audits.

Run from the qec-correlated-noise project root:
    python export_distance3_conflict_audit.py

Writes:
    results/distance3_conflict_audit/summary.csv
    results/distance3_conflict_audit/signatures.csv
    results/distance3_conflict_audit/components.csv

The export tests two circuit base profiles because the bundled audit scripts
are not perfectly aligned:
  * project_standard: make_base_circuit(..., p=P_BASE), which sets all four
    Stim generated noise parameters to P_BASE;
  * audit_edges_legacy: the historical audit profile with
    after_clifford_depolarization=0 and the other three parameters P_BASE.

This distinction is intentional and recorded in every row. A '^' component
is a decomposition of one complete DEM instruction, not an independent event.
Do not sum component probabilities as if they were independent mechanisms.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCES = (3, 5, 7)
OUT_DIR = Path("results") / "distance3_conflict_audit"


def make_legacy_audit_base(distance: int):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0,
        before_round_data_depolarization=P_BASE,
        after_reset_flip_probability=P_BASE,
        before_measure_flip_probability=P_BASE,
    )


def parse_dem_components(dem, distance, p_corr, model, base_profile):
    rows = []
    non_graphlike_count = 0
    text_lines = str(dem.flattened()).splitlines()
    for line_index, line in enumerate(text_lines):
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue
        probability = float(match.group(1))
        body = match.group(2)
        pieces = re.split(r"\s+\^\s+", body)
        is_decomposed = len(pieces) > 1
        for component_index, piece in enumerate(pieces):
            detectors = tuple(sorted(int(x) for x in re.findall(r"\bD(\d+)\b", piece)))
            observables = tuple(sorted(int(x) for x in re.findall(r"\bL(\d+)\b", piece)))
            if not detectors:
                continue
            if len(detectors) > 2:
                non_graphlike_count += 1
                continue
            signature = ";".join(f"D{x}" for x in detectors)
            labels = ";".join(f"L{x}" for x in observables) if observables else "NONE"
            rows.append({
                "base_profile": base_profile,
                "distance": distance,
                "rounds": distance,
                "p_base": P_BASE,
                "p_corr": p_corr,
                "physical_model": model,
                "signature": signature,
                "num_detectors_in_signature": len(detectors),
                "logical_labels": labels,
                "probability": probability,
                "dem_line_index": line_index,
                "component_index_within_instruction": component_index,
                "is_decomposed_instruction": is_decomposed,
                "full_dem_instruction": line.strip(),
                "component_term": piece.strip(),
            })
    return rows, non_graphlike_count


def summarize_condition(component_rows, non_graphlike_count, base_profile, distance, p_corr, model):
    by_signature = defaultdict(list)
    for row in component_rows:
        by_signature[(row["num_detectors_in_signature"], row["signature"])].append(row)

    duplicate_count = 0
    conflict_count = 0
    singleton_conflicts = 0
    for (n_detectors, _signature), rows in by_signature.items():
        if len(rows) > 1:
            duplicate_count += 1
        labels = {row["logical_labels"] for row in rows}
        if len(labels) > 1:
            conflict_count += 1
            if n_detectors == 1:
                singleton_conflicts += 1

    return {
        "base_profile": base_profile,
        "distance": distance,
        "rounds": distance,
        "p_base": P_BASE,
        "p_corr": p_corr,
        "physical_model": model,
        "graphlike_component_rows": len(component_rows),
        "unique_graphlike_signatures": len(by_signature),
        "duplicate_signatures": duplicate_count,
        "conflicting_signatures_one_or_two_detectors": conflict_count,
        "conflicting_singleton_signatures": singleton_conflicts,
        "non_graphlike_components": non_graphlike_count,
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_component_rows = []
    all_signature_rows = []
    all_summary_rows = []

    for base_profile in ("project_standard", "audit_edges_legacy"):
        for distance in DISTANCES:
            if base_profile == "project_standard":
                base = make_base_circuit(distance=distance, rounds=distance, p=P_BASE)
            else:
                base = make_legacy_audit_base(distance)

            data_by_xy, data_qubits = get_data_layout(base)
            pairs = make_correlated_pairs(data_by_xy)

            for p_corr in P_CORR_VALUES:
                for model in ("matched_independent", "correlated"):
                    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, model)
                    dem = circuit.detector_error_model(decompose_errors=True).flattened()
                    rows, non_graphlike_count = parse_dem_components(
                        dem, distance, p_corr, model, base_profile
                    )
                    all_component_rows.extend(rows)
                    summary = summarize_condition(
                        rows, non_graphlike_count, base_profile, distance, p_corr, model
                    )
                    summary["data_qubits"] = len(data_qubits)
                    summary["vertical_pairs_per_round"] = len(pairs)
                    summary["dem_num_detectors"] = dem.num_detectors
                    summary["dem_num_observables"] = dem.num_observables
                    all_summary_rows.append(summary)

                    by_signature = defaultdict(list)
                    for row in rows:
                        by_signature[(row["num_detectors_in_signature"], row["signature"])].append(row)
                    for (n_detectors, signature), sig_rows in sorted(by_signature.items()):
                        labels = sorted({r["logical_labels"] for r in sig_rows})
                        if len(labels) <= 1:
                            continue
                        all_signature_rows.append({
                            "base_profile": base_profile,
                            "distance": distance,
                            "rounds": distance,
                            "p_base": P_BASE,
                            "p_corr": p_corr,
                            "physical_model": model,
                            "signature": signature,
                            "num_detectors_in_signature": n_detectors,
                            "logical_label_variants": " | ".join(labels),
                            "component_row_count": len(sig_rows),
                            "distinct_source_instruction_count": len({r['dem_line_index'] for r in sig_rows}),
                            "contains_decomposed_source": any(r["is_decomposed_instruction"] for r in sig_rows),
                        })
                    print(
                        f"{base_profile:20s} d={distance} p_corr={p_corr:g} "
                        f"{model:18s} singleton_conflicts={summary['conflicting_singleton_signatures']} "
                        f"all_conflicts={summary['conflicting_signatures_one_or_two_detectors']}"
                    )

    write_csv(OUT_DIR / "summary.csv", all_summary_rows)
    write_csv(OUT_DIR / "signatures.csv", all_signature_rows)
    write_csv(OUT_DIR / "components.csv", all_component_rows)
    print("\nWrote:")
    print(f"  {OUT_DIR / 'summary.csv'}")
    print(f"  {OUT_DIR / 'signatures.csv'}")
    print(f"  {OUT_DIR / 'components.csv'}")
    print("\nImportant: '^' components share a complete DEM mechanism; component probabilities are not independent events.")


if __name__ == "__main__":
    main()
