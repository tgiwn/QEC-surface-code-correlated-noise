"""Deterministic, no-Monte-Carlo audit of distance-3 DEM order sensitivity.

Run from the QEC project root with the project's Python environment:
    & $python .\\audit_minimal_d3_order_reproducer.py

The script does NOT modify the original CSVs, and it does NOT draw physical
circuit samples. It has two checks:
  1) a tiny synthetic DEM with two parallel singleton mechanisms carrying
     competing logical labels;
  2) the project's actual d=3 correlated-memory DEM at p_corr=1e-4 and 2e-4,
     comparing original instruction order against canonical probability order
     on all weight-1 and weight-2 detector syndromes (24 + C(24,2) = 300).

A synthetic result is only an API/graph-behaviour probe; the project DEM audit
is the evidence relevant to the research claim. Different package versions may
behave differently, so the exact versions are recorded in the outputs.
"""
from __future__ import annotations

import csv
import importlib.metadata
import re
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCE = 3
ROUNDS = 3
EXPECTED_STIM = "1.16.0"
EXPECTED_PYMATCHING = "2.4.0"
OUT_DIR = Path("results/minimal_d3_order_reproducer")


def version_of(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def error_positions(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort complete error instructions by descending probability, then text.

    Whole error instructions are moved intact; no mechanism is split on '^'.
    """
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_positions(lines)
    error_lines = [lines[i] for i in positions]

    def sort_key(line: str) -> tuple[float, str]:
        match = re.match(r"\s*error\(([^)]+)\)", line)
        if match is None:
            raise ValueError(f"Could not parse DEM instruction: {line.strip()}")
        return (-float(match.group(1)), line.strip())

    ordered = sorted(error_lines, key=sort_key)
    for pos, line in zip(positions, ordered):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (
        flat.num_detectors, flat.num_observables
    ):
        raise ValueError("Sorting changed the DEM dimensions")
    return result


def predict_one(matching: pymatching.Matching, syndrome: np.ndarray) -> list[int]:
    result = matching.decode(syndrome)
    return [int(x) for x in np.asarray(result, dtype=np.uint8).reshape(-1)]


def synthetic_parallel_label_probe() -> dict[str, object]:
    """Probe two graphlike error mechanisms with the same D0 syndrome."""
    mechanisms = [
        "error(0.01) D0",
        "error(0.001) D0 L0",
    ]
    syndrome = np.asarray([1], dtype=np.uint8)
    records: list[dict[str, object]] = []
    predictions: dict[str, list[int]] = {}

    for name, ordered_mechanisms in (
        ("no_logical_label_first", mechanisms),
        ("logical_label_first", list(reversed(mechanisms))),
    ):
        dem = stim.DetectorErrorModel("\n".join(ordered_mechanisms))
        matching = pymatching.Matching(dem)
        pred = predict_one(matching, syndrome)
        predictions[name] = pred
        records.append({
            "case": name,
            "dem": str(dem).replace("\n", " | "),
            "syndrome": "D0=1",
            "predicted_observables": "".join(map(str, pred)),
            "matching_edges": repr(matching.edges()),
        })

    return {
        "records": records,
        "predictions": predictions,
        "order_sensitive": predictions["no_logical_label_first"]
        != predictions["logical_label_first"],
    }


def make_low_weight_syndromes(num_detectors: int) -> tuple[np.ndarray, list[str]]:
    syndromes: list[np.ndarray] = []
    labels: list[str] = []
    for d in range(num_detectors):
        s = np.zeros(num_detectors, dtype=np.uint8)
        s[d] = 1
        syndromes.append(s)
        labels.append(f"D{d}")
    for d1, d2 in combinations(range(num_detectors), 2):
        s = np.zeros(num_detectors, dtype=np.uint8)
        s[d1] = 1
        s[d2] = 1
        syndromes.append(s)
        labels.append(f"D{d1}+D{d2}")
    return np.asarray(syndromes, dtype=np.uint8), labels


def build_actual_dem(p_corr: float) -> stim.DetectorErrorModel:
    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
    return circuit.detector_error_model(decompose_errors=True).flattened()


def run_project_dem_audit(stamp: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []

    for p_corr in P_CORR_VALUES:
        original_dem = build_actual_dem(p_corr)
        sorted_dem = canonical_probability_sorted_dem(original_dem)
        if original_dem.num_detectors != 24 or original_dem.num_observables != 1:
            raise ValueError(
                f"Expected d=3 DEM with 24 detectors/1 observable; found "
                f"{original_dem.num_detectors}/{original_dem.num_observables}"
            )

        syndromes, syndrome_labels = make_low_weight_syndromes(original_dem.num_detectors)
        assert len(syndromes) == 300

        for decoder_mode in ("ordinary", "correlation_aware"):
            use_correlations = decoder_mode == "correlation_aware"
            original_matching = pymatching.Matching(
                original_dem, enable_correlations=use_correlations
            )
            sorted_matching = pymatching.Matching(
                sorted_dem, enable_correlations=use_correlations
            )

            original_predictions = as_2d(original_matching.decode_batch(
                syndromes, enable_correlations=use_correlations
            ))
            sorted_predictions = as_2d(sorted_matching.decode_batch(
                syndromes, enable_correlations=use_correlations
            ))
            changed_mask = np.any(original_predictions != sorted_predictions, axis=1)
            changed_indices = np.flatnonzero(changed_mask)

            summary_rows.append({
                "timestamp": stamp,
                "distance": DISTANCE,
                "rounds": ROUNDS,
                "p_base": P_BASE,
                "p_corr": p_corr,
                "physical_model": "correlated",
                "decoder_mode": decoder_mode,
                "num_detectors": original_dem.num_detectors,
                "num_observables": original_dem.num_observables,
                "num_dem_error_instructions": sum(
                    line.lstrip().startswith("error(") for line in str(original_dem).splitlines()
                ),
                "num_syndromes_tested": len(syndromes),
                "changed_predictions": len(changed_indices),
                "changed_fraction": len(changed_indices) / len(syndromes),
                "original_edges": repr(original_matching.edges()),
                "sorted_edges": repr(sorted_matching.edges()),
            })

            for index in changed_indices:
                difference_rows.append({
                    "timestamp": stamp,
                    "distance": DISTANCE,
                    "p_corr": p_corr,
                    "decoder_mode": decoder_mode,
                    "syndrome": syndrome_labels[index],
                    "syndrome_weight": int(np.sum(syndromes[index])),
                    "original_prediction": "".join(map(str, original_predictions[index].tolist())),
                    "sorted_prediction": "".join(map(str, sorted_predictions[index].tolist())),
                })

            print(
                f"d={DISTANCE} p_corr={p_corr:g} {decoder_mode:17s}: "
                f"changed {len(changed_indices)}/{len(syndromes)} low-weight syndromes"
            )

    return summary_rows, difference_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("no_rows\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stim_version = version_of("stim")
    pymatching_version = version_of("PyMatching")
    numpy_version = version_of("numpy")
    print("=== Minimal distance-3 DEM order reproducer ===")
    print(f"Stim={stim_version}; PyMatching={pymatching_version}; NumPy={numpy_version}")
    if stim_version != EXPECTED_STIM or pymatching_version != EXPECTED_PYMATCHING:
        print(
            "WARNING: environment differs from the recorded October 14 run "
            f"(Stim {EXPECTED_STIM}, PyMatching {EXPECTED_PYMATCHING}); "
            "record the versions when interpreting differences."
        )

    synthetic = synthetic_parallel_label_probe()
    print("\nSynthetic competing-label DEM probe:")
    for row in synthetic["records"]:
        print(
            f"  {row['case']}: predicted L={row['predicted_observables']}; "
            f"edges={row['matching_edges']}"
        )
    print(f"  order_sensitive={synthetic['order_sensitive']}")

    summary_rows, difference_rows = run_project_dem_audit(stamp)
    metadata_path = OUT_DIR / f"metadata_{stamp}.txt"
    metadata_path.write_text(
        "Minimal distance-3 DEM order reproducer\n"
        f"Timestamp: {datetime.now().astimezone().isoformat()}\n"
        f"Python: {__import__('sys').version.replace(chr(10), ' ')}\n"
        f"Stim: {stim_version}\n"
        f"PyMatching: {pymatching_version}\n"
        f"NumPy: {numpy_version}\n"
        f"d={DISTANCE}; rounds={ROUNDS}; p_base={P_BASE}; p_corr={P_CORR_VALUES}\n"
        "No physical-circuit Monte Carlo shots were sampled.\n"
        "Project DEM comparison uses all 24 weight-1 and all 276 weight-2 syndromes.\n"
        f"Synthetic test shows order sensitivity: {synthetic['order_sensitive']}\n",
        encoding="utf-8",
    )
    synthetic_path = OUT_DIR / f"synthetic_probe_{stamp}.csv"
    write_csv(synthetic_path, synthetic["records"])
    summary_path = OUT_DIR / f"summary_{stamp}.csv"
    differences_path = OUT_DIR / f"syndrome_differences_{stamp}.csv"
    write_csv(summary_path, summary_rows)
    write_csv(differences_path, difference_rows)

    print("\nWrote files:")
    for p in (metadata_path, synthetic_path, summary_path, differences_path):
        print(f"  {p}")
    print("\nNo Monte Carlo simulation was run and no existing results were overwritten.")


if __name__ == "__main__":
    main()
