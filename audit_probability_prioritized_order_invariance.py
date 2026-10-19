"""Test the reported probability-prioritized d=3 graph under DEM line permutations.

Uses the project's exact_dem_map_decoder.build_probability_prioritized_matching
helper, the original d=3 physical-circuit seeds, and shared samples for all graph
constructions. Prints results only; it does not create or overwrite CSV files.
Run from the project root so correlated_experiment.py and exact_dem_map_decoder.py
are importable:
    & $python .\audit_probability_prioritized_order_invariance.py
"""
from __future__ import annotations

import math
import re
import platform
from importlib.metadata import version

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)
from exact_dem_map_decoder import build_probability_prioritized_matching

P_BASE = 0.001
DISTANCE = 3
ROUNDS = 3
SHOTS = 500_000
CONDITIONS = ((0.0001, 20261080), (0.0002, 20261081))
ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")


def error_positions(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]


def reverse_error_order(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Reverse complete error instructions while keeping non-error lines fixed."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_positions(lines)
    original = [lines[i] for i in positions]
    for pos, line in zip(positions, reversed(original)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Reversing DEM error instructions changed dimensions")
    return result


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort complete error instructions by descending p, then full-line text."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_positions(lines)
    errors = [lines[i] for i in positions]

    def key(line: str) -> tuple[float, str]:
        match = ERROR_RE.match(line)
        if match is None:
            raise ValueError(f"Could not parse DEM error instruction: {line!r}")
        return (-float(match.group(1)), line.strip())

    ordered = sorted(errors, key=key)
    for pos, line in zip(positions, ordered):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Sorting DEM error instructions changed dimensions")
    return result


def as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def fail_mask(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    pred = as_2d(predictions)
    truth = as_2d(actual)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape mismatch: {pred.shape} vs {truth.shape}")
    return np.any(pred != truth, axis=1)


def paired_delta_ci(candidate_fail: np.ndarray, baseline_fail: np.ndarray) -> tuple[float, float, float]:
    """Return candidate-minus-baseline pp and paired-normal 95% CI in pp."""
    diff = candidate_fail.astype(np.int8) - baseline_fail.astype(np.int8)
    delta_pp = float(np.mean(diff) * 100)
    se_pp = float(np.std(diff, ddof=1) / math.sqrt(len(diff)) * 100)
    return delta_pp, delta_pp - 1.959963984540054 * se_pp, delta_pp + 1.959963984540054 * se_pp


def main() -> int:
    print("Probability-prioritized d=3 order-invariance diagnostic")
    print(f"Python: {platform.python_version()}")
    for package in ("stim", "pymatching", "numpy"):
        print(f"{package}: {version(package)}")
    print(f"distance={DISTANCE}; rounds={ROUNDS}; p_base={P_BASE}; shots/condition={SHOTS:,}")
    print("Uses the saved circuit seeds; no output CSV is written.")

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)

    for p_corr, seed in CONDITIONS:
        print("\n" + "=" * 78)
        print(f"p_corr={p_corr:g}; seed={seed}")
        circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
        dem = circuit.detector_error_model(decompose_errors=True).flattened()
        reversed_dem = reverse_error_order(dem)
        sorted_dem = canonical_probability_sorted_dem(dem)
        sorted_from_reversed = canonical_probability_sorted_dem(reversed_dem)
        canonical_text_equal = str(sorted_dem) == str(sorted_from_reversed)
        print(f"dem detectors={dem.num_detectors}; observables={dem.num_observables}; canonical text stable={canonical_text_equal}")
        if not canonical_text_equal:
            raise AssertionError("Canonical sorting did not normalize reversed input")

        priority_original = build_probability_prioritized_matching(dem)
        priority_reversed = build_probability_prioritized_matching(reversed_dem)
        priority_from_sorted = build_probability_prioritized_matching(sorted_dem)
        ordinary_sorted = pymatching.Matching(sorted_dem)

        sampler = circuit.compile_detector_sampler(seed=seed)
        syndrome, actual = sampler.sample(shots=SHOTS, separate_observables=True)
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        actual = as_2d(actual)

        pred_original = as_2d(priority_original.decode_batch(syndrome))
        pred_reversed = as_2d(priority_reversed.decode_batch(syndrome))
        pred_priority_sorted = as_2d(priority_from_sorted.decode_batch(syndrome))
        pred_ordinary_sorted = as_2d(ordinary_sorted.decode_batch(syndrome))

        fail_original = fail_mask(pred_original, actual)
        fail_reversed = fail_mask(pred_reversed, actual)
        fail_priority_sorted = fail_mask(pred_priority_sorted, actual)
        fail_ordinary_sorted = fail_mask(pred_ordinary_sorted, actual)

        changed_reversed = int(np.count_nonzero(np.any(pred_original != pred_reversed, axis=1)))
        changed_priority_sorted = int(np.count_nonzero(np.any(pred_original != pred_priority_sorted, axis=1)))
        changed_ordinary_sorted = int(np.count_nonzero(np.any(pred_original != pred_ordinary_sorted, axis=1)))
        d1, lo1, hi1 = paired_delta_ci(fail_reversed, fail_original)
        d2, lo2, hi2 = paired_delta_ci(fail_priority_sorted, fail_original)

        print(f"priority / original DEM failures: {int(fail_original.sum()):,}/{SHOTS:,} ({100*fail_original.mean():.6f}%)")
        print(f"priority / reversed DEM failures: {int(fail_reversed.sum()):,}/{SHOTS:,} ({100*fail_reversed.mean():.6f}%)")
        print(f"priority / sorted DEM failures: {int(fail_priority_sorted.sum()):,}/{SHOTS:,} ({100*fail_priority_sorted.mean():.6f}%)")
        print(f"ordinary / sorted DEM failures: {int(fail_ordinary_sorted.sum()):,}/{SHOTS:,} ({100*fail_ordinary_sorted.mean():.6f}%)")
        print(f"priority predictions changed (original vs reversed): {changed_reversed:,}/{SHOTS:,}")
        print(f"priority predictions changed (original vs sorted input): {changed_priority_sorted:,}/{SHOTS:,}")
        print(f"priority vs ordinary-sorted prediction disagreements: {changed_ordinary_sorted:,}/{SHOTS:,}")
        print(f"paired failure delta (priority-reversed minus priority-original): {d1:+.6f} pp; 95% CI [{lo1:+.6f}, {hi1:+.6f}] pp")
        print(f"paired failure delta (priority-built-from-sorted minus original-input): {d2:+.6f} pp; 95% CI [{lo2:+.6f}, {hi2:+.6f}] pp")

        if changed_reversed != 0 or changed_priority_sorted != 0:
            print("WARNING: priority graph predictions were not invariant across these DEM input permutations.")
        else:
            print("PASS: priority graph predictions were invariant across tested DEM input permutations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
