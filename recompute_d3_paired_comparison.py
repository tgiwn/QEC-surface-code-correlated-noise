"""Recompute paired uncertainty for the saved d=3 DEM-order comparison.

This script reproduces the original 500,000-shot conditions (seeds 20261080 and
20261081), comparing the original-order DEM to the repository's canonical
probability-sorted DEM. It prints paired failure discordances, paired-normal
95% confidence intervals, and exact two-sided McNemar p-values. It does not
write or overwrite any CSV files.

Run from the project root so correlated_experiment.py is importable:
    & $python .\recompute_d3_paired_comparison.py

The Bonferroni-adjusted p-values are across these two conditions only. They do
not correct for other decoder variants or exploratory analyses in the paper.
"""
from __future__ import annotations

import math
import platform
import re
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pymatching
import stim
from scipy.stats import binomtest

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

P_BASE = 0.001
DISTANCE = 3
ROUNDS = 3
SHOTS = 500_000
CONDITIONS = ((0.0001, 20261080), (0.0002, 20261081))
Z_975 = 1.959963984540054
ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Use the same sorting rule in analyze_correlation_aware_win_loss.py."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    err_lines = [lines[i] for i in positions]

    def key(line: str) -> tuple[float, str]:
        match = ERROR_RE.match(line)
        if match is None:
            raise ValueError(f"Unable to parse DEM error instruction: {line!r}")
        return (-float(match.group(1)), line.strip())

    for pos, line in zip(positions, sorted(err_lines, key=key)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise AssertionError("Sorting changed DEM dimensions")
    return result


def as_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def main() -> int:
    print("Paired d=3 original-order vs canonical-sorted DEM audit")
    print(f"Python: {platform.python_version()}")
    for package in ("stim", "pymatching", "numpy", "scipy"):
        print(f"{package}: {version(package)}")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        print(f"Git HEAD: {commit}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git HEAD: unavailable (run from a Git checkout to record provenance)")

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    print(f"distance={DISTANCE}; rounds={ROUNDS}; p_base={P_BASE}; shots/condition={SHOTS:,}; pairs/round={len(pairs)}")
    results = []

    for p_corr, seed in CONDITIONS:
        print("\n" + "=" * 78)
        print(f"p_corr={p_corr:g}; seed={seed}")
        circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
        raw_dem = circuit.detector_error_model(decompose_errors=True).flattened()
        sorted_dem = canonical_probability_sorted_dem(raw_dem)
        original = pymatching.Matching(raw_dem)
        canonical = pymatching.Matching(sorted_dem)
        sampler = circuit.compile_detector_sampler(seed=seed)
        syndrome, actual = sampler.sample(shots=SHOTS, separate_observables=True)
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        actual = as_2d(actual)
        pred_original = as_2d(original.decode_batch(syndrome))
        pred_sorted = as_2d(canonical.decode_batch(syndrome))
        if pred_original.shape != actual.shape or pred_sorted.shape != actual.shape:
            raise ValueError(
                f"Prediction shape mismatch: original={pred_original.shape}, "
                f"sorted={pred_sorted.shape}, actual={actual.shape}"
            )

        fail_original = np.any(pred_original != actual, axis=1)
        fail_sorted = np.any(pred_sorted != actual, axis=1)
        original_only = int(np.count_nonzero(fail_original & ~fail_sorted))
        sorted_only = int(np.count_nonzero(fail_sorted & ~fail_original))
        both_fail = int(np.count_nonzero(fail_original & fail_sorted))
        both_correct = int(np.count_nonzero(~fail_original & ~fail_sorted))
        pred_changed = int(np.count_nonzero(np.any(pred_original != pred_sorted, axis=1)))
        f_original = int(np.count_nonzero(fail_original))
        f_sorted = int(np.count_nonzero(fail_sorted))

        differences = fail_sorted.astype(np.int8) - fail_original.astype(np.int8)
        delta_pp = float(np.mean(differences) * 100)
        se_pp = float(np.std(differences, ddof=1) / math.sqrt(SHOTS) * 100)
        ci_low, ci_high = delta_pp - Z_975 * se_pp, delta_pp + Z_975 * se_pp
        discordant = original_only + sorted_only
        mcnemar_p = (
            float(binomtest(sorted_only, n=discordant, p=0.5, alternative="two-sided").pvalue)
            if discordant else 1.0
        )
        results.append({"p_corr": p_corr, "p": mcnemar_p})

        print(f"Detectors: {raw_dem.num_detectors}; observables: {raw_dem.num_observables}")
        print(f"Original-order failures: {f_original:,}/{SHOTS:,} ({100*f_original/SHOTS:.6f}%)")
        print(f"Canonical-sorted failures: {f_sorted:,}/{SHOTS:,} ({100*f_sorted/SHOTS:.6f}%)")
        print(f"Predictions changed: {pred_changed:,}/{SHOTS:,}")
        print(f"Original-only failures: {original_only:,}")
        print(f"Sorted-only failures: {sorted_only:,}")
        print(f"Both fail: {both_fail:,}; both correct: {both_correct:,}")
        print(f"Paired delta (sorted - original): {delta_pp:+.6f} percentage points")
        print(f"Paired-normal 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}] percentage points")
        print(f"Exact two-sided McNemar p-value: {mcnemar_p:.12g}")
        print(f"Bonferroni-adjusted p-value (2 conditions only): {min(1.0, 2*mcnemar_p):.12g}")

    print("\nInterpretation guardrails:")
    print("- Different decoder predictions are not the same as discordant logical-failure outcomes.")
    print("- The paired CI and McNemar test use the same circuit shots for both decoders.")
    print("- The Bonferroni adjustment here covers only these two p_corr contrasts.")
    print("- This does not test the separate probability-prioritized decoder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
