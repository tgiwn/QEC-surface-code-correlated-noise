"""Stratify where PyMatching correlation-aware decoding helps or hurts.

Uses the project's physical circuit builder and compares ordinary MWPM against
PyMatching two-pass correlated matching on identical circuit shots. The DEM uses
Stim's default, topology-aware decompositions; no arbitrary hyperedge rewrites or
probability scaling are applied.

Outputs:
  correlation_aware_win_loss_summary.csv
  correlation_aware_win_loss_by_weight.csv

Run in the project directory containing correlated_experiment.py:
    & $python .\\analyze_correlation_aware_win_loss.py
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Any

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
DISTANCES = (5, 7)
SHOTS_BY_DISTANCE = {5: 1_000_000, 7: 2_000_000}
CHUNK_SIZE = 25_000
SEED_BASE = 202_710_00
SUMMARY_CSV = Path("correlation_aware_win_loss_summary.csv")
WEIGHT_CSV = Path("correlation_aware_win_loss_by_weight.csv")

ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")


def as_2d(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def failure_mask(pred: Any, actual: Any) -> np.ndarray:
    p, y = as_2d(pred), as_2d(actual)
    if p.shape != y.shape:
        raise ValueError(f"Prediction/observable shape mismatch: {p.shape} vs {y.shape}")
    return np.any(p != y, axis=1)


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort complete flattened error instructions without changing their bodies."""
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


def ci_for_paired_binary(candidate_only: int, baseline_only: int, n: int, z: float = 1.959963984540054):
    """Paired normal CI for candidate-minus-baseline failure probability."""
    delta_pp = 100.0 * (candidate_only - baseline_only) / n
    discordant = candidate_only + baseline_only
    if n <= 1:
        return delta_pp, float("nan"), float("nan")
    # Variance of the mean of paired binary differences candidate - baseline.
    var = max(0.0, (discordant - (candidate_only - baseline_only) ** 2 / n) / (n - 1))
    se_pp = 100.0 * math.sqrt(var / n)
    return delta_pp, delta_pp - z * se_pp, delta_pp + z * se_pp


def weight_bin(weight: int) -> str:
    if weight == 0:
        return "w0"
    if weight == 1:
        return "w1"
    if weight == 2:
        return "w2"
    if weight <= 5:
        return "w3_5"
    if weight <= 10:
        return "w6_10"
    return "w11plus"


def main() -> None:
    summary_fields = [
        "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "detectors",
        "observable_flip_count", "ordinary_failures", "correlation_aware_failures",
        "ordinary_rate_pct", "correlation_aware_rate_pct", "delta_corr_minus_ordinary_pp",
        "delta_ci_low_pp", "delta_ci_high_pp", "prediction_disagreements",
        "correlation_aware_only_correct", "ordinary_only_correct", "both_fail", "both_correct",
        "dem_error_instructions", "dem_separator_count", "dem_graphlike_components",
    ]
    weight_fields = [
        "distance", "p_base", "p_corr", "physical_model", "weight_bin", "shots_in_bin",
        "ordinary_failures", "correlation_aware_failures", "ordinary_rate_pct",
        "correlation_aware_rate_pct", "delta_corr_minus_ordinary_pp", "delta_ci_low_pp",
        "delta_ci_high_pp", "correlation_aware_only_correct", "ordinary_only_correct",
        "both_fail", "both_correct", "prediction_disagreements",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as sf, WEIGHT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as wf:
        sw, ww = csv.DictWriter(sf, fieldnames=summary_fields), csv.DictWriter(wf, fieldnames=weight_fields)
        sw.writeheader(); ww.writeheader()
        print("Correlation-aware MWPM: paired win/loss analysis by syndrome weight")
        print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}")
        print(f"shots={SHOTS_BY_DISTANCE}; chunk={CHUNK_SIZE:,}; seed base={SEED_BASE}")
        print("Physical circuits are sampled once per condition; both decoders see identical shots.")
        print("Uses Stim default decomposition and canonical instruction ordering; no component rewrites.")

        for distance in DISTANCES:
            shots = SHOTS_BY_DISTANCE[distance]
            base = make_base_circuit(distance=distance, rounds=distance, p=P_BASE)
            data_by_xy, data_qubits = get_data_layout(base)
            pairs = make_correlated_pairs(data_by_xy)

            for pc_index, p_corr in enumerate(P_CORR_VALUES):
                for model_index, (model_name, mode) in enumerate([
                    ("matched-independent-control", "matched_independent"),
                    ("fully-correlated", "correlated"),
                ]):
                    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
                    raw_dem = circuit.detector_error_model(decompose_errors=True).flattened()
                    dem = canonical_probability_sorted_dem(raw_dem)
                    ordinary = pymatching.Matching(dem)
                    corr = pymatching.Matching(dem, enable_correlations=True)
                    stats = {"error_instructions": 0, "separator_count": 0, "graphlike_components": 0}
                    for line in str(dem).splitlines():
                        m = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
                        if not m or float(m.group(1)) <= 0:
                            continue
                        stats["error_instructions"] += 1
                        parts = re.split(r"\s+\^\s+", m.group(2))
                        stats["separator_count"] += max(0, len(parts) - 1)
                        stats["graphlike_components"] += sum(
                            1 for part in parts if re.search(r"\bD\d+\b", part)
                            and len(re.findall(r"\bD\d+\b", part)) <= 2
                        )

                    seed = SEED_BASE + distance * 1000 + pc_index * 100 + model_index
                    sampler = circuit.compile_detector_sampler(seed=seed)
                    names = ("w0", "w1", "w2", "w3_5", "w6_10", "w11plus")
                    bins = {b: {
                        "n": 0, "ord_fail": 0, "corr_fail": 0, "corr_only": 0,
                        "ord_only": 0, "both_fail": 0, "both_correct": 0,
                        "disagree": 0,
                    } for b in names}
                    totals = {k: 0 for k in [
                        "ord_fail", "corr_fail", "corr_only", "ord_only", "both_fail",
                        "both_correct", "disagree", "obs_flip",
                    ]}
                    start = time.time(); processed = 0
                    print("\n" + "=" * 96)
                    print(f"d={distance}; p_corr={p_corr:g}; physical={model_name}; seed={seed}; shots={shots:,}; pairs={len(pairs)}")
                    while processed < shots:
                        n_batch = min(CHUNK_SIZE, shots - processed)
                        syndrome, actual = sampler.sample(shots=n_batch, separate_observables=True)
                        syndrome = np.asarray(syndrome, dtype=np.uint8)
                        actual = as_2d(actual)
                        pred_o = as_2d(ordinary.decode_batch(syndrome))
                        pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                        fail_o = failure_mask(pred_o, actual)
                        fail_c = failure_mask(pred_c, actual)
                        disagree = np.any(pred_o != pred_c, axis=1)
                        weights = np.sum(syndrome, axis=1)
                        obs_flip = np.any(actual != 0, axis=1)

                        totals["ord_fail"] += int(np.count_nonzero(fail_o))
                        totals["corr_fail"] += int(np.count_nonzero(fail_c))
                        totals["corr_only"] += int(np.count_nonzero(fail_c & ~fail_o))
                        totals["ord_only"] += int(np.count_nonzero(fail_o & ~fail_c))
                        totals["both_fail"] += int(np.count_nonzero(fail_o & fail_c))
                        totals["both_correct"] += int(np.count_nonzero(~fail_o & ~fail_c))
                        totals["disagree"] += int(np.count_nonzero(disagree))
                        totals["obs_flip"] += int(np.count_nonzero(obs_flip))
                        for i, w in enumerate(weights):
                            b = bins[weight_bin(int(w))]
                            b["n"] += 1
                            b["ord_fail"] += int(fail_o[i])
                            b["corr_fail"] += int(fail_c[i])
                            b["corr_only"] += int(fail_c[i] and not fail_o[i])
                            b["ord_only"] += int(fail_o[i] and not fail_c[i])
                            b["both_fail"] += int(fail_o[i] and fail_c[i])
                            b["both_correct"] += int(not fail_o[i] and not fail_c[i])
                            b["disagree"] += int(disagree[i])
                        processed += n_batch
                        if processed % 200_000 == 0 or processed == shots:
                            print(f"  processed {processed:,}/{shots:,}")

                    delta, lo, hi = ci_for_paired_binary(totals["corr_only"], totals["ord_only"], shots)
                    print(f"Observable flips: {totals['obs_flip']:,}/{shots:,} ({totals['obs_flip']/shots*100:.5f}%)")
                    print(f"Ordinary MWPM failures: {totals['ord_fail']:,} ({totals['ord_fail']/shots*100:.5f}%)")
                    print(f"Correlation-aware failures: {totals['corr_fail']:,} ({totals['corr_fail']/shots*100:.5f}%)")
                    print(f"Paired delta corr-aware - ordinary: {delta:+.6f} pp [{lo:+.6f}, {hi:+.6f}]")
                    print(f"Corr-aware-only failures={totals['corr_only']:,}; ordinary-only failures={totals['ord_only']:,}; prediction disagreements={totals['disagree']:,}")
                    print("By detector weight: bin       n       ordinary%   corr-aware%   delta pp [95% CI]       C-only / O-only")
                    for bname in names:
                        b = bins[bname]
                        if not b["n"]:
                            continue
                        bd, blo, bhi = ci_for_paired_binary(b["corr_only"], b["ord_only"], b["n"])
                        print(f"  {bname:7s} {b['n']:8,d} {100*b['ord_fail']/b['n']:10.5f} {100*b['corr_fail']/b['n']:12.5f} {bd:+.6f} [{blo:+.6f},{bhi:+.6f}] {b['corr_only']:5d}/{b['ord_only']:5d}")

                    sw.writerow({
                        "distance": distance, "p_base": P_BASE, "p_corr": p_corr, "physical_model": model_name,
                        "seed": seed, "shots": shots, "detectors": dem.num_detectors,
                        "observable_flip_count": totals["obs_flip"], "ordinary_failures": totals["ord_fail"],
                        "correlation_aware_failures": totals["corr_fail"],
                        "ordinary_rate_pct": 100*totals["ord_fail"]/shots,
                        "correlation_aware_rate_pct": 100*totals["corr_fail"]/shots,
                        "delta_corr_minus_ordinary_pp": delta, "delta_ci_low_pp": lo, "delta_ci_high_pp": hi,
                        "prediction_disagreements": totals["disagree"],
                        "correlation_aware_only_correct": totals["ord_only"],
                        "ordinary_only_correct": totals["corr_only"],
                        "both_fail": totals["both_fail"], "both_correct": totals["both_correct"],
                        "dem_error_instructions": stats["error_instructions"],
                        "dem_separator_count": stats["separator_count"],
                        "dem_graphlike_components": stats["graphlike_components"],
                    })
                    for bname, b in bins.items():
                        if not b["n"]:
                            continue
                        bd, blo, bhi = ci_for_paired_binary(b["corr_only"], b["ord_only"], b["n"])
                        ww.writerow({
                            "distance": distance, "p_base": P_BASE, "p_corr": p_corr,
                            "physical_model": model_name, "weight_bin": bname, "shots_in_bin": b["n"],
                            "ordinary_failures": b["ord_fail"], "correlation_aware_failures": b["corr_fail"],
                            "ordinary_rate_pct": 100*b["ord_fail"]/b["n"],
                            "correlation_aware_rate_pct": 100*b["corr_fail"]/b["n"],
                            "delta_corr_minus_ordinary_pp": bd, "delta_ci_low_pp": blo, "delta_ci_high_pp": bhi,
                            "correlation_aware_only_correct": b["ord_only"],
                            "ordinary_only_correct": b["corr_only"], "both_fail": b["both_fail"],
                            "both_correct": b["both_correct"], "prediction_disagreements": b["disagree"],
                        })
                    sf.flush(); wf.flush()
                    print(f"Condition elapsed={time.time()-start:.1f}s; outputs checkpointed")

    print("\nCompleted.")
    print(f"Summary: {SUMMARY_CSV.resolve()}")
    print(f"By syndrome weight: {WEIGHT_CSV.resolve()}")
    print("Negative paired delta means correlation-aware MWPM has fewer logical failures.")
    print("Weight-stratified intervals are descriptive and are not corrected for selection across bins.")


if __name__ == "__main__":
    main()
