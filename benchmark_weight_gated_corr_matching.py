"""Screen and independently validate a weight-gated correlation-aware decoder.

Motivation: paired win/loss analysis found that correlation-aware matching's gains
are concentrated in medium/high detector-weight syndromes. This script screens a
simple policy (use ordinary MWPM below a detector-weight threshold, correlation-
aware MWPM at or above it), selecting one threshold per distance from fully
correlated screening samples pooled across both p_corr values. A fresh validation
sample then compares the selected policy against ordinary and always-correlation-
aware MWPM on both fully-correlated and matched-independent physical circuits.

The physical circuit and Stim's default DEM decomposition are unchanged. DEM error
instructions are sorted canonically only; no arbitrary component rewrites occur.

Run from the project directory containing correlated_experiment.py:
    & $python .\\benchmark_weight_gated_corr_matching.py
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
VALIDATION_SHOTS = {5: 1_000_000, 7: 2_000_000}
SCREEN_SHOTS = {d: VALIDATION_SHOTS[d] // 2 for d in DISTANCES}
CHUNK_SIZE = 25_000
SEED_SCREEN_BASE = 202_730_00
SEED_VALID_BASE = 202_740_00
# Apply correlation-aware decoding when detector weight >= threshold.
# 0 means always correlation-aware; 999 means always ordinary for these codes.
THRESHOLDS = (0, 3, 4, 5, 6, 8, 11, 999)
MAX_SCREEN_EXTRA_FAILURE_FRACTION = 0.10
SCREEN_CSV = Path("weight_gated_corr_screen.csv")
THRESHOLD_CSV = Path("weight_gated_corr_threshold_selection.csv")
VALIDATION_CSV = Path("weight_gated_corr_validation.csv")
ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")


def as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def failure_mask(predictions: Any, actual: Any) -> np.ndarray:
    p, y = as_2d(predictions), as_2d(actual)
    if p.shape != y.shape:
        raise ValueError(f"prediction/observable shape mismatch: {p.shape} vs {y.shape}")
    return np.any(p != y, axis=1)


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    error_lines = [lines[i] for i in positions]

    def key(line: str) -> tuple[float, str]:
        match = ERROR_RE.match(line)
        if match is None:
            raise ValueError(f"could not parse DEM error instruction: {line!r}")
        return (-float(match.group(1)), line.strip())

    for pos, line in zip(positions, sorted(error_lines, key=key)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise AssertionError("canonical sorting changed DEM dimensions")
    return result


def paired_delta_ci(candidate_only: int, baseline_only: int, n: int, z: float = 1.959963984540054):
    """Paired normal CI in percentage points for candidate minus baseline failure rate."""
    delta_fraction = (candidate_only - baseline_only) / n
    discordant = candidate_only + baseline_only
    if n <= 1:
        return delta_fraction * 100, float("nan"), float("nan")
    var_mean = max(0.0, (discordant - n * delta_fraction * delta_fraction) / (n - 1) / n)
    se_pp = 100 * math.sqrt(var_mean)
    delta_pp = delta_fraction * 100
    return delta_pp, delta_pp - z * se_pp, delta_pp + z * se_pp


def construct_condition(distance: int, p_corr: float, mode: str):
    base = make_base_circuit(distance=distance, rounds=distance, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
    dem = canonical_probability_sorted_dem(circuit.detector_error_model(decompose_errors=True).flattened())
    ordinary = pymatching.Matching(dem)
    corr = pymatching.Matching(dem, enable_correlations=True)
    return circuit, dem, ordinary, corr, len(pairs)


def screen_thresholds() -> dict[int, int]:
    totals: dict[int, dict[int, int]] = {
        d: {t: 0 for t in THRESHOLDS} for d in DISTANCES
    }
    total_corr_failures = {d: 0 for d in DISTANCES}
    total_routed = {d: {t: 0 for t in THRESHOLDS} for d in DISTANCES}
    screen_fields = [
        "distance", "p_base", "p_corr", "physical_model", "seed", "shots",
        "threshold", "ordinary_failures", "correlation_aware_failures",
        "gated_failures", "extra_failures_vs_corr_aware", "shots_routed_to_corr",
        "fraction_routed_to_corr",
    ]
    print("Weight-gated correlated matching: screen + independent validation")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}")
    print(f"validation shots/condition={VALIDATION_SHOTS}; screen shots={SCREEN_SHOTS}; chunk={CHUNK_SIZE:,}")
    print(f"thresholds={THRESHOLDS}; screen non-inferiority budget={MAX_SCREEN_EXTRA_FAILURE_FRACTION:.0%} over always-correlation-aware")
    print("Screen threshold is selected per distance using only fully-correlated screening shots pooled across both p_corr values.")

    with SCREEN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=screen_fields)
        writer.writeheader()
        for distance in DISTANCES:
            for pc_index, p_corr in enumerate(P_CORR_VALUES):
                shots = SCREEN_SHOTS[distance]
                seed = SEED_SCREEN_BASE + distance * 100 + pc_index
                circuit, dem, ordinary, corr, pair_count = construct_condition(distance, p_corr, "correlated")
                sampler = circuit.compile_detector_sampler(seed=seed)
                per_t = {t: {"gate_fail": 0, "routed": 0} for t in THRESHOLDS}
                ordinary_failures = corr_failures = processed = 0
                print("\n" + "=" * 94)
                print(f"SCREEN d={distance}; p_corr={p_corr:g}; seed={seed}; shots={shots:,}; detectors={dem.num_detectors}; pairs={pair_count}")
                while processed < shots:
                    n_batch = min(CHUNK_SIZE, shots - processed)
                    synd, actual = sampler.sample(shots=n_batch, separate_observables=True)
                    synd = np.asarray(synd, dtype=np.uint8)
                    actual = as_2d(actual)
                    pred_o = as_2d(ordinary.decode_batch(synd))
                    pred_c = as_2d(corr.decode_batch(synd, enable_correlations=True))
                    fail_o = failure_mask(pred_o, actual)
                    fail_c = failure_mask(pred_c, actual)
                    weights = np.sum(synd, axis=1)
                    ordinary_failures += int(np.count_nonzero(fail_o))
                    corr_failures += int(np.count_nonzero(fail_c))
                    for threshold in THRESHOLDS:
                        high = weights >= threshold
                        gate_fail = np.where(high, fail_c, fail_o)
                        per_t[threshold]["gate_fail"] += int(np.count_nonzero(gate_fail))
                        per_t[threshold]["routed"] += int(np.count_nonzero(high))
                    processed += n_batch
                    if processed % 200_000 == 0 or processed == shots:
                        print(f"  sampled {processed:,}/{shots:,}")
                total_corr_failures[distance] += corr_failures
                print(f"  ordinary failures={ordinary_failures:,}; correlation-aware failures={corr_failures:,}")
                for threshold in THRESHOLDS:
                    gated = per_t[threshold]["gate_fail"]
                    totals[distance][threshold] += gated
                    routed = per_t[threshold]["routed"]
                    total_routed[distance][threshold] += routed
                    writer.writerow({
                        "distance": distance, "p_base": P_BASE, "p_corr": p_corr,
                        "physical_model": "fully-correlated-screen", "seed": seed,
                        "shots": shots, "threshold": threshold,
                        "ordinary_failures": ordinary_failures,
                        "correlation_aware_failures": corr_failures,
                        "gated_failures": gated,
                        "extra_failures_vs_corr_aware": gated - corr_failures,
                        "shots_routed_to_corr": routed,
                        "fraction_routed_to_corr": routed / shots,
                    })
                f.flush()

    selected: dict[int, int] = {}
    with THRESHOLD_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = ["distance", "screen_shots_pooled", "screen_corr_aware_failures", "threshold",
                  "gated_failures", "extra_failures_vs_corr_aware", "allowed_extra_failures",
                  "fraction_extra_over_corr_baseline", "selection_rule"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for distance in DISTANCES:
            baseline = total_corr_failures[distance]
            allowed = MAX_SCREEN_EXTRA_FAILURE_FRACTION * baseline
            feasible = [t for t in THRESHOLDS if totals[distance][t] <= baseline + allowed]
            threshold = max(feasible) if feasible else 0
            selected[distance] = threshold
            gated = totals[distance][threshold]
            denom = baseline if baseline else 1
            writer.writerow({
                "distance": distance, "screen_shots_pooled": sum(SCREEN_SHOTS[distance] for _ in P_CORR_VALUES),
                "screen_corr_aware_failures": baseline, "threshold": threshold,
                "gated_failures": gated, "extra_failures_vs_corr_aware": gated - baseline,
                "allowed_extra_failures": allowed,
                "fraction_extra_over_corr_baseline": (gated - baseline) / denom,
                "selection_rule": "largest threshold whose pooled failures are within 10% of always-correlation-aware",
            })
            pooled_screen_shots = sum(SCREEN_SHOTS[distance] for _ in P_CORR_VALUES)
            routed_fraction = total_routed[distance][threshold] / pooled_screen_shots
            print(f"\nSELECTED d={distance}: use correlation-aware iff weight >= {threshold}; "
                  f"screen CA failures={baseline}, gated={gated}, extra={gated-baseline} "
                  f"({(gated-baseline)/denom:.1%}); routed-to-CA fraction={routed_fraction:.1%}")
    print(f"Screen table: {SCREEN_CSV.resolve()}")
    print(f"Selected thresholds: {THRESHOLD_CSV.resolve()}")
    return selected


def validate(selected: dict[int, int]) -> None:
    fields = [
        "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "gate_threshold",
        "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "ordinary_rate_pct", "correlation_aware_rate_pct", "gated_rate_pct",
        "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "shots_routed_to_corr", "fraction_routed_to_corr", "ordinary_decode_seconds",
        "correlation_aware_decode_seconds", "gated_decode_seconds", "speedup_vs_full_corr",
        "dem_error_instructions", "dem_separator_count",
    ]
    with VALIDATION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for distance in DISTANCES:
            shots = VALIDATION_SHOTS[distance]
            for pc_index, p_corr in enumerate(P_CORR_VALUES):
                for model_index, (model_name, mode) in enumerate([
                    ("matched-independent-control", "matched_independent"),
                    ("fully-correlated", "correlated"),
                ]):
                    seed = SEED_VALID_BASE + distance * 1000 + pc_index * 10 + model_index
                    circuit, dem, ordinary, corr, pair_count = construct_condition(distance, p_corr, mode)
                    sampler = circuit.compile_detector_sampler(seed=seed)
                    threshold = selected[distance]
                    counts = {
                        "o_fail": 0, "c_fail": 0, "g_fail": 0,
                        "gate_only_vs_o": 0, "o_only_vs_gate": 0,
                        "gate_only_vs_c": 0, "c_only_vs_gate": 0,
                        "routed": 0,
                    }
                    ord_time = corr_time = gate_time = 0.0
                    processed = 0
                    dem_errors = dem_separators = 0
                    for line in str(dem).splitlines():
                        m = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
                        if m and float(m.group(1)) > 0:
                            dem_errors += 1
                            dem_separators += len(re.findall(r"\s+\^\s+", m.group(2)))
                    print("\n" + "=" * 94)
                    print(f"VALIDATE d={distance}; p_corr={p_corr:g}; physical={model_name}; seed={seed}; "
                          f"shots={shots:,}; gate uses corr-aware iff weight >= {threshold}")
                    while processed < shots:
                        n_batch = min(CHUNK_SIZE, shots - processed)
                        synd, actual = sampler.sample(shots=n_batch, separate_observables=True)
                        synd = np.asarray(synd, dtype=np.uint8)
                        actual = as_2d(actual)
                        weights = np.sum(synd, axis=1)

                        t0 = time.perf_counter()
                        pred_o = as_2d(ordinary.decode_batch(synd))
                        ord_time += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        pred_c = as_2d(corr.decode_batch(synd, enable_correlations=True))
                        corr_time += time.perf_counter() - t0

                        high = weights >= threshold
                        low = ~high
                        pred_g = np.zeros_like(pred_o, dtype=np.uint8)
                        t0 = time.perf_counter()
                        if np.any(low):
                            pred_g[low] = as_2d(ordinary.decode_batch(synd[low]))
                        if np.any(high):
                            pred_g[high] = as_2d(corr.decode_batch(synd[high], enable_correlations=True))
                        gate_time += time.perf_counter() - t0

                        fail_o = failure_mask(pred_o, actual)
                        fail_c = failure_mask(pred_c, actual)
                        fail_g = failure_mask(pred_g, actual)
                        counts["o_fail"] += int(np.count_nonzero(fail_o))
                        counts["c_fail"] += int(np.count_nonzero(fail_c))
                        counts["g_fail"] += int(np.count_nonzero(fail_g))
                        counts["gate_only_vs_o"] += int(np.count_nonzero(fail_g & ~fail_o))
                        counts["o_only_vs_gate"] += int(np.count_nonzero(fail_o & ~fail_g))
                        counts["gate_only_vs_c"] += int(np.count_nonzero(fail_g & ~fail_c))
                        counts["c_only_vs_gate"] += int(np.count_nonzero(fail_c & ~fail_g))
                        counts["routed"] += int(np.count_nonzero(high))
                        processed += n_batch
                        if processed % 200_000 == 0 or processed == shots:
                            print(f"  sampled/decoded {processed:,}/{shots:,}")

                    d_o, lo_o, hi_o = paired_delta_ci(counts["gate_only_vs_o"], counts["o_only_vs_gate"], shots)
                    d_c, lo_c, hi_c = paired_delta_ci(counts["gate_only_vs_c"], counts["c_only_vs_gate"], shots)
                    route_fraction = counts["routed"] / shots
                    speedup = corr_time / gate_time if gate_time > 0 else float("nan")
                    print(f"  ordinary={counts['o_fail']:,} ({counts['o_fail']/shots*100:.5f}%) | "
                          f"full corr-aware={counts['c_fail']:,} ({counts['c_fail']/shots*100:.5f}%) | "
                          f"gated={counts['g_fail']:,} ({counts['g_fail']/shots*100:.5f}%)")
                    print(f"  gate - ordinary={d_o:+.6f} pp [{lo_o:+.6f},{hi_o:+.6f}]; "
                          f"gate - full corr-aware={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}]")
                    print(f"  routed to corr-aware={route_fraction:.1%}; measured decode times: ordinary={ord_time:.3f}s, "
                          f"full corr-aware={corr_time:.3f}s, gated={gate_time:.3f}s; full/gated speed ratio={speedup:.2f}x")
                    writer.writerow({
                        "distance": distance, "p_base": P_BASE, "p_corr": p_corr,
                        "physical_model": model_name, "seed": seed, "shots": shots,
                        "gate_threshold": threshold,
                        "ordinary_failures": counts["o_fail"], "correlation_aware_failures": counts["c_fail"],
                        "gated_failures": counts["g_fail"],
                        "ordinary_rate_pct": counts["o_fail"] / shots * 100,
                        "correlation_aware_rate_pct": counts["c_fail"] / shots * 100,
                        "gated_rate_pct": counts["g_fail"] / shots * 100,
                        "gate_vs_ordinary_delta_pp": d_o, "gate_vs_ordinary_ci_low_pp": lo_o,
                        "gate_vs_ordinary_ci_high_pp": hi_o,
                        "gate_vs_corr_delta_pp": d_c, "gate_vs_corr_ci_low_pp": lo_c,
                        "gate_vs_corr_ci_high_pp": hi_c,
                        "shots_routed_to_corr": counts["routed"], "fraction_routed_to_corr": route_fraction,
                        "ordinary_decode_seconds": ord_time, "correlation_aware_decode_seconds": corr_time,
                        "gated_decode_seconds": gate_time, "speedup_vs_full_corr": speedup,
                        "dem_error_instructions": dem_errors, "dem_separator_count": dem_separators,
                    })
                    f.flush()
    print(f"\nValidation complete: {VALIDATION_CSV.resolve()}")
    print("Negative gate-minus-baseline delta means fewer failures for the gated policy.")
    print("Selected thresholds were chosen on separate fully-correlated screening samples; validation uses fresh seeds.")


def main() -> None:
    selected = screen_thresholds()
    validate(selected)


if __name__ == "__main__":
    main()
