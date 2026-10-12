"""Independent multi-seed confirmation of the screened weight-gated decoder.

The thresholds are frozen from the previous screen (distance 5: use correlated
matching when detector weight >= 5; distance 7: >= 8). No retuning occurs here.
Each physical circuit sample is decoded by ordinary MWPM, full correlation-aware
MWPM, and the gated policy. Outputs include replicate-level and pooled paired
statistics. Run from the project directory containing correlated_experiment.py.
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
SHOTS = {5: 1_000_000, 7: 2_000_000}
CHUNK_SIZE = 25_000
REPLICATES = 3
SEED_BASE = 202_830_00
FROZEN_THRESHOLDS = {5: 5, 7: 8}
REPLICATE_CSV = Path("weight_gated_corr_multiseed_replicates.csv")
SUMMARY_CSV = Path("weight_gated_corr_multiseed_summary.csv")
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
    variance_of_mean = max(0.0, (discordant - n * delta_fraction * delta_fraction) / ((n - 1) * n))
    se_pp = 100 * math.sqrt(variance_of_mean)
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


def count_dem_instructions(dem: stim.DetectorErrorModel) -> tuple[int, int]:
    errors = separators = 0
    for line in str(dem).splitlines():
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if match and float(match.group(1)) > 0:
            errors += 1
            separators += len(re.findall(r"\s+\^\s+", match.group(2)))
    return errors, separators


def main() -> None:
    print("Frozen-threshold multi-seed validation of weight-gated correlation-aware MWPM")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; shots/replicate={SHOTS}")
    print(f"replicates={REPLICATES}; seed base={SEED_BASE}; frozen thresholds={FROZEN_THRESHOLDS}; chunk={CHUNK_SIZE:,}")
    print("Thresholds were selected in an earlier screen and are NOT retuned here.")
    print("Negative gated-minus-baseline delta means fewer logical failures for the gated policy.")

    replicate_fields = [
        "replicate", "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "gate_threshold",
        "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "gate_only_vs_ordinary", "ordinary_only_vs_gate", "gate_only_vs_corr", "corr_only_vs_gate",
        "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "shots_routed_to_corr", "fraction_routed_to_corr", "ordinary_decode_seconds",
        "correlation_aware_decode_seconds", "gated_decode_seconds", "speedup_vs_full_corr",
    ]
    aggregate: dict[tuple[int, float, str], dict[str, float | int]] = {}
    with REPLICATE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=replicate_fields)
        writer.writeheader()
        for distance in DISTANCES:
            threshold = FROZEN_THRESHOLDS[distance]
            for p_index, p_corr in enumerate(P_CORR_VALUES):
                for model_index, (model_name, mode) in enumerate([
                    ("matched-independent-control", "matched_independent"),
                    ("fully-correlated", "correlated"),
                ]):
                    key = (distance, p_corr, model_name)
                    agg = aggregate.setdefault(key, {
                        "shots": 0, "ordinary_failures": 0, "correlation_aware_failures": 0,
                        "gated_failures": 0, "gate_only_vs_ordinary": 0, "ordinary_only_vs_gate": 0,
                        "gate_only_vs_corr": 0, "corr_only_vs_gate": 0, "routed": 0,
                        "ordinary_decode_seconds": 0.0, "correlation_aware_decode_seconds": 0.0,
                        "gated_decode_seconds": 0.0,
                    })
                    for rep in range(REPLICATES):
                        seed = SEED_BASE + rep * 100_000 + distance * 1000 + p_index * 10 + model_index
                        circuit, dem, ordinary, corr, pair_count = construct_condition(distance, p_corr, mode)
                        sampler = circuit.compile_detector_sampler(seed=seed)
                        n_shots = SHOTS[distance]
                        counts = {
                            "o_fail": 0, "c_fail": 0, "g_fail": 0,
                            "gate_only_vs_o": 0, "o_only_vs_gate": 0,
                            "gate_only_vs_c": 0, "c_only_vs_gate": 0, "routed": 0,
                        }
                        ordinary_seconds = corr_seconds = gated_seconds = 0.0
                        processed = 0
                        print("\n" + "=" * 94)
                        print(f"d={distance}; p_corr={p_corr:g}; physical={model_name}; replicate={rep+1}/{REPLICATES}; "
                              f"seed={seed}; shots={n_shots:,}; pairs={pair_count}; threshold={threshold}")
                        while processed < n_shots:
                            batch = min(CHUNK_SIZE, n_shots - processed)
                            syndrome, actual = sampler.sample(shots=batch, separate_observables=True)
                            syndrome = np.asarray(syndrome, dtype=np.uint8)
                            actual = as_2d(actual)
                            weights = np.sum(syndrome, axis=1)

                            t0 = time.perf_counter()
                            pred_o = as_2d(ordinary.decode_batch(syndrome))
                            ordinary_seconds += time.perf_counter() - t0
                            t0 = time.perf_counter()
                            pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                            corr_seconds += time.perf_counter() - t0

                            high = weights >= threshold
                            low = ~high
                            pred_g = np.zeros_like(pred_o, dtype=np.uint8)
                            t0 = time.perf_counter()
                            if np.any(low):
                                pred_g[low] = as_2d(ordinary.decode_batch(syndrome[low]))
                            if np.any(high):
                                pred_g[high] = as_2d(corr.decode_batch(syndrome[high], enable_correlations=True))
                            gated_seconds += time.perf_counter() - t0

                            fail_o = failure_mask(pred_o, actual)
                            fail_c = failure_mask(pred_c, actual)
                            fail_g = failure_mask(pred_g, actual)
                            counts["o_fail"] += int(fail_o.sum())
                            counts["c_fail"] += int(fail_c.sum())
                            counts["g_fail"] += int(fail_g.sum())
                            counts["gate_only_vs_o"] += int(np.count_nonzero(fail_g & ~fail_o))
                            counts["o_only_vs_gate"] += int(np.count_nonzero(fail_o & ~fail_g))
                            counts["gate_only_vs_c"] += int(np.count_nonzero(fail_g & ~fail_c))
                            counts["c_only_vs_gate"] += int(np.count_nonzero(fail_c & ~fail_g))
                            counts["routed"] += int(high.sum())
                            processed += batch
                            if processed % 200_000 == 0 or processed == n_shots:
                                print(f"  decoded {processed:,}/{n_shots:,}")

                        d_o, lo_o, hi_o = paired_delta_ci(counts["gate_only_vs_o"], counts["o_only_vs_gate"], n_shots)
                        d_c, lo_c, hi_c = paired_delta_ci(counts["gate_only_vs_c"], counts["c_only_vs_gate"], n_shots)
                        routed_fraction = counts["routed"] / n_shots
                        speedup = corr_seconds / gated_seconds if gated_seconds > 0 else float("nan")
                        print(f"ordinary={counts['o_fail']:,} ({counts['o_fail']/n_shots*100:.5f}%) | "
                              f"full CA={counts['c_fail']:,} ({counts['c_fail']/n_shots*100:.5f}%) | "
                              f"gated={counts['g_fail']:,} ({counts['g_fail']/n_shots*100:.5f}%)")
                        print(f"gate-ordinary={d_o:+.6f} pp [{lo_o:+.6f},{hi_o:+.6f}]; "
                              f"gate-full-CA={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}]; "
                              f"routed={routed_fraction:.1%}; speedup={speedup:.2f}x")
                        writer.writerow({
                            "replicate": rep + 1, "distance": distance, "p_base": P_BASE, "p_corr": p_corr,
                            "physical_model": model_name, "seed": seed, "shots": n_shots, "gate_threshold": threshold,
                            "ordinary_failures": counts["o_fail"], "correlation_aware_failures": counts["c_fail"],
                            "gated_failures": counts["g_fail"], "gate_only_vs_ordinary": counts["gate_only_vs_o"],
                            "ordinary_only_vs_gate": counts["o_only_vs_gate"], "gate_only_vs_corr": counts["gate_only_vs_c"],
                            "corr_only_vs_gate": counts["c_only_vs_gate"], "gate_vs_ordinary_delta_pp": d_o,
                            "gate_vs_ordinary_ci_low_pp": lo_o, "gate_vs_ordinary_ci_high_pp": hi_o,
                            "gate_vs_corr_delta_pp": d_c, "gate_vs_corr_ci_low_pp": lo_c,
                            "gate_vs_corr_ci_high_pp": hi_c, "shots_routed_to_corr": counts["routed"],
                            "fraction_routed_to_corr": routed_fraction,
                            "ordinary_decode_seconds": ordinary_seconds,
                            "correlation_aware_decode_seconds": corr_seconds,
                            "gated_decode_seconds": gated_seconds, "speedup_vs_full_corr": speedup,
                        })
                        f.flush()
                        # Accumulate sufficient counts for pooled paired uncertainty.
                        agg["shots"] += n_shots
                        for source, target in [
                            ("o_fail", "ordinary_failures"), ("c_fail", "correlation_aware_failures"),
                            ("g_fail", "gated_failures"), ("gate_only_vs_o", "gate_only_vs_ordinary"),
                            ("o_only_vs_gate", "ordinary_only_vs_gate"), ("gate_only_vs_c", "gate_only_vs_corr"),
                            ("c_only_vs_gate", "corr_only_vs_gate"), ("routed", "routed"),
                        ]:
                            agg[target] += counts[source]
                        agg["ordinary_decode_seconds"] += ordinary_seconds
                        agg["correlation_aware_decode_seconds"] += corr_seconds
                        agg["gated_decode_seconds"] += gated_seconds

    summary_fields = [
        "distance", "p_base", "p_corr", "physical_model", "replicates", "pooled_shots", "gate_threshold",
        "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "fraction_routed_to_corr", "ordinary_decode_seconds", "correlation_aware_decode_seconds",
        "gated_decode_seconds", "speedup_vs_full_corr",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for key in sorted(aggregate):
            distance, p_corr, model_name = key
            agg = aggregate[key]
            n = int(agg["shots"])
            d_o, lo_o, hi_o = paired_delta_ci(int(agg["gate_only_vs_ordinary"]), int(agg["ordinary_only_vs_gate"]), n)
            d_c, lo_c, hi_c = paired_delta_ci(int(agg["gate_only_vs_corr"]), int(agg["corr_only_vs_gate"]), n)
            ca_seconds = float(agg["correlation_aware_decode_seconds"])
            gate_seconds = float(agg["gated_decode_seconds"])
            row = {
                "distance": distance, "p_base": P_BASE, "p_corr": p_corr, "physical_model": model_name,
                "replicates": REPLICATES, "pooled_shots": n, "gate_threshold": FROZEN_THRESHOLDS[distance],
                "ordinary_failures": agg["ordinary_failures"], "correlation_aware_failures": agg["correlation_aware_failures"],
                "gated_failures": agg["gated_failures"], "gate_vs_ordinary_delta_pp": d_o,
                "gate_vs_ordinary_ci_low_pp": lo_o, "gate_vs_ordinary_ci_high_pp": hi_o,
                "gate_vs_corr_delta_pp": d_c, "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                "fraction_routed_to_corr": float(agg["routed"]) / n,
                "ordinary_decode_seconds": agg["ordinary_decode_seconds"],
                "correlation_aware_decode_seconds": ca_seconds, "gated_decode_seconds": gate_seconds,
                "speedup_vs_full_corr": ca_seconds / gate_seconds if gate_seconds else float("nan"),
            }
            writer.writerow(row)
            print(f"\nPOOLED d={distance}; p_corr={p_corr:g}; {model_name}; n={n:,}; threshold={FROZEN_THRESHOLDS[distance]}")
            print(f"  ordinary={int(agg['ordinary_failures'])/n*100:.5f}% | full CA={int(agg['correlation_aware_failures'])/n*100:.5f}% | "
                  f"gated={int(agg['gated_failures'])/n*100:.5f}%")
            print(f"  gate-full-CA={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}]; "
                  f"gate-ordinary={d_o:+.6f} pp [{lo_o:+.6f},{hi_o:+.6f}]")
            print(f"  routed={float(agg['routed'])/n:.1%}; measured full/gated speedup={ca_seconds/gate_seconds:.2f}x")
            f.flush()

    print(f"\nReplicate rows: {REPLICATE_CSV.resolve()}")
    print(f"Pooled summary: {SUMMARY_CSV.resolve()}")
    print("Interpretation: thresholds are fixed before these new seeds; pooled paired CIs are primary.")


if __name__ == "__main__":
    main()
