"""Screen and independently validate weight-gated correlation-aware MWPM at d=9.

Uses the project's correlated_experiment.py API. The physical model is always
left unchanged; both ordinary and correlation-aware decoders are built from the
same canonical probability-sorted, Stim-default decomposed DEM. A detector-weight
threshold is selected only from fully-correlated screening samples, then frozen
for independent validation against fully-correlated and matched-independent
controls at both p_corr values.
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
DISTANCE = 9
SCREEN_SHOTS_PER_P = 5_000_000
VALIDATION_SHOTS_PER_REPLICATE = 2_000_000
VALIDATION_REPLICATES = 3
CHUNK_SIZE = 25_000
SEED_BASE = 202_850_00
THRESHOLDS = (6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 999)
SCREEN_NONINFERIORITY_BUDGET = 0.10

SCREEN_CSV = Path("weight_gated_d9_screen.csv")
THRESHOLD_CSV = Path("weight_gated_d9_threshold_selection.csv")
REPLICATE_CSV = Path("weight_gated_d9_validation_replicates.csv")
SUMMARY_CSV = Path("weight_gated_d9_validation_summary.csv")
ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")


def as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def fail_mask(predictions: Any, actual: Any) -> np.ndarray:
    p = as_2d(predictions)
    y = as_2d(actual)
    if p.shape != y.shape:
        raise ValueError(f"prediction/observable shape mismatch: {p.shape} vs {y.shape}")
    return np.any(p != y, axis=1)


def paired_delta_ci(candidate_only: int, baseline_only: int, n: int, z: float = 1.959963984540054):
    """Paired normal CI in percentage points for candidate-minus-baseline failure rate."""
    delta_fraction = (candidate_only - baseline_only) / n
    discordant = candidate_only + baseline_only
    var = max(0.0, (discordant - n * delta_fraction * delta_fraction) / ((n - 1) * n)) if n > 1 else float("nan")
    delta_pp = 100 * delta_fraction
    se_pp = 100 * math.sqrt(var) if n > 1 else float("nan")
    return delta_pp, delta_pp - z * se_pp, delta_pp + z * se_pp


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


def make_condition(p_corr: float, mode: str):
    base = make_base_circuit(distance=DISTANCE, rounds=DISTANCE, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
    dem = canonical_probability_sorted_dem(circuit.detector_error_model(decompose_errors=True).flattened())
    ordinary = pymatching.Matching(dem)
    corr = pymatching.Matching(dem, enable_correlations=True)
    return circuit, dem, ordinary, corr, len(pairs)


def print_progress(processed: int, total: int) -> None:
    if processed % 250_000 == 0 or processed == total:
        print(f"  decoded {processed:,}/{total:,}")


def screen_threshold() -> int:
    """Select largest threshold within budget using independent screen samples."""
    screen_fields = [
        "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "threshold",
        "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "gated_extra_failures_vs_corr", "fraction_routed_to_corr",
    ]
    pooled = {t: {"ordinary": 0, "corr": 0, "gated": 0, "routed": 0, "shots": 0} for t in THRESHOLDS}
    with SCREEN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=screen_fields)
        writer.writeheader()
        for p_index, p_corr in enumerate(P_CORR_VALUES):
            seed = SEED_BASE + 50_000 + p_index
            circuit, dem, ordinary, corr, pair_count = make_condition(p_corr, "correlated")
            sampler = circuit.compile_detector_sampler(seed=seed)
            print("\n" + "=" * 92)
            print(f"SCREEN d=9; p_corr={p_corr:g}; fully-correlated; seed={seed}; shots={SCREEN_SHOTS_PER_P:,}; pairs={pair_count}")
            print(f"DEM dimensions=({dem.num_detectors},{dem.num_observables}); error instructions={sum(1 for x in str(dem).splitlines() if x.lstrip().startswith('error('))}")
            local = {t: {"ordinary": 0, "corr": 0, "gated": 0, "routed": 0} for t in THRESHOLDS}
            processed = 0
            while processed < SCREEN_SHOTS_PER_P:
                batch = min(CHUNK_SIZE, SCREEN_SHOTS_PER_P - processed)
                syndrome, actual = sampler.sample(shots=batch, separate_observables=True)
                syndrome = np.asarray(syndrome, dtype=np.uint8)
                actual = as_2d(actual)
                weight = np.sum(syndrome, axis=1)
                pred_o = as_2d(ordinary.decode_batch(syndrome))
                pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                fo = fail_mask(pred_o, actual)
                fc = fail_mask(pred_c, actual)
                for t in THRESHOLDS:
                    high = weight >= t
                    fg = np.where(high, fc, fo)
                    local[t]["ordinary"] += int(fo.sum())
                    local[t]["corr"] += int(fc.sum())
                    local[t]["gated"] += int(fg.sum())
                    local[t]["routed"] += int(high.sum())
                processed += batch
                print_progress(processed, SCREEN_SHOTS_PER_P)
            for t in THRESHOLDS:
                s = local[t]
                pooled[t]["ordinary"] += s["ordinary"]
                pooled[t]["corr"] += s["corr"]
                pooled[t]["gated"] += s["gated"]
                pooled[t]["routed"] += s["routed"]
                pooled[t]["shots"] += SCREEN_SHOTS_PER_P
                writer.writerow({
                    "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr,
                    "physical_model": "fully-correlated", "seed": seed, "shots": SCREEN_SHOTS_PER_P,
                    "threshold": t, "ordinary_failures": s["ordinary"],
                    "correlation_aware_failures": s["corr"], "gated_failures": s["gated"],
                    "gated_extra_failures_vs_corr": s["gated"] - s["corr"],
                    "fraction_routed_to_corr": s["routed"] / SCREEN_SHOTS_PER_P,
                })
                f.flush()

    full_ca = pooled[THRESHOLDS[0]]["corr"]  # threshold 6 routes almost all, but not all: don't use this as denominator.
    # Full correlation-aware failures are threshold independent; every threshold's stored corr total is identical.
    full_ca = pooled[THRESHOLDS[0]]["corr"]
    total_shots = pooled[THRESHOLDS[0]]["shots"]
    budget_failures = full_ca * (1 + SCREEN_NONINFERIORITY_BUDGET)
    eligible = [t for t in THRESHOLDS if pooled[t]["gated"] <= budget_failures]
    selected = max(eligible) if eligible else THRESHOLDS[0]

    fields = ["distance", "p_base", "p_corr_screen_pooled", "screen_shots", "threshold",
              "ordinary_failures", "correlation_aware_failures", "gated_failures",
              "extra_failures", "extra_fraction_vs_ca", "routed_fraction", "within_budget"]
    with THRESHOLD_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in THRESHOLDS:
            p = pooled[t]
            extra = p["gated"] - p["corr"]
            writer.writerow({
                "distance": DISTANCE, "p_base": P_BASE, "p_corr_screen_pooled": ";".join(map(str, P_CORR_VALUES)),
                "screen_shots": p["shots"], "threshold": t, "ordinary_failures": p["ordinary"],
                "correlation_aware_failures": p["corr"], "gated_failures": p["gated"],
                "extra_failures": extra, "extra_fraction_vs_ca": extra / p["corr"] if p["corr"] else float("nan"),
                "routed_fraction": p["routed"] / p["shots"], "within_budget": p["gated"] <= budget_failures,
            })
    print("\nSCREEN THRESHOLD CURVE (pooled over both fully-correlated conditions)")
    print(f"full correlation-aware failures={full_ca:,}/{total_shots:,}; permitted gated failures <= {budget_failures:.1f} ({SCREEN_NONINFERIORITY_BUDGET:.0%} excess budget)")
    print("threshold   ordinary_fail   full_CA_fail   gated_fail   extra_vs_CA   routed")
    for t in THRESHOLDS:
        p = pooled[t]
        print(f"{t:9} {p['ordinary']:15,} {p['corr']:14,} {p['gated']:12,} {p['gated']-p['corr']:+12,} {p['routed']/p['shots']:8.1%}")
    print(f"\nSELECTED frozen d=9 threshold: use correlation-aware iff detector weight >= {selected}")
    print(f"Selected screen extra failures={pooled[selected]['gated']-pooled[selected]['corr']:,} ({(pooled[selected]['gated']-pooled[selected]['corr'])/full_ca:.1%} of CA failures); routed={pooled[selected]['routed']/pooled[selected]['shots']:.1%}")
    print(f"Threshold screen: {SCREEN_CSV.resolve()}\nSelection table: {THRESHOLD_CSV.resolve()}")
    return selected


def validate(selected: int) -> None:
    replicate_fields = [
        "replicate", "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "gate_threshold",
        "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "gate_only_vs_ordinary", "ordinary_only_vs_gate", "gate_only_vs_corr", "corr_only_vs_gate",
        "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "shots_routed_to_corr", "fraction_routed_to_corr", "ordinary_decode_seconds",
        "correlation_aware_decode_seconds", "gated_decode_seconds", "speedup_vs_full_corr",
    ]
    # Sufficient counts allow correctly pooled paired uncertainty across replicates.
    agg: dict[tuple[float, str], dict[str, float | int]] = {}
    with REPLICATE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=replicate_fields)
        writer.writeheader()
        for p_index, p_corr in enumerate(P_CORR_VALUES):
            for model_index, (model_name, mode) in enumerate([
                ("matched-independent-control", "matched_independent"),
                ("fully-correlated", "correlated"),
            ]):
                k = (p_corr, model_name)
                a = agg.setdefault(k, {"shots": 0, "o": 0, "c": 0, "g": 0, "g_only_o": 0,
                                       "o_only_g": 0, "g_only_c": 0, "c_only_g": 0, "routed": 0,
                                       "o_sec": 0.0, "c_sec": 0.0, "g_sec": 0.0})
                for rep in range(VALIDATION_REPLICATES):
                    seed = SEED_BASE + 100_000 + rep * 1_000_000 + p_index * 100 + model_index
                    circuit, dem, ordinary, corr, pair_count = make_condition(p_corr, mode)
                    sampler = circuit.compile_detector_sampler(seed=seed)
                    n_shots = VALIDATION_SHOTS_PER_REPLICATE
                    counts = {name: 0 for name in ["o", "c", "g", "g_only_o", "o_only_g", "g_only_c", "c_only_g", "routed"]}
                    o_seconds = c_seconds = g_seconds = 0.0
                    processed = 0
                    print("\n" + "=" * 92)
                    print(f"VALIDATE d=9; p_corr={p_corr:g}; physical={model_name}; replicate={rep+1}/{VALIDATION_REPLICATES}; seed={seed}; shots={n_shots:,}; threshold={selected}")
                    while processed < n_shots:
                        batch = min(CHUNK_SIZE, n_shots - processed)
                        syndrome, actual = sampler.sample(shots=batch, separate_observables=True)
                        syndrome = np.asarray(syndrome, dtype=np.uint8)
                        actual = as_2d(actual)
                        weights = np.sum(syndrome, axis=1)

                        t0 = time.perf_counter()
                        pred_o = as_2d(ordinary.decode_batch(syndrome))
                        o_seconds += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                        c_seconds += time.perf_counter() - t0

                        fo = fail_mask(pred_o, actual)
                        fc = fail_mask(pred_c, actual)
                        high = weights >= selected
                        fg = np.where(high, fc, fo)
                        counts["o"] += int(fo.sum())
                        counts["c"] += int(fc.sum())
                        counts["g"] += int(fg.sum())
                        counts["g_only_o"] += int(np.count_nonzero(fg & ~fo))
                        counts["o_only_g"] += int(np.count_nonzero(fo & ~fg))
                        counts["g_only_c"] += int(np.count_nonzero(fg & ~fc))
                        counts["c_only_g"] += int(np.count_nonzero(fc & ~fg))
                        counts["routed"] += int(high.sum())

                        # Time the true gated call path separately, not the already-computed full predictions.
                        low = ~high
                        t0 = time.perf_counter()
                        if np.any(low):
                            _ = ordinary.decode_batch(syndrome[low])
                        if np.any(high):
                            _ = corr.decode_batch(syndrome[high], enable_correlations=True)
                        g_seconds += time.perf_counter() - t0
                        processed += batch
                        print_progress(processed, n_shots)

                    do, lo_o, hi_o = paired_delta_ci(counts["g_only_o"], counts["o_only_g"], n_shots)
                    dc, lo_c, hi_c = paired_delta_ci(counts["g_only_c"], counts["c_only_g"], n_shots)
                    routed_frac = counts["routed"] / n_shots
                    speedup = c_seconds / g_seconds if g_seconds else float("nan")
                    print(f"ordinary={counts['o']:,} ({counts['o']/n_shots*100:.5f}%) | full CA={counts['c']:,} ({counts['c']/n_shots*100:.5f}%) | gated={counts['g']:,} ({counts['g']/n_shots*100:.5f}%)")
                    print(f"gate-ordinary={do:+.6f} pp [{lo_o:+.6f},{hi_o:+.6f}]; gate-full-CA={dc:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}]; routed={routed_frac:.1%}; speedup={speedup:.2f}x")
                    writer.writerow({
                        "replicate": rep + 1, "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr,
                        "physical_model": model_name, "seed": seed, "shots": n_shots, "gate_threshold": selected,
                        "ordinary_failures": counts["o"], "correlation_aware_failures": counts["c"], "gated_failures": counts["g"],
                        "gate_only_vs_ordinary": counts["g_only_o"], "ordinary_only_vs_gate": counts["o_only_g"],
                        "gate_only_vs_corr": counts["g_only_c"], "corr_only_vs_gate": counts["c_only_g"],
                        "gate_vs_ordinary_delta_pp": do, "gate_vs_ordinary_ci_low_pp": lo_o, "gate_vs_ordinary_ci_high_pp": hi_o,
                        "gate_vs_corr_delta_pp": dc, "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                        "shots_routed_to_corr": counts["routed"], "fraction_routed_to_corr": routed_frac,
                        "ordinary_decode_seconds": o_seconds, "correlation_aware_decode_seconds": c_seconds,
                        "gated_decode_seconds": g_seconds, "speedup_vs_full_corr": speedup,
                    })
                    f.flush()
                    a["shots"] += n_shots
                    for src, dst in [("o", "o"), ("c", "c"), ("g", "g"), ("g_only_o", "g_only_o"),
                                     ("o_only_g", "o_only_g"), ("g_only_c", "g_only_c"), ("c_only_g", "c_only_g"), ("routed", "routed")]:
                        a[dst] += counts[src]
                    a["o_sec"] += o_seconds
                    a["c_sec"] += c_seconds
                    a["g_sec"] += g_seconds

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
        for (p_corr, model_name), a in sorted(agg.items()):
            n = int(a["shots"])
            do, lo_o, hi_o = paired_delta_ci(int(a["g_only_o"]), int(a["o_only_g"]), n)
            dc, lo_c, hi_c = paired_delta_ci(int(a["g_only_c"]), int(a["c_only_g"]), n)
            cs = float(a["c_sec"]); gs = float(a["g_sec"])
            row = {
                "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr, "physical_model": model_name,
                "replicates": VALIDATION_REPLICATES, "pooled_shots": n, "gate_threshold": selected,
                "ordinary_failures": a["o"], "correlation_aware_failures": a["c"], "gated_failures": a["g"],
                "gate_vs_ordinary_delta_pp": do, "gate_vs_ordinary_ci_low_pp": lo_o, "gate_vs_ordinary_ci_high_pp": hi_o,
                "gate_vs_corr_delta_pp": dc, "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                "fraction_routed_to_corr": int(a["routed"]) / n, "ordinary_decode_seconds": a["o_sec"],
                "correlation_aware_decode_seconds": cs, "gated_decode_seconds": gs,
                "speedup_vs_full_corr": cs / gs if gs else float("nan"),
            }
            writer.writerow(row)
            print("\nPOOLED d=9; p_corr={:.4g}; {}; n={:,}; threshold={}".format(p_corr, model_name, n, selected))
            print("  ordinary={:.5f}% | full CA={:.5f}% | gated={:.5f}%".format(a["o"] / n * 100, a["c"] / n * 100, a["g"] / n * 100))
            print("  gate-full-CA={:+.6f} pp [{:+.6f},{:+.6f}]; gate-ordinary={:+.6f} pp [{:+.6f},{:+.6f}]".format(dc, lo_c, hi_c, do, lo_o, hi_o))
            print("  routed={:.1%}; measured speedup={:.2f}x".format(int(a["routed"]) / n, cs / gs if gs else float("nan")))
    print(f"\nReplicates: {REPLICATE_CSV.resolve()}\nPooled summary: {SUMMARY_CSV.resolve()}")


def main() -> None:
    print("Distance-9 screen + frozen-threshold validation of weight-gated correlation-aware MWPM")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; screen shots/p_corr={SCREEN_SHOTS_PER_P:,}; validation={VALIDATION_REPLICATES} x {VALIDATION_SHOTS_PER_REPLICATE:,}/condition")
    print(f"candidate thresholds={THRESHOLDS}; screening risk budget={SCREEN_NONINFERIORITY_BUDGET:.0%}; seed base={SEED_BASE}; chunk={CHUNK_SIZE:,}")
    print("Threshold is selected only from fully-correlated screen samples and frozen before all validation samples.")
    selected = screen_threshold()
    validate(selected)
    print("\nCompleted d=9 weight-gate scaling test. Results are model- and parameter-specific; no threshold claim follows from this single distance.")


if __name__ == "__main__":
    main()
