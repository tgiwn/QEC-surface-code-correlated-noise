"""Calibrate the strength assigned to correlated-pair mechanisms in the decoder DEM.

The physical circuit ALWAYS uses the target p_corr with ALL correlated pairs.
For each target condition, candidate decoder DEMs use model_p_corr = scale * target_p_corr,
with the base noise P_BASE unchanged. This is a decoder-model calibration experiment,
not a change to the sampled physical noise.

Tuning/validation protocol:
  * Each condition uses one fixed fresh circuit sample of SHOTS shots.
  * First half is a screening split used to select one global multiplier.
  * Second half is held out and used to evaluate that selected multiplier.
  * All decoder models see identical detector/observable samples within each condition.

Run from the project directory containing correlated_experiment.py.
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from statistics import NormalDist
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

# ----------------------------- Settings -----------------------------------
P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCES = (5, 7)
SHOTS = 1_000_000  # exactly half screen / half held-out
CHUNK_SIZE = 25_000
SCALE_FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
SEED_BASE = 202_641_00
OUTPUT_CSV = Path("correlated_pair_dem_strength_calibration.csv")
ALPHA = 0.05


def as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def failure_mask(predictions: Any, actual: Any) -> np.ndarray:
    pred, obs = as_2d(predictions), as_2d(actual)
    if pred.shape != obs.shape:
        raise ValueError(f"Prediction/observable shape mismatch: {pred.shape} vs {obs.shape}")
    return np.any(pred != obs, axis=1)


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort complete flattened error instructions by probability, then by full line text."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    errors = [lines[i] for i in positions]

    def key(line: str):
        match = re.match(r"\s*error\(([^)]+)\)", line)
        if not match:
            raise ValueError(f"Could not parse DEM instruction: {line.strip()}")
        return (-float(match.group(1)), line.strip())

    for position, line in zip(positions, sorted(errors, key=key)):
        lines[position] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Canonical sorting changed DEM dimensions")
    return result


def paired_stats(candidate_failures: int, baseline_failures: int,
                 candidate_only: int, baseline_only: int, n: int, z: float) -> dict[str, float]:
    delta_count = candidate_failures - baseline_failures
    delta_pp = 100.0 * delta_count / n
    discordant = candidate_only + baseline_only
    # Variance of the paired per-shot difference in { -1, 0, +1 }.
    variance_mean = max(0.0, discordant / n - (delta_count / n) ** 2) / n
    se_pp = 100.0 * math.sqrt(variance_mean)
    return {
        "baseline_failures": baseline_failures,
        "candidate_failures": candidate_failures,
        "baseline_rate_pct": 100.0 * baseline_failures / n,
        "candidate_rate_pct": 100.0 * candidate_failures / n,
        "delta_vs_true_model_pp": delta_pp,
        "ci95_low_pp": delta_pp - 1.959963984540054 * se_pp,
        "ci95_high_pp": delta_pp + 1.959963984540054 * se_pp,
        "bonferroni_low_pp": delta_pp - z * se_pp,
        "bonferroni_high_pp": delta_pp + z * se_pp,
        "candidate_only_failures": candidate_only,
        "true_model_only_failures": baseline_only,
        "changed_predictions": candidate_only + baseline_only,
        "shots": n,
    }


def build_decoder(base, data_qubits, pairs, model_p_corr: float, expected_dims: tuple[int, int]):
    model_circuit = add_noise_layer(base, data_qubits, pairs, model_p_corr, "correlated")
    dem = model_circuit.detector_error_model(decompose_errors=True).flattened()
    dims = (dem.num_detectors, dem.num_observables)
    if dims != expected_dims:
        raise ValueError(f"Decoder DEM dimensions changed: expected {expected_dims}, got {dims}")
    decoder = pymatching.Matching(canonical_probability_sorted_dem(dem), enable_correlations=True)
    n_err = sum(line.lstrip().startswith("error(") for line in str(dem).splitlines())
    return decoder, dem, n_err


def main() -> None:
    if SHOTS % 2:
        raise ValueError("SHOTS must be even for the pre-registered 50/50 screen/holdout split")

    n_scales = len(SCALE_FACTORS)
    n_nonbaseline = n_scales - 1
    total_familywise_tests = len(DISTANCES) * len(P_CORR_VALUES) * n_nonbaseline
    z_bonf = NormalDist().inv_cdf(1 - ALPHA / (2 * total_familywise_tests))
    half = SHOTS // 2

    print("Correlated-pair decoder DEM-strength calibration")
    print(f"P_BASE={P_BASE}; target p_corr={P_CORR_VALUES}; distances={DISTANCES}")
    print(f"shots/condition={SHOTS:,}; split={half:,} screen + {half:,} held-out; seed base={SEED_BASE}")
    print(f"decoder multiplier candidates={SCALE_FACTORS}")
    print("Physical circuits always use ALL correlated pairs at the target p_corr.")
    print("Only the decoder DEM's correlated-pair probability is scaled; P_BASE is unchanged.")
    print("One global multiplier is selected from the combined screening halves and tested on held-out halves.")
    print(f"Nominal family-wise comparison count={total_familywise_tests}; Bonferroni z={z_bonf:.3f}")

    setups = {}
    for d in DISTANCES:
        base = make_base_circuit(distance=d, rounds=d, p=P_BASE)
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)
        setups[d] = (base, data_qubits, pairs)
        print(f"distance={d}: data qubits={len(data_qubits)}; correlated pairs={len(pairs)}")

    fieldnames = [
        "distance", "p_base", "target_p_corr", "decoder_scale", "decoder_model_p_corr", "seed",
        "split", "shots", "dem_error_instructions", "baseline_failures", "candidate_failures",
        "baseline_rate_pct", "candidate_rate_pct", "delta_vs_true_model_pp", "ci95_low_pp",
        "ci95_high_pp", "bonferroni_low_pp", "bonferroni_high_pp", "candidate_only_failures",
        "true_model_only_failures", "changed_predictions",
    ]
    screen_totals = {scale: 0 for scale in SCALE_FACTORS}
    baseline_screen_total = 0
    heldout_combined = {scale: {k: 0 for k in ("baseline_failures", "candidate_failures", "candidate_only", "baseline_only", "shots")} for scale in SCALE_FACTORS}

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for d in DISTANCES:
            base, data_qubits, pairs = setups[d]
            for pc_idx, target_pc in enumerate(P_CORR_VALUES):
                seed = int(SEED_BASE + d * 100 + pc_idx)
                t0 = time.perf_counter()
                physical_circuit = add_noise_layer(base, data_qubits, pairs, target_pc, "correlated")
                physical_dem = physical_circuit.detector_error_model(decompose_errors=True).flattened()
                expected_dims = (physical_dem.num_detectors, physical_dem.num_observables)
                baseline_decoder = pymatching.Matching(
                    canonical_probability_sorted_dem(physical_dem), enable_correlations=True
                )
                baseline_instruction_count = sum(
                    line.lstrip().startswith("error(") for line in str(physical_dem).splitlines()
                )

                decoders = {}
                instruction_counts = {}
                for scale in SCALE_FACTORS:
                    model_pc = target_pc * scale
                    if math.isclose(scale, 1.0):
                        decoders[scale] = baseline_decoder
                        instruction_counts[scale] = baseline_instruction_count
                    else:
                        decoder, _dem, n_err = build_decoder(base, data_qubits, pairs, model_pc, expected_dims)
                        decoders[scale] = decoder
                        instruction_counts[scale] = n_err

                print("\n" + "=" * 96)
                print(f"d={d}; target p_corr={target_pc:g}; seed={seed}; shots={SHOTS:,}; DEM dimensions={expected_dims}")
                print(f"physical/true-model DEM error instructions={baseline_instruction_count}; decoder models={len(decoders)}")

                # Aggregates for each split and multiplier.
                stats = {
                    split: {
                        scale: {"candidate_failures": 0, "candidate_only": 0, "baseline_only": 0, "shots": 0}
                        for scale in SCALE_FACTORS
                    }
                    for split in ("screen", "heldout")
                }
                baseline_fail_counts = {"screen": 0, "heldout": 0}

                sampler = physical_circuit.compile_detector_sampler(seed=seed)
                processed = 0
                while processed < SHOTS:
                    n = min(CHUNK_SIZE, SHOTS - processed)
                    syndrome, actual = sampler.sample(shots=n, separate_observables=True)
                    syndrome = np.asarray(syndrome, dtype=np.uint8)
                    actual2d = as_2d(actual)
                    base_pred = as_2d(baseline_decoder.decode_batch(syndrome, enable_correlations=True))
                    base_fail = failure_mask(base_pred, actual2d)
                    split_key = "screen" if processed < half else "heldout"
                    # SHOTS and CHUNK_SIZE are divisible to ensure no chunk crosses the split boundary.
                    if processed < half < processed + n:
                        raise ValueError("A sampler chunk crossed the screen/holdout boundary")
                    baseline_fail_counts[split_key] += int(np.count_nonzero(base_fail))
                    stats[split_key][1.0]["candidate_failures"] += int(np.count_nonzero(base_fail))
                    stats[split_key][1.0]["shots"] += n
                    for scale, decoder in decoders.items():
                        if math.isclose(scale, 1.0):
                            continue
                        pred = as_2d(decoder.decode_batch(syndrome, enable_correlations=True))
                        fail = failure_mask(pred, actual2d)
                        stats[split_key][scale]["candidate_failures"] += int(np.count_nonzero(fail))
                        stats[split_key][scale]["candidate_only"] += int(np.count_nonzero(fail & ~base_fail))
                        stats[split_key][scale]["baseline_only"] += int(np.count_nonzero(base_fail & ~fail))
                        stats[split_key][scale]["shots"] += n
                    processed += n
                    if processed % (CHUNK_SIZE * 4) == 0 or processed == SHOTS:
                        print(f"  sampled/decoded {processed:,}/{SHOTS:,}", flush=True)

                for split_key in ("screen", "heldout"):
                    baseline_n = stats[split_key][1.0]["shots"]
                    baseline_count = baseline_fail_counts[split_key]
                    for scale in SCALE_FACTORS:
                        st = stats[split_key][scale]
                        if scale == 1.0:
                            candidate_only = baseline_only = 0
                        else:
                            candidate_only = st["candidate_only"]
                            baseline_only = st["baseline_only"]
                        summary = paired_stats(
                            st["candidate_failures"], baseline_count,
                            candidate_only, baseline_only, baseline_n, z_bonf
                        )
                        row = {
                            "distance": d, "p_base": P_BASE, "target_p_corr": target_pc,
                            "decoder_scale": scale, "decoder_model_p_corr": target_pc * scale,
                            "seed": seed, "split": split_key, "shots": baseline_n,
                            "dem_error_instructions": instruction_counts[scale], **summary,
                        }
                        writer.writerow(row)
                        f.flush()

                        if split_key == "screen":
                            screen_totals[scale] += st["candidate_failures"]
                            if math.isclose(scale, 1.0):
                                baseline_screen_total += st["candidate_failures"]
                        else:
                            agg = heldout_combined[scale]
                            agg["baseline_failures"] += baseline_count
                            agg["candidate_failures"] += st["candidate_failures"]
                            agg["candidate_only"] += candidate_only
                            agg["baseline_only"] += baseline_only
                            agg["shots"] += baseline_n

                screen_best = min(SCALE_FACTORS, key=lambda s: screen_totals[s])
                print(f"Condition elapsed={time.perf_counter() - t0:.1f}s; global screening leader so far: scale={screen_best:g}")

    # One global multiplier selected using aggregate screening failures over all four conditions.
    selected_scale = min(SCALE_FACTORS, key=lambda s: screen_totals[s])
    true_scale = 1.0
    print("\n" + "=" * 96)
    print("GLOBAL SCREENING RESULT")
    print(f"Aggregate screen failures at multiplier 1.0 (true decoder model): {screen_totals[true_scale]:,}")
    print("Aggregate screening failures by multiplier:")
    for scale in SCALE_FACTORS:
        tag = "  <-- selected globally" if scale == selected_scale else ""
        print(f"  scale={scale:>4g}: {screen_totals[scale]:,}{tag}")
    print("Selection is based only on the first half of each condition.")

    print("\nHELD-OUT VALIDATION OF THE SCREEN-SELECTED GLOBAL MULTIPLIER")
    pooled = heldout_combined[selected_scale]
    pooled_stats = paired_stats(
        pooled["candidate_failures"], pooled["baseline_failures"],
        pooled["candidate_only"], pooled["baseline_only"], pooled["shots"], z_bonf,
    )
    print(f"selected scale={selected_scale:g}; pooled shots={pooled['shots']:,}")
    print(f"held-out baseline failures={pooled['baseline_failures']:,}; candidate failures={pooled['candidate_failures']:,}")
    print(f"pooled delta={pooled_stats['delta_vs_true_model_pp']:+.6f} pp; nominal 95% CI="
          f"[{pooled_stats['ci95_low_pp']:+.6f}, {pooled_stats['ci95_high_pp']:+.6f}] pp; "
          f"Bonferroni CI=[{pooled_stats['bonferroni_low_pp']:+.6f}, {pooled_stats['bonferroni_high_pp']:+.6f}] pp")
    print("Per-condition held-out results for the selected global multiplier:")
    # Recover selected-scale held-out rows from CSV for exact condition-level reporting.
    with OUTPUT_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "heldout" and math.isclose(float(row["decoder_scale"]), selected_scale):
                delta = float(row["delta_vs_true_model_pp"])
                low = float(row["ci95_low_pp"])
                high = float(row["ci95_high_pp"])
                print(f"  d={row['distance']} target p_corr={float(row['target_p_corr']):g}: "
                      f"{row['baseline_rate_pct']}% -> {row['candidate_rate_pct']}%; "
                      f"delta={delta:+.6f} pp [{low:+.6f}, {high:+.6f}]")

    print(f"\nCompleted. Full split-by-scale results: {OUTPUT_CSV.resolve()}")
    print("Interpretation: negative delta means the calibrated DEM reduced logical failures vs the true-strength DEM.")
    print("Held-out evaluation of the single screening-selected scale is the primary result; individual scale CIs are exploratory.")


if __name__ == "__main__":
    main()
