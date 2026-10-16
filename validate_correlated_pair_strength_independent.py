"""Independent-seed validation of correlated-pair decoder-strength calibration.

Reads correlated_pair_dem_strength_calibration.csv from the screening/held-out
calibration experiment. It selects the best two multipliers using SCREEN rows
only, then evaluates those fixed choices on entirely new circuit seeds. The
true-strength model (scale=1.0) is the paired baseline in every condition.

Physical samples always contain all correlated pairs at the target p_corr.
Only the decoder DEM uses target_p_corr * scale.

Run from the project directory containing correlated_experiment.py and the
calibration CSV.
"""
from __future__ import annotations

import csv
import math
import re
import time
from collections import defaultdict
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

P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCES = (5, 7)
SHOTS = 1_000_000
CHUNK_SIZE = 25_000
SEED_BASE = 202_651_00
CALIBRATION_CSV = Path("correlated_pair_dem_strength_calibration.csv")
OUTPUT_CSV = Path("correlated_pair_dem_strength_independent_validation.csv")
ALPHA = 0.05


def as_2d(values: Any) -> np.ndarray:
    a = np.asarray(values, dtype=np.uint8)
    return a[:, None] if a.ndim == 1 else a


def failure_mask(predictions: Any, actual: Any) -> np.ndarray:
    p, y = as_2d(predictions), as_2d(actual)
    if p.shape != y.shape:
        raise ValueError(f"prediction/observable shape mismatch: {p.shape} vs {y.shape}")
    return np.any(p != y, axis=1)


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    errors = [lines[i] for i in positions]

    def key(line: str):
        m = re.match(r"\s*error\(([^)]+)\)", line)
        if not m:
            raise ValueError(f"could not parse DEM instruction: {line.strip()}")
        return (-float(m.group(1)), line.strip())

    for pos, line in zip(positions, sorted(errors, key=key)):
        lines[pos] = line
    sorted_dem = stim.DetectorErrorModel("\n".join(lines))
    if (sorted_dem.num_detectors, sorted_dem.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("canonical sorting changed DEM dimensions")
    return sorted_dem


def paired_stats(candidate_failures: int, baseline_failures: int,
                 candidate_only: int, baseline_only: int, n: int, z: float) -> dict[str, float | int]:
    delta_count = candidate_failures - baseline_failures
    delta_pp = 100.0 * delta_count / n
    discordant = candidate_only + baseline_only
    variance_mean = max(0.0, discordant / n - (delta_count / n) ** 2) / n
    se_pp = 100.0 * math.sqrt(variance_mean)
    return {
        "shots": n,
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
    }


def choose_scales_from_screen(path: Path) -> tuple[list[float], dict[float, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. First run calibrate_correlated_pair_dem_strength.py to create it."
        )
    scores: dict[float, int] = defaultdict(int)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    screen_rows = [r for r in rows if r.get("split") == "screen"]
    if not screen_rows:
        raise ValueError(f"No split=screen rows found in {path}")
    seen_conditions = {(int(r["distance"]), float(r["target_p_corr"])) for r in screen_rows}
    expected = {(d, p) for d in DISTANCES for p in P_CORR_VALUES}
    if not expected.issubset(seen_conditions):
        raise ValueError(f"Calibration CSV missing screen conditions: {sorted(expected - seen_conditions)}")
    for r in screen_rows:
        scores[float(r["decoder_scale"])] += int(r["candidate_failures"])
    ordered = sorted(scores, key=lambda scale: (scores[scale], scale))
    if len(ordered) < 2:
        raise ValueError("Need at least two distinct decoder scales in screen rows")
    return ordered[:2], dict(scores)


def build_decoder(base, data_qubits, pairs, model_p_corr: float, expected_dims: tuple[int, int]):
    model_circuit = add_noise_layer(base, data_qubits, pairs, model_p_corr, "correlated")
    dem = model_circuit.detector_error_model(decompose_errors=True).flattened()
    dims = (dem.num_detectors, dem.num_observables)
    if dims != expected_dims:
        raise ValueError(f"decoder DEM dimensions changed: expected {expected_dims}, got {dims}")
    n_errors = sum(line.lstrip().startswith("error(") for line in str(dem).splitlines())
    return pymatching.Matching(canonical_probability_sorted_dem(dem), enable_correlations=True), n_errors


def main() -> None:
    top_two, screen_scores = choose_scales_from_screen(CALIBRATION_CSV)
    # Scale=1.0 is the baseline, so only evaluate non-baseline candidates.
    candidate_scales = [s for s in top_two if not math.isclose(s, 1.0)]
    unique_scales = [1.0] + candidate_scales
    n_comparisons = max(1, len(candidate_scales) * len(DISTANCES) * len(P_CORR_VALUES))
    z_bonf = NormalDist().inv_cdf(1 - ALPHA / (2 * n_comparisons))

    print("Independent-seed validation of correlated-pair decoder strength")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; shots/condition={SHOTS:,}")
    print(f"New seed base={SEED_BASE}; input={CALIBRATION_CSV}; output={OUTPUT_CSV}")
    print("Top two scales selected using SCREEN rows only:")
    for rank, scale in enumerate(top_two, 1):
        print(f"  rank {rank}: scale={scale:g}; aggregated screen failures={screen_scores[scale]:,}")
    print(f"Fresh-sample comparisons: baseline scale=1.0 vs {candidate_scales or 'no non-baseline scale selected'}")
    print("Physical circuits always use the full target p_corr; only decoder DEM probabilities are scaled.")

    setups = {}
    for d in DISTANCES:
        base = make_base_circuit(distance=d, rounds=d, p=P_BASE)
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)
        setups[d] = (base, data_qubits, pairs)

    fields = [
        "distance", "p_base", "target_p_corr", "decoder_scale", "decoder_model_p_corr", "seed",
        "shots", "dem_error_instructions", "baseline_failures", "candidate_failures",
        "baseline_rate_pct", "candidate_rate_pct", "delta_vs_true_model_pp", "ci95_low_pp",
        "ci95_high_pp", "bonferroni_low_pp", "bonferroni_high_pp", "candidate_only_failures",
        "true_model_only_failures", "changed_predictions",
    ]
    overall = {s: {k: 0 for k in ("baseline_failures", "candidate_failures", "candidate_only", "baseline_only", "shots")} for s in candidate_scales}

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for d in DISTANCES:
            base, data_qubits, pairs = setups[d]
            for pc_idx, target_pc in enumerate(P_CORR_VALUES):
                seed = int(SEED_BASE + d * 100 + pc_idx)
                physical = add_noise_layer(base, data_qubits, pairs, target_pc, "correlated")
                physical_dem = physical.detector_error_model(decompose_errors=True).flattened()
                dims = (physical_dem.num_detectors, physical_dem.num_observables)
                baseline_nerr = sum(line.lstrip().startswith("error(") for line in str(physical_dem).splitlines())
                baseline = pymatching.Matching(canonical_probability_sorted_dem(physical_dem), enable_correlations=True)
                decoders = {1.0: baseline}
                nerr = {1.0: baseline_nerr}
                for scale in candidate_scales:
                    decoders[scale], nerr[scale] = build_decoder(base, data_qubits, pairs, target_pc * scale, dims)

                print("\n" + "=" * 88)
                print(f"d={d}; target p_corr={target_pc:g}; seed={seed}; detectors/observables={dims}; models={unique_scales}")
                stats = {s: {"candidate_failures": 0, "candidate_only": 0, "baseline_only": 0} for s in candidate_scales}
                baseline_failures = 0
                processed = 0
                sampler = physical.compile_detector_sampler(seed=seed)
                start = time.perf_counter()
                while processed < SHOTS:
                    n = min(CHUNK_SIZE, SHOTS - processed)
                    syndrome, actual = sampler.sample(shots=n, separate_observables=True)
                    syndrome = np.asarray(syndrome, dtype=np.uint8)
                    actual2d = as_2d(actual)
                    base_pred = as_2d(baseline.decode_batch(syndrome, enable_correlations=True))
                    base_fail = failure_mask(base_pred, actual2d)
                    baseline_failures += int(np.count_nonzero(base_fail))
                    for scale in candidate_scales:
                        pred = as_2d(decoders[scale].decode_batch(syndrome, enable_correlations=True))
                        fail = failure_mask(pred, actual2d)
                        stats[scale]["candidate_failures"] += int(np.count_nonzero(fail))
                        stats[scale]["candidate_only"] += int(np.count_nonzero(fail & ~base_fail))
                        stats[scale]["baseline_only"] += int(np.count_nonzero(base_fail & ~fail))
                    processed += n
                    if processed % (CHUNK_SIZE * 4) == 0 or processed == SHOTS:
                        print(f"  sampled/decoded {processed:,}/{SHOTS:,}", flush=True)

                for scale in candidate_scales:
                    st = stats[scale]
                    summary = paired_stats(st["candidate_failures"], baseline_failures,
                                            st["candidate_only"], st["baseline_only"], SHOTS, z_bonf)
                    row = {
                        "distance": d, "p_base": P_BASE, "target_p_corr": target_pc,
                        "decoder_scale": scale, "decoder_model_p_corr": target_pc * scale,
                        "seed": seed, "dem_error_instructions": nerr[scale], **summary,
                    }
                    writer.writerow(row)
                    f.flush()
                    agg = overall[scale]
                    agg["baseline_failures"] += baseline_failures
                    agg["candidate_failures"] += st["candidate_failures"]
                    agg["candidate_only"] += st["candidate_only"]
                    agg["baseline_only"] += st["baseline_only"]
                    agg["shots"] += SHOTS
                    print(f"  scale={scale:g}: baseline={100*baseline_failures/SHOTS:.5f}%; "
                          f"candidate={100*st['candidate_failures']/SHOTS:.5f}%; "
                          f"delta={summary['delta_vs_true_model_pp']:+.6f} pp "
                          f"95% CI=[{summary['ci95_low_pp']:+.6f},{summary['ci95_high_pp']:+.6f}] pp")
                print(f"Condition elapsed={time.perf_counter()-start:.1f}s; checkpointed {OUTPUT_CSV}")

    if candidate_scales:
        print("\nPOOLED INDEPENDENT-SEED RESULTS (equal number of shots per condition)")
        for scale in candidate_scales:
            a = overall[scale]
            summary = paired_stats(a["candidate_failures"], a["baseline_failures"],
                                   a["candidate_only"], a["baseline_only"], a["shots"], z_bonf)
            print(f"  scale={scale:g}; candidate={summary['candidate_rate_pct']:.6f}%; "
                  f"baseline={summary['baseline_rate_pct']:.6f}%; "
                  f"delta={summary['delta_vs_true_model_pp']:+.6f} pp; "
                  f"nominal CI=[{summary['ci95_low_pp']:+.6f},{summary['ci95_high_pp']:+.6f}] pp; "
                  f"Bonferroni CI=[{summary['bonferroni_low_pp']:+.6f},{summary['bonferroni_high_pp']:+.6f}] pp")
    else:
        print("\nThe screen-selected top two are both scale=1.0 (or no alternative was available); no non-baseline confirmation was needed.")
    print(f"\nCompleted. Independent validation results: {OUTPUT_CSV.resolve()}")
    print("This fresh-seed run is the confirmatory check for the top screen-selected non-baseline candidate(s).")


if __name__ == "__main__":
    main()
