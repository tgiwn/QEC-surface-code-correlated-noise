"""Compare physical-circuit and flattened-DEM sampling at distances 5 and 7.

The circuit sampler and DEM sampler are independent sources from the same
nominal noise configuration. This tests whether the generated flattened DEM
reproduces detector/observable diagnostics and decoder failure rates at larger
distances; it does not prove equality of the full distributions.

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
SHOTS = 1_000_000
CHUNK_SIZE = 25_000
SEED_BASE = 202_662_00
SUMMARY_CSV = Path("dem_fidelity_validation_summary.csv")
MARGINALS_CSV = Path("dem_fidelity_detector_marginals.csv")
ALPHA = 0.05


def as_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort whole flattened error instructions by descending p and text.

    Complete error instructions are moved intact; terms separated by '^' are
    never split into independent instructions.
    """
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    errors = [lines[i] for i in positions]

    def key(line: str):
        match = re.match(r"\s*error\(([^)]+)\)", line)
        if not match:
            raise ValueError(f"Could not parse DEM instruction: {line.strip()}")
        return (-float(match.group(1)), line.strip())

    for pos, line in zip(positions, sorted(errors, key=key)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Canonical sorting changed DEM dimensions")
    return result


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def diff_ci_pp(a_count: int, b_count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Unpaired a-minus-b difference and normal CI, in percentage points."""
    pa, pb = a_count / n, b_count / n
    delta = 100.0 * (pa - pb)
    se = 100.0 * math.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
    return delta, delta - z * se, delta + z * se


def rate_pct(count: int, n: int) -> float:
    return 100.0 * count / n


def sample_chunk_circuit(sampler, n: int) -> tuple[np.ndarray, np.ndarray]:
    det, obs = sampler.sample(shots=n, separate_observables=True)
    return np.asarray(det, dtype=np.uint8), as_2d(obs)


def sample_chunk_dem(sampler, n: int) -> tuple[np.ndarray, np.ndarray]:
    result = sampler.sample(shots=n)
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("Unexpected return value from Stim DEM sampler")
    return np.asarray(result[0], dtype=np.uint8), as_2d(result[1])


def weight_bin_counts(det: np.ndarray) -> np.ndarray:
    weights = np.count_nonzero(det, axis=1)
    # Mutually exclusive, exhaustive bins.
    ids = np.select(
        [weights == 0, weights == 1, weights == 2, (weights >= 3) & (weights <= 5),
         (weights >= 6) & (weights <= 10), weights >= 11],
        [0, 1, 2, 3, 4, 5],
        default=5,
    )
    return np.bincount(ids, minlength=6).astype(np.int64)


def process_sample(
    det: np.ndarray,
    obs: np.ndarray,
    ordinary: pymatching.Matching,
    corr: pymatching.Matching,
) -> dict[str, Any]:
    ordinary_pred = as_2d(ordinary.decode_batch(det))
    corr_pred = as_2d(corr.decode_batch(det, enable_correlations=True))
    actual = as_2d(obs)
    return {
        "detector_counts": np.count_nonzero(det, axis=0).astype(np.int64),
        "observable_count": int(np.count_nonzero(actual[:, 0])),
        "weight_bins": weight_bin_counts(det),
        "ordinary_failures": int(np.count_nonzero(np.any(ordinary_pred != actual, axis=1))),
        "correlation_aware_failures": int(np.count_nonzero(np.any(corr_pred != actual, axis=1))),
        "shots": int(det.shape[0]),
    }


def main() -> None:
    print("Physical-circuit vs flattened-DEM fidelity check at distances 5 and 7")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; shots/source/condition={SHOTS:,}")
    print(f"chunk={CHUNK_SIZE:,}; seed base={SEED_BASE}")
    print("Circuit and DEM samplers use independent seeds; rate differences are unpaired.")
    print("Complete error instructions are kept intact when sorting the decoder DEM.")

    summary_fields = [
        "distance", "p_base", "p_corr", "seed_circuit", "seed_dem", "shots_per_source",
        "n_detectors", "n_observables", "dem_error_instructions", "dem_normalization_note",
        "circuit_observable_rate_pct", "dem_observable_rate_pct", "observable_delta_circuit_minus_dem_pp",
        "observable_ci95_low_pp", "observable_ci95_high_pp", "circuit_mean_detector_weight",
        "dem_mean_detector_weight", "weight_bin_deltas_pp", "mean_abs_detector_marginal_delta_pp",
        "max_abs_detector_marginal_delta_pp", "max_detector_marginal_z", "bonferroni_detector_z_threshold",
        "num_detector_marginals_beyond_bonferroni", "ordinary_circuit_failures", "ordinary_dem_failures",
        "ordinary_circuit_rate_pct", "ordinary_dem_rate_pct", "ordinary_delta_circuit_minus_dem_pp",
        "ordinary_ci95_low_pp", "ordinary_ci95_high_pp", "corr_circuit_failures", "corr_dem_failures",
        "corr_circuit_rate_pct", "corr_dem_rate_pct", "corr_delta_circuit_minus_dem_pp",
        "corr_ci95_low_pp", "corr_ci95_high_pp",
    ]
    marginal_fields = [
        "distance", "p_base", "p_corr", "detector_id", "circuit_fires", "dem_fires",
        "circuit_rate_pct", "dem_rate_pct", "delta_circuit_minus_dem_pp", "z_score",
        "bonferroni_marginal_flag",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fs, MARGINALS_CSV.open("w", newline="", encoding="utf-8") as fm:
        sw = csv.DictWriter(fs, fieldnames=summary_fields)
        mw = csv.DictWriter(fm, fieldnames=marginal_fields)
        sw.writeheader(); mw.writeheader(); fs.flush(); fm.flush()

        for d in DISTANCES:
            base = make_base_circuit(distance=d, rounds=d, p=P_BASE)
            data_by_xy, data_qubits = get_data_layout(base)
            pairs = make_correlated_pairs(data_by_xy)

            for pc_idx, pc in enumerate(P_CORR_VALUES):
                seed_circuit = int(SEED_BASE + d * 100 + pc_idx * 2)
                seed_dem = seed_circuit + 1
                t0 = time.perf_counter()
                circuit = add_noise_layer(base, data_qubits, pairs, pc, "correlated")
                physical_dem = circuit.detector_error_model(decompose_errors=True).flattened()
                sorted_dem = canonical_probability_sorted_dem(physical_dem)
                dims = (physical_dem.num_detectors, physical_dem.num_observables)
                ordinary = pymatching.Matching(sorted_dem)
                corr = pymatching.Matching(sorted_dem, enable_correlations=True)

                circuit_sampler = circuit.compile_detector_sampler(seed=seed_circuit)
                dem_sampler = physical_dem.compile_sampler(seed=seed_dem)
                nc = nd = 0
                c_det_counts = np.zeros(physical_dem.num_detectors, dtype=np.int64)
                d_det_counts = np.zeros(physical_dem.num_detectors, dtype=np.int64)
                c_obs = d_obs = 0
                c_wbins = np.zeros(6, dtype=np.int64)
                d_wbins = np.zeros(6, dtype=np.int64)
                c_ord = d_ord = c_corr = d_corr = 0
                c_weight_sum = d_weight_sum = 0

                print("\n" + "=" * 92)
                n_err = sum(line.lstrip().startswith("error(") for line in str(physical_dem).splitlines())
                print(f"d={d}; p_corr={pc:g}; dims={dims}; pairs={len(pairs)}; DEM error instructions={n_err}")
                print(f"circuit seed={seed_circuit}; DEM seed={seed_dem}; sampling {SHOTS:,} from each source")

                while nc < SHOTS or nd < SHOTS:
                    if nc < SHOTS:
                        n = min(CHUNK_SIZE, SHOTS - nc)
                        det, obs = sample_chunk_circuit(circuit_sampler, n)
                        met = process_sample(det, obs, ordinary, corr)
                        c_det_counts += met["detector_counts"]
                        c_obs += met["observable_count"]
                        c_wbins += met["weight_bins"]
                        c_ord += met["ordinary_failures"]
                        c_corr += met["correlation_aware_failures"]
                        c_weight_sum += int(np.count_nonzero(det, axis=1).sum())
                        nc += n
                    if nd < SHOTS:
                        n = min(CHUNK_SIZE, SHOTS - nd)
                        det, obs = sample_chunk_dem(dem_sampler, n)
                        met = process_sample(det, obs, ordinary, corr)
                        d_det_counts += met["detector_counts"]
                        d_obs += met["observable_count"]
                        d_wbins += met["weight_bins"]
                        d_ord += met["ordinary_failures"]
                        d_corr += met["correlation_aware_failures"]
                        d_weight_sum += int(np.count_nonzero(det, axis=1).sum())
                        nd += n
                    if nc % 200_000 == 0 or nc == SHOTS:
                        print(f"  processed circuit={nc:,}/{SHOTS:,}; DEM={nd:,}/{SHOTS:,}")

                # Detector marginal comparison, with Bonferroni familywise control.
                cp = c_det_counts / SHOTS
                dp = d_det_counts / SHOTS
                se = np.sqrt(cp * (1 - cp) / SHOTS + dp * (1 - dp) / SHOTS)
                z_scores = np.divide(np.abs(cp - dp), se, out=np.zeros_like(cp, dtype=float), where=se > 0)
                z_bonf = NormalDist().inv_cdf(1 - ALPHA / (2 * physical_dem.num_detectors))
                abs_delta_pp = np.abs(cp - dp) * 100
                marg_sig = z_scores > z_bonf
                for det_id in range(physical_dem.num_detectors):
                    delta = float(100 * (cp[det_id] - dp[det_id]))
                    mw.writerow({
                        "distance": d, "p_base": P_BASE, "p_corr": pc, "detector_id": det_id,
                        "circuit_fires": int(c_det_counts[det_id]), "dem_fires": int(d_det_counts[det_id]),
                        "circuit_rate_pct": rate_pct(int(c_det_counts[det_id]), SHOTS),
                        "dem_rate_pct": rate_pct(int(d_det_counts[det_id]), SHOTS),
                        "delta_circuit_minus_dem_pp": delta, "z_score": float(z_scores[det_id]),
                        "bonferroni_marginal_flag": bool(marg_sig[det_id]),
                    })
                fm.flush()

                obs_delta, obs_lo, obs_hi = diff_ci_pp(c_obs, d_obs, SHOTS)
                ord_delta, ord_lo, ord_hi = diff_ci_pp(c_ord, d_ord, SHOTS)
                corr_delta, corr_lo, corr_hi = diff_ci_pp(c_corr, d_corr, SHOTS)
                c_mean_w, d_mean_w = c_weight_sum / SHOTS, d_weight_sum / SHOTS
                bin_names = ("w0", "w1", "w2", "w3_5", "w6_10", "w11plus")
                bin_delta_strings = []
                for idx, label in enumerate(bin_names):
                    delta, lo, hi = diff_ci_pp(int(c_wbins[idx]), int(d_wbins[idx]), SHOTS)
                    bin_delta_strings.append(f"{label}:{delta:+.6f}[{lo:+.6f},{hi:+.6f}]")

                summary = {
                    "distance": d, "p_base": P_BASE, "p_corr": pc, "seed_circuit": seed_circuit,
                    "seed_dem": seed_dem, "shots_per_source": SHOTS, "n_detectors": dims[0],
                    "n_observables": dims[1], "dem_error_instructions": n_err,
                    "dem_normalization_note": "empirically sampled; normalization not directly computed",
                    "circuit_observable_rate_pct": rate_pct(c_obs, SHOTS),
                    "dem_observable_rate_pct": rate_pct(d_obs, SHOTS),
                    "observable_delta_circuit_minus_dem_pp": obs_delta,
                    "observable_ci95_low_pp": obs_lo, "observable_ci95_high_pp": obs_hi,
                    "circuit_mean_detector_weight": c_mean_w, "dem_mean_detector_weight": d_mean_w,
                    "weight_bin_deltas_pp": ";".join(bin_delta_strings),
                    "mean_abs_detector_marginal_delta_pp": float(np.mean(abs_delta_pp)),
                    "max_abs_detector_marginal_delta_pp": float(np.max(abs_delta_pp)),
                    "max_detector_marginal_z": float(np.max(z_scores)),
                    "bonferroni_detector_z_threshold": z_bonf,
                    "num_detector_marginals_beyond_bonferroni": int(np.count_nonzero(marg_sig)),
                    "ordinary_circuit_failures": c_ord, "ordinary_dem_failures": d_ord,
                    "ordinary_circuit_rate_pct": rate_pct(c_ord, SHOTS),
                    "ordinary_dem_rate_pct": rate_pct(d_ord, SHOTS),
                    "ordinary_delta_circuit_minus_dem_pp": ord_delta,
                    "ordinary_ci95_low_pp": ord_lo, "ordinary_ci95_high_pp": ord_hi,
                    "corr_circuit_failures": c_corr, "corr_dem_failures": d_corr,
                    "corr_circuit_rate_pct": rate_pct(c_corr, SHOTS),
                    "corr_dem_rate_pct": rate_pct(d_corr, SHOTS),
                    "corr_delta_circuit_minus_dem_pp": corr_delta,
                    "corr_ci95_low_pp": corr_lo, "corr_ci95_high_pp": corr_hi,
                }
                sw.writerow(summary); fs.flush()

                print(f"P(L0=1): circuit={rate_pct(c_obs, SHOTS):.6f}%; DEM={rate_pct(d_obs, SHOTS):.6f}%; "
                      f"delta={obs_delta:+.6f} pp [{obs_lo:+.6f},{obs_hi:+.6f}]")
                print(f"Mean detector marginal |delta|={np.mean(abs_delta_pp):.6f} pp; max={np.max(abs_delta_pp):.6f} pp; "
                      f"Bonferroni outliers={np.count_nonzero(marg_sig)}/{dims[0]}; max |z|={np.max(z_scores):.2f} (threshold {z_bonf:.2f})")
                print("Syndrome-weight bin circuit-minus-DEM differences (pp, 95% CI):")
                for item in bin_delta_strings:
                    print("  " + item)
                print(f"ordinary/sorted MWPM: circuit={rate_pct(c_ord, SHOTS):.6f}%; DEM={rate_pct(d_ord, SHOTS):.6f}%; "
                      f"delta={ord_delta:+.6f} pp [{ord_lo:+.6f},{ord_hi:+.6f}]")
                print(f"correlation-aware/sorted: circuit={rate_pct(c_corr, SHOTS):.6f}%; DEM={rate_pct(d_corr, SHOTS):.6f}%; "
                      f"delta={corr_delta:+.6f} pp [{corr_lo:+.6f},{corr_hi:+.6f}]")
                print(f"Condition elapsed={time.perf_counter() - t0:.1f}s; checkpointed outputs")

    print("\nCompleted.")
    print(f"Summary: {SUMMARY_CSV.resolve()}")
    print(f"Detector marginals: {MARGINALS_CSV.resolve()}")
    print("These are finite-sample distribution checks, not a proof that circuit and DEM distributions are identical.")
    print("The Bonferroni flags refer only to detector marginal tests; dependence between detectors remains.")


if __name__ == "__main__":
    main()
