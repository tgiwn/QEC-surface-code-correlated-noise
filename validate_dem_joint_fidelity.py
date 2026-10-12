"""Compare circuit and flattened-DEM joint syndrome statistics at d=5,7.

This extends a marginal-only fidelity check by comparing:
  * covariance of pairs of detector bits that occur in a common complete DEM instruction;
  * covariance between each detector bit and the logical-observable flip bit;
  * the raw logical-observable flip probability.

Circuit and DEM samples are independent. Complete flattened error instructions are
kept intact; terms separated by '^' are not interpreted as independent events.
This is a targeted second-order diagnostic, not a proof of equality of full joint laws.
"""
from __future__ import annotations

import csv
import itertools
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

# ---------------------------- Settings -------------------------------------
P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCES = (5, 7)
SHOTS = 2_000_000
CHUNK_SIZE = 25_000
SEED_BASE = 202_672_00
ALPHA = 0.05
PAIR_BLOCK = 1024
SUMMARY_CSV = Path("dem_fidelity_joint_summary.csv")
DETAIL_CSV = Path("dem_fidelity_joint_details.csv")

POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
Z95 = NormalDist().inv_cdf(1 - ALPHA / 2)


def as_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def complete_error_instruction_detector_pairs(dem: stim.DetectorErrorModel) -> list[tuple[int, int]]:
    """Return detector pairs jointly present anywhere in one complete `error(...)` line."""
    support: set[tuple[int, int]] = set()
    for line in str(dem.flattened()).splitlines():
        if not line.lstrip().startswith("error("):
            continue
        ids = sorted(set(int(x) for x in re.findall(r"\bD(\d+)\b", line)))
        for i, j in itertools.combinations(ids, 2):
            support.add((i, j))
    return sorted(support)


def make_samplers(circuit: stim.Circuit, dem: stim.DetectorErrorModel, seed_c: int, seed_d: int):
    return circuit.compile_detector_sampler(seed=seed_c), dem.compile_sampler(seed=seed_d)


def sample_circuit(sampler, n: int) -> tuple[np.ndarray, np.ndarray]:
    det, obs = sampler.sample(shots=n, separate_observables=True)
    return np.asarray(det, dtype=np.uint8), as_2d(obs)


def sample_dem(sampler, n: int) -> tuple[np.ndarray, np.ndarray]:
    out = sampler.sample(shots=n)
    if not isinstance(out, tuple) or len(out) < 2:
        raise RuntimeError("Unexpected result from Stim DEM sampler")
    return np.asarray(out[0], dtype=np.uint8), as_2d(out[1])


def accumulate_batch(
    det: np.ndarray,
    obs: np.ndarray,
    det_counts: np.ndarray,
    det_obs_counts: np.ndarray,
    pair_counts: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[int, int]:
    n = int(det.shape[0])
    det_counts += det.sum(axis=0, dtype=np.int64)
    obs_bit = obs[:, 0]
    obs_count = int(obs_bit.sum())
    if np.any(obs_bit):
        det_obs_counts += det[obs_bit.astype(bool)].sum(axis=0, dtype=np.int64)

    # Pack the shot dimension to bytes; count simultaneous firing for only those
    # detector pairs that co-occur in at least one whole DEM error instruction.
    packed = np.packbits(det, axis=0, bitorder="little")
    for start in range(0, pair_i.size, PAIR_BLOCK):
        end = min(pair_i.size, start + PAIR_BLOCK)
        common = np.bitwise_and(packed[:, pair_i[start:end]], packed[:, pair_j[start:end]])
        pair_counts[start:end] += POPCOUNT[common].sum(axis=0, dtype=np.uint64).astype(np.int64)
    return n, obs_count


def covariance_and_se(n: int, count_x: int, count_y: int, count_xy: int) -> tuple[float, float]:
    """Cov(X,Y) and plug-in SE from the 2x2 contingency table."""
    px = count_x / n
    py = count_y / n
    pxy = count_xy / n
    cov = pxy - px * py

    # Influence function for covariance: (X-px)(Y-py) - cov.
    cells = (
        (n - count_x - count_y + count_xy, px * py),
        (count_x - count_xy, (1 - px) * (-py)),
        (count_y - count_xy, (-px) * (1 - py)),
        (count_xy, (1 - px) * (1 - py)),
    )
    e_g2 = sum(c * (g ** 2) for c, g in cells) / n
    var_one = max(0.0, e_g2 - cov * cov)
    return cov, math.sqrt(var_one / n)


def compare_cov(
    n: int,
    x_c: int,
    y_c: int,
    xy_c: int,
    x_d: int,
    y_d: int,
    xy_d: int,
) -> dict[str, float]:
    cov_c, se_c = covariance_and_se(n, x_c, y_c, xy_c)
    cov_d, se_d = covariance_and_se(n, x_d, y_d, xy_d)
    delta = cov_c - cov_d
    se = math.sqrt(se_c * se_c + se_d * se_d)
    z = delta / se if se > 0 else (0.0 if delta == 0 else math.copysign(math.inf, delta))
    return {
        "cov_circuit": cov_c,
        "cov_dem": cov_d,
        "delta_cov": delta,
        "se_delta_cov": se,
        "z_score": z,
        "ci95_low": delta - Z95 * se,
        "ci95_high": delta + Z95 * se,
    }


def probability_delta_ci(k_c: int, k_d: int, n: int) -> tuple[float, float, float, float]:
    """Independent-sample probability difference and z, returned as percentage points."""
    p_c = k_c / n
    p_d = k_d / n
    delta = p_c - p_d
    se = math.sqrt(p_c * (1 - p_c) / n + p_d * (1 - p_d) / n)
    z = delta / se if se > 0 else (0.0 if delta == 0 else math.copysign(math.inf, delta))
    return 100 * delta, 100 * (delta - Z95 * se), 100 * (delta + Z95 * se), z


def main() -> None:
    print("Circuit-vs-DEM joint-fidelity check: detector pairs and detector-observable dependence")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; shots/source/condition={SHOTS:,}")
    print(f"chunk={CHUNK_SIZE:,}; seed base={SEED_BASE}; pair block={PAIR_BLOCK}")
    print("Circuit and DEM seeds are independent. Whole error instructions are retained intact.")
    print("Tests second-order statistics; this is not a proof of equality of full distributions.")

    summary_fields = [
        "distance", "p_base", "p_corr", "seed_circuit", "seed_dem", "shots_per_source",
        "n_detectors", "n_dem_supported_detector_pairs", "observable_circuit_count", "observable_dem_count",
        "observable_rate_circuit_pct", "observable_rate_dem_pct", "observable_delta_pp",
        "observable_ci95_low_pp", "observable_ci95_high_pp", "observable_z",
        "observable_bonferroni_across_four_conditions_significant",
        "detector_observable_tests", "detector_observable_bonferroni_z", "detector_observable_flags",
        "max_abs_detector_observable_z", "detector_pair_tests", "detector_pair_bonferroni_z",
        "detector_pair_flags", "max_abs_detector_pair_z",
    ]
    detail_fields = [
        "distance", "p_corr", "test_family", "detector_i", "detector_j_or_observable",
        "circuit_covariance", "dem_covariance", "delta_covariance_x1e6", "se_delta_covariance_x1e6",
        "ci95_low_x1e6", "ci95_high_x1e6", "z_score", "bonferroni_z_threshold", "bonferroni_significant",
        "circuit_joint_count", "dem_joint_count", "circuit_rate_i_pct", "dem_rate_i_pct",
        "circuit_rate_j_pct", "dem_rate_j_pct",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fs, DETAIL_CSV.open("w", newline="", encoding="utf-8") as fd:
        sw = csv.DictWriter(fs, fieldnames=summary_fields)
        dw = csv.DictWriter(fd, fieldnames=detail_fields)
        sw.writeheader(); dw.writeheader(); fs.flush(); fd.flush()

        condition_index = 0
        for d in DISTANCES:
            base = make_base_circuit(distance=d, rounds=d, p=P_BASE)
            data_by_xy, data_qubits = get_data_layout(base)
            pairs = make_correlated_pairs(data_by_xy)
            for pc_idx, pc in enumerate(P_CORR_VALUES):
                seed_c = int(SEED_BASE + d * 100 + pc_idx * 2)
                seed_d = seed_c + 1
                start_t = time.perf_counter()
                circuit = add_noise_layer(base, data_qubits, pairs, pc, "correlated")
                dem = circuit.detector_error_model(decompose_errors=True).flattened()
                n_det = dem.num_detectors
                if dem.num_observables < 1:
                    raise RuntimeError("Expected at least one logical observable in DEM")
                supported_pairs = complete_error_instruction_detector_pairs(dem)
                pair_arr = np.asarray(supported_pairs, dtype=np.int64).reshape((-1, 2))
                pair_i = pair_arr[:, 0] if pair_arr.size else np.zeros(0, dtype=np.int64)
                pair_j = pair_arr[:, 1] if pair_arr.size else np.zeros(0, dtype=np.int64)

                circuit_sampler, dem_sampler = make_samplers(circuit, dem, seed_c, seed_d)
                # Independent sufficient statistics for circuit and DEM samples.
                stats = {}
                for source in ("circuit", "dem"):
                    stats[source] = {
                        "n": 0,
                        "det_counts": np.zeros(n_det, dtype=np.int64),
                        "obs_count": 0,
                        "det_obs_counts": np.zeros(n_det, dtype=np.int64),
                        "pair_counts": np.zeros(len(supported_pairs), dtype=np.int64),
                    }

                print("\n" + "=" * 94)
                print(f"d={d}; p_corr={pc:g}; detectors={n_det}; whole DEM error instructions="
                      f"{sum(line.lstrip().startswith('error(') for line in str(dem).splitlines())}; "
                      f"candidate detector pairs={len(supported_pairs):,}")
                print(f"circuit seed={seed_c}; DEM seed={seed_d}; {SHOTS:,} samples per source")

                while stats["circuit"]["n"] < SHOTS or stats["dem"]["n"] < SHOTS:
                    for source in ("circuit", "dem"):
                        st = stats[source]
                        if st["n"] >= SHOTS:
                            continue
                        take = min(CHUNK_SIZE, SHOTS - st["n"])
                        if source == "circuit":
                            det, obs = sample_circuit(circuit_sampler, take)
                        else:
                            det, obs = sample_dem(dem_sampler, take)
                        got, obs_count = accumulate_batch(
                            det, obs, st["det_counts"], st["det_obs_counts"], st["pair_counts"], pair_i, pair_j
                        )
                        st["n"] += got
                        st["obs_count"] += obs_count
                    current = stats["circuit"]["n"]
                    if current and (current % 250_000 == 0 or current == SHOTS):
                        print(f"  processed circuit={current:,}/{SHOTS:,}; DEM={stats['dem']['n']:,}/{SHOTS:,}")

                c = stats["circuit"]
                m = stats["dem"]
                obs_delta, obs_lo, obs_hi, obs_z = probability_delta_ci(c["obs_count"], m["obs_count"], SHOTS)
                # Four-condition familywise threshold for the observable marginal.
                obs_bonf_z = NormalDist().inv_cdf(1 - ALPHA / (2 * len(DISTANCES) * len(P_CORR_VALUES)))
                obs_bonf_sig = abs(obs_z) > obs_bonf_z

                # Detector-observable covariance family: D_i with L_0.
                det_obs_rows = []
                for i in range(n_det):
                    cm = compare_cov(
                        SHOTS, int(c["det_counts"][i]), int(c["obs_count"]), int(c["det_obs_counts"][i]),
                        int(m["det_counts"][i]), int(m["obs_count"]), int(m["det_obs_counts"][i]),
                    )
                    det_obs_rows.append((i, cm))
                dz = NormalDist().inv_cdf(1 - ALPHA / (2 * max(1, n_det)))
                det_obs_flags = sum(abs(row[1]["z_score"]) > dz for row in det_obs_rows)
                max_det_obs_z = max((abs(row[1]["z_score"]) for row in det_obs_rows), default=0.0)
                for i, res in det_obs_rows:
                    dw.writerow({
                        "distance": d, "p_corr": pc, "test_family": "detector_observable",
                        "detector_i": i, "detector_j_or_observable": "L0",
                        "circuit_covariance": res["cov_circuit"], "dem_covariance": res["cov_dem"],
                        "delta_covariance_x1e6": res["delta_cov"] * 1e6,
                        "se_delta_covariance_x1e6": res["se_delta_cov"] * 1e6,
                        "ci95_low_x1e6": res["ci95_low"] * 1e6, "ci95_high_x1e6": res["ci95_high"] * 1e6,
                        "z_score": res["z_score"], "bonferroni_z_threshold": dz,
                        "bonferroni_significant": abs(res["z_score"]) > dz,
                        "circuit_joint_count": int(c["det_obs_counts"][i]),
                        "dem_joint_count": int(m["det_obs_counts"][i]),
                        "circuit_rate_i_pct": 100 * c["det_counts"][i] / SHOTS,
                        "dem_rate_i_pct": 100 * m["det_counts"][i] / SHOTS,
                        "circuit_rate_j_pct": 100 * c["obs_count"] / SHOTS,
                        "dem_rate_j_pct": 100 * m["obs_count"] / SHOTS,
                    })
                fd.flush()

                # Detector-pair covariance family, restricted to detector pairs jointly
                # present in a whole flattened DEM error instruction.
                pair_rows = []
                for k, (i, j) in enumerate(supported_pairs):
                    res = compare_cov(
                        SHOTS, int(c["det_counts"][i]), int(c["det_counts"][j]), int(c["pair_counts"][k]),
                        int(m["det_counts"][i]), int(m["det_counts"][j]), int(m["pair_counts"][k]),
                    )
                    pair_rows.append((i, j, k, res))
                pair_bonf_z = NormalDist().inv_cdf(1 - ALPHA / (2 * max(1, len(supported_pairs))))
                pair_flags = sum(abs(row[3]["z_score"]) > pair_bonf_z for row in pair_rows)
                max_pair_z = max((abs(row[3]["z_score"]) for row in pair_rows), default=0.0)
                for i, j, k, res in pair_rows:
                    dw.writerow({
                        "distance": d, "p_corr": pc, "test_family": "detector_pair",
                        "detector_i": i, "detector_j_or_observable": j,
                        "circuit_covariance": res["cov_circuit"], "dem_covariance": res["cov_dem"],
                        "delta_covariance_x1e6": res["delta_cov"] * 1e6,
                        "se_delta_covariance_x1e6": res["se_delta_cov"] * 1e6,
                        "ci95_low_x1e6": res["ci95_low"] * 1e6, "ci95_high_x1e6": res["ci95_high"] * 1e6,
                        "z_score": res["z_score"], "bonferroni_z_threshold": pair_bonf_z,
                        "bonferroni_significant": abs(res["z_score"]) > pair_bonf_z,
                        "circuit_joint_count": int(c["pair_counts"][k]), "dem_joint_count": int(m["pair_counts"][k]),
                        "circuit_rate_i_pct": 100 * c["det_counts"][i] / SHOTS,
                        "dem_rate_i_pct": 100 * m["det_counts"][i] / SHOTS,
                        "circuit_rate_j_pct": 100 * c["det_counts"][j] / SHOTS,
                        "dem_rate_j_pct": 100 * m["det_counts"][j] / SHOTS,
                    })
                fd.flush()

                print(f"P(L0=1): circuit={100*c['obs_count']/SHOTS:.6f}%; DEM={100*m['obs_count']/SHOTS:.6f}%; "
                      f"delta={obs_delta:+.6f} pp [{obs_lo:+.6f},{obs_hi:+.6f}], z={obs_z:+.2f}; "
                      f"4-condition Bonferroni={'FLAG' if obs_bonf_sig else 'no flag'} (|z|>{obs_bonf_z:.2f})")
                print(f"Detector-observable covariance: flags={det_obs_flags}/{n_det} after Bonferroni; "
                      f"max |z|={max_det_obs_z:.2f} (threshold {dz:.2f})")
                print(f"Detector-pair covariance on DEM-supported pairs: flags={pair_flags}/{len(supported_pairs):,}; "
                      f"max |z|={max_pair_z:.2f} (threshold {pair_bonf_z:.2f})")

                # Print top diagnostic residuals for context, preserving corrected tests above.
                print("  largest detector-observable residuals:")
                for i, res in sorted(det_obs_rows, key=lambda x: abs(x[1]["z_score"]), reverse=True)[:5]:
                    print(f"    D{i:3d}: delta_cov*1e6={res['delta_cov']*1e6:+.3f}; z={res['z_score']:+.2f}; "
                          f"raw p(D&L)_C={c['det_obs_counts'][i]/SHOTS:.6g}, p(D&L)_DEM={m['det_obs_counts'][i]/SHOTS:.6g}")
                print("  largest detector-pair residuals:")
                for i, j, k, res in sorted(pair_rows, key=lambda x: abs(x[3]["z_score"]), reverse=True)[:5]:
                    print(f"    (D{i},D{j}): delta_cov*1e6={res['delta_cov']*1e6:+.3f}; z={res['z_score']:+.2f}; "
                          f"joint_C={c['pair_counts'][k]}, joint_DEM={m['pair_counts'][k]}")

                sw.writerow({
                    "distance": d, "p_base": P_BASE, "p_corr": pc, "seed_circuit": seed_c, "seed_dem": seed_d,
                    "shots_per_source": SHOTS, "n_detectors": n_det, "n_dem_supported_detector_pairs": len(supported_pairs),
                    "observable_circuit_count": c["obs_count"], "observable_dem_count": m["obs_count"],
                    "observable_rate_circuit_pct": 100*c["obs_count"]/SHOTS,
                    "observable_rate_dem_pct": 100*m["obs_count"]/SHOTS,
                    "observable_delta_pp": obs_delta, "observable_ci95_low_pp": obs_lo,
                    "observable_ci95_high_pp": obs_hi, "observable_z": obs_z,
                    "observable_bonferroni_across_four_conditions_significant": obs_bonf_sig,
                    "detector_observable_tests": n_det, "detector_observable_bonferroni_z": dz,
                    "detector_observable_flags": det_obs_flags, "max_abs_detector_observable_z": max_det_obs_z,
                    "detector_pair_tests": len(supported_pairs), "detector_pair_bonferroni_z": pair_bonf_z,
                    "detector_pair_flags": pair_flags, "max_abs_detector_pair_z": max_pair_z,
                })
                fs.flush()
                print(f"Condition elapsed={time.perf_counter()-start_t:.1f}s; checkpointed outputs")

    print("\nCompleted.")
    print(f"Summary: {SUMMARY_CSV.resolve()}")
    print(f"Pairwise detail: {DETAIL_CSV.resolve()}")
    print("No flags means no second-order discrepancy was detected within the tested families and power.")
    print("This does not certify equality of the complete circuit and DEM joint distributions.")


if __name__ == "__main__":
    main()
