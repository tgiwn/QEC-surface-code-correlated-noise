"""Leave-one-correlated-pair-out decoder-model ablation at d=5 and d=7.

Purpose:
  Keep the *sampled physical circuit* at the full correlated-noise model, but
  remove one vertical correlated pair at a time from the decoder's DEM. This
  isolates whether representing each pair helps or hurts the current
  correlation-aware MWPM decoder.

For each target distance and p_corr:
  - sample fresh shots once from the full correlated circuit;
  - compare full-pair correlation-aware MWPM with each leave-one-pair-out
    decoder and a no-correlated-pair decoder on those exact same shots;
  - report paired failure-rate differences and a global Bonferroni CI.

This is a decoder-model ablation, NOT a physical-noise ablation. The physical
circuit always contains all correlated pairs.

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
SHOTS = 500_000  # per distance/p_corr
CHUNK_SIZE = 25_000
SEED_BASE = 202_621_00
OUTPUT_CSV = Path("correlated_pair_decoder_model_ablation.csv")
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
    """Sort complete error instructions by probability, then full text."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    errors = [lines[i] for i in positions]

    def key(line: str):
        m = re.match(r"\s*error\(([^)]+)\)", line)
        if not m:
            raise ValueError(f"Could not parse DEM instruction: {line.strip()}")
        return (-float(m.group(1)), line.strip())

    for pos, line in zip(positions, sorted(errors, key=key)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Canonical sorting changed DEM dimensions")
    return result


def build_decoder(base, data_qubits, selected_pairs, p_corr: float, expected_dims: tuple[int, int]):
    model_circuit = add_noise_layer(base, data_qubits, selected_pairs, p_corr, "correlated")
    dem = model_circuit.detector_error_model(decompose_errors=True).flattened()
    dims = (dem.num_detectors, dem.num_observables)
    if dims != expected_dims:
        raise ValueError(f"Decoder DEM dimensions changed: expected {expected_dims}, got {dims}")
    sorted_dem = canonical_probability_sorted_dem(dem)
    return pymatching.Matching(sorted_dem, enable_correlations=True), dem


def paired_stats(candidate_fail: np.ndarray, full_fail: np.ndarray) -> dict[str, float | int]:
    """Candidate-minus-full paired failure delta and normal CI."""
    candidate_only = int(np.count_nonzero(candidate_fail & ~full_fail))
    full_only = int(np.count_nonzero(full_fail & ~candidate_fail))
    n = int(len(full_fail))
    delta = (candidate_only - full_only) / n
    discordant = candidate_only + full_only
    se = math.sqrt(max(0.0, discordant / n - delta * delta) / n)
    z95 = NormalDist().inv_cdf(0.975)
    return {
        "candidate_only_failures": candidate_only,
        "full_only_failures": full_only,
        "delta": delta,
        "ci95_low": delta - z95 * se,
        "ci95_high": delta + z95 * se,
        "se": se,
    }


def main() -> None:
    print("Correlated-pair decoder-model ablation")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; shots/condition={SHOTS:,}")
    print("Physical samples always include ALL correlated pairs; only decoder DEMs are ablated.")
    print("Negative delta means omitting that pair improves the decoder vs the full-pair model.")

    setups = {}
    pair_counts_by_d = {}
    for d in DISTANCES:
        base = make_base_circuit(distance=d, rounds=d, p=P_BASE)
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)
        qubit_to_xy = {q: xy for xy, q in data_by_xy.items()}
        setups[d] = (base, data_qubits, pairs, qubit_to_xy)
        pair_counts_by_d[d] = len(pairs)
        print(f"distance={d}: detectors per d-round circuit will be model-derived; vertical pairs={len(pairs)}")

    total_pair_tests = sum(pair_counts_by_d[d] * len(P_CORR_VALUES) for d in DISTANCES)
    z_bonf = NormalDist().inv_cdf(1 - ALPHA / (2 * total_pair_tests))
    print(f"Pair comparisons={total_pair_tests}; global Bonferroni 95% CI z={z_bonf:.3f}")

    fieldnames = [
        "distance", "p_base", "p_corr", "seed", "shots", "decoder_model", "omitted_pair_index",
        "pair_qubits", "pair_coordinates", "failures", "rate_pct", "delta_vs_full_pp",
        "ci95_low_pp", "ci95_high_pp", "bonferroni_low_pp", "bonferroni_high_pp",
        "candidate_only_failures", "full_only_failures", "changed_predictions",
    ]
    rows: list[dict[str, Any]] = []

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for d in DISTANCES:
            base, data_qubits, pairs, q_to_xy = setups[d]
            for pc_idx, p_corr in enumerate(P_CORR_VALUES):
                seed = int(SEED_BASE + d * 100 + pc_idx)
                t0 = time.perf_counter()
                full_circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
                full_dem = full_circuit.detector_error_model(decompose_errors=True).flattened()
                expected_dims = (full_dem.num_detectors, full_dem.num_observables)
                full_sorted = canonical_probability_sorted_dem(full_dem)
                full_decoder = pymatching.Matching(full_sorted, enable_correlations=True)

                no_pair_decoder, no_pair_dem = build_decoder(base, data_qubits, [], p_corr, expected_dims)
                decoder_specs: list[tuple[str, int | None, tuple[int, int] | None, pymatching.Matching]] = [
                    ("no-correlated-pairs", None, None, no_pair_decoder),
                    ("full-pairs", None, None, full_decoder),
                ]
                for i, pair in enumerate(pairs):
                    remaining = [q_pair for j, q_pair in enumerate(pairs) if j != i]
                    decoder, _ = build_decoder(base, data_qubits, remaining, p_corr, expected_dims)
                    coords = (q_to_xy[pair[0]], q_to_xy[pair[1]])
                    decoder_specs.append((f"leave-out-pair-{i+1:02d}", i + 1, pair, decoder))

                print("\n" + "=" * 92)
                print(f"d={d}; p_corr={p_corr:g}; seed={seed}; shots={SHOTS:,}; pairs={len(pairs)}")
                print(f"DEM dimensions={expected_dims}; full error instructions={sum(x.lstrip().startswith('error(') for x in str(full_dem).splitlines())}; "
                      f"no-pair instructions={sum(x.lstrip().startswith('error(') for x in str(no_pair_dem).splitlines())}")
                print(f"Built {len(decoder_specs)} decoders; sampling the full physical circuit once for paired evaluation.")

                fail_counts = {name: 0 for name, *_ in decoder_specs}
                changed_counts = {name: 0 for name, *_ in decoder_specs}
                cand_only_counts = {name: 0 for name, *_ in decoder_specs}
                full_only_counts = {name: 0 for name, *_ in decoder_specs}
                sampler = full_circuit.compile_detector_sampler(seed=seed)
                processed = 0
                while processed < SHOTS:
                    n = min(CHUNK_SIZE, SHOTS - processed)
                    syndrome, actual = sampler.sample(shots=n, separate_observables=True)
                    syndrome = np.asarray(syndrome, dtype=np.uint8)
                    actual2d = as_2d(actual)
                    full_pred = as_2d(full_decoder.decode_batch(syndrome, enable_correlations=True))
                    full_fail = failure_mask(full_pred, actual2d)
                    fail_counts["full-pairs"] += int(np.count_nonzero(full_fail))

                    for name, omitted_idx, pair, decoder in decoder_specs:
                        if name == "full-pairs":
                            continue
                        pred = as_2d(decoder.decode_batch(syndrome, enable_correlations=True))
                        fail = failure_mask(pred, actual2d)
                        fail_counts[name] += int(np.count_nonzero(fail))
                        changed_counts[name] += int(np.count_nonzero(np.any(pred != full_pred, axis=1)))
                        cand_only_counts[name] += int(np.count_nonzero(fail & ~full_fail))
                        full_only_counts[name] += int(np.count_nonzero(full_fail & ~fail))
                    processed += n
                    if processed % (CHUNK_SIZE * 4) == 0 or processed == SHOTS:
                        print(f"  sampled/decoded {processed:,}/{SHOTS:,}", flush=True)

                # Write full model row and all ablations with paired confidence intervals.
                full_failures = fail_counts["full-pairs"]
                for name, omitted_idx, pair, decoder in decoder_specs:
                    failures = fail_counts[name]
                    rate_pct = 100 * failures / SHOTS
                    if name == "full-pairs":
                        stats = {"delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
                                 "candidate_only_failures": 0, "full_only_failures": 0}
                    else:
                        # Counts were accumulated over exactly paired shots.
                        cand_only = cand_only_counts[name]
                        full_only = full_only_counts[name]
                        delta = (cand_only - full_only) / SHOTS
                        discordant = cand_only + full_only
                        se = math.sqrt(max(0.0, discordant / SHOTS - delta * delta) / SHOTS)
                        z95 = NormalDist().inv_cdf(0.975)
                        stats = {
                            "delta": delta,
                            "ci95_low": delta - z95 * se,
                            "ci95_high": delta + z95 * se,
                            "candidate_only_failures": cand_only,
                            "full_only_failures": full_only,
                        }
                    if pair is None:
                        pair_qubits, pair_coords = "", ""
                    else:
                        pair_qubits = f"{pair[0]},{pair[1]}"
                        pair_coords = f"{q_to_xy[pair[0]]};{q_to_xy[pair[1]]}"
                    row = {
                        "distance": d, "p_base": P_BASE, "p_corr": p_corr, "seed": seed, "shots": SHOTS,
                        "decoder_model": name, "omitted_pair_index": omitted_idx or "",
                        "pair_qubits": pair_qubits, "pair_coordinates": pair_coords,
                        "failures": failures, "rate_pct": rate_pct,
                        "delta_vs_full_pp": 100 * float(stats["delta"]),
                        "ci95_low_pp": 100 * float(stats["ci95_low"]),
                        "ci95_high_pp": 100 * float(stats["ci95_high"]),
                        "bonferroni_low_pp": 100 * (float(stats["delta"]) - z_bonf * (se if name != "full-pairs" else 0.0)),
                        "bonferroni_high_pp": 100 * (float(stats["delta"]) + z_bonf * (se if name != "full-pairs" else 0.0)),
                        "candidate_only_failures": stats["candidate_only_failures"],
                        "full_only_failures": stats["full_only_failures"],
                        "changed_predictions": changed_counts[name],
                    }
                    # For writing only. Recompute paired SE without relying on a last-loop variable.
                    if name != "full-pairs":
                        discordant = int(cand_only_counts[name] + full_only_counts[name])
                        d_frac = float(stats["delta"])
                        se_this = math.sqrt(max(0.0, discordant / SHOTS - d_frac * d_frac) / SHOTS)
                        row["bonferroni_low_pp"] = 100 * (d_frac - z_bonf * se_this)
                        row["bonferroni_high_pp"] = 100 * (d_frac + z_bonf * se_this)
                    writer.writerow(row)
                    rows.append(row)
                f.flush()

                omissions = [r for r in rows if r["distance"] == d and r["p_corr"] == p_corr
                             and str(r["decoder_model"]).startswith("leave-out-pair-")]
                omissions.sort(key=lambda r: float(r["delta_vs_full_pp"]))
                print(f"Full-pair decoder failures={full_failures}/{SHOTS} ({100*full_failures/SHOTS:.5f}%)")
                print("Pairs whose omission most improves the decoder (negative delta is good):")
                for r in omissions[:8]:
                    sig = "Bonferroni CI excludes 0" if (float(r["bonferroni_high_pp"]) < 0 or float(r["bonferroni_low_pp"]) > 0) else "not Bonferroni-significant"
                    print(f"  {r['decoder_model']:18s} coords={r['pair_coordinates']:20s} rate={float(r['rate_pct']):.5f}% "
                          f"delta={float(r['delta_vs_full_pp']):+.5f} pp "
                          f"CI=[{float(r['ci95_low_pp']):+.5f},{float(r['ci95_high_pp']):+.5f}] "
                          f"global={sig}")
                print("Pairs whose omission most harms the decoder (positive delta is bad):")
                for r in omissions[-5:][::-1]:
                    sig = "Bonferroni CI excludes 0" if (float(r["bonferroni_high_pp"]) < 0 or float(r["bonferroni_low_pp"]) > 0) else "not Bonferroni-significant"
                    print(f"  {r['decoder_model']:18s} coords={r['pair_coordinates']:20s} rate={float(r['rate_pct']):.5f}% "
                          f"delta={float(r['delta_vs_full_pp']):+.5f} pp "
                          f"CI=[{float(r['ci95_low_pp']):+.5f},{float(r['ci95_high_pp']):+.5f}] "
                          f"global={sig}")
                print(f"Condition elapsed={time.perf_counter()-t0:.1f}s; checkpointed {OUTPUT_CSV.resolve()}")

    print("\nCompleted.")
    print(f"Results: {OUTPUT_CSV.resolve()}")
    print("Interpretation: if omitting a pair helps, the current graph decoder may be using that pair's DEM representation poorly; if omission hurts, its inclusion was useful under the model.")
    print("A single leave-one-out result is a sensitivity diagnostic, not proof of pair causality or an optimal decoder.")


if __name__ == "__main__":
    main()
