"""Grouped correlated-pair decoder-model ablation at distances 5 and 7.

The sampled physical circuit always contains ALL correlated pairs. Each candidate
removes a structured group of pair mechanisms from the decoder DEM only. This is
intended to detect group-level effects that individual leave-one-out tests may
not resolve at 500k shots.

Groups tested:
  * no-correlated-pairs (all pair mechanisms omitted);
  * all pairs sharing a data-layout x coordinate (vertical column groups);
  * all pairs sharing a lower y coordinate (horizontal edge-row groups);
  * boundary pairs and interior pairs;
  * two checkerboard partitions of the pair lattice.

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
SHOTS = 500_000
CHUNK_SIZE = 25_000
SEED_BASE = 202_631_00
OUTPUT_CSV = Path("correlated_pair_group_ablation.csv")
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
    """Sort complete error instructions by probability, then by full line text."""
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


def coord_key(value: float) -> str:
    return f"{value:g}"


def make_pair_groups(pairs, qubit_to_xy: dict[int, tuple[float, float]]) -> dict[str, set[int]]:
    """Build structured pair-index groups from the data-qubit lattice geometry."""
    pair_coords: list[tuple[tuple[float, float], tuple[float, float]]] = []
    xs = set()
    ys = set()
    for pair in pairs:
        p0 = tuple(map(float, qubit_to_xy[pair[0]]))
        p1 = tuple(map(float, qubit_to_xy[pair[1]]))
        pair_coords.append((p0, p1))
        xs.update((p0[0], p1[0]))
        ys.update((p0[1], p1[1]))

    x_values = sorted(xs)
    y_values = sorted(ys)
    x_index = {x: i for i, x in enumerate(x_values)}
    y_index = {y: i for i, y in enumerate(y_values)}
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    groups: dict[str, set[int]] = {"no-correlated-pairs": set(range(len(pairs)))}
    by_x: dict[float, set[int]] = {}
    by_y_edge: dict[float, set[int]] = {}
    boundary: set[int] = set()
    interior: set[int] = set()
    checker_a: set[int] = set()
    checker_b: set[int] = set()

    for i, (p0, p1) in enumerate(pair_coords):
        # Pair mechanisms in this layout connect qubits along one coordinate axis.
        if p0[0] == p1[0]:
            x = p0[0]
            by_x.setdefault(x, set()).add(i)
        else:
            # Defensive fallback for a horizontal pair layout.
            x = min(p0[0], p1[0])
            by_x.setdefault(x, set()).add(i)

        y_lo = min(p0[1], p1[1])
        by_y_edge.setdefault(y_lo, set()).add(i)

        on_boundary = (
            p0[0] in (x_min, x_max) or p1[0] in (x_min, x_max)
            or p0[1] in (y_min, y_max) or p1[1] in (y_min, y_max)
        )
        (boundary if on_boundary else interior).add(i)

        xi = x_index[x]
        yi = y_index[y_lo]
        if (xi + yi) % 2 == 0:
            checker_a.add(i)
        else:
            checker_b.add(i)

    for x, indices in sorted(by_x.items()):
        groups[f"column-x-{coord_key(x)}"] = indices
    for y, indices in sorted(by_y_edge.items()):
        groups[f"row-edge-y-{coord_key(y)}"] = indices
    groups["boundary-pairs"] = boundary
    groups["interior-pairs"] = interior
    groups["checkerboard-even"] = checker_a
    groups["checkerboard-odd"] = checker_b

    # Drop empty groups and any group that removes every pair except the separately named control.
    groups = {name: indices for name, indices in groups.items() if indices}
    # Check the intended pair lattice groups cover every pair, with no empty partition.
    if not groups["no-correlated-pairs"]:
        raise ValueError("No correlated pairs were generated")
    return groups


def build_decoder(base, data_qubits, selected_pairs, p_corr: float, expected_dims: tuple[int, int]):
    model_circuit = add_noise_layer(base, data_qubits, selected_pairs, p_corr, "correlated")
    dem = model_circuit.detector_error_model(decompose_errors=True).flattened()
    dims = (dem.num_detectors, dem.num_observables)
    if dims != expected_dims:
        raise ValueError(f"Decoder DEM dimensions changed: expected {expected_dims}, got {dims}")
    decoder = pymatching.Matching(canonical_probability_sorted_dem(dem), enable_correlations=True)
    return decoder, dem


def main() -> None:
    print("Grouped correlated-pair decoder-model ablation")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; shots/condition={SHOTS:,}")
    print("Physical samples always include ALL correlated pairs; only decoder DEM groups are omitted.")
    print("Negative delta = omission improves versus full-pair decoder; positive delta = omission harms.")

    setups = {}
    groups_by_d = {}
    for d in DISTANCES:
        base = make_base_circuit(distance=d, rounds=d, p=P_BASE)
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)
        qubit_to_xy = {q: tuple(map(float, xy)) for xy, q in data_by_xy.items()}
        groups = make_pair_groups(pairs, qubit_to_xy)
        setups[d] = (base, data_qubits, pairs, qubit_to_xy)
        groups_by_d[d] = groups
        print(f"distance={d}: pairs={len(pairs)}; structured decoder ablations={len(groups)}")
        for name, members in groups.items():
            print(f"  {name:25s}: removes {len(members):2d}/{len(pairs)} pairs")

    total_comparisons = sum(len(groups_by_d[d]) * len(P_CORR_VALUES) for d in DISTANCES)
    z_bonf = NormalDist().inv_cdf(1 - ALPHA / (2 * total_comparisons))
    print(f"Group comparisons={total_comparisons}; global Bonferroni 95% CI z={z_bonf:.3f}")

    fieldnames = [
        "distance", "p_base", "p_corr", "seed", "shots", "group_name", "omitted_pair_count",
        "omitted_pair_indices_1based", "omitted_pair_coordinates", "full_failures", "candidate_failures",
        "full_rate_pct", "candidate_rate_pct", "delta_vs_full_pp", "ci95_low_pp", "ci95_high_pp",
        "bonferroni_low_pp", "bonferroni_high_pp", "candidate_only_failures", "full_only_failures",
        "changed_predictions",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for d in DISTANCES:
            base, data_qubits, pairs, q_to_xy = setups[d]
            groups = groups_by_d[d]
            for pc_idx, p_corr in enumerate(P_CORR_VALUES):
                seed = int(SEED_BASE + d * 100 + pc_idx)
                t0 = time.perf_counter()
                full_circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
                full_dem = full_circuit.detector_error_model(decompose_errors=True).flattened()
                expected_dims = (full_dem.num_detectors, full_dem.num_observables)
                full_decoder = pymatching.Matching(canonical_probability_sorted_dem(full_dem), enable_correlations=True)

                specs = []
                for group_name, omitted_indices in groups.items():
                    selected_pairs = [pair for i, pair in enumerate(pairs) if i not in omitted_indices]
                    decoder, dem = build_decoder(base, data_qubits, selected_pairs, p_corr, expected_dims)
                    coords = []
                    for i in sorted(omitted_indices):
                        pair = pairs[i]
                        p0, p1 = q_to_xy[pair[0]], q_to_xy[pair[1]]
                        coords.append(f"{p0};{p1}")
                    specs.append((group_name, omitted_indices, decoder, coords, dem))

                print("\n" + "=" * 96)
                print(f"d={d}; p_corr={p_corr:g}; seed={seed}; shots={SHOTS:,}; dimensions={expected_dims}")
                print(f"Full DEM instructions={sum(line.lstrip().startswith('error(') for line in str(full_dem).splitlines())}; "
                      f"built {len(specs)} grouped ablation decoders")

                fail_counts = {name: 0 for name, *_ in specs}
                changed_counts = {name: 0 for name, *_ in specs}
                candidate_only_counts = {name: 0 for name, *_ in specs}
                full_only_counts = {name: 0 for name, *_ in specs}
                full_failures = 0
                sampler = full_circuit.compile_detector_sampler(seed=seed)
                processed = 0
                while processed < SHOTS:
                    n = min(CHUNK_SIZE, SHOTS - processed)
                    syndrome, actual = sampler.sample(shots=n, separate_observables=True)
                    syndrome = np.asarray(syndrome, dtype=np.uint8)
                    actual2d = as_2d(actual)
                    full_pred = as_2d(full_decoder.decode_batch(syndrome, enable_correlations=True))
                    full_fail = failure_mask(full_pred, actual2d)
                    full_failures += int(np.count_nonzero(full_fail))

                    for name, _omitted, decoder, _coords, _dem in specs:
                        pred = as_2d(decoder.decode_batch(syndrome, enable_correlations=True))
                        fail = failure_mask(pred, actual2d)
                        fail_counts[name] += int(np.count_nonzero(fail))
                        changed_counts[name] += int(np.count_nonzero(np.any(pred != full_pred, axis=1)))
                        candidate_only_counts[name] += int(np.count_nonzero(fail & ~full_fail))
                        full_only_counts[name] += int(np.count_nonzero(full_fail & ~fail))
                    processed += n
                    if processed % (CHUNK_SIZE * 4) == 0 or processed == SHOTS:
                        print(f"  sampled/decoded {processed:,}/{SHOTS:,}", flush=True)

                rows_here = []
                for name, omitted, decoder, coords, dem in specs:
                    candidate_only = candidate_only_counts[name]
                    full_only = full_only_counts[name]
                    delta = (candidate_only - full_only) / SHOTS
                    discordant = candidate_only + full_only
                    se = math.sqrt(max(0.0, discordant / SHOTS - delta * delta) / SHOTS)
                    z95 = NormalDist().inv_cdf(0.975)
                    candidate_failures = fail_counts[name]
                    row = {
                        "distance": d,
                        "p_base": P_BASE,
                        "p_corr": p_corr,
                        "seed": seed,
                        "shots": SHOTS,
                        "group_name": name,
                        "omitted_pair_count": len(omitted),
                        "omitted_pair_indices_1based": ";".join(str(i + 1) for i in sorted(omitted)),
                        "omitted_pair_coordinates": " | ".join(coords),
                        "full_failures": full_failures,
                        "candidate_failures": candidate_failures,
                        "full_rate_pct": 100 * full_failures / SHOTS,
                        "candidate_rate_pct": 100 * candidate_failures / SHOTS,
                        "delta_vs_full_pp": 100 * delta,
                        "ci95_low_pp": 100 * (delta - z95 * se),
                        "ci95_high_pp": 100 * (delta + z95 * se),
                        "bonferroni_low_pp": 100 * (delta - z_bonf * se),
                        "bonferroni_high_pp": 100 * (delta + z_bonf * se),
                        "candidate_only_failures": candidate_only,
                        "full_only_failures": full_only,
                        "changed_predictions": changed_counts[name],
                    }
                    writer.writerow(row)
                    rows_here.append(row)
                f.flush()

                print(f"Full-pair decoder failures={full_failures}/{SHOTS} ({100*full_failures/SHOTS:.5f}%)")
                rows_here.sort(key=lambda r: float(r["delta_vs_full_pp"]))
                print("Largest apparent improvements from group omission (negative is good):")
                for row in rows_here[:5]:
                    significant = float(row["bonferroni_high_pp"]) < 0 or float(row["bonferroni_low_pp"]) > 0
                    print(f"  {row['group_name']:25s} omitted={int(row['omitted_pair_count']):2d} "
                          f"delta={float(row['delta_vs_full_pp']):+.5f} pp "
                          f"CI=[{float(row['ci95_low_pp']):+.5f},{float(row['ci95_high_pp']):+.5f}] "
                          f"Bonferroni={'significant' if significant else 'n.s.'}")
                print("Largest apparent harms from group omission (positive is bad):")
                for row in rows_here[-5:][::-1]:
                    significant = float(row["bonferroni_high_pp"]) < 0 or float(row["bonferroni_low_pp"]) > 0
                    print(f"  {row['group_name']:25s} omitted={int(row['omitted_pair_count']):2d} "
                          f"delta={float(row['delta_vs_full_pp']):+.5f} pp "
                          f"CI=[{float(row['ci95_low_pp']):+.5f},{float(row['ci95_high_pp']):+.5f}] "
                          f"Bonferroni={'significant' if significant else 'n.s.'}")
                print(f"Condition elapsed={time.perf_counter()-t0:.1f}s; checkpointed {OUTPUT_CSV.resolve()}")

    print("\nCompleted.")
    print(f"Results: {OUTPUT_CSV.resolve()}")
    print("This tests structured decoder-model ablations on the same full-correlated physical samples.")
    print("Group-level effects are decoder sensitivity, not causal attribution or proof of MAP optimality.")


if __name__ == "__main__":
    main()
