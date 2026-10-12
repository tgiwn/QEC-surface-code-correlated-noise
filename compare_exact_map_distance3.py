"""Fresh-seed distance-3 decoder comparison against exact DEM-MAP.

Compares on identical physical-circuit shots:
  * ordinary MWPM using the original DEM instruction order;
  * ordinary MWPM after canonical probability sorting;
  * probability-prioritized graph MWPM heuristic;
  * correlation-aware MWPM using the original DEM order;
  * correlation-aware MWPM after canonical probability sorting;
  * exact DEM-MAP, computed from the full flattened DEM mechanism distribution.

The exact posterior implementation is imported from exact_dem_map_decoder.py.
That implementation treats every full error(p) instruction as one Bernoulli
mechanism and does not split instructions at '^'. The dense MAP calculation is
restricted to distance 3 (24 detectors, one logical observable).

Run from the project directory containing correlated_experiment.py and
exact_dem_map_decoder.py:
    & $python .\\compare_exact_map_distance3.py

Outputs (overwritten at start, written after each completed condition):
    exact_map_distance3_comparison_raw.csv
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

import exact_dem_map_decoder as exact
from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

# --------------------------- Experiment settings ---------------------------
DISTANCE = 3
ROUNDS = 3
P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
SHOTS = 500_000
SEED_BASE = 20_261_080
RAW_CSV = Path("exact_map_distance3_comparison_raw.csv")

DECODER_ORDER = (
    "ordinary/original",
    "ordinary/canonical-sorted",
    "probability-prioritized",
    "correlation-aware/original",
    "correlation-aware/canonical-sorted",
    "exact DEM-MAP",
)


def as_2d(values):
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def failure_mask(predictions, actual):
    pred = as_2d(predictions)
    obs = as_2d(actual)
    if pred.shape != obs.shape:
        raise ValueError(f"prediction/observable shape mismatch: {pred.shape} vs {obs.shape}")
    return np.any(pred != obs, axis=1)


def paired_delta_ci(candidate_fail, baseline_fail):
    """Paired candidate-minus-baseline risk difference and normal 95% CI."""
    candidate = np.asarray(candidate_fail, dtype=np.uint8)
    baseline = np.asarray(baseline_fail, dtype=np.uint8)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("Paired failure masks must be same-length 1-D arrays")
    diff = candidate.astype(np.int8) - baseline.astype(np.int8)
    delta = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return delta, (delta - 1.96 * se, delta + 1.96 * se)


def error_positions(lines):
    return [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]


def reorder_error_instructions(dem, reverse=False):
    """Return a DEM with only complete error instructions reordered."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_positions(lines)
    errors = [lines[i] for i in positions]
    if reverse:
        ordered = list(reversed(errors))
    else:
        def key(line):
            match = re.match(r"\s*error\(([^)]+)\)", line)
            if not match:
                raise ValueError(f"Cannot parse error instruction: {line.strip()}")
            return (-float(match.group(1)), line.strip())
        ordered = sorted(errors, key=key)
    for pos, line in zip(positions, ordered):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Reordering changed DEM dimensions")
    return result


def decode_in_chunks(decoder, detector_bits, enable_correlations=False):
    outputs = []
    for start in range(0, len(detector_bits), 25_000):
        batch = detector_bits[start:start + 25_000]
        if enable_correlations:
            decoded = decoder.decode_batch(batch, enable_correlations=True)
        else:
            decoded = decoder.decode_batch(batch)
        outputs.append(as_2d(decoded))
    return np.concatenate(outputs, axis=0)


def append_rows(path, rows, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    if DISTANCE != 3:
        raise ValueError("This comparison's exact DEM-MAP is intentionally distance-3 only")
    print("Fresh-seed exact DEM-MAP comparison at distance 3")
    print(f"distance={DISTANCE}; rounds={ROUNDS}; P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; shots/condition={SHOTS:,}")
    print("DEM-MAP preserves each complete flattened error instruction as one Bernoulli event.")
    print("All decoders are tested against the same circuit samples within each condition.\n")

    if RAW_CSV.exists():
        RAW_CSV.unlink()

    fieldnames = [
        "distance", "rounds", "p_base", "p_corr", "seed", "shots", "decoder",
        "failures", "logical_failure_rate", "exact_map_bayes_risk", "excess_vs_exact_map",
        "excess_vs_exact_map_ci_low", "excess_vs_exact_map_ci_high",
        "delta_vs_ordinary_original", "delta_vs_ordinary_original_ci_low",
        "delta_vs_ordinary_original_ci_high", "prediction_disagreements_vs_exact_map",
        "prediction_disagreements_vs_ordinary_original", "exact_map_ties_on_sample",
        "num_dem_mechanisms", "joint_distribution_normalization", "decode_seconds",
        "canonical_equal_from_reversed",
    ]

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)

    for p_idx, p_corr in enumerate(P_CORR_VALUES):
        seed = SEED_BASE + p_idx
        condition_started = time.time()
        print("\n" + "=" * 82)
        print(f"CONDITION p_corr={p_corr:g}; circuit sampler seed={seed}")

        circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
        dem = circuit.detector_error_model(decompose_errors=True).flattened()
        if dem.num_detectors != 24 or dem.num_observables != 1:
            raise ValueError(
                f"Expected distance-3 DEM with 24 detectors and one observable; got "
                f"{dem.num_detectors} detectors, {dem.num_observables} observables"
            )

        sorted_dem = reorder_error_instructions(dem, reverse=False)
        sorted_from_reversed = reorder_error_instructions(
            reorder_error_instructions(dem, reverse=True), reverse=False
        )
        canonical_equal = str(sorted_dem) == str(sorted_from_reversed)
        if not canonical_equal:
            raise AssertionError("Canonical sorting failed to normalize the reversed DEM")
        print(f"DEM instructions: {sum(line.lstrip().startswith('error(') for line in str(dem).splitlines()):,}; canonical sort invariant={canonical_equal}")

        # The exact joint distribution and risk are model-based, not estimated
        # from the sampled circuit shots.
        q0, q1, normalization, bayes_risk = exact.exact_joint_distribution(dem)
        mechanisms, _zero_p, _neutral = exact.parse_full_dem_mechanisms(dem)
        n_mechanisms = len(mechanisms)

        print("\nBuilding decoders...")
        decoders = {
            "ordinary/original": (pymatching.Matching(dem), False),
            "ordinary/canonical-sorted": (pymatching.Matching(sorted_dem), False),
            "probability-prioritized": (exact.build_probability_prioritized_matching(sorted_dem), False),
        }
        correlation_aware_available = True
        try:
            decoders["correlation-aware/original"] = (pymatching.Matching(dem, enable_correlations=True), True)
            decoders["correlation-aware/canonical-sorted"] = (pymatching.Matching(sorted_dem, enable_correlations=True), True)
        except (TypeError, ValueError) as exc:
            correlation_aware_available = False
            decoders.pop("correlation-aware/original", None)
            decoders.pop("correlation-aware/canonical-sorted", None)
            print(f"WARNING: correlation-aware MWPM unavailable in installed PyMatching: {exc}")

        print("Sampling fresh circuit shots...")
        sampler = circuit.compile_detector_sampler(seed=seed)
        detector_bits, observable_bits = sampler.sample(shots=SHOTS, separate_observables=True)
        detector_bits = np.asarray(detector_bits, dtype=np.uint8)
        actual = as_2d(observable_bits)
        syndrome_indices = exact.encode_syndromes(detector_bits)

        prob0 = q0[syndrome_indices]
        prob1 = q1[syndrome_indices]
        map_pred = (prob1 > prob0).astype(np.uint8)[:, None]  # deterministic tie -> L0=0
        map_ties = int(np.count_nonzero(prob0 == prob1))

        predictions = {}
        decode_times = {}
        for name in DECODER_ORDER:
            if name == "exact DEM-MAP":
                predictions[name] = map_pred
                decode_times[name] = 0.0
                continue
            if name not in decoders:
                continue
            decoder, use_correlations = decoders[name]
            t0 = time.time()
            predictions[name] = decode_in_chunks(decoder, detector_bits, use_correlations)
            decode_times[name] = time.time() - t0
            print(f"  decoded {name:36s} in {decode_times[name]:.2f}s")

        if "exact DEM-MAP" not in predictions or "ordinary/original" not in predictions:
            raise RuntimeError("Required decoder predictions missing")
        failure_masks = {
            name: failure_mask(pred, actual)
            for name, pred in predictions.items()
        }
        map_fail = failure_masks["exact DEM-MAP"]
        original_fail = failure_masks["ordinary/original"]
        original_pred = predictions["ordinary/original"]

        print("\nDecoder results (paired on identical circuit shots)")
        print("decoder                                failures      rate       delta vs MAP (95% CI)          MAP disagreements")
        print("-" * 120)
        rows = []
        for name in DECODER_ORDER:
            if name not in predictions:
                continue
            pred = predictions[name]
            fail = failure_masks[name]
            n_fail = int(np.count_nonzero(fail))
            rate = float(np.mean(fail))
            delta_map, ci_map = paired_delta_ci(fail, map_fail)
            delta_original, ci_original = paired_delta_ci(fail, original_fail)
            disagreements_map = int(np.count_nonzero(np.any(pred != map_pred, axis=1)))
            disagreements_original = int(np.count_nonzero(np.any(pred != original_pred, axis=1)))
            print(
                f"{name:38s} {n_fail:8,d}  {rate:9.5%}  "
                f"{delta_map * 100:+.5f} pp [{ci_map[0] * 100:+.5f}, {ci_map[1] * 100:+.5f}]  "
                f"{disagreements_map:8,d}"
            )
            rows.append({
                "distance": DISTANCE,
                "rounds": ROUNDS,
                "p_base": P_BASE,
                "p_corr": p_corr,
                "seed": seed,
                "shots": SHOTS,
                "decoder": name,
                "failures": n_fail,
                "logical_failure_rate": rate,
                "exact_map_bayes_risk": bayes_risk,
                "excess_vs_exact_map": delta_map,
                "excess_vs_exact_map_ci_low": ci_map[0],
                "excess_vs_exact_map_ci_high": ci_map[1],
                "delta_vs_ordinary_original": delta_original,
                "delta_vs_ordinary_original_ci_low": ci_original[0],
                "delta_vs_ordinary_original_ci_high": ci_original[1],
                "prediction_disagreements_vs_exact_map": disagreements_map,
                "prediction_disagreements_vs_ordinary_original": disagreements_original,
                "exact_map_ties_on_sample": map_ties,
                "num_dem_mechanisms": n_mechanisms,
                "joint_distribution_normalization": normalization,
                "decode_seconds": decode_times[name],
                "canonical_equal_from_reversed": canonical_equal,
            })

        append_rows(RAW_CSV, rows, fieldnames)
        print(f"\nExact DEM-MAP Bayes risk: {bayes_risk:.6%} ({bayes_risk * 100:.6f} percentage points)")
        print(f"Observed exact-MAP failures: {int(np.count_nonzero(map_fail)):,}/{SHOTS:,} ({np.mean(map_fail):.5%})")
        print(f"MAP ties on sampled syndromes: {map_ties:,}; DEM normalization={normalization:.16f}")
        print(f"Condition elapsed: {time.time() - condition_started:.1f}s")
        print(f"Raw results saved incrementally to: {RAW_CSV.resolve()}")

        # Release the dense posterior tables before the next p_corr condition.
        del q0, q1, prob0, prob1, map_pred, syndrome_indices, predictions, failure_masks
        del detector_bits, observable_bits, actual, decoders, rows, circuit, dem, sorted_dem, sorted_from_reversed

    print("\nCompleted. Interpret paired delta as decoder failure risk minus exact-MAP failure risk on these same circuit shots.")
    print("Exact DEM-MAP is Bayes-optimal under the flattened DEM distribution; the circuit-sample rate is an empirical transfer check.")
    print("The result is distance-specific and is not a threshold or asymptotic-scaling estimate.")
    print(f"Results file: {RAW_CSV.resolve()}")


if __name__ == "__main__":
    main()
