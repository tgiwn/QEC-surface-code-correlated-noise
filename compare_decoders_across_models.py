"""Compare ordinary and probability-prioritized matching across models/distances.

Diagnostic only: the probability-prioritized graph is an order-invariant
heuristic for choosing labels on parallel graphlike components. It is NOT an
exact maximum-likelihood decoder, and decomposed components from one DEM
instruction remain correlated in the physical model.

Run from the qec-correlated-noise project directory so that
correlated_experiment.py is importable.
"""

import math
import re
from collections import defaultdict

import numpy as np
import pymatching
from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

P_BASE = 0.001
P_CORR = 0.0001
DISTANCES = (3, 5, 7)
SHOTS = 200_000
SEED = 20261018


def as_2d(values):
    values = np.asarray(values, dtype=np.uint8)
    return values[:, None] if values.ndim == 1 else values


def failure_mask(predictions, actual):
    predictions = as_2d(predictions)
    actual = as_2d(actual)
    if predictions.shape != actual.shape:
        raise ValueError(
            f"Prediction/observable shape mismatch: {predictions.shape} vs {actual.shape}"
        )
    return np.any(predictions != actual, axis=1)


def paired_delta_ci(candidate_failures, baseline_failures):
    """Paired difference in binary failure outcomes, candidate minus baseline."""
    delta_samples = (
        candidate_failures.astype(np.int8) - baseline_failures.astype(np.int8)
    )
    delta = float(np.mean(delta_samples))
    if len(delta_samples) < 2:
        return delta, (float("nan"), float("nan"))
    se = float(np.std(delta_samples, ddof=1) / np.sqrt(len(delta_samples)))
    return delta, (delta - 1.96 * se, delta + 1.96 * se)


def reverse_error_order(dem):
    """Reverse flattened error instructions without changing mechanism text."""
    flattened = dem.flattened()
    lines = str(flattened).splitlines()
    error_positions = [
        i for i, line in enumerate(lines) if line.lstrip().startswith("error(")
    ]
    old = [lines[i] for i in error_positions]
    for i, line in zip(error_positions, reversed(old)):
        lines[i] = line
    import stim
    reversed_dem = stim.DetectorErrorModel("\n".join(lines))
    if (flattened.num_detectors != reversed_dem.num_detectors or
            flattened.num_observables != reversed_dem.num_observables):
        raise ValueError("Reordering unexpectedly changed DEM dimensions")
    return reversed_dem


def extract_components(dem):
    """Returns graphlike DEM components (p, detectors, labels, source line)."""
    records = []
    ignored_non_graphlike = 0
    zero_probability = 0

    for line in str(dem.flattened()).splitlines():
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue
        p = float(match.group(1))
        body = match.group(2)
        if p == 0:
            zero_probability += 1
            continue
        if not 0 < p < 0.5:
            raise ValueError(f"Expected 0 < p < 0.5, got {p}: {line.strip()}")

        for piece in re.split(r"\s+\^\s+", body):
            detectors = tuple(sorted(
                int(x) for x in re.findall(r"\bD(\d+)\b", piece)
            ))
            labels = tuple(sorted(
                int(x) for x in re.findall(r"\bL(\d+)\b", piece)
            ))
            if len(detectors) not in (1, 2):
                if detectors:
                    ignored_non_graphlike += 1
                continue
            records.append((p, detectors, labels, line.strip()))

    return records, ignored_non_graphlike, zero_probability


def build_probability_prioritized_matching(dem):
    """Build diagnostic graph, sorting graphlike components by descending p."""
    records, ignored, zeros = extract_components(dem)
    records.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))

    labels_by_signature = defaultdict(set)
    for p, detectors, labels, _source in records:
        labels_by_signature[detectors].add(labels)
    conflicts = sum(1 for labels in labels_by_signature.values() if len(labels) > 1)

    top_tie_classes = 0
    by_signature = defaultdict(list)
    for p, detectors, labels, source in records:
        by_signature[detectors].append((p, labels, source))
    for signature_records in by_signature.values():
        max_p = max(item[0] for item in signature_records)
        top_labels = {
            item[1] for item in signature_records
            if math.isclose(item[0], max_p, rel_tol=1e-12, abs_tol=1e-15)
        }
        if len(top_labels) > 1:
            top_tie_classes += 1

    matching = pymatching.Matching()
    for p, detectors, labels, _source in records:
        weight = math.log((1 - p) / p)
        fault_ids = set(labels)
        if len(detectors) == 1:
            matching.add_boundary_edge(
                detectors[0], fault_ids=fault_ids, weight=weight,
                error_probability=p, merge_strategy="independent",
            )
        else:
            matching.add_edge(
                detectors[0], detectors[1], fault_ids=fault_ids, weight=weight,
                error_probability=p, merge_strategy="independent",
            )

    stats = {
        "graphlike_components": len(records),
        "conflicting_signatures": conflicts,
        "top_probability_tie_classes": top_tie_classes,
        "ignored_non_graphlike_components": ignored,
        "zero_probability_instructions": zeros,
    }
    return matching, stats


def main():
    print(f"Shots per condition: {SHOTS:,}; seed base: {SEED}")
    print(f"P_BASE={P_BASE}; P_CORR={P_CORR}")
    print("Compare ordinary original/reversed DEMs and probability-prioritized graph.")
    print("All decoder comparisons within each condition use the same sampled shots.")

    for distance in DISTANCES:
        base = make_base_circuit(
            distance=distance, rounds=distance, p=P_BASE
        )
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)

        for model_index, (model_name, mode) in enumerate([
            ("CONTROL matched-independent", "matched_independent"),
            ("CORRELATED", "correlated"),
        ]):
            circuit = add_noise_layer(
                base, data_qubits, pairs, P_CORR, mode
            )
            dem = circuit.detector_error_model(decompose_errors=True).flattened()
            reversed_dem = reverse_error_order(dem)

            ordinary_original = pymatching.Matching(dem)
            ordinary_reversed = pymatching.Matching(reversed_dem)
            prioritized_original, stats = build_probability_prioritized_matching(dem)
            prioritized_reversed, _ = build_probability_prioritized_matching(reversed_dem)

            sampler = circuit.compile_detector_sampler(
                seed=SEED + 100 * distance + model_index
            )
            detector_samples, actual = sampler.sample(
                shots=SHOTS, separate_observables=True
            )
            detector_samples = np.asarray(detector_samples, dtype=np.uint8)
            actual = as_2d(actual)

            decoders = {
                "ordinary/original": ordinary_original,
                "ordinary/reversed": ordinary_reversed,
                "probability-prioritized": prioritized_original,
                "priority from reversed DEM": prioritized_reversed,
            }
            predictions = {
                name: as_2d(decoder.decode_batch(detector_samples))
                for name, decoder in decoders.items()
            }
            failures = {
                name: failure_mask(pred, actual)
                for name, pred in predictions.items()
            }
            baseline_name = "ordinary/original"
            baseline_fail = failures[baseline_name]

            print(f"\n========== DISTANCE {distance} | {model_name} ==========")
            print(
                f"detectors={dem.num_detectors}; observables={dem.num_observables}; "
                f"vertical pairs={len(pairs)}"
            )
            print("Graph diagnostics: " + ", ".join(
                f"{key}={value}" for key, value in stats.items()
            ))
            print("decoder                         failures       rate       predictions changed   paired delta vs ordinary/original (95% CI)")
            print("-" * 118)
            for name in decoders:
                fail = failures[name]
                changed = int(np.count_nonzero(np.any(
                    predictions[name] != predictions[baseline_name], axis=1
                )))
                delta, ci = paired_delta_ci(fail, baseline_fail)
                count = int(np.count_nonzero(fail))
                print(
                    f"{name:30s} {count:7,d}   {count / SHOTS:8.5%}"
                    f"        {changed:7,d}              {delta * 100:+.5f} pp "
                    f"[{ci[0] * 100:+.5f}, {ci[1] * 100:+.5f}]"
                )

            priority_order_changes = int(np.count_nonzero(np.any(
                predictions["probability-prioritized"]
                != predictions["priority from reversed DEM"], axis=1
            )))
            print(
                "Probability-prioritized predictions changed by DEM order: "
                f"{priority_order_changes:,} / {SHOTS:,}"
            )


if __name__ == "__main__":
    main()
