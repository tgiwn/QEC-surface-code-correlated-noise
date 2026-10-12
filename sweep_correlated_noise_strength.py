"""Sweep injected pair-correlation strength to quantify the residual penalty.

Uses the same P_BASE and circuit construction as correlated_experiment.py.
At distance 3, compares ordinary MWPM with the probability-prioritized graph
because the ordinary DEM has conflicting singleton logical labels. At distances
5 and 7, compares ordinary MWPM with correlation-aware matching.

This is a diagnostic, not an exact maximum-likelihood decoder. In particular,
the probability-prioritized construction treats graphlike decomposition
components as matching edges and does not preserve all source-level correlations.
Run from the qec-correlated-noise project directory.
"""

import math
import re
from collections import defaultdict

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
P_CORR_VALUES = (0.0, 0.00005, 0.0001, 0.0002)
DISTANCES = (3, 5, 7)
SHOTS_BY_DISTANCE = {3: 250_000, 5: 250_000, 7: 1_000_000}
CHUNK_SIZE = 25_000
SEED_BASE = 20261022


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


def paired_delta_ci(candidate_fail, baseline_fail):
    """Normal approximate CI for paired candidate-minus-baseline failure rate."""
    diff = candidate_fail.astype(np.int8) - baseline_fail.astype(np.int8)
    delta = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return delta, (delta - 1.96 * se, delta + 1.96 * se)


def wilson_interval(successes, trials, z=1.96):
    """Wilson score interval for one binomial rate."""
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_difference_ci(a, n_a, b, n_b, z=1.96):
    """Newcombe hybrid-score CI for independent rates p_a - p_b."""
    p_a = a / n_a
    p_b = b / n_b
    la, ua = wilson_interval(a, n_a, z)
    lb, ub = wilson_interval(b, n_b, z)
    diff = p_a - p_b
    low = diff - math.sqrt((p_a - la) ** 2 + (ub - p_b) ** 2)
    high = diff + math.sqrt((ua - p_a) ** 2 + (p_b - lb) ** 2)
    return diff, (low, high)


def reverse_error_order(dem):
    """Reverse flattened error instructions; leave their contents unchanged."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    old = [lines[i] for i in positions]
    for i, line in zip(positions, reversed(old)):
        lines[i] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (flat.num_detectors != result.num_detectors or
            flat.num_observables != result.num_observables):
        raise ValueError("Reordering changed DEM dimensions")
    return result


def extract_graphlike_components(dem):
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
            raise ValueError(f"Expected 0 < DEM probability < 0.5, got {p}: {line.strip()}")
        for piece in re.split(r"\s+\^\s+", body):
            dets = tuple(sorted(int(x) for x in re.findall(r"\bD(\d+)\b", piece)))
            labels = tuple(sorted(int(x) for x in re.findall(r"\bL(\d+)\b", piece)))
            if len(dets) not in (1, 2):
                if dets:
                    ignored_non_graphlike += 1
                continue
            records.append((p, dets, labels, line.strip()))
    return records, ignored_non_graphlike, zero_probability


def build_probability_prioritized_matching(dem):
    """Order-independent heuristic for retaining labels on parallel edges."""
    records, ignored, zeros = extract_graphlike_components(dem)
    records.sort(key=lambda rec: (-rec[0], rec[1], rec[2], rec[3]))
    labels_by_signature = defaultdict(set)
    top_tie_classes = 0
    grouped = defaultdict(list)
    for p, dets, labels, source in records:
        labels_by_signature[dets].add(labels)
        grouped[dets].append((p, labels))
    for sig_records in grouped.values():
        max_p = max(p for p, _ in sig_records)
        top_labels = {label for p, label in sig_records if math.isclose(p, max_p, rel_tol=1e-12, abs_tol=1e-15)}
        if len(top_labels) > 1:
            top_tie_classes += 1
    conflicts = sum(len(labels) > 1 for labels in labels_by_signature.values())

    matching = pymatching.Matching()
    for p, dets, labels, _source in records:
        weight = math.log((1 - p) / p)
        fault_ids = set(labels)
        if len(dets) == 1:
            matching.add_boundary_edge(
                dets[0], fault_ids=fault_ids, weight=weight,
                error_probability=p, merge_strategy="independent",
            )
        else:
            matching.add_edge(
                dets[0], dets[1], fault_ids=fault_ids, weight=weight,
                error_probability=p, merge_strategy="independent",
            )
    return matching, {
        "graphlike_components": len(records),
        "conflicting_signatures": conflicts,
        "top_probability_tie_classes": top_tie_classes,
        "ignored_non_graphlike_components": ignored,
        "zero_probability_instructions": zeros,
    }


def decode_batch(decoder, syndrome, correlated=False):
    return as_2d(decoder.decode_batch(syndrome, enable_correlations=correlated))


def main():
    print("Correlation-strength sweep: correlated vs matched-independent noise")
    print(f"P_BASE={P_BASE}; P_CORR_VALUES={P_CORR_VALUES}; seed base={SEED_BASE}")
    print(f"Shots by distance={SHOTS_BY_DISTANCE}; chunk size={CHUNK_SIZE:,}")
    print("Decoder choice: distance 3 uses probability-prioritized MWPM; distances 5/7 use correlation-aware matching.")
    print("Both noise models use the same decoder choice at a given distance. Control-vs-correlated samples are independent; their difference CI is unpaired.")

    summaries = {}
    for distance in DISTANCES:
        shots = SHOTS_BY_DISTANCE[distance]
        base = make_base_circuit(distance=distance, rounds=distance, p=P_BASE)
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)

        for p_idx, p_corr in enumerate(P_CORR_VALUES):
            for model_idx, (model_name, mode) in enumerate([
                ("CONTROL", "matched_independent"),
                ("CORRELATED", "correlated"),
            ]):
                circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
                dem = circuit.detector_error_model(decompose_errors=True).flattened()
                reversed_dem = reverse_error_order(dem)
                ordinary = pymatching.Matching(dem)
                ordinary_reversed = pymatching.Matching(reversed_dem)

                if distance == 3:
                    selected, graph_stats = build_probability_prioritized_matching(dem)
                    selected_reversed, _ = build_probability_prioritized_matching(reversed_dem)
                    selected_is_correlated = False
                    selected_name = "priority"
                else:
                    selected = pymatching.Matching(dem, enable_correlations=True)
                    selected_reversed = pymatching.Matching(reversed_dem, enable_correlations=True)
                    selected_is_correlated = True
                    selected_name = "correlation-aware"
                    graph_stats = {"conflicting_signatures": 0}

                seed = SEED_BASE + 10_000 * distance + 100 * p_idx + model_idx
                sampler = circuit.compile_detector_sampler(seed=seed)
                ordinary_failures = 0
                selected_failures = 0
                selected_reverse_failures = 0
                selected_changed_prediction_count = 0
                priority_vs_ordinary_disagree = 0
                delta_samples = []
                processed = 0

                while processed < shots:
                    batch_size = min(CHUNK_SIZE, shots - processed)
                    syndrome, actual_obs = sampler.sample(
                        shots=batch_size, separate_observables=True
                    )
                    syndrome = np.asarray(syndrome, dtype=np.uint8)
                    actual = as_2d(actual_obs)
                    ordinary_pred = decode_batch(ordinary, syndrome)
                    ordinary_rev_pred = decode_batch(ordinary_reversed, syndrome)
                    if selected_is_correlated:
                        chosen_pred = decode_batch(selected, syndrome, correlated=True)
                        chosen_rev_pred = decode_batch(selected_reversed, syndrome, correlated=True)
                    else:
                        chosen_pred = decode_batch(selected, syndrome)
                        chosen_rev_pred = decode_batch(selected_reversed, syndrome)

                    ordinary_fail = failure_mask(ordinary_pred, actual)
                    chosen_fail = failure_mask(chosen_pred, actual)
                    chosen_rev_fail = failure_mask(chosen_rev_pred, actual)
                    ordinary_failures += int(np.count_nonzero(ordinary_fail))
                    selected_failures += int(np.count_nonzero(chosen_fail))
                    selected_reverse_failures += int(np.count_nonzero(chosen_rev_fail))
                    selected_changed_prediction_count += int(np.count_nonzero(np.any(chosen_pred != chosen_rev_pred, axis=1)))
                    priority_vs_ordinary_disagree += int(np.count_nonzero(np.any(chosen_pred != ordinary_pred, axis=1)))
                    delta_samples.append(chosen_fail.astype(np.int8) - ordinary_fail.astype(np.int8))
                    processed += batch_size

                delta_vec = np.concatenate(delta_samples)
                delta_pp = float(np.mean(delta_vec)) * 100
                delta_se = float(np.std(delta_vec, ddof=1) / math.sqrt(shots)) * 100
                paired_ci_pp = (delta_pp - 1.96 * delta_se, delta_pp + 1.96 * delta_se)
                rate_ci = wilson_interval(selected_failures, shots)
                summaries[(distance, p_corr, model_name)] = {
                    "shots": shots,
                    "ordinary_failures": ordinary_failures,
                    "selected_failures": selected_failures,
                    "selected_rate": selected_failures / shots,
                    "selected_ci": rate_ci,
                    "selected_reverse_failures": selected_reverse_failures,
                    "selected_order_disagreements": selected_changed_prediction_count,
                    "selected_vs_ordinary_disagreements": priority_vs_ordinary_disagree,
                    "paired_delta_pp": delta_pp,
                    "paired_delta_ci_pp": paired_ci_pp,
                }

                print(f"\nD={distance} | P_CORR={p_corr:.8g} | {model_name} | shots={shots:,} | seed={seed}")
                print(f"detectors={dem.num_detectors}; pairs={len(pairs)}; decoder={selected_name}; conflicting signatures in priority audit={graph_stats.get('conflicting_signatures', 'n/a')}")
                print(f"ordinary/original: {ordinary_failures:,} ({ordinary_failures / shots:.6%})")
                print(f"selected/original: {selected_failures:,} ({selected_failures / shots:.6%}); Wilson 95% CI [{rate_ci[0]:.6%}, {rate_ci[1]:.6%}]")
                print(f"selected/reversed: {selected_reverse_failures:,} ({selected_reverse_failures / shots:.6%})")
                print(f"selected - ordinary paired delta: {delta_pp:+.5f} pp [{paired_ci_pp[0]:+.5f}, {paired_ci_pp[1]:+.5f}] pp")
                print(f"selected predictions differing from ordinary: {priority_vs_ordinary_disagree:,}/{shots:,}")
                print(f"selected predictions differing by DEM order: {selected_changed_prediction_count:,}/{shots:,}")

    print("\n\n========== CORRELATED MINUS CONTROL: SELECTED DECODER ==========")
    print("Unpaired Newcombe 95% confidence intervals for difference in rates; positive means correlated has a higher failure rate.")
    print("distance | P_CORR | control failures/rate | correlated failures/rate | excess correlated rate (95% CI)")
    print("-" * 124)
    for distance in DISTANCES:
        for p_corr in P_CORR_VALUES:
            control = summaries[(distance, p_corr, "CONTROL")]
            corr = summaries[(distance, p_corr, "CORRELATED")]
            diff, ci = newcombe_difference_ci(
                corr["selected_failures"], corr["shots"],
                control["selected_failures"], control["shots"],
            )
            print(
                f"{distance:8d} | {p_corr:7.5f} | "
                f"{control['selected_failures']:7,d}/{control['selected_rate']:.6%} | "
                f"{corr['selected_failures']:7,d}/{corr['selected_rate']:.6%} | "
                f"{diff * 100:+.5f} pp [{ci[0] * 100:+.5f}, {ci[1] * 100:+.5f}] pp"
            )

    print("\nInterpretation cautions:")
    print("- This is a dose-response diagnostic, not a threshold or scaling study.")
    print("- The distance-3 priority graph is a heuristic, not exact maximum-likelihood decoding.")
    print("- Correlation-aware matching is also not exact maximum-likelihood decoding.")
    print("- DEM components separated by '^' remain correlated in the physical mechanism; the priority graph does not fully model that correlation.")


if __name__ == "__main__":
    main()
