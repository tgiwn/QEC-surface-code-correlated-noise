"""Test whether canonical probability-ordering stabilizes the DEM-based MWPM decoder.

The test is motivated by strong order sensitivity observed at distance 3. It compares:
  * ordinary MWPM from the original and reversed DEM;
  * ordinary MWPM from a canonical descending-probability DEM;
  * correlation-aware MWPM from original, reversed, and canonical sorted DEMs;
  * the previously tested manually probability-prioritized graph heuristic.

Each decoder receives identical sampled syndromes and actual observables for a condition.
This is a finite-distance diagnostic, not a threshold estimate.

Run from the project directory containing correlated_experiment.py:
    & $python .\\test_probability_sorted_dem_order.py
"""
from __future__ import annotations

import csv
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

# --------------------------- Experiment settings ---------------------------
P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCES = (3, 5)
REPLICATES = 3
SHOTS_BY_DISTANCE = {3: 500_000, 5: 500_000}
CHUNK_SIZE = 25_000
SEED_BASE = 31_000_000
RAW_CSV = Path("sorted_dem_decoder_raw.csv")
SUMMARY_CSV = Path("sorted_dem_decoder_summary.csv")
MODEL_CONFIGS = (
    ("CONTROL matched-independent", "matched_independent"),
    ("CORRELATED", "correlated"),
)
DECODERS = (
    "ordinary/original",
    "ordinary/reversed",
    "ordinary/probability-sorted",
    "correlation-aware/original",
    "correlation-aware/reversed",
    "correlation-aware/probability-sorted",
    "priority/original",
    "priority/reversed",
)
BASELINE = "ordinary/original"


def as_2d(values):
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def failure_mask(predictions, actual):
    pred = as_2d(predictions)
    obs = as_2d(actual)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, actual={obs.shape}")
    return np.any(pred != obs, axis=1)


def paired_delta_ci(candidate_only: int, baseline_only: int, shots: int):
    """Paired normal 95% CI for candidate-minus-baseline risk."""
    delta = (candidate_only - baseline_only) / shots
    discordant = candidate_only + baseline_only
    if shots <= 1:
        return delta, (float("nan"), float("nan"))
    sample_var = max(0.0, (discordant - shots * delta * delta) / (shots - 1))
    se = math.sqrt(sample_var / shots)
    return delta, (delta - 1.96 * se, delta + 1.96 * se)


def wilson95(k: int, n: int):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def error_instruction_positions(lines):
    return [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]


def reverse_error_order(dem):
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_instruction_positions(lines)
    saved = [lines[i] for i in positions]
    for pos, line in zip(positions, reversed(saved)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Reversing DEM order changed its dimensions")
    return result


def canonical_probability_sorted_dem(dem):
    """Sort complete error instructions by descending p, then canonical line text.

    Whole error instructions are moved intact; terms separated by '^' are never split.
    The line text tie-break makes the result identical regardless of original order.
    """
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_instruction_positions(lines)
    error_lines = [lines[i] for i in positions]

    def sort_key(line):
        match = re.match(r"\s*error\(([^)]+)\)", line)
        if not match:
            raise ValueError(f"Could not parse DEM error instruction: {line}")
        return (-float(match.group(1)), line.strip())

    ordered = sorted(error_lines, key=sort_key)
    for pos, line in zip(positions, ordered):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Sorting DEM order changed its dimensions")
    return result


def extract_components(dem):
    records = []
    ignored_non_graphlike = 0
    for line in str(dem.flattened()).splitlines():
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue
        p = float(match.group(1))
        body = match.group(2)
        if p == 0:
            continue
        if not 0 < p < 0.5:
            raise ValueError(f"Expected 0 < p < 0.5, got {p}: {line.strip()}")
        for piece in re.split(r"\s+\^\s+", body):
            detectors = tuple(sorted(int(x) for x in re.findall(r"\bD(\d+)\b", piece)))
            labels = tuple(sorted(int(x) for x in re.findall(r"\bL(\d+)\b", piece)))
            if len(detectors) not in (1, 2):
                if detectors:
                    ignored_non_graphlike += 1
                continue
            records.append((p, detectors, labels, line.strip()))
    return records, ignored_non_graphlike


def build_probability_prioritized_matching(dem):
    records, ignored = extract_components(dem)
    # Canonical deterministic ordering: higher p first, with stable tie-breaking.
    records.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    matching = pymatching.Matching()
    labels_by_signature = defaultdict(set)
    for p, detectors, labels, _source in records:
        labels_by_signature[detectors].add(labels)
        weight = math.log((1 - p) / p)
        if len(detectors) == 1:
            matching.add_boundary_edge(
                detectors[0], fault_ids=set(labels), weight=weight,
                error_probability=p, merge_strategy="independent",
            )
        else:
            matching.add_edge(
                detectors[0], detectors[1], fault_ids=set(labels), weight=weight,
                error_probability=p, merge_strategy="independent",
            )
    conflict_classes = sum(len(labels) > 1 for labels in labels_by_signature.values())
    return matching, len(records), conflict_classes, ignored


def count_conflicts(dem):
    labels_by_signature = defaultdict(set)
    for p, detectors, labels, _source in extract_components(dem)[0]:
        if detectors:
            labels_by_signature[detectors].add(labels)
    return sum(len(labels) > 1 for labels in labels_by_signature.values())


def get_seed(distance, p_idx, replicate, model_idx):
    return int(SEED_BASE + distance * 1_000_000 + p_idx * 100_000 + replicate * 1_000 + model_idx)


def evaluate_condition(distance, p_corr, p_idx, replicate, model_name, mode, model_idx):
    shots = SHOTS_BY_DISTANCE[distance]
    seed = get_seed(distance, p_idx, replicate, model_idx)

    base = make_base_circuit(distance=distance, rounds=distance, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
    dem = circuit.detector_error_model(decompose_errors=True).flattened()
    reversed_dem = reverse_error_order(dem)
    sorted_dem = canonical_probability_sorted_dem(dem)
    sorted_from_reversed = canonical_probability_sorted_dem(reversed_dem)
    sorted_dem_canonical_equal = str(sorted_dem) == str(sorted_from_reversed)
    if not sorted_dem_canonical_equal:
        raise AssertionError("Canonical sorting did not produce the same DEM from reversed input")

    priority_original, priority_n, priority_conflicts, ignored = build_probability_prioritized_matching(dem)
    priority_reversed, _, _, _ = build_probability_prioritized_matching(reversed_dem)

    decoders = {
        "ordinary/original": pymatching.Matching(dem),
        "ordinary/reversed": pymatching.Matching(reversed_dem),
        "ordinary/probability-sorted": pymatching.Matching(sorted_dem),
        "correlation-aware/original": pymatching.Matching(dem, enable_correlations=True),
        "correlation-aware/reversed": pymatching.Matching(reversed_dem, enable_correlations=True),
        "correlation-aware/probability-sorted": pymatching.Matching(sorted_dem, enable_correlations=True),
        "priority/original": priority_original,
        "priority/reversed": priority_reversed,
    }

    failures = {name: 0 for name in DECODERS}
    runtimes = {name: 0.0 for name in DECODERS}
    candidate_only = {name: 0 for name in DECODERS}
    baseline_only = {name: 0 for name in DECODERS}
    prediction_changes_vs_baseline = {name: 0 for name in DECODERS}
    order_disagreements = {
        "ordinary": 0,
        "correlation-aware": 0,
        "priority": 0,
    }

    sampler = circuit.compile_detector_sampler(seed=seed)
    processed = 0
    start = time.perf_counter()
    while processed < shots:
        n = min(CHUNK_SIZE, shots - processed)
        syndrome, actual_obs = sampler.sample(shots=n, separate_observables=True)
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        actual = as_2d(actual_obs)

        preds = {}
        for name, decoder in decoders.items():
            t0 = time.perf_counter()
            if name.startswith("correlation-aware"):
                pred = decoder.decode_batch(syndrome, enable_correlations=True)
            else:
                pred = decoder.decode_batch(syndrome)
            runtimes[name] += time.perf_counter() - t0
            preds[name] = as_2d(pred)

        masks = {name: failure_mask(preds[name], actual) for name in DECODERS}
        base_mask = masks[BASELINE]
        base_pred = preds[BASELINE]
        for name in DECODERS:
            failures[name] += int(np.count_nonzero(masks[name]))
            prediction_changes_vs_baseline[name] += int(np.count_nonzero(np.any(preds[name] != base_pred, axis=1)))
            candidate_only[name] += int(np.count_nonzero(masks[name] & ~base_mask))
            baseline_only[name] += int(np.count_nonzero(base_mask & ~masks[name]))

        for family in order_disagreements:
            order_disagreements[family] += int(np.count_nonzero(np.any(
                preds[f"{family}/original"] != preds[f"{family}/reversed"], axis=1
            )))
        processed += n

    elapsed = time.perf_counter() - start
    row = {
        "distance": distance,
        "p_base": P_BASE,
        "p_corr": p_corr,
        "model": model_name,
        "mode": mode,
        "replicate": replicate,
        "seed": seed,
        "shots": shots,
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "num_pairs": len(pairs),
        "dem_error_instructions": sum(1 for line in str(dem).splitlines() if line.lstrip().startswith("error(")),
        "conflicting_signatures": count_conflicts(dem),
        "priority_graphlike_components": priority_n,
        "priority_conflicting_signatures": priority_conflicts,
        "priority_ignored_non_graphlike_components": ignored,
        "sorted_dem_identical_from_reversed_input": sorted_dem_canonical_equal,
        "elapsed_seconds": elapsed,
    }
    for name in DECODERS:
        key = name.replace("/", "_").replace("-", "_").replace(" ", "_")
        lo, hi = wilson95(failures[name], shots)
        delta, (dlo, dhi) = paired_delta_ci(candidate_only[name], baseline_only[name], shots)
        row[f"{key}_failures"] = failures[name]
        row[f"{key}_rate_pct"] = 100 * failures[name] / shots
        row[f"{key}_wilson_lo_pct"] = 100 * lo
        row[f"{key}_wilson_hi_pct"] = 100 * hi
        row[f"{key}_prediction_changes_vs_baseline"] = prediction_changes_vs_baseline[name]
        row[f"{key}_candidate_only_vs_baseline"] = candidate_only[name]
        row[f"{key}_baseline_only_vs_candidate"] = baseline_only[name]
        row[f"{key}_delta_vs_baseline_pp"] = 100 * delta
        row[f"{key}_paired_ci_lo_pp"] = 100 * dlo
        row[f"{key}_paired_ci_hi_pp"] = 100 * dhi
        row[f"{key}_decode_seconds"] = runtimes[name]
    for family, count in order_disagreements.items():
        row[f"{family.replace('-', '_')}_order_disagreements"] = count

    print(
        f"  d={distance} p_corr={p_corr:.5g} | {model_name:29s} | rep={replicate} "
        f"seed={seed} | ordinary={failures[BASELINE]}/{shots} ({100*failures[BASELINE]/shots:.5f}%) | "
        f"ordinary-sorted={failures['ordinary/probability-sorted']}/{shots} "
        f"({100*failures['ordinary/probability-sorted']/shots:.5f}%) | "
        f"CA-original={failures['correlation-aware/original']}/{shots} "
        f"({100*failures['correlation-aware/original']/shots:.5f}%) | "
        f"CA-sorted={failures['correlation-aware/probability-sorted']}/{shots} "
        f"({100*failures['correlation-aware/probability-sorted']/shots:.5f}%) | "
        f"priority={failures['priority/original']}/{shots} ({100*failures['priority/original']/shots:.5f}%) | "
        f"order O/CA/P={order_disagreements['ordinary']}/{order_disagreements['correlation-aware']}/"
        f"{order_disagreements['priority']} | {elapsed:.1f}s",
        flush=True,
    )
    return row


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(int(row["distance"]), float(row["p_corr"]), row["model"])].append(row)
    families = (
        "ordinary/reversed",
        "ordinary/probability-sorted",
        "correlation-aware/original",
        "correlation-aware/reversed",
        "correlation-aware/probability-sorted",
        "priority/original",
        "priority/reversed",
    )
    summaries = []
    for (distance, p_corr, model_name), rows in sorted(grouped.items()):
        n = sum(int(r["shots"]) for r in rows)
        baseline_fail = sum(int(r["ordinary_original_failures"]) for r in rows)
        record = {
            "distance": distance,
            "p_base": P_BASE,
            "p_corr": p_corr,
            "model": model_name,
            "replicates": len(rows),
            "total_shots": n,
            "ordinary_original_failures": baseline_fail,
            "ordinary_original_rate_pct": 100 * baseline_fail / n,
            "sorted_dem_identical_from_reversed_for_all_replicates": all(bool(r["sorted_dem_identical_from_reversed_input"]) for r in rows),
        }
        lo, hi = wilson95(baseline_fail, n)
        record["ordinary_original_wilson_lo_pct"] = 100 * lo
        record["ordinary_original_wilson_hi_pct"] = 100 * hi
        for name in families:
            key = name.replace("/", "_").replace("-", "_").replace(" ", "_")
            fail = sum(int(r[f"{key}_failures"]) for r in rows)
            cand_only = sum(int(r[f"{key}_candidate_only_vs_baseline"]) for r in rows)
            base_only = sum(int(r[f"{key}_baseline_only_vs_candidate"]) for r in rows)
            delta, (dlo, dhi) = paired_delta_ci(cand_only, base_only, n)
            rlo, rhi = wilson95(fail, n)
            record[f"{key}_failures"] = fail
            record[f"{key}_rate_pct"] = 100 * fail / n
            record[f"{key}_wilson_lo_pct"] = 100 * rlo
            record[f"{key}_wilson_hi_pct"] = 100 * rhi
            record[f"{key}_delta_vs_ordinary_pp"] = 100 * delta
            record[f"{key}_paired_ci_lo_pp"] = 100 * dlo
            record[f"{key}_paired_ci_hi_pp"] = 100 * dhi
            record[f"{key}_prediction_changes_total"] = sum(int(r[f"{key}_prediction_changes_vs_baseline"]) for r in rows)
        for family in ("ordinary", "correlation_aware", "priority"):
            record[f"{family}_order_disagreements_total"] = sum(int(r[f"{family}_order_disagreements"]) for r in rows)
        summaries.append(record)
    return summaries


def main():
    total_shots = REPLICATES * len(P_CORR_VALUES) * len(MODEL_CONFIGS) * sum(SHOTS_BY_DISTANCE.values())
    conditions = len(DISTANCES) * len(P_CORR_VALUES) * len(MODEL_CONFIGS) * REPLICATES
    print("Canonical probability-sorted DEM test: ordinary/correlation-aware/priority MWPM")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances={DISTANCES}; replicates={REPLICATES}")
    print(f"Shots={SHOTS_BY_DISTANCE}; chunk={CHUNK_SIZE:,}; conditions={conditions}; total circuit shots={total_shots:,}")
    print("Tests whether sorting complete error instructions by descending probability stabilizes distance-3 decoding.")
    print("The sorted DEM tie-break is canonical by instruction text; '^'-joined mechanisms remain intact.")
    print(f"Raw output: {RAW_CSV.resolve()}")

    raw_rows = []
    with RAW_CSV.open("w", newline="", encoding="utf-8") as raw_file:
        writer = None
        index = 0
        for distance in DISTANCES:
            for p_idx, p_corr in enumerate(P_CORR_VALUES):
                for replicate in range(1, REPLICATES + 1):
                    for model_idx, (model_name, mode) in enumerate(MODEL_CONFIGS):
                        index += 1
                        print(
                            f"\n[{index}/{conditions}] d={distance}, p_corr={p_corr:.5g}, "
                            f"replicate={replicate}, model={model_name}", flush=True
                        )
                        row = evaluate_condition(
                            distance, p_corr, p_idx, replicate, model_name, mode, model_idx
                        )
                        raw_rows.append(row)
                        if writer is None:
                            writer = csv.DictWriter(raw_file, fieldnames=list(row.keys()))
                            writer.writeheader()
                        writer.writerow(row)
                        raw_file.flush()

    summaries = summarize(raw_rows)
    write_csv(SUMMARY_CSV, summaries)
    print("\n================ POOLED SUMMARY ================")
    for row in summaries:
        print(
            f"d={row['distance']} p_corr={row['p_corr']:.5g} {row['model']}: "
            f"ordinary={row['ordinary_original_rate_pct']:.5f}%; "
            f"ordinary-sorted={row['ordinary_probability_sorted_rate_pct']:.5f}% "
            f"delta={row['ordinary_probability_sorted_delta_vs_ordinary_pp']:+.5f} pp "
            f"[{row['ordinary_probability_sorted_paired_ci_lo_pp']:+.5f}, "
            f"{row['ordinary_probability_sorted_paired_ci_hi_pp']:+.5f}]; "
            f"CA-original={row['correlation_aware_original_rate_pct']:.5f}%; "
            f"CA-sorted={row['correlation_aware_probability_sorted_rate_pct']:.5f}% "
            f"delta={row['correlation_aware_probability_sorted_delta_vs_ordinary_pp']:+.5f} pp "
            f"[{row['correlation_aware_probability_sorted_paired_ci_lo_pp']:+.5f}, "
            f"{row['correlation_aware_probability_sorted_paired_ci_hi_pp']:+.5f}]; "
            f"priority={row['priority_original_rate_pct']:.5f}% "
            f"delta={row['priority_original_delta_vs_ordinary_pp']:+.5f} pp "
            f"[{row['priority_original_paired_ci_lo_pp']:+.5f}, "
            f"{row['priority_original_paired_ci_hi_pp']:+.5f}]"
        )
        print(
            f"  order disagreements across pooled shots: ordinary={row['ordinary_order_disagreements_total']}, "
            f"correlation-aware={row['correlation_aware_order_disagreements_total']}, "
            f"priority={row['priority_order_disagreements_total']}; "
            f"canonical DEM equality checks passed={row['sorted_dem_identical_from_reversed_for_all_replicates']}"
        )
    print(f"\nRaw results: {RAW_CSV.resolve()}")
    print(f"Pooled summary: {SUMMARY_CSV.resolve()}")
    print("Interpretation warning: positive delta means worse than ordinary/original; negative favors candidate.")
    print("This is a decoder-order diagnostic, not a threshold or asymptotic-scaling study.")


if __name__ == "__main__":
    main()
