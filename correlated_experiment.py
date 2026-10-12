
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pymatching
import stim


def make_base_circuit(distance, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )


def get_data_layout(circuit):
    # In Stim's generated rotated surface-code layout, data
    # qubits have odd integer x and y coordinates.
    coordinates = {}

    for instruction in circuit:
        if getattr(instruction, "name", None) != "QUBIT_COORDS":
            continue

        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        if len(targets) == 1 and len(args) >= 2:
            q = targets[0].value
            x = int(round(args[0]))
            y = int(round(args[1]))
            coordinates[q] = (x, y)

    data_by_xy = {
        (x, y): q
        for q, (x, y) in coordinates.items()
        if x % 2 == 1 and y % 2 == 1
    }

    if not data_by_xy:
        raise RuntimeError("Could not identify data qubits in QUBIT_COORDS.")

    return data_by_xy, set(data_by_xy.values())


def make_correlated_pairs(data_by_xy):
    """Pair vertically adjacent data qubits, counting each pair once."""
    pairs = []

    for (x, y), q1 in sorted(data_by_xy.items()):
        q2 = data_by_xy.get((x, y + 2))

        if q2 is not None:
            pairs.append((q1, q2))

    if not pairs:
        raise RuntimeError("No adjacent data-qubit pairs were found.")

    return pairs

def is_data_noise_layer(instruction, data_qubits):
    # Locate the DEPOLARIZE1 instruction that targets all
    # data qubits. This is the round-boundary data-noise layer.
    if getattr(instruction, "name", None) != "DEPOLARIZE1":
        return False

    targets = instruction.targets_copy()

    return (
        len(targets) == len(data_qubits)
        and {t.value for t in targets} == data_qubits
    )


def count_expanded(circuit, predicate):
    # Count instructions after accounting for REPEAT-block
    # repetition counts.
    count = 0

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            count += instruction.repeat_count * count_expanded(
                instruction.body_copy(), predicate
            )
        elif predicate(instruction):
            count += 1

    return count


def add_noise_layer(base, data_qubits, pairs, p_corr, mode):
    degrees = {q: 0 for q in data_qubits}

    for q1, q2 in pairs:
        degrees[q1] += 1
        degrees[q2] += 1

    def rewrite(source):
        result = stim.Circuit()

        for instruction in source:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                result.append(
                    stim.CircuitRepeatBlock(
                        instruction.repeat_count,
                        rewrite(instruction.body_copy()),
                    )
                )
                continue

            result.append(instruction)

            if not is_data_noise_layer(instruction, data_qubits):
                continue

            if mode == "correlated":
                # Each pair experiences a joint XX event with
                # probability p_corr, independently of other events.
                for q1, q2 in pairs:
                    result.append(
                        "CORRELATED_ERROR",
                        [stim.target_x(q1), stim.target_x(q2)],
                        p_corr,
                    )

            elif mode == "matched_independent":
                # Replace the joint events with independent
                # X errors having identical per-qubit X marginals.
                for degree in sorted(set(degrees.values())):
                    if degree == 0:
                        continue

                    probability = (
                        1 - (1 - 2 * p_corr) ** degree
                    ) / 2

                    qubits = sorted(
                        q for q in data_qubits
                        if degrees[q] == degree
                    )

                    result.append("X_ERROR", qubits, probability)

            else:
                raise ValueError(f"Unknown noise mode: {mode}")

        return result

    return rewrite(base)


def wilson_interval(failures, shots, z=1.96):
    rate = failures / shots
    denominator = 1 + z**2 / shots

    center = (
        rate + z**2 / (2 * shots)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            rate * (1 - rate) / shots
            + z**2 / (4 * shots**2)
        )
        / denominator
    )

    return max(0.0, center - half_width), min(
        1.0, center + half_width
    )


def evaluate(actual, predicted, shots):
    failures = int(
        np.count_nonzero(
            np.any(predicted != actual, axis=1)
        )
    )

    rate = failures / shots
    low, high = wilson_interval(failures, shots)

    return {
        "logical_failures": failures,
        "logical_error_rate": rate,
        "ci95_low": low,
        "ci95_high": high,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--p-corr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    rounds = args.rounds if args.rounds is not None else args.distance

    if args.distance < 3 or args.distance % 2 == 0:
        raise ValueError("Distance must be odd and at least 3.")
    if rounds < 1 or args.shots < 1:
        raise ValueError("Rounds and shots must be positive.")
    if not 0 < args.p < 1:
        raise ValueError("p must be between 0 and 1.")
    if not 0 < args.p_corr < 1:
        raise ValueError("p-corr must be between 0 and 1.")

    start = time.perf_counter()

    base = make_base_circuit(args.distance, rounds, args.p)

    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)

    base_layers = count_expanded(
        base,
        lambda inst: is_data_noise_layer(inst, data_qubits),
    )

    if base_layers != rounds:
        raise RuntimeError(
            f"Expected {rounds} data-noise layers, found {base_layers}. "
            "Inspect the generated circuit before proceeding."
        )

    correlated = add_noise_layer(
        base, data_qubits, pairs, args.p_corr, "correlated"
    )

    matched_independent = add_noise_layer(
        base, data_qubits, pairs, args.p_corr, "matched_independent"
    )

    # Verify that the correlated layer was inserted in every
    # physical round, including rounds inside REPEAT blocks.
    injected_events = count_expanded(
        correlated,
        lambda inst: getattr(inst, "name", None) in {"E", "CORRELATED_ERROR"},
    )

    expected_events = len(pairs) * rounds

    if injected_events != expected_events:
        raise RuntimeError(
            f"Expected {expected_events} correlated pair events "
            f"across all rounds, found {injected_events}."
        )

    # Save all circuit variants for reproducibility.
    circuits_dir = Path("circuits")
    circuits_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"d{args.distance}_r{rounds}"
        f"_p{args.p:g}_pc{args.p_corr:g}"
    )

    base_path = circuits_dir / f"base_{tag}.stim"
    control_path = circuits_dir / f"matched_independent_{tag}.stim"
    corr_path = circuits_dir / f"correlated_XX_{tag}.stim"

    base.to_file(str(base_path))
    matched_independent.to_file(str(control_path))
    correlated.to_file(str(corr_path))

    # Construct a decoder from each physical noise model.
    base_dem = base.detector_error_model(decompose_errors=True)
    control_dem = matched_independent.detector_error_model(
        decompose_errors=True
    )
    corr_dem = correlated.detector_error_model(
        decompose_errors=True
    )

    baseline_decoder = pymatching.Matching.from_detector_error_model(
        base_dem
    )

    independent_decoder = (
        pymatching.Matching.from_detector_error_model(control_dem)
    )

    correlated_model_mwpm = (
        pymatching.Matching.from_detector_error_model(corr_dem)
    )

    # This decoder additionally uses the correlation information
    # represented by decomposable hyperedges in the error model.
    correlated_decoder = (
        pymatching.Matching.from_detector_error_model(
            corr_dem,
            enable_correlations=True,
        )
    )

    # Sample each physical model independently.
    base_detectors, base_observables = (
        base.compile_detector_sampler(seed=args.seed).sample(
            shots=args.shots,
            separate_observables=True,
        )
    )

    control_detectors, control_observables = (
        matched_independent.compile_detector_sampler(
            seed=args.seed + 1
        ).sample(
            shots=args.shots,
            separate_observables=True,
        )
    )

    corr_detectors, corr_observables = (
        correlated.compile_detector_sampler(
            seed=args.seed + 2
        ).sample(
            shots=args.shots,
            separate_observables=True,
        )
    )

    # Decode the samples. These comparisons distinguish a decoder
    # trained on independent noise from one that knows the changed
    # marginal rates and one that also exploits correlations.

    metrics = [
        (
            "baseline decoder on baseline noise",
            base_observables,
            baseline_decoder.decode_batch(base_detectors),
        ),
        (
            "matched-independent decoder on control noise",
            control_observables,
            independent_decoder.decode_batch(control_detectors),
        ),
        (
            "baseline decoder on correlated noise",
            corr_observables,
            baseline_decoder.decode_batch(corr_detectors),
        ),
        (
            "matched-independent decoder on correlated noise",
            corr_observables,
            independent_decoder.decode_batch(corr_detectors),
        ),
        (
            "correlated-model ordinary MWPM",
            corr_observables,
            correlated_model_mwpm.decode_batch(corr_detectors),
        ),
        (
            "correlated-aware MWPM",
            corr_observables,
            correlated_decoder.decode_batch(
                corr_detectors,
                enable_correlations=True,
            ),
        ),
    ]

    # Paired decoder comparison: both decoders see the same shots.
    ordinary_pred = metrics[4][2]
    aware_pred = metrics[5][2]
    actual = corr_observables

    ordinary_fail = np.any(ordinary_pred != actual, axis=1)
    aware_fail = np.any(aware_pred != actual, axis=1)

    both_fail = ordinary_fail & aware_fail
    ordinary_only = ordinary_fail & ~aware_fail
    aware_only = ~ordinary_fail & aware_fail
    both_succeed = ~ordinary_fail & ~aware_fail

    print("\n=== Paired decoder comparison ===")
    print(f"Both fail:                  {np.count_nonzero(both_fail):,}")
    print(f"Ordinary only fails:        {np.count_nonzero(ordinary_only):,}")
    print(f"Correlation-aware only fails: {np.count_nonzero(aware_only):,}")
    print(f"Both succeed:               {np.count_nonzero(both_succeed):,}")

    # Check whether Stim's detector error model contains explicitly
    # decomposed correlated mechanisms that PyMatching can exploit.
    decomposed_terms = [
        line
        for line in str(corr_dem).splitlines()
        if line.strip().startswith("error(") and "^" in line
    ]

    print("\n=== Detector error model diagnostic ===")
    print(f"Decomposed error terms containing '^': {len(decomposed_terms)}")

    for line in decomposed_terms[:10]:
        print(line)
    print("\n=== Correlated Pauli noise experiment ===")
    print(f"Distance: {args.distance}")
    print(f"Rounds: {rounds}")
    print(f"Shots per model: {args.shots:,}")
    print(f"Background p: {args.p:g}")
    print(f"Pair-event probability: {args.p_corr:g}")
    print(f"Distinct data-qubit pairs per round: {len(pairs)}")
    print(f"Pair-event opportunities per circuit: {expected_events}")

    rows = []

    for label, actual, predicted in metrics:
        result = evaluate(actual, predicted, args.shots)

        print(
            f"\n{label}\n"
            f"  Failures: {result['logical_failures']:,}/"
            f"{args.shots:,}\n"
            f"  Logical error rate: "
            f"{result['logical_error_rate']:.6g}\n"
            f"  Approx. 95% Wilson CI: "
            f"[{result['ci95_low']:.6g}, "
            f"{result['ci95_high']:.6g}]"
        )

        rows.append({
            "metric": label,
            **result,
        })

    elapsed = time.perf_counter() - start
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "correlated_comparison.csv"

    fields = [
        "timestamp",
        "distance",
        "rounds",
        "shots",
        "p_independent",
        "p_corr",
        "pair_count",
        "pair_event_opportunities",
        "metric",
        "logical_failures",
        "logical_error_rate",
        "ci95_low",
        "ci95_high",
        "elapsed_seconds",
    ]

    write_header = (
        not csv_path.exists() or csv_path.stat().st_size == 0
    )

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "distance": args.distance,
                "rounds": rounds,
                "shots": args.shots,
                "p_independent": args.p,
                "p_corr": args.p_corr,
                "pair_count": len(pairs),
                "pair_event_opportunities": expected_events,
                "elapsed_seconds": elapsed,
                **row,
            })

    print(f"\nCircuits saved under: {circuits_dir}")
    print(f"Results appended to: {csv_path}")
    print(f"Total elapsed time: {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()