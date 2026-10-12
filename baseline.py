
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pymatching
import stim


def run_experiment(distance, rounds, shots, p, seed):
    if distance < 3 or distance % 2 == 0:
        raise ValueError("Use an odd code distance >= 3, such as 3 or 5.")
    if rounds < 1 or shots < 1:
        raise ValueError("Rounds and shots must be positive.")
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1.")

    start = time.perf_counter()

    # Stage 1: Generate a noisy surface-code memory circuit.
    # This is an independent-noise baseline, not yet the
    # correlated Pauli noise model for the research study.
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
    )

    # Stage 2: Convert the circuit's error mechanisms into a
    # detector error model and construct the MWPM decoder.
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    # Save the generated circuit so the experiment is inspectable.
    circuits_dir = Path("circuits")
    circuits_dir.mkdir(parents=True, exist_ok=True)

    circuit_path = circuits_dir / (
        f"rotated_memory_z_d{distance}_r{rounds}_p{p:g}.stim"
    )
    circuit.to_file(str(circuit_path))

    # Stage 3: Sample syndromes and actual logical observables.
    sampler = circuit.compile_detector_sampler(seed=seed)
    syndrome, actual_observables = sampler.sample(
        shots=shots,
        separate_observables=True,
    )

    # Stage 4: Decode the sampled syndromes in a batch.
    predicted_observables = matching.decode_batch(syndrome)

    # A logical failure occurs when the predicted logical
    # observable differs from the actual sampled observable.
    failures = int(
        np.count_nonzero(
            np.any(predicted_observables != actual_observables, axis=1)
        )
    )

    elapsed = time.perf_counter() - start
    logical_error_rate = failures / shots
    shots_per_second = shots / elapsed if elapsed > 0 else float("inf")

    print("\n=== Surface-code baseline ===")
    print(f"Code distance:       {distance}")
    print(f"Memory rounds:       {rounds}")
    print(f"Shots:               {shots:,}")
    print(f"Noise parameter p:   {p:g}")
    print(f"Logical failures:    {failures:,}")
    print(f"Estimated logical error rate: {logical_error_rate:.6g}")
    print(f"Total elapsed time:  {elapsed:.3f} seconds")
    print(f"End-to-end shots/s:  {shots_per_second:.1f}")
    print(f"Circuit saved to:    {circuit_path}")

    if failures == 0:
        print(
            "Note: zero observed failures does not mean the true "
            "logical error rate is zero."
        )
        print(
            f"Approximate 95% upper bound from the rule of three: "
            f"{3 / shots:.3g}"
        )

    # Record the run so later experiments can be compared fairly.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "baseline.csv"

    fields = [
        "timestamp",
        "distance",
        "rounds",
        "shots",
        "noise_parameter_p",
        "logical_failures",
        "logical_error_rate",
        "elapsed_seconds",
        "shots_per_second",
        "seed",
    ]

    write_header = (
        not results_file.exists() or results_file.stat().st_size == 0
    )

    with results_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "distance": distance,
                "rounds": rounds,
                "shots": shots,
                "noise_parameter_p": p,
                "logical_failures": failures,
                "logical_error_rate": logical_error_rate,
                "elapsed_seconds": elapsed,
                "shots_per_second": shots_per_second,
                "seed": seed,
            }
        )

    print(f"Results recorded in: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a surface-code memory baseline experiment."
    )
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rounds = args.rounds if args.rounds is not None else args.distance

    run_experiment(
        distance=args.distance,
        rounds=rounds,
        shots=args.shots,
        p=args.p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()