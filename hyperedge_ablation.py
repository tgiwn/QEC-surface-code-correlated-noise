
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
)


DISTANCE = 5
ROUNDS = 5
P = 0.001
P_CORRS = (0.0004,)
SHOTS = 50_000_000
CHUNK = 250_000
SEED = 20261020

OUTPUT = Path("results/hyperedge_ablation_pair12_34_confirm.csv")


def make_base():
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=DISTANCE,
        rounds=ROUNDS,
        after_clifford_depolarization=0,
        before_round_data_depolarization=P,
        after_reset_flip_probability=P,
        before_measure_flip_probability=P,
    )


def interval(k, n, z=1.96):
    rate = k / n
    denom = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denom
    half = (
        z
        * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar(ordinary_only, full_model_only):
    # Exact two-sided binomial test on the discordant paired outcomes.
    n = ordinary_only + full_model_only
    if n == 0:
        return 1.0

    k = min(ordinary_only, full_model_only)
    log_two = math.log(2)
    tail = 0.0

    for i in range(k + 1):
        log_probability = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - n * log_two
        )
        tail += math.exp(log_probability)

    return min(1.0, 2 * tail)


def main():
    base = make_base()
    data_by_xy, data_qubits = get_data_layout(base)

    pair = [(data_by_xy[(1, 3)], data_by_xy[(1, 7)])]
    assert pair[0] == (12, 34), f"Unexpected pair: {pair[0]}"

    print("=== Correlated-model decoder ablation ===")
    print(f"Distance/rounds: {DISTANCE}/{ROUNDS}")
    print(f"Pair qubit IDs: {pair[0]}")
    print(f"Shots per condition: {SHOTS:,}")
    print(f"Batch size: {CHUNK:,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp", "distance", "rounds", "shots",
        "p", "p_corr", "metric", "failures",
        "logical_error_rate", "ci95_low", "ci95_high",
        "control_aware_only_failures", "full_model_aware_only_failures",
        "exact_mcnemar_p",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for index, p_corr in enumerate(P_CORRS):
            if p_corr == 0:
                control = base
                correlated = base
            else:
                control = add_noise_layer(
                    base, data_qubits, pair, p_corr,
                    "matched_independent",
                )
                correlated = add_noise_layer(
                    base, data_qubits, pair, p_corr,
                    "correlated",
                )

            control_dem = control.detector_error_model(
                decompose_errors=True
            )
            corr_dem = correlated.detector_error_model(
                decompose_errors=True
            )

            # All decoders are tested on the same correlated samples.
            ordinary = pymatching.Matching.from_detector_error_model(
                corr_dem
            )

            control_aware = (
                pymatching.Matching.from_detector_error_model(
                    control_dem,
                    enable_correlations=True,
                )
            )

            full_model_aware = (
                pymatching.Matching.from_detector_error_model(
                    corr_dem,
                    enable_correlations=True,
                )
            )

            sampler = correlated.compile_detector_sampler(
                seed=SEED + index
            )

            failures = {
                "ordinary_mwpm": 0,
                "control_model_aware": 0,
                "full_model_aware": 0,
            }

            control_aware_only = 0
            full_model_aware_only = 0

            for start in range(0, SHOTS, CHUNK):
                batch = min(CHUNK, SHOTS - start)

                detectors, actual = sampler.sample(
                    shots=batch,
                    separate_observables=True,
                )

                ordinary_pred = ordinary.decode_batch(detectors)

                control_aware_pred = control_aware.decode_batch(
                    detectors,
                    enable_correlations=True,
                )

                full_model_pred = full_model_aware.decode_batch(
                    detectors,
                    enable_correlations=True,
                )

                ordinary_fail = np.any(
                    ordinary_pred != actual, axis=1
                )
                control_fail = np.any(
                    control_aware_pred != actual, axis=1
                )
                full_fail = np.any(
                    full_model_pred != actual, axis=1
                )

                failures["ordinary_mwpm"] += int(
                    np.count_nonzero(ordinary_fail)
                )
                failures["control_model_aware"] += int(
                    np.count_nonzero(control_fail)
                )
                failures["full_model_aware"] += int(
                    np.count_nonzero(full_fail)
                )

                control_aware_only += int(
                    np.count_nonzero(control_fail & ~full_fail)
                )
                full_model_aware_only += int(
                    np.count_nonzero(~control_fail & full_fail)
                )

            print(f"\n=== p_corr={p_corr:g} ===")

            timestamp = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

            for label, count in failures.items():
                rate = count / SHOTS
                low, high = interval(count, SHOTS)

                print(
                    f"{label}: {count:,}/{SHOTS:,}; "
                    f"rate={rate:.6g}; "
                    f"95% CI=[{low:.6g}, {high:.6g}]"
                )

                writer.writerow({
                    "timestamp": timestamp,
                    "distance": DISTANCE,
                    "rounds": ROUNDS,
                    "shots": SHOTS,
                    "p": P,
                    "p_corr": p_corr,
                    "metric": label,
                    "failures": count,
                    "logical_error_rate": rate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "control_aware_only_failures": "",
                    "full_model_aware_only_failures": "",
                    "exact_mcnemar_p": "",
                })

            p_value = exact_mcnemar(
                control_aware_only,
                full_model_aware_only,
            )

            print(
                f"Control-aware only fails: {control_aware_only:,}"
            )
            print(
                f"Full-model-aware only fails: {full_model_aware_only:,}"
            )
            print(f"Exact paired McNemar p-value: {p_value:.6g}")

            writer.writerow({
                "timestamp": timestamp,
                "distance": DISTANCE,
                "rounds": ROUNDS,
                "shots": SHOTS,
                "p": P,
                "p_corr": p_corr,
                "metric": "paired_ablation_test",
                "control_aware_only_failures": control_aware_only,
                "full_model_aware_only_failures": full_model_aware_only,
                "exact_mcnemar_p": p_value,
            })

            f.flush()

    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()