
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pymatching

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)


P = 0.001
SHOTS = 500_000
DISTANCES = (3, 5, 7)
P_CORRS = (0.0, 0.00005, 0.0001, 0.0002)
BASE_SEED = 20261010


def wilson_interval(failures, shots, z=1.96):
    rate = failures / shots
    denom = 1 + z**2 / shots
    center = (rate + z**2 / (2 * shots)) / denom
    half = (
        z
        * np.sqrt(
            rate * (1 - rate) / shots
            + z**2 / (4 * shots**2)
        )
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def evaluate(circuit, decoder, shots, seed):
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, actual = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    predicted = decoder.decode_batch(detectors)

    failures = int(
        np.count_nonzero(
            np.any(predicted != actual, axis=1)
        )
    )

    low, high = wilson_interval(failures, shots)

    return {
        "failures": failures,
        "rate": failures / shots,
        "ci95_low": low,
        "ci95_high": high,
    }


def main():
    output = Path("results/correlation_effect_sweep_v2.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp",
        "distance",
        "rounds",
        "shots",
        "p",
        "p_corr",
        "pair_count",
        "physical_model",
        "logical_failures",
        "logical_error_rate",
        "ci95_low",
        "ci95_high",
        "decoder",
    ]

    write_header = not output.exists() or output.stat().st_size == 0
    case_index = 0

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        for distance in DISTANCES:
            rounds = distance
            base = make_base_circuit(distance, rounds, P)
            data_by_xy, data_qubits = get_data_layout(base)
            pairs = make_correlated_pairs(data_by_xy)

            for p_corr in P_CORRS:
                control = add_noise_layer(
                    base,
                    data_qubits,
                    pairs,
                    p_corr,
                    "matched_independent",
                )

                correlated = add_noise_layer(
                    base,
                    data_qubits,
                    pairs,
                    p_corr,
                    "correlated",
                )

                # Train one ordinary decoder on the independent
                # model with the same single-qubit X marginals.
                control_dem = control.detector_error_model(
                    decompose_errors=True
                )
                decoder = pymatching.Matching.from_detector_error_model(
                    control_dem
                )

                control_result = evaluate(
                    control, decoder, SHOTS, BASE_SEED + 2 * case_index
                )
                correlated_result = evaluate(
                    correlated,
                    decoder,
                    SHOTS,
                    BASE_SEED + 2 * case_index + 1,
                )

                ratio = (
                    correlated_result["rate"] / control_result["rate"]
                    if control_result["rate"] > 0
                    else float("nan")
                )

                print(
                    f"\nd={distance}, p={P}, p_corr={p_corr}, "
                    f"pairs/round={len(pairs)}"
                )
                print(
                    f"  Independent control: "
                    f"{control_result['failures']}/{SHOTS}, "
                    f"rate={control_result['rate']:.6g}, "
                    f"95% CI=[{control_result['ci95_low']:.6g}, "
                    f"{control_result['ci95_high']:.6g}]"
                )
                print(
                    f"  Correlated noise:    "
                    f"{correlated_result['failures']}/{SHOTS}, "
                    f"rate={correlated_result['rate']:.6g}, "
                    f"95% CI=[{correlated_result['ci95_low']:.6g}, "
                    f"{correlated_result['ci95_high']:.6g}]"
                )
                print(f"  Correlated/control rate ratio: {ratio:.3g}")

                timestamp = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )

                for model_name, result in [
                    ("matched_independent_control", control_result),
                    ("correlated_XX", correlated_result),
                ]:
                    writer.writerow({
                        "timestamp": timestamp,
                        "distance": distance,
                        "rounds": rounds,
                        "shots": SHOTS,
                        "p": P,
                        "p_corr": p_corr,
                        "pair_count": len(pairs),
                        "physical_model": model_name,
                        "logical_failures": result["failures"],
                        "logical_error_rate": result["rate"],
                        "ci95_low": result["ci95_low"],
                        "ci95_high": result["ci95_high"],
                        "decoder": "matched-independent MWPM",
                    })

                f.flush()
                case_index += 1

    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()