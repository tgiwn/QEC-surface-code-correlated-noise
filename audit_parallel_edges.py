
import re
from collections import defaultdict

import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_correlated_pairs,
)

DISTANCES = (3, 5, 7)
P = 0.001
P_CORRS = (0.0001, 0.0002)


def make_base(distance):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0,
        before_round_data_depolarization=P,
        after_reset_flip_probability=P,
        before_measure_flip_probability=P,
    )


def inspect_dem(dem):
    # A graphlike component contains one or two detectors.
    # Record its detector signature and logical-observable labels.
    components = defaultdict(list)
    total_components = 0
    non_graphlike = 0

    for line in str(dem).splitlines():
        line = line.strip()

        match = re.match(r"error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue

        probability = float(match.group(1))
        target_text = match.group(2)

        # ^ separates components of one decomposed error mechanism.
        pieces = re.split(r"\s+\^\s+", target_text)

        for piece in pieces:
            detectors = tuple(sorted(
                int(x) for x in re.findall(r"\bD(\d+)\b", piece)
            ))
            observables = tuple(sorted(
                int(x) for x in re.findall(r"\bL(\d+)\b", piece)
            ))

            if not detectors:
                continue

            if len(detectors) > 2:
                non_graphlike += 1
                continue

            total_components += 1
            components[detectors].append({
                "probability": probability,
                "observables": observables,
                "line": line,
            })

    duplicate_signatures = {
        signature: entries
        for signature, entries in components.items()
        if len(entries) > 1
    }

    label_conflicts = {
        signature: entries
        for signature, entries in duplicate_signatures.items()
        if len({entry["observables"] for entry in entries}) > 1
    }

    return {
        "total_components": total_components,
        "unique_signatures": len(components),
        "duplicate_signatures": duplicate_signatures,
        "label_conflicts": label_conflicts,
        "non_graphlike": non_graphlike,
    }


def main():
    for distance in DISTANCES:
        base = make_base(distance)
        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)

        print(f"\n========== DISTANCE {distance} ==========")
        print(f"Data qubits: {len(data_qubits)}")
        print(f"Vertical pairs per round: {len(pairs)}")

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

            print(f"\n--- p_corr={p_corr:g} ---")

            for label, circuit in [
                ("Control", control),
                ("Correlated", correlated),
            ]:
                dem = circuit.detector_error_model(
                    decompose_errors=True
                )
                result = inspect_dem(dem)

                print(f"\n{label} model")
                print(
                    "Graphlike components: "
                    f"{result['total_components']}"
                )
                print(
                    "Unique detector signatures: "
                    f"{result['unique_signatures']}"
                )
                print(
                    "Duplicate signatures: "
                    f"{len(result['duplicate_signatures'])}"
                )
                print(
                    "Potential logical-label conflicts: "
                    f"{len(result['label_conflicts'])}"
                )
                print(
                    "Non-graphlike components after decomposition: "
                    f"{result['non_graphlike']}"
                )

                for signature, entries in list(
                    result["label_conflicts"].items()
                )[:5]:
                    print(f"  Signature D{signature}")
                    for entry in entries[:4]:
                        print(
                            f"    p={entry['probability']:.6g}, "
                            f"logical labels={entry['observables']}, "
                            f"source={entry['line']}"
                        )


if __name__ == "__main__":
    main()