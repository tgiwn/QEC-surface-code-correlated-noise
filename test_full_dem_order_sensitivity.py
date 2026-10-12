import itertools

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

P_CORR = 0.0001
DISTANCES = (3, 5, 7)


def reverse_error_order(dem):
    """Reverse error instructions within each detector-shift region."""
    lines = str(dem.flattened()).splitlines()
    positions = []

    def flush():
        original = [lines[i] for i in positions]
        for i, line in zip(positions, reversed(original)):
            lines[i] = line

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        if stripped.startswith("shift_detectors"):
            flush()
            positions = []
        elif stripped.startswith("error("):
            positions.append(i)

    flush()
    return stim.DetectorErrorModel("\n".join(lines))


def test_syndromes(num_detectors, include_pairs):
    """Generate one-detector syndromes, plus pairs when requested."""
    for i in range(num_detectors):
        yield (i,)

    if include_pairs:
        yield from itertools.combinations(range(num_detectors), 2)


def predict(matching, num_detectors, detector_ids):
    syndrome = np.zeros(num_detectors, dtype=np.uint8)
    syndrome[list(detector_ids)] = 1

    return tuple(int(x) for x in matching.decode(syndrome))


def main():
    for distance in DISTANCES:
        base = make_base_circuit(
            distance=distance,
            rounds=distance,
            p=0.001,
        )

        data_by_xy, data_qubits = get_data_layout(base)
        pairs = make_correlated_pairs(data_by_xy)

        circuit = add_noise_layer(
            base,
            data_qubits,
            pairs,
            P_CORR,
            "correlated",
        )

        dem = circuit.detector_error_model(
            decompose_errors=True
        ).flattened()

        reversed_dem = reverse_error_order(dem)

        if (
            dem.num_detectors != reversed_dem.num_detectors
            or dem.num_observables != reversed_dem.num_observables
        ):
            raise ValueError("The reordered DEM changed its dimensions.")

        normal_decoder = pymatching.Matching(dem)
        reversed_decoder = pymatching.Matching(reversed_dem)

        num_detectors = dem.num_detectors
        include_pairs = True

        changed = []
        tested = 0

        for detector_ids in test_syndromes(
            num_detectors,
            include_pairs,
        ):
            tested += 1

            normal_prediction = predict(
                normal_decoder,
                num_detectors,
                detector_ids,
            )

            reversed_prediction = predict(
                reversed_decoder,
                num_detectors,
                detector_ids,
            )

            if normal_prediction != reversed_prediction:
                changed.append(
                    (
                        detector_ids,
                        normal_prediction,
                        reversed_prediction,
                    )
                )

        print(f"\n========== DISTANCE {distance} ==========")
        print(f"Detector count: {num_detectors}")
        print(f"Syndromes tested: {tested}")
        print(f"Predictions changed: {len(changed)}")

        for detector_ids, before, after in changed[:20]:
            print(
                f"  Detectors {detector_ids}: "
                f"original={before}, reordered={after}"
            )

        if len(changed) > 20:
            print(f"  ... and {len(changed) - 20} more changes")


if __name__ == "__main__":
    main()