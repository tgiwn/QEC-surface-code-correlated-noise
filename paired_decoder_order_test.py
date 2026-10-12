"""Paired logical-failure comparison for two DEM instruction orderings.

Run from the qec-correlated-noise project directory so that
correlated_experiment.py is importable.
"""

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
DISTANCES = (3,)
SHOTS = 100_000
SEED = 20261011


def reverse_error_order(dem):
    """Reverse error instructions within each shift_detectors region."""
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
    reordered = stim.DetectorErrorModel("\n".join(lines))

    if (dem.num_detectors != reordered.num_detectors or
            dem.num_observables != reordered.num_observables):
        raise ValueError("Reordering changed DEM dimensions.")

    return reordered


def failure_mask(predictions, actual_observables):
    predictions = np.asarray(predictions, dtype=np.uint8)
    actual_observables = np.asarray(actual_observables, dtype=np.uint8)

    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if actual_observables.ndim == 1:
        actual_observables = actual_observables[:, None]

    if predictions.shape != actual_observables.shape:
        raise ValueError(
            "Decoder predictions and sampled observables have different "
            f"shapes: {predictions.shape} vs {actual_observables.shape}. "
            "Check the logical-observable IDs in the DEM."
        )

    return np.any(predictions != actual_observables, axis=1)


def main():
    print(f"Shots per distance: {SHOTS:,}")
    print("Noise model: correlated, same sampled shots for both decoders")

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

        dem = circuit.detector_error_model(decompose_errors=True).flattened()
        reordered_dem = reverse_error_order(dem)

        decoder_original = pymatching.Matching(dem)
        decoder_reordered = pymatching.Matching(reordered_dem)

        sampler = circuit.compile_detector_sampler(seed=SEED + distance)
        detector_samples, observable_flips = sampler.sample(
            shots=SHOTS,
            separate_observables=True,
        )
        detector_samples = np.asarray(detector_samples, dtype=np.uint8)

        predicted_original = decoder_original.decode_batch(detector_samples)
        predicted_reordered = decoder_reordered.decode_batch(detector_samples)

        pred_original = np.asarray(predicted_original, dtype=np.uint8)
        pred_reordered = np.asarray(predicted_reordered, dtype=np.uint8)
        actual = np.asarray(observable_flips, dtype=np.uint8)

        if pred_original.ndim == 1:
            pred_original = pred_original[:, None]
        if pred_reordered.ndim == 1:
            pred_reordered = pred_reordered[:, None]
        if actual.ndim == 1:
            actual = actual[:, None]

        original_fail = failure_mask(pred_original, actual)
        reordered_fail = failure_mask(pred_reordered, actual)
        prediction_changed = np.any(pred_original != pred_reordered, axis=1)

        original_only = int(np.count_nonzero(original_fail & ~reordered_fail))
        reordered_only = int(np.count_nonzero(reordered_fail & ~original_fail))
        original_count = int(np.count_nonzero(original_fail))
        reordered_count = int(np.count_nonzero(reordered_fail))

        p_original = original_count / SHOTS
        p_reordered = reordered_count / SHOTS
        differences = reordered_fail.astype(np.int8) - original_fail.astype(np.int8)
        delta = float(np.mean(differences))
        se_delta = float(np.std(differences, ddof=1) / np.sqrt(SHOTS))

        print(f"\n========== DISTANCE {distance} ==========")
        print(f"Detectors: {dem.num_detectors}; observables: {dem.num_observables}")
        print(f"Sampled syndromes with different decoder predictions: "
              f"{int(np.count_nonzero(prediction_changed)):,} / {SHOTS:,} "
              f"({np.mean(prediction_changed):.6%})")
        print(f"Original-order logical failures:  {original_count:,} / {SHOTS:,} "
              f"({p_original:.6%})")
        print(f"Reordered logical failures:       {reordered_count:,} / {SHOTS:,} "
              f"({p_reordered:.6%})")
        print(f"Original-only failures: {original_only:,}")
        print(f"Reordered-only failures: {reordered_only:,}")
        print(f"Paired rate difference (reordered - original): {delta:+.6%}")
        print(f"Approx. paired 95% CI: [{delta - 1.96 * se_delta:+.6%}, "
              f"{delta + 1.96 * se_delta:+.6%}]")


if __name__ == "__main__":
    main()
