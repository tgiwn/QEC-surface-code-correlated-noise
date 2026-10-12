
from pathlib import Path
import difflib

import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
)


DISTANCE = 5
ROUNDS = 5
P = 0.001
P_CORR = 0.0004


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


def main():
    base = make_base()
    data_by_xy, data_qubits = get_data_layout(base)

    # Original pair used in the ablation.
    pair = [(data_by_xy[(1, 3)], data_by_xy[(1, 7)])]
    assert pair[0] == (12, 34), f"Unexpected pair: {pair}"

    control = add_noise_layer(
        base, data_qubits, pair, P_CORR, "matched_independent"
    )

    correlated = add_noise_layer(
        base, data_qubits, pair, P_CORR, "correlated"
    )

    output = Path("results/dem_audit_pair12_34")
    output.mkdir(parents=True, exist_ok=True)

    for decompose in (False, True):
        suffix = "decomposed" if decompose else "raw"

        control_dem = control.detector_error_model(
            decompose_errors=decompose
        )
        corr_dem = correlated.detector_error_model(
            decompose_errors=decompose
        )

        control_text = str(control_dem)
        corr_text = str(corr_dem)

        (output / f"control_{suffix}.dem").write_text(
            control_text, encoding="utf-8"
        )
        (output / f"correlated_{suffix}.dem").write_text(
            corr_text, encoding="utf-8"
        )

        diff = list(difflib.unified_diff(
            control_text.splitlines(),
            corr_text.splitlines(),
            fromfile=f"control_{suffix}.dem",
            tofile=f"correlated_{suffix}.dem",
            lineterm="",
            n=2,
        ))

        print(f"\n=== {suffix.upper()} DEM comparison ===")
        print(f"Control DEM lines: {len(control_text.splitlines())}")
        print(f"Correlated DEM lines: {len(corr_text.splitlines())}")
        print(f"Diff lines: {len(diff)}")

        for line in diff[:100]:
            print(line)

        if len(diff) > 100:
            print("... diff truncated; complete DEM files were saved.")

    print(f"\nFull DEM files saved to: {output}")


if __name__ == "__main__":
    main()