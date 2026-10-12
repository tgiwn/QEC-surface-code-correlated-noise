"""Source-aware audit of conflicting logical labels in a Stim DEM.

This is a deterministic structural audit; it does not sample Monte Carlo shots.
It keeps each DEM error instruction intact and marks graphlike components
originating from a decomposed instruction (the '^' convention), so those
components are not accidentally described as independent physical faults.
"""

import re
from collections import defaultdict

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

DISTANCE = 3
P_BASE = 0.001
P_CORR = 0.0001


def parse_components(dem):
    records = []
    for line_index, line in enumerate(str(dem.flattened()).splitlines()):
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue

        p = float(match.group(1))
        body = match.group(2)
        pieces = re.split(r"\s+\^\s+", body)
        is_decomposed = len(pieces) > 1

        for component_index, piece in enumerate(pieces):
            detectors = tuple(sorted(
                int(x) for x in re.findall(r"\bD(\d+)\b", piece)
            ))
            labels = tuple(sorted(
                int(x) for x in re.findall(r"\bL(\d+)\b", piece)
            ))

            # Matching components must contain one or two detectors.
            if len(detectors) not in (1, 2):
                continue

            records.append({
                "signature": detectors,
                "labels": labels,
                "p": p,
                "line_index": line_index,
                "component_index": component_index,
                "is_decomposed": is_decomposed,
                "full_line": line.strip(),
                "component": piece.strip(),
            })
    return records


def main():
    base = make_base_circuit(
        distance=DISTANCE,
        rounds=DISTANCE,
        p=P_BASE,
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
    records = parse_components(dem)

    by_signature = defaultdict(list)
    for record in records:
        by_signature[record["signature"]].append(record)

    conflicts = {
        signature: rows
        for signature, rows in by_signature.items()
        if len({row["labels"] for row in rows}) > 1
    }

    print(f"Distance: {DISTANCE}; rounds: {DISTANCE}")
    print(f"P_BASE={P_BASE}; P_CORR={P_CORR}")
    print(f"Flattened DEM graphlike components: {len(records)}")
    print(f"Conflicting detector signatures: {len(conflicts)}")
    print("\nEach row below is a DEM component, not necessarily a separate physical event.")
    print("A '^' source instruction decomposes one mechanism into correlated graphlike components.")

    for signature in sorted(conflicts):
        rows = sorted(
            conflicts[signature],
            key=lambda row: (-row["p"], row["labels"], row["line_index"], row["component_index"]),
        )
        labels = sorted({row["labels"] for row in rows})
        print(f"\n=== Signature {signature}; labels {labels} ===")

        for label in labels:
            label_rows = [row for row in rows if row["labels"] == label]
            unique_instructions = {row["line_index"] for row in label_rows}
            simple_count = sum(not row["is_decomposed"] for row in label_rows)
            decomposed_count = sum(row["is_decomposed"] for row in label_rows)
            max_p = max(row["p"] for row in label_rows)
            print(
                f"  label={label}: components={len(label_rows)}, "
                f"source instructions={len(unique_instructions)}, "
                f"simple components={simple_count}, decomposed components={decomposed_count}, "
                f"max individual p={max_p:.12g}"
            )
            for row in label_rows:
                source_type = "DECOMPOSED/correlated" if row["is_decomposed"] else "single instruction"
                print(
                    f"    p={row['p']:.12g}; source_line={row['line_index']}; "
                    f"component={row['component_index']}; {source_type}"
                )
                print(f"      component terms: {row['component']}")
                print(f"      full DEM instruction: {row['full_line']}")

    print("\nImportant: do not sum these component probabilities blindly.")
    print("Different error instructions represent independent mechanisms, but components split by '^' belong to one decomposed mechanism.")
    print("This audit reports evidence for a later parity/posterior calculation; it does not itself calculate the optimal logical label.")


if __name__ == "__main__":
    main()
