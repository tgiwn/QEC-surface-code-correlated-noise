"""Audit PyMatching's parallel-edge label handling in the installed environment.

This is a diagnostic/reproduction script, not a proposed decoder fix. It separates:
(1) the public Stim DEM -> PyMatching graph-loading path, and
(2) manually adding duplicate graph edges with explicitly selected merge strategies.
These are related but not equivalent code paths.
"""
from __future__ import annotations

import math
import platform
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pymatching
import stim


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def report_dem_order(label: str, dem_text: str) -> None:
    print(f"\n--- DEM-loading path: {label} ---")
    print(dem_text)
    dem = stim.DetectorErrorModel(dem_text)
    matching = pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=False
    )
    syndrome = np.array([1], dtype=np.uint8)
    try:
        prediction = matching.decode(syndrome)
        logical0 = int(prediction[0]) if len(prediction) else "no fault-id output"
    except Exception as exc:  # Report a diagnostic without hiding later tests.
        logical0 = f"decode raised {type(exc).__name__}: {exc}"
    try:
        boundary_edge = matching.get_boundary_edge_data(0)
    except Exception as exc:
        boundary_edge = f"could not inspect boundary edge: {type(exc).__name__}: {exc}"
    print(f"num_detectors={matching.num_detectors}; num_fault_ids={matching.num_fault_ids}")
    print(f"boundary_edge_data={boundary_edge}")
    print(f"decode(syndrome=[1]) logical prediction={logical0}")


def p_to_weight(p: float) -> float:
    return math.log((1.0 - p) / p)


def report_manual_strategy(strategy: str, order: str) -> None:
    print(f"\n--- Manual graph path: strategy={strategy}; order={order} ---")
    matching = pymatching.Matching(num_detectors=1, num_fault_ids=1)
    no_logical = {"fault_ids": set(), "p": 1e-3}
    with_logical = {"fault_ids": {0}, "p": 1e-4}
    edges = [no_logical, with_logical]
    if order == "L0 then no-L0":
        edges.reverse()

    for idx, edge in enumerate(edges):
        try:
            matching.add_boundary_edge(
                0,
                fault_ids=edge["fault_ids"],
                weight=p_to_weight(edge["p"]),
                error_probability=edge["p"],
                merge_strategy=strategy,
            )
            print(
                f"  added edge {idx + 1}: fault_ids={edge['fault_ids']}, "
                f"p={edge['p']}"
            )
        except Exception as exc:
            print(f"  add edge {idx + 1} raised {type(exc).__name__}: {exc}")
            break

    try:
        edge_data = matching.get_boundary_edge_data(0)
    except Exception as exc:
        edge_data = f"could not inspect boundary edge: {type(exc).__name__}: {exc}"
    prediction = matching.decode(np.array([1], dtype=np.uint8))
    logical0 = int(prediction[0]) if len(prediction) else "no fault-id output"
    print(f"final_boundary_edge_data={edge_data}")
    print(f"decode(syndrome=[1]) logical prediction={logical0}")


def main() -> None:
    print("PyMatching parallel-edge logical-label audit")
    print(f"Python: {platform.python_version()}")
    print(f"Stim: {package_version('stim')}")
    print(f"PyMatching: {package_version('pymatching')}")
    print(f"NumPy: {package_version('numpy')}")
    print("\nPART A: the Stim DEM -> PyMatching path used by the experiment")

    report_dem_order(
        "no-L0 mechanism first",
        "error(0.001) D0\nerror(0.0001) D0 L0",
    )
    report_dem_order(
        "L0 mechanism first",
        "error(0.0001) D0 L0\nerror(0.001) D0",
    )

    print("\nPART B: manual graph API (not the same as DEM loading)")
    print(
        "Manual edge probabilities are deliberately distinct: p(no-L0)=0.001; "
        "p(L0)=0.0001. The tests show documented merge-policy semantics, not "
        "which policy is scientifically correct for the full DEM."
    )
    strategies = ["disallow", "independent", "smallest-weight", "keep-original", "replace"]
    for strategy in strategies:
        for order in ["no-L0 then L0", "L0 then no-L0"]:
            report_manual_strategy(strategy, order)

    print("\nInterpretation guardrails:")
    print("- Part A tests the actual DEM-loading pathway and order sensitivity.")
    print("- Part B concerns manually added graph edges; it is not an alternative setting passed to from_detector_error_model().")
    print("- Do not choose a merge policy solely because it gives the smallest failure rate.")


if __name__ == "__main__":
    main()
