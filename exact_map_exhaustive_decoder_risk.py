r"""Exhaustively evaluate distance-3 decoders under the flattened DEM distribution.

Unlike shot-based validation, this enumerates every one of the 2^24 detector
syndromes and computes each decoder's exact logical-failure risk weighted by
the exact joint distribution from exact_dem_map_decoder.py.

The DEM is never resampled or altered except for reordering complete error
instructions for the canonical-sorted condition. Each full error(p) instruction
remains one Bernoulli mechanism in the MAP convolution.

Run from the project directory containing:
  - correlated_experiment.py
  - exact_dem_map_decoder.py

PowerShell:
  & $python .\exact_map_exhaustive_decoder_risk.py

Writes exact_map_exhaustive_decoder_risk.csv (overwritten at start).
This is distance-3-only: 24 detectors permit exhaustive enumeration in 2^24 states.
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

import exact_dem_map_decoder as exact
from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

DISTANCE = 3
ROUNDS = 3
P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
CHUNK = 50_000
OUT_CSV = Path("exact_map_exhaustive_decoder_risk.csv")


def error_positions(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]


def canonical_sort_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Sort full flattened error instructions, preserving each instruction intact."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_positions(lines)
    errors = [lines[i] for i in positions]

    def key(line: str):
        match = re.match(r"\s*error\(([^)]+)\)", line)
        if not match:
            raise ValueError(f"Cannot parse DEM instruction: {line.strip()}")
        return (-float(match.group(1)), line.strip())

    ordered = sorted(errors, key=key)
    for pos, line in zip(positions, ordered):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Canonical sort changed DEM dimensions")
    return result


def as_2d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def build_decoder_set(dem, sorted_dem):
    decoders = {
        "ordinary/original": (pymatching.Matching(dem), False),
        "ordinary/canonical-sorted": (pymatching.Matching(sorted_dem), False),
        "probability-prioritized": (exact.build_probability_prioritized_matching(sorted_dem), False),
    }
    try:
        decoders["correlation-aware/original"] = (pymatching.Matching(dem, enable_correlations=True), True)
        decoders["correlation-aware/canonical-sorted"] = (
            pymatching.Matching(sorted_dem, enable_correlations=True), True
        )
    except (TypeError, ValueError) as exc:
        decoders.pop("correlation-aware/original", None)
        decoders.pop("correlation-aware/canonical-sorted", None)
        print(f"WARNING: correlation-aware MWPM unavailable: {exc}")
    return decoders


def main() -> None:
    if OUT_CSV.exists():
        OUT_CSV.unlink()
    if DISTANCE != 3:
        raise ValueError("Exhaustive state enumeration here is intentionally limited to distance 3")

    fields = [
        "distance", "rounds", "p_base", "p_corr", "num_detectors", "num_states",
        "normalization", "map_bayes_risk", "map_bayes_risk_percent",
        "decoder", "decoder_dem_risk", "decoder_dem_risk_percent",
        "excess_vs_map", "excess_vs_map_pp", "supported_syndrome_disagreements",
        "all_state_disagreements", "dem_probability_mass_on_disagreements",
        "weighted_excess_check", "decode_seconds",
    ]

    print("Exhaustive distance-3 decoder risk under the exact flattened DEM distribution")
    print(f"distance={DISTANCE}; rounds={ROUNDS}; P_BASE={P_BASE}; p_corr={P_CORR_VALUES}")
    print("Enumerating all 2^24 detector syndromes; no circuit Monte Carlo is used.")
    print("Each complete flattened error instruction remains one Bernoulli event.\n")

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    shifts = np.arange(24, dtype=np.uint32)
    num_states = 1 << 24

    for p_corr in P_CORR_VALUES:
        condition_start = time.time()
        print("\n" + "=" * 88)
        print(f"CONDITION p_corr={p_corr:g}")
        circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
        dem = circuit.detector_error_model(decompose_errors=True).flattened()
        sorted_dem = canonical_sort_dem(dem)
        if (dem.num_detectors, dem.num_observables) != (24, 1):
            raise ValueError(
                f"Expected 24 detectors and one observable; got {dem.num_detectors} detectors, "
                f"{dem.num_observables} observables"
            )
        canonical_again = canonical_sort_dem(dem)
        if str(sorted_dem) != str(canonical_again):
            raise AssertionError("Canonical sorting is not deterministic")

        q0, q1, normalization, bayes_risk = exact.exact_joint_distribution(dem)
        if len(q0) != num_states or len(q1) != num_states:
            raise ValueError(f"Expected {num_states} state probabilities, got {len(q0)}, {len(q1)}")
        total_state_mass = q0 + q1
        supported = total_state_mass > 0
        map_pred_all = q1 > q0  # MAP tie is resolved to L0=0.
        map_risk_check = float(np.minimum(q0, q1).sum(dtype=np.float64) / normalization)
        if not math.isclose(map_risk_check, bayes_risk, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f"Independent MAP risk check failed: {map_risk_check} vs {bayes_risk}")

        decoders = build_decoder_set(dem, sorted_dem)
        names = list(decoders)
        print(f"DEM mechanisms: {sum(x.lstrip().startswith('error(') for x in str(dem).splitlines())}")
        print(f"State space: {num_states:,}; supported states: {int(np.count_nonzero(supported)):,}")
        print(f"Exact DEM-MAP Bayes risk: {bayes_risk * 100:.8f}%")
        print(f"Decoders: {', '.join(names)}")

        accum = {
            name: {
                "risk_mass": 0.0,
                "map_disagreement_mass": 0.0,
                "weighted_excess_mass": 0.0,
                "supported_disagreements": 0,
                "all_state_disagreements": 0,
                "seconds": 0.0,
            }
            for name in names
        }
        sort_priority_disagreements = 0
        sort_priority_weighted_disagreement_mass = 0.0
        started_loop = time.time()
        for start in range(0, num_states, CHUNK):
            stop = min(start + CHUNK, num_states)
            states = np.arange(start, stop, dtype=np.uint32)
            detector_bits = ((states[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
            probs0 = q0[start:stop]
            probs1 = q1[start:stop]
            state_mass = probs0 + probs1
            map_pred = map_pred_all[start:stop]

            chunk_preds = {}
            for name, (decoder, enable_corr) in decoders.items():
                t0 = time.time()
                if enable_corr:
                    pred = decoder.decode_batch(detector_bits, enable_correlations=True)
                else:
                    pred = decoder.decode_batch(detector_bits)
                pred = as_2d(pred)[:, 0].astype(bool)
                accum[name]["seconds"] += time.time() - t0
                chunk_preds[name] = pred

                # Prediction 0 fails when the true logical bit is 1 (mass q1);
                # prediction 1 fails when the true logical bit is 0 (mass q0).
                fail_mass = np.where(pred, probs0, probs1)
                accum[name]["risk_mass"] += float(fail_mass.sum(dtype=np.float64))
                diff = pred != map_pred
                diff_supported = diff & (state_mass > 0)
                accum[name]["all_state_disagreements"] += int(np.count_nonzero(diff))
                accum[name]["supported_disagreements"] += int(np.count_nonzero(diff_supported))
                accum[name]["map_disagreement_mass"] += float(state_mass[diff].sum(dtype=np.float64))
                # Exact excess above MAP equals |q0-q1| over syndromes where prediction differs.
                accum[name]["weighted_excess_mass"] += float(
                    np.abs(probs0[diff] - probs1[diff]).sum(dtype=np.float64)
                )

            if "ordinary/canonical-sorted" in chunk_preds and "probability-prioritized" in chunk_preds:
                pair_diff = chunk_preds["ordinary/canonical-sorted"] != chunk_preds["probability-prioritized"]
                sort_priority_disagreements += int(np.count_nonzero(pair_diff))
                sort_priority_weighted_disagreement_mass += float(state_mass[pair_diff].sum(dtype=np.float64))

            if stop == num_states or stop % (CHUNK * 64) == 0:
                done = stop
                print(
                    f"  decoded {done:,}/{num_states:,} syndromes; "
                    f"elapsed {time.time() - started_loop:.1f}s"
                )

        rows = []
        print("\nExact decoder risks over the full syndrome space:")
        print(f"{'decoder':38s} {'DEM risk':>12s} {'excess vs MAP':>15s} {'weighted syndromes':>19s} {'all diff states':>16s}")
        print("-" * 108)
        for name in names:
            a = accum[name]
            risk = a["risk_mass"] / normalization
            excess = risk - bayes_risk
            weighted_excess_check = a["weighted_excess_mass"] / normalization
            weighted_disagreement_mass = a["map_disagreement_mass"] / normalization
            # Check the exact identity: non-MAP excess risk equals weighted |q0-q1|
            # on states where decoder's prediction differs from MAP.
            if not math.isclose(excess, weighted_excess_check, rel_tol=2e-7, abs_tol=2e-10):
                print(
                    f"WARNING: small numerical risk identity discrepancy for {name}: "
                    f"risk gap={excess:.12g}; weighted check={weighted_excess_check:.12g}"
                )
            row = {
                "distance": DISTANCE,
                "rounds": ROUNDS,
                "p_base": P_BASE,
                "p_corr": p_corr,
                "num_detectors": 24,
                "num_states": num_states,
                "normalization": normalization,
                "map_bayes_risk": bayes_risk,
                "map_bayes_risk_percent": bayes_risk * 100.0,
                "decoder": name,
                "decoder_dem_risk": risk,
                "decoder_dem_risk_percent": risk * 100.0,
                "excess_vs_map": excess,
                "excess_vs_map_pp": excess * 100.0,
                "supported_syndrome_disagreements": a["supported_disagreements"],
                "all_state_disagreements": a["all_state_disagreements"],
                "dem_probability_mass_on_disagreements": weighted_disagreement_mass,
                "weighted_excess_check": weighted_excess_check,
                "decode_seconds": a["seconds"],
            }
            rows.append(row)
            print(
                f"{name:38s} {risk * 100:11.7f}% {excess * 100:14.7f} pp "
                f"{weighted_disagreement_mass:18.7%} {a['all_state_disagreements']:16,d}"
            )

        print(
            "\nCanonical-sorted ordinary vs probability-prioritized: "
            f"{sort_priority_disagreements:,}/{num_states:,} state predictions differ; "
            f"DEM mass on differences={sort_priority_weighted_disagreement_mass / normalization:.12g}"
        )
        print(f"Condition total elapsed: {time.time() - condition_start:.1f}s")
        with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if f.tell() == 0 and OUT_CSV.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(rows)

        del q0, q1, total_state_mass, supported, map_pred_all, decoders, accum
        del sorted_dem, dem, circuit, rows

    print(f"\nCompleted. Wrote exact risks to: {OUT_CSV.resolve()}")
    print("Interpretation: risks are exact under the flattened DEM distribution at distance 3.")
    print("This is not a circuit-level Monte Carlo estimate or a threshold/scaling result.")


if __name__ == "__main__":
    main()
