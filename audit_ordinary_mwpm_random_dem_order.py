"""Audit ordinary MWPM sensitivity to random DEM error-instruction order.

Runs the same two fixed-seed distance-3 correlated-noise memory conditions used by
`audit_probability_prioritized_order_invariance.py`. For each condition, it samples
one shared physical-circuit dataset and evaluates ordinary PyMatching MWPM on the
original DEM, reversed-error-order DEM, canonical probability-sorted DEM, and a
specified number of deterministic random permutations of the DEM's complete
`error(...)` instructions. Non-error lines remain in their original positions.

This is an empirical finite-sample diagnostic, not a proof about all syndromes.
Run from the project root so the project's circuit-generation modules are importable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)
from audit_probability_prioritized_order_invariance import (
    as_2d,
    canonical_probability_sorted_dem,
    error_positions,
    fail_mask,
    paired_delta_ci,
    reverse_error_order,
)

P_BASE = 0.001
DISTANCE = 3
ROUNDS = 3
CONDITIONS = ((0.0001, 20261080), (0.0002, 20261081))
BASE_ORDER_SEED = 202610200


def random_error_order_dem(dem: stim.DetectorErrorModel, seed: int) -> stim.DetectorErrorModel:
    """Return a DEM with complete error lines shuffled among error-line slots."""
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = error_positions(lines)
    errors = [lines[i] for i in positions]
    if len(errors) < 2:
        raise ValueError(f"Need at least two error instructions; found {len(errors)}")
    shuffled = list(errors)
    np.random.default_rng(seed).shuffle(shuffled)
    for pos, line in zip(positions, shuffled):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise ValueError("Permutation changed detector or observable dimensions")
    return result


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=500_000, help="Shots per physical-circuit condition (default: 500000)")
    parser.add_argument("--permutations", type=int, default=50, help="Random DEM orders per condition (default: 50)")
    parser.add_argument("--output-dir", default="results/priority_order_random_order_20261020")
    args = parser.parse_args()

    if args.shots < 1 or args.permutations < 1:
        parser.error("--shots and --permutations must both be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"ordinary_mwpm_random_order_{timestamp}.csv"
    metadata_path = out_dir / f"ordinary_mwpm_random_order_{timestamp}.json"
    log_path = out_dir / f"ordinary_mwpm_random_order_{timestamp}.log"

    script_path = Path(__file__).resolve()
    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "stim": version("stim"),
        "pymatching": version("pymatching"),
        "numpy": version("numpy"),
        "distance": DISTANCE,
        "rounds": ROUNDS,
        "p_base": P_BASE,
        "shots_per_condition": args.shots,
        "random_permutations_per_condition": args.permutations,
        "physical_sample_seeds": {str(p): seed for p, seed in CONDITIONS},
        "random_order_seed_rule": "202610200 + condition_index*10000 + permutation_index (0-indexed)",
        "error_order_policy": "shuffle complete flattened DEM error instruction lines among existing error-line positions; non-error lines unchanged",
        "randomized_orders_are_not_physical_noise_resamples": True,
        "script_sha256": sha256_file(script_path),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_branch": subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=False).stdout.strip(),
    }

    columns = [
        "p_corr", "physical_sample_seed", "order_kind", "permutation_index", "order_seed",
        "shots", "error_instruction_count", "num_detectors", "num_observables",
        "failure_count", "failure_rate_percent",
        "prediction_disagreements_vs_canonical", "disagreement_rate_percent_vs_canonical",
        "prediction_disagreements_vs_original", "disagreement_rate_percent_vs_original",
        "failure_delta_pp_vs_canonical", "failure_delta_ci_low_pp_vs_canonical", "failure_delta_ci_high_pp_vs_canonical",
    ]
    rows: list[dict[str, object]] = []
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    log("Ordinary MWPM randomized DEM error-order sensitivity audit")
    log(f"Python={platform.python_version()} Stim={version('stim')} PyMatching={version('pymatching')} NumPy={version('numpy')}")
    log(f"d={DISTANCE}; rounds={ROUNDS}; p_base={P_BASE}; shots/condition={args.shots:,}; random permutations/condition={args.permutations}")
    log(f"Script SHA256={metadata['script_sha256']}")

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)

    for condition_index, (p_corr, physical_seed) in enumerate(CONDITIONS):
        log("\n" + "=" * 78)
        log(f"p_corr={p_corr:g}; physical sample seed={physical_seed}")
        circuit = add_noise_layer(base, data_qubits, pairs, p_corr, "correlated")
        dem = circuit.detector_error_model(decompose_errors=True).flattened()
        reversed_dem = reverse_error_order(dem)
        canonical_dem = canonical_probability_sorted_dem(dem)
        if str(canonical_dem) != str(canonical_probability_sorted_dem(reversed_dem)):
            raise AssertionError("Canonical-sorted DEM differs when derived from reversed order")

        lines = str(dem).splitlines()
        n_errors = len(error_positions(lines))
        log(f"DEM detectors={dem.num_detectors}; observables={dem.num_observables}; error instructions={n_errors}")

        sampler = circuit.compile_detector_sampler(seed=physical_seed)
        syndrome, actual = sampler.sample(shots=args.shots, separate_observables=True)
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        actual = as_2d(actual)

        # Evaluate canonical first so every later configuration has a canonical
        # prediction vector and paired failure baseline available for comparison.
        models: list[tuple[str, int | str, stim.DetectorErrorModel]] = [
            ("canonical_sorted", "", canonical_dem),
            ("original", "", dem),
            ("reversed", "", reversed_dem),
        ]
        for perm_index in range(args.permutations):
            order_seed = BASE_ORDER_SEED + condition_index * 10000 + perm_index
            models.append(("random", order_seed, random_error_order_dem(dem, order_seed)))

        canonical_pred: np.ndarray | None = None
        original_pred: np.ndarray | None = None
        canonical_fail: np.ndarray | None = None
        summary_random_failures: list[int] = []
        summary_random_disagreements: list[int] = []

        for order_kind, order_seed, model_dem in models:
            matcher = pymatching.Matching(model_dem)
            pred = as_2d(matcher.decode_batch(syndrome))
            failures = fail_mask(pred, actual)

            if order_kind == "canonical_sorted":
                canonical_pred = pred.copy()
                canonical_fail = failures.copy()
            elif order_kind == "original":
                original_pred = pred.copy()

            if canonical_pred is None or canonical_fail is None:
                if order_kind != "original":
                    raise AssertionError("Canonical baseline must be evaluated before comparison")
                disagree_canonical = -1
                disagree_original = 0
                delta, lo, hi = (float("nan"),) * 3
            else:
                disagree_canonical = int(np.count_nonzero(np.any(pred != canonical_pred, axis=1)))
                disagree_original = (
                    int(np.count_nonzero(np.any(pred != original_pred, axis=1)))
                    if original_pred is not None else ""
                )
                delta, lo, hi = paired_delta_ci(failures, canonical_fail)

            row = {
                "p_corr": p_corr,
                "physical_sample_seed": physical_seed,
                "order_kind": order_kind,
                "permutation_index": (order_seed - (BASE_ORDER_SEED + condition_index * 10000)) if order_kind == "random" else "",
                "order_seed": order_seed,
                "shots": args.shots,
                "error_instruction_count": n_errors,
                "num_detectors": dem.num_detectors,
                "num_observables": dem.num_observables,
                "failure_count": int(failures.sum()),
                "failure_rate_percent": 100 * float(failures.mean()),
                "prediction_disagreements_vs_canonical": disagree_canonical,
                "disagreement_rate_percent_vs_canonical": 100 * disagree_canonical / args.shots if disagree_canonical >= 0 else "",
                "prediction_disagreements_vs_original": disagree_original,
                "disagreement_rate_percent_vs_original": (
                    100 * disagree_original / args.shots if disagree_original != "" else ""
                ),
                "failure_delta_pp_vs_canonical": delta,
                "failure_delta_ci_low_pp_vs_canonical": lo,
                "failure_delta_ci_high_pp_vs_canonical": hi,
            }
            rows.append(row)
            log(f"{order_kind:16s} seed={str(order_seed):>10s} failures={int(failures.sum()):>6,}/{args.shots:,} ({100*float(failures.mean()):.6f}%) disagreements_vs_sorted={disagree_canonical if disagree_canonical >= 0 else 'baseline'}")

            if order_kind == "random":
                summary_random_failures.append(int(failures.sum()))
                summary_random_disagreements.append(disagree_canonical)

            del matcher, pred, failures

        values = np.asarray(summary_random_failures, dtype=float)
        changes = np.asarray(summary_random_disagreements, dtype=float)
        log("\nRandom-order distribution (each permutation decodes the SAME sampled syndromes):")
        log(f"failure count min/median/max: {int(values.min())}/{float(np.median(values)):g}/{int(values.max())}")
        log(f"failure rate % min/median/max: {values.min()/args.shots*100:.6f}/{np.median(values)/args.shots*100:.6f}/{values.max()/args.shots*100:.6f}")
        log(f"prediction disagreement vs canonical min/median/max: {int(changes.min())}/{float(np.median(changes)):g}/{int(changes.max())}")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    metadata["csv_file"] = csv_path.name
    metadata["log_file"] = log_path.name
    metadata["csv_sha256"] = sha256_file(csv_path)
    metadata["rows"] = len(rows)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("\n=== Saved artifacts ===")
    for path in (csv_path, metadata_path, log_path):
        print(f"{path}: {path.stat().st_size} bytes; SHA256 {sha256_file(path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
