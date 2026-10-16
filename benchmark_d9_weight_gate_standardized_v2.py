#!/usr/bin/env python3
"""Standardized independent timing/accuracy validation for the d=9 weight gate.

This script benchmarks a fixed detector-weight gate (weight >= 12) against
full correlation-aware MWPM using the same sampled syndrome batches.

Timed path definitions are deliberately explicit:
  * full-CA: decode the entire batch with correlation-aware MWPM;
  * gated: compute detector weights, route weight < 12 to ordinary MWPM, and
    route weight >= 12 to correlation-aware MWPM. Only the selected subset is
    decoded by each decoder. Syndrome generation and failure-count bookkeeping
    are excluded from both timings.

Threshold 12 is treated as frozen from the earlier screening study. This script
runs fresh validation seeds; it does not rewrite any existing project outputs.
Run it from the project root containing correlated_experiment.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymatching
import stim

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

DISTANCE = 9
ROUNDS = 9
P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
GATE_THRESHOLD = 12
SEED_BASE = 2026101400
CHUNK_SIZE = 25_000
WARMUP_SHOTS = 2_048
Z95 = 1.959963984540054
MODEL_VARIANTS = (
    ("matched-independent-control", "matched_independent"),
    ("fully-correlated", "correlated"),
)


def as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def failure_mask(predictions: Any, actual: Any) -> np.ndarray:
    pred = as_2d(predictions)
    obs = as_2d(actual)
    if pred.shape != obs.shape:
        raise ValueError(f"prediction/observable shape mismatch: {pred.shape} vs {obs.shape}")
    return np.any(pred != obs, axis=1)


def paired_delta_ci(candidate_only: int, baseline_only: int, n: int) -> tuple[float, float, float]:
    """Paired candidate-minus-baseline risk difference and normal 95% CI, in pp."""
    if n <= 1:
        return float("nan"), float("nan"), float("nan")
    delta_fraction = (candidate_only - baseline_only) / n
    discordant = candidate_only + baseline_only
    var = max(0.0, (discordant - n * delta_fraction * delta_fraction) / ((n - 1) * n))
    delta_pp = 100.0 * delta_fraction
    se_pp = 100.0 * math.sqrt(var)
    return delta_pp, delta_pp - Z95 * se_pp, delta_pp + Z95 * se_pp


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def git_output(args: list[str], cwd: Path) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, timeout=5, check=False)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    error_lines = [lines[i] for i in positions]

    def key(line: str) -> tuple[float, str]:
        left = line.strip().split("(", 1)[1].split(")", 1)[0]
        return (-float(left), line.strip())

    for pos, line in zip(positions, sorted(error_lines, key=key)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise AssertionError("Canonical sorting unexpectedly changed DEM dimensions")
    return result


def build_condition(p_corr: float, mode: str):
    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
    dem = canonical_probability_sorted_dem(circuit.detector_error_model(decompose_errors=True).flattened())
    ordinary = pymatching.Matching(dem)
    corr = pymatching.Matching(dem, enable_correlations=True)
    return circuit, dem, ordinary, corr, len(pairs)


def gated_decode(syndrome: np.ndarray, ordinary: pymatching.Matching,
                 corr: pymatching.Matching, threshold: int) -> tuple[np.ndarray, int]:
    """Run the actual threshold-gated path and return predictions/routed count."""
    weights = np.sum(syndrome, axis=1)
    high = weights >= threshold
    low = ~high
    predictions = np.empty((len(syndrome), 1), dtype=np.uint8)
    if np.any(low):
        predictions[low] = as_2d(ordinary.decode_batch(syndrome[low]))
    if np.any(high):
        predictions[high] = as_2d(corr.decode_batch(syndrome[high], enable_correlations=True))
    return predictions, int(np.count_nonzero(high))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots-per-replicate", type=int, default=2_000_000,
                    help="fresh circuit shots per replicate and condition (default: 2,000,000)")
    ap.add_argument("--replicates", type=int, default=3,
                    help="independent replicates per condition (default: 3)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                    help="number of shots decoded per batch (default: 25,000)")
    ap.add_argument("--seed-base", type=int, default=2026101600,
                    help="base seed for this benchmark run; choose a fresh value for an independent run")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.shots_per_replicate < 1 or args.replicates < 1 or args.chunk_size < 1:
        raise ValueError("shots, replicates, and chunk size must be positive")

    root = Path.cwd().resolve()
    if not (root / "correlated_experiment.py").exists():
        raise FileNotFoundError("Run this script from the project root containing correlated_experiment.py")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "results" / "paper_evidence" / f"d9_gate_standardized_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    replicate_csv = out_dir / "replicates.csv"
    summary_csv = out_dir / "summary.csv"
    metadata_txt = out_dir / "run_metadata.txt"

    replicate_fields = [
        "replicate", "distance", "rounds", "p_base", "p_corr", "physical_model", "seed", "shots",
        "gate_threshold", "ordinary_failures", "full_ca_failures", "gated_failures",
        "gate_only_vs_full_ca", "full_ca_only_vs_gate", "routed_fraction",
        "full_ca_seconds", "gated_seconds", "speedup_full_ca_over_gate",
        "gate_minus_full_ca_delta_pp", "gate_minus_full_ca_ci_low_pp", "gate_minus_full_ca_ci_high_pp",
        "gate_minus_ordinary_delta_pp", "gate_minus_ordinary_ci_low_pp", "gate_minus_ordinary_ci_high_pp",
    ]

    # Aggregate counts and timing separately for each physical model / strength.
    aggregates: dict[tuple[float, str], dict[str, float | int]] = {}
    print("=== Standardized distance-9 detector-weight gate benchmark ===")
    print(f"Stim={package_version('stim')}; PyMatching={package_version('pymatching')}; NumPy={package_version('numpy')}")
    print(f"d={DISTANCE}, rounds={ROUNDS}, p_base={P_BASE}, threshold={GATE_THRESHOLD}")
    print(f"Replicates={args.replicates}; shots/replicate={args.shots_per_replicate:,}; chunk={args.chunk_size:,}")
    print("Timing excludes sampling and statistical bookkeeping; includes gate-weight calculation and selected decoder calls.")
    print("Full-CA timing decodes all syndromes; gated timing decodes ordinary only on weight < 12 and CA only on weight >= 12.")

    with replicate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=replicate_fields)
        writer.writeheader()

        for p_index, p_corr in enumerate(P_CORR_VALUES):
            for model_index, (model_name, mode) in enumerate(MODEL_VARIANTS):
                key = (p_corr, model_name)
                agg = aggregates.setdefault(key, {
                    "shots": 0, "ordinary_failures": 0, "full_ca_failures": 0, "gated_failures": 0,
                    "gate_only_vs_full_ca": 0, "full_ca_only_vs_gate": 0,
                    "gate_only_vs_ordinary": 0, "ordinary_only_vs_gate": 0, "routed": 0,
                    "full_ca_seconds": 0.0, "gated_seconds": 0.0,
                })
                for rep in range(args.replicates):
                    seed = args.seed_base + rep * 1_000_000 + p_index * 100 + model_index
                    circuit, dem, ordinary, corr, pair_count = build_condition(p_corr, mode)
                    sampler = circuit.compile_detector_sampler(seed=seed)
                    warm_sampler = circuit.compile_detector_sampler(seed=seed + 8_000_000_000)
                    warm_syndrome, _ = warm_sampler.sample(
                        shots=min(WARMUP_SHOTS, args.chunk_size), separate_observables=True)
                    warm_syndrome = np.asarray(warm_syndrome, dtype=np.uint8)
                    # Warm each path once; discard warmup timings and data.
                    _ = corr.decode_batch(warm_syndrome, enable_correlations=True)
                    _ = gated_decode(warm_syndrome, ordinary, corr, GATE_THRESHOLD)
                    _ = ordinary.decode_batch(warm_syndrome)

                    counts = {
                        "ordinary_failures": 0, "full_ca_failures": 0, "gated_failures": 0,
                        "gate_only_vs_full_ca": 0, "full_ca_only_vs_gate": 0,
                        "gate_only_vs_ordinary": 0, "ordinary_only_vs_gate": 0, "routed": 0,
                    }
                    full_ca_seconds = gated_seconds = 0.0
                    processed = chunk_no = 0
                    print(f"\nCondition p_corr={p_corr:g}; model={model_name}; replicate={rep + 1}/{args.replicates}; seed={seed}; pair opportunities={pair_count}; shots={args.shots_per_replicate:,}")
                    while processed < args.shots_per_replicate:
                        n = min(args.chunk_size, args.shots_per_replicate - processed)
                        syndrome, actual = sampler.sample(shots=n, separate_observables=True)
                        syndrome = np.asarray(syndrome, dtype=np.uint8)
                        actual = as_2d(actual)

                        # Alternate timing order across chunks to reduce systematic order bias.
                        if chunk_no % 2 == 0:
                            t0 = time.perf_counter()
                            pred_ca = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                            full_ca_seconds += time.perf_counter() - t0
                            t0 = time.perf_counter()
                            pred_gate, routed = gated_decode(syndrome, ordinary, corr, GATE_THRESHOLD)
                            gated_seconds += time.perf_counter() - t0
                        else:
                            t0 = time.perf_counter()
                            pred_gate, routed = gated_decode(syndrome, ordinary, corr, GATE_THRESHOLD)
                            gated_seconds += time.perf_counter() - t0
                            t0 = time.perf_counter()
                            pred_ca = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                            full_ca_seconds += time.perf_counter() - t0

                        # Ordinary full-batch reference is for accuracy only; excluded from timed paths.
                        pred_o = as_2d(ordinary.decode_batch(syndrome))
                        f_ca = failure_mask(pred_ca, actual)
                        f_gate = failure_mask(pred_gate, actual)
                        f_ord = failure_mask(pred_o, actual)
                        counts["ordinary_failures"] += int(f_ord.sum())
                        counts["full_ca_failures"] += int(f_ca.sum())
                        counts["gated_failures"] += int(f_gate.sum())
                        counts["gate_only_vs_full_ca"] += int(np.count_nonzero(f_gate & ~f_ca))
                        counts["full_ca_only_vs_gate"] += int(np.count_nonzero(f_ca & ~f_gate))
                        counts["gate_only_vs_ordinary"] += int(np.count_nonzero(f_gate & ~f_ord))
                        counts["ordinary_only_vs_gate"] += int(np.count_nonzero(f_ord & ~f_gate))
                        counts["routed"] += routed
                        processed += n
                        chunk_no += 1
                        if processed % 250_000 == 0 or processed == args.shots_per_replicate:
                            print(f"  scored {processed:,}/{args.shots_per_replicate:,} shots")

                    n_total = args.shots_per_replicate
                    d_ca, lo_ca, hi_ca = paired_delta_ci(counts["gate_only_vs_full_ca"], counts["full_ca_only_vs_gate"], n_total)
                    d_o, lo_o, hi_o = paired_delta_ci(counts["gate_only_vs_ordinary"], counts["ordinary_only_vs_gate"], n_total)
                    speedup = full_ca_seconds / gated_seconds if gated_seconds else float("nan")
                    row = {
                        "replicate": rep + 1, "distance": DISTANCE, "rounds": ROUNDS, "p_base": P_BASE,
                        "p_corr": p_corr, "physical_model": model_name, "seed": seed, "shots": n_total,
                        "gate_threshold": GATE_THRESHOLD, "ordinary_failures": counts["ordinary_failures"],
                        "full_ca_failures": counts["full_ca_failures"], "gated_failures": counts["gated_failures"],
                        "gate_only_vs_full_ca": counts["gate_only_vs_full_ca"],
                        "full_ca_only_vs_gate": counts["full_ca_only_vs_gate"],
                        "routed_fraction": counts["routed"] / n_total,
                        "full_ca_seconds": full_ca_seconds, "gated_seconds": gated_seconds,
                        "speedup_full_ca_over_gate": speedup,
                        "gate_minus_full_ca_delta_pp": d_ca, "gate_minus_full_ca_ci_low_pp": lo_ca,
                        "gate_minus_full_ca_ci_high_pp": hi_ca,
                        "gate_minus_ordinary_delta_pp": d_o, "gate_minus_ordinary_ci_low_pp": lo_o,
                        "gate_minus_ordinary_ci_high_pp": hi_o,
                    }
                    writer.writerow(row)
                    f.flush()
                    for field in counts:
                        agg[field] = int(agg[field]) + int(counts[field])
                    agg["shots"] = int(agg["shots"]) + n_total
                    agg["full_ca_seconds"] = float(agg["full_ca_seconds"]) + full_ca_seconds
                    agg["gated_seconds"] = float(agg["gated_seconds"]) + gated_seconds
                    print(f"  ordinary={counts['ordinary_failures']/n_total*100:.5f}% | full CA={counts['full_ca_failures']/n_total*100:.5f}% | gated={counts['gated_failures']/n_total*100:.5f}%")
                    print(f"  routed={counts['routed']/n_total:.1%}; speedup={speedup:.3f}x; gate-full-CA={d_ca:+.6f} pp [{lo_ca:+.6f}, {hi_ca:+.6f}]")

    summary_fields = [
        "distance", "rounds", "p_base", "p_corr", "physical_model", "replicates", "pooled_shots",
        "gate_threshold", "ordinary_failures", "full_ca_failures", "gated_failures", "routed_fraction",
        "full_ca_seconds", "gated_seconds", "speedup_full_ca_over_gate",
        "gate_minus_full_ca_delta_pp", "gate_minus_full_ca_ci_low_pp", "gate_minus_full_ca_ci_high_pp",
        "gate_minus_ordinary_delta_pp", "gate_minus_ordinary_ci_low_pp", "gate_minus_ordinary_ci_high_pp",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (p_corr, model_name), agg in sorted(aggregates.items()):
            n = int(agg["shots"])
            d_ca, lo_ca, hi_ca = paired_delta_ci(int(agg["gate_only_vs_full_ca"]), int(agg["full_ca_only_vs_gate"]), n)
            d_o, lo_o, hi_o = paired_delta_ci(int(agg["gate_only_vs_ordinary"]), int(agg["ordinary_only_vs_gate"]), n)
            full_ca_seconds = float(agg["full_ca_seconds"])
            gated_seconds = float(agg["gated_seconds"])
            writer.writerow({
                "distance": DISTANCE, "rounds": ROUNDS, "p_base": P_BASE, "p_corr": p_corr,
                "physical_model": model_name, "replicates": args.replicates, "pooled_shots": n,
                "gate_threshold": GATE_THRESHOLD, "ordinary_failures": agg["ordinary_failures"],
                "full_ca_failures": agg["full_ca_failures"], "gated_failures": agg["gated_failures"],
                "routed_fraction": int(agg["routed"]) / n, "full_ca_seconds": full_ca_seconds,
                "gated_seconds": gated_seconds,
                "speedup_full_ca_over_gate": full_ca_seconds / gated_seconds if gated_seconds else float("nan"),
                "gate_minus_full_ca_delta_pp": d_ca, "gate_minus_full_ca_ci_low_pp": lo_ca,
                "gate_minus_full_ca_ci_high_pp": hi_ca,
                "gate_minus_ordinary_delta_pp": d_o, "gate_minus_ordinary_ci_low_pp": lo_o,
                "gate_minus_ordinary_ci_high_pp": hi_o,
            })

    script_path = Path(__file__).resolve()
    project_module = root / "correlated_experiment.py"
    meta = [
        "Standardized d=9 detector-weight gate timing and accuracy validation",
        f"generated_local: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"project_root: {root}",
        f"git_commit: {git_output(['rev-parse', 'HEAD'], root)}",
        f"git_status_short: {git_output(['status', '--short'], root)}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"stim: {package_version('stim')}",
        f"pymatching: {package_version('pymatching')}",
        f"numpy: {package_version('numpy')}",
        f"distance: {DISTANCE}", f"rounds: {ROUNDS}", f"p_base: {P_BASE}",
        f"p_corr_values: {P_CORR_VALUES}", f"physical_modes: {MODEL_VARIANTS}",
        f"gate_threshold: {GATE_THRESHOLD}", f"replicates: {args.replicates}",
        f"seed_base: {args.seed_base}",
        f"shots_per_replicate_condition: {args.shots_per_replicate}", f"chunk_size: {args.chunk_size}",
        "timing_protocol: warm each decoder path on separate 2048-shot warmup; excluded warmup; sample generation and failure bookkeeping excluded; gate weight computation plus selective decode calls included; alternating call order by chunk.",
        "full_ca_path: corr.decode_batch(all syndrome rows, enable_correlations=True)",
        "gated_path: detector weight <12 -> ordinary.decode_batch(low subset); weight >=12 -> corr.decode_batch(high subset, enable_correlations=True)",
        f"benchmark_script_sha256: {sha256_file(script_path)}",
        f"correlated_experiment_sha256: {sha256_file(project_module)}",
        f"replicates_csv: {replicate_csv.relative_to(root).as_posix()}",
        f"summary_csv: {summary_csv.relative_to(root).as_posix()}",
    ]
    metadata_txt.write_text("\n".join(meta) + "\n", encoding="utf-8")
    print("\nCompleted standardized validation.")
    print(f"Output directory: {out_dir}")
    print(f"Replicate data: {replicate_csv}")
    print(f"Pooled summary: {summary_csv}")
    print(f"Metadata: {metadata_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
