"""Independent validation of stricter risk-budgeted MWPM score gates at d=9.

Uses the saved screening frontier from test_matching_cost_gate_d9.py, so it does
not repeat the expensive 10-million-shot screening run. For total MWPM cost and
cost-per-detector, it selects cutoffs under 2.5%, 5%, and 10% *screening*
excess-failure budgets, then evaluates all frozen gates on fresh circuit seeds.

The physical circuit always retains the full target correlated-pair noise. Gates
only choose whether to use ordinary or correlation-aware matching for a shot.
"""
from __future__ import annotations

import csv
import math
import re
import time
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

P_BASE = 0.001
P_CORR_VALUES = (0.0001, 0.0002)
DISTANCE = 9
VALIDATION_SHOTS_PER_REPLICATE = 2_000_000
VALIDATION_REPLICATES = 3
CHUNK_SIZE = 25_000
SEED_BASE = 20_400_000
RISK_BUDGETS = (0.025, 0.05, 0.10)
FROZEN_WEIGHT_THRESHOLD = 12
BENCHMARK_SHOTS_PER_REPLICATE = 250_000

SOURCE_SCREEN_CSV = Path("matching_cost_gate_d9_screen.csv")
SELECTION_CSV = Path("matching_cost_gate_risk_budget_selection.csv")
REPLICATE_CSV = Path("matching_cost_gate_risk_budget_validation_replicates.csv")
SUMMARY_CSV = Path("matching_cost_gate_risk_budget_validation_summary.csv")
ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")
METRICS = ("mwpm_total_cost", "mwpm_cost_per_detector")


def as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8)
    return arr[:, None] if arr.ndim == 1 else arr


def fail_mask(predictions: Any, actual: Any) -> np.ndarray:
    p = as_2d(predictions)
    y = as_2d(actual)
    if p.shape != y.shape:
        raise ValueError(f"prediction/observable shape mismatch: {p.shape} vs {y.shape}")
    return np.any(p != y, axis=1)


def paired_delta_ci(candidate_only: int, baseline_only: int, n: int, z: float = 1.959963984540054):
    """Candidate-minus-baseline paired failure-rate difference, percentage points."""
    d = (candidate_only - baseline_only) / n
    discordant = candidate_only + baseline_only
    var = max(0.0, (discordant - n * d * d) / ((n - 1) * n)) if n > 1 else float("nan")
    delta_pp = 100 * d
    se_pp = 100 * math.sqrt(var) if n > 1 else float("nan")
    return delta_pp, delta_pp - z * se_pp, delta_pp + z * se_pp


def canonical_probability_sorted_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    flat = dem.flattened()
    lines = str(flat).splitlines()
    positions = [i for i, line in enumerate(lines) if line.lstrip().startswith("error(")]
    errors = [lines[i] for i in positions]

    def key(line: str) -> tuple[float, str]:
        m = ERROR_RE.match(line)
        if m is None:
            raise ValueError(f"Could not parse DEM error instruction: {line!r}")
        return (-float(m.group(1)), line.strip())

    for pos, line in zip(positions, sorted(errors, key=key)):
        lines[pos] = line
    result = stim.DetectorErrorModel("\n".join(lines))
    if (result.num_detectors, result.num_observables) != (flat.num_detectors, flat.num_observables):
        raise AssertionError("canonical sorting changed DEM dimensions")
    return result


def make_condition(p_corr: float, mode: str):
    base = make_base_circuit(distance=DISTANCE, rounds=DISTANCE, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, p_corr, mode)
    dem = canonical_probability_sorted_dem(circuit.detector_error_model(decompose_errors=True).flattened())
    ordinary = pymatching.Matching(dem)
    corr = pymatching.Matching(dem, enable_correlations=True)
    return circuit, dem, ordinary, corr, len(pairs)


def score_for(metric: str, costs: np.ndarray, detector_weight: np.ndarray) -> np.ndarray:
    if metric == "mwpm_total_cost":
        return costs
    if metric == "mwpm_cost_per_detector":
        return costs / np.maximum(detector_weight.astype(np.float64), 1.0)
    raise KeyError(metric)


def policy_name(metric: str, budget: float) -> str:
    return f"{metric}_budget_{budget:g}"


def load_selection_from_saved_screen() -> dict[str, dict[str, float]]:
    if not SOURCE_SCREEN_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_SCREEN_CSV}. Run test_matching_cost_gate_d9.py first; "
            "this script intentionally does not repeat its expensive screening run."
        )
    with SOURCE_SCREEN_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # The source script appends pooled frontier rows with p_corr='pooled'.
    pooled: dict[str, list[dict[str, Any]]] = {m: [] for m in METRICS}
    for row in rows:
        if str(row.get("p_corr", "")).strip().lower() != "pooled":
            continue
        metric = str(row.get("metric", ""))
        if metric not in pooled:
            continue
        try:
            cutoff = float(row["cutoff"])
            route_fraction = float(row["route_fraction"])
            extra = int(float(row["extra_failures_vs_corr"]))
            ca_failures = int(float(row["correlation_aware_failures"]))
            gated_failures = int(float(row["gated_failures"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Malformed pooled screening row: {row}") from exc
        pooled[metric].append({
            "cutoff": cutoff,
            "route_fraction": route_fraction,
            "extra_failures": extra,
            "ca_failures": ca_failures,
            "gated_failures": gated_failures,
        })

    selected: dict[str, dict[str, float]] = {}
    selection_fields = [
        "distance", "p_base", "metric", "risk_budget", "cutoff", "screen_route_fraction",
        "screen_ca_failures", "screen_gated_failures", "screen_extra_failures", "screen_extra_fraction_vs_ca",
        "source_screen_csv",
    ]
    with SELECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selection_fields)
        writer.writeheader()
        for metric in METRICS:
            if not pooled[metric]:
                raise ValueError(f"No pooled frontier rows for {metric} in {SOURCE_SCREEN_CSV}")
            for budget in RISK_BUDGETS:
                candidates = [r for r in pooled[metric]
                              if r["ca_failures"] > 0 and r["extra_failures"] <= budget * r["ca_failures"]]
                if not candidates:
                    raise RuntimeError(f"No eligible cutoff for {metric} at risk budget {budget:.1%}")
                # Use the least expensive eligible routing policy; ties prefer fewer failures.
                best = min(candidates, key=lambda r: (r["route_fraction"], r["gated_failures"]))
                selected.setdefault(metric, {})[str(budget)] = float(best["cutoff"])
                writer.writerow({
                    "distance": DISTANCE,
                    "p_base": P_BASE,
                    "metric": metric,
                    "risk_budget": budget,
                    "cutoff": best["cutoff"],
                    "screen_route_fraction": best["route_fraction"],
                    "screen_ca_failures": best["ca_failures"],
                    "screen_gated_failures": best["gated_failures"],
                    "screen_extra_failures": best["extra_failures"],
                    "screen_extra_fraction_vs_ca": best["extra_failures"] / best["ca_failures"],
                    "source_screen_csv": str(SOURCE_SCREEN_CSV),
                })
                print(
                    f"SCREEN-SELECTED {metric}, budget={budget:.1%}: cutoff={best['cutoff']:.8g}; "
                    f"routed={best['route_fraction']:.1%}; extra={best['extra_failures']}/"
                    f"{best['ca_failures']} ({best['extra_failures']/best['ca_failures']:.1%})"
                )
    print(f"Selection table written to {SELECTION_CSV.resolve()}")
    return selected


def main() -> None:
    print("Distance-9 risk-budgeted matching-cost gates (reusing previous screening data)")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; distances=(9); budgets={RISK_BUDGETS}")
    print(f"fresh validation={VALIDATION_REPLICATES} x {VALIDATION_SHOTS_PER_REPLICATE:,}/condition; seed base={SEED_BASE}")
    print(f"Source screen: {SOURCE_SCREEN_CSV.resolve()}")
    print("Thresholds are selected from saved SCREEN rows, then validated on new circuit seeds.")
    selected = load_selection_from_saved_screen()

    policies: list[tuple[str, str, float]] = []
    for metric in METRICS:
        for budget in RISK_BUDGETS:
            policies.append((policy_name(metric, budget), metric, selected[metric][str(budget)]))
    policy_names = [p[0] for p in policies] + ["detector_weight_ge_12"]

    replicate_fields = [
        "replicate", "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "policy", "score_metric",
        "risk_budget", "cutoff", "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "routed_count", "routed_fraction", "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp",
        "gate_vs_ordinary_ci_high_pp", "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp",
        "gate_vs_corr_ci_high_pp", "ordinary_seconds_bench", "full_ca_seconds_bench", "gated_seconds_bench",
        "speedup_vs_full_ca_bench",
    ]
    summary_agg: dict[tuple[float, str, str], dict[str, float | int]] = {}

    with REPLICATE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=replicate_fields)
        writer.writeheader()
        f.flush()
        for p_index, p_corr in enumerate(P_CORR_VALUES):
            for model_index, (model_name, mode) in enumerate([
                ("matched-independent-control", "matched_independent"),
                ("fully-correlated", "correlated"),
            ]):
                for rep in range(VALIDATION_REPLICATES):
                    seed = SEED_BASE + 100_000 + rep * 1_000_000 + p_index * 100 + model_index
                    circuit, dem, ordinary, corr, pair_count = make_condition(p_corr, mode)
                    sampler = circuit.compile_detector_sampler(seed=seed)
                    counts = {
                        name: {k: 0 for k in ("ordinary", "corr", "gated", "routed", "g_only_o", "o_only_g", "g_only_c", "c_only_g")}
                        for name in policy_names
                    }
                    bench_times = {
                        name: {"ordinary": 0.0, "ca": 0.0, "gated": 0.0, "shots": 0}
                        for name in policy_names
                    }
                    processed = 0
                    print("\n" + "=" * 100)
                    print(f"VALIDATE d=9; p_corr={p_corr:g}; physical={model_name}; replicate={rep+1}/{VALIDATION_REPLICATES}; "
                          f"seed={seed}; shots={VALIDATION_SHOTS_PER_REPLICATE:,}; pairs={pair_count}")
                    while processed < VALIDATION_SHOTS_PER_REPLICATE:
                        batch = min(CHUNK_SIZE, VALIDATION_SHOTS_PER_REPLICATE - processed)
                        syndrome, actual = sampler.sample(shots=batch, separate_observables=True)
                        syndrome = np.asarray(syndrome, dtype=np.uint8)
                        actual = as_2d(actual)
                        detector_weight = np.sum(syndrome, axis=1).astype(np.uint16)

                        # One shared set of predictions is used for all paired accuracy comparisons.
                        pred_o, costs = ordinary.decode_batch(syndrome, return_weights=True)
                        pred_o = as_2d(pred_o)
                        costs = np.asarray(costs, dtype=np.float64).reshape(-1)
                        pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                        fo, fc = fail_mask(pred_o, actual), fail_mask(pred_c, actual)
                        scores = {
                            metric: (costs if metric == "mwpm_total_cost" else costs / np.maximum(detector_weight.astype(np.float64), 1.0))
                            for metric in METRICS
                        }
                        routes: dict[str, np.ndarray] = {}
                        for name, metric, cutoff in policies:
                            routes[name] = scores[metric] >= cutoff
                        routes["detector_weight_ge_12"] = detector_weight >= FROZEN_WEIGHT_THRESHOLD

                        for name in policy_names:
                            route = routes[name]
                            fg = np.where(route, fc, fo)
                            c = counts[name]
                            c["ordinary"] += int(fo.sum())
                            c["corr"] += int(fc.sum())
                            c["gated"] += int(fg.sum())
                            c["routed"] += int(route.sum())
                            c["g_only_o"] += int(np.count_nonzero(fg & ~fo))
                            c["o_only_g"] += int(np.count_nonzero(fo & ~fg))
                            c["g_only_c"] += int(np.count_nonzero(fg & ~fc))
                            c["c_only_g"] += int(np.count_nonzero(fc & ~fg))

                        # Time true decoding paths only on the first BENCHMARK_SHOTS_PER_REPLICATE
                        # shots of each replicate. Timings are not used for threshold selection.
                        if processed < BENCHMARK_SHOTS_PER_REPLICATE:
                            bench_t0 = time.perf_counter()
                            _ = ordinary.decode_batch(syndrome)
                            ordinary_time = time.perf_counter() - bench_t0
                            bench_t0 = time.perf_counter()
                            _ = corr.decode_batch(syndrome, enable_correlations=True)
                            ca_time = time.perf_counter() - bench_t0
                            for name in policy_names:
                                bench_t0 = time.perf_counter()
                                if name == "detector_weight_ge_12":
                                    _ = ordinary.decode_batch(syndrome)
                                else:
                                    # Cost gates need the score-producing ordinary pass on all shots.
                                    _ = ordinary.decode_batch(syndrome, return_weights=True)
                                route = routes[name]
                                if np.any(route):
                                    _ = corr.decode_batch(syndrome[route], enable_correlations=True)
                                gated_time = time.perf_counter() - bench_t0
                                bt = bench_times[name]
                                bt["ordinary"] += ordinary_time
                                bt["ca"] += ca_time
                                bt["gated"] += gated_time
                                bt["shots"] += batch

                        processed += batch
                        if processed % 250_000 == 0 or processed == VALIDATION_SHOTS_PER_REPLICATE:
                            print(f"  decoded {processed:,}/{VALIDATION_SHOTS_PER_REPLICATE:,}")

                    for name, metric, cutoff in policies:
                        c = counts[name]
                        n = VALIDATION_SHOTS_PER_REPLICATE
                        d_o, lo_o, hi_o = paired_delta_ci(c["g_only_o"], c["o_only_g"], n)
                        d_c, lo_c, hi_c = paired_delta_ci(c["g_only_c"], c["c_only_g"], n)
                        bt = bench_times[name]
                        speedup = bt["ca"] / bt["gated"] if bt["gated"] else float("nan")
                        budget = next(b for b in RISK_BUDGETS if policy_name(metric, b) == name)
                        row = {
                            "replicate": rep + 1, "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr,
                            "physical_model": model_name, "seed": seed, "shots": n, "policy": name,
                            "score_metric": metric, "risk_budget": budget, "cutoff": cutoff,
                            "ordinary_failures": c["ordinary"], "correlation_aware_failures": c["corr"],
                            "gated_failures": c["gated"], "routed_count": c["routed"], "routed_fraction": c["routed"] / n,
                            "gate_vs_ordinary_delta_pp": d_o, "gate_vs_ordinary_ci_low_pp": lo_o,
                            "gate_vs_ordinary_ci_high_pp": hi_o, "gate_vs_corr_delta_pp": d_c,
                            "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                            "ordinary_seconds_bench": bt["ordinary"], "full_ca_seconds_bench": bt["ca"],
                            "gated_seconds_bench": bt["gated"], "speedup_vs_full_ca_bench": speedup,
                        }
                        writer.writerow(row)
                        f.flush()
                        key = (p_corr, model_name, name)
                        a = summary_agg.setdefault(key, {
                            "shots": 0, "ordinary": 0, "corr": 0, "gated": 0, "routed": 0,
                            "g_only_o": 0, "o_only_g": 0, "g_only_c": 0, "c_only_g": 0,
                            "ordinary_sec": 0.0, "ca_sec": 0.0, "gated_sec": 0.0,
                        })
                        a["shots"] += n
                        for field in ("ordinary", "corr", "gated", "routed", "g_only_o", "o_only_g", "g_only_c", "c_only_g"):
                            a[field] += c[field]
                        a["ordinary_sec"] += bt["ordinary"]
                        a["ca_sec"] += bt["ca"]
                        a["gated_sec"] += bt["gated"]
                        print(f"  {name}: gated={c['gated']}/{n} ({c['gated']/n*100:.5f}%), routed={c['routed']/n:.1%}, "
                              f"delta vs full CA={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}], "
                              f"bench speedup={speedup:.2f}x")

                    # Log the fixed detector-weight comparator once per condition/replicate.
                    name = "detector_weight_ge_12"
                    c = counts[name]
                    n = VALIDATION_SHOTS_PER_REPLICATE
                    d_o, lo_o, hi_o = paired_delta_ci(c["g_only_o"], c["o_only_g"], n)
                    d_c, lo_c, hi_c = paired_delta_ci(c["g_only_c"], c["c_only_g"], n)
                    bt = bench_times[name]
                    speedup = bt["ca"] / bt["gated"] if bt["gated"] else float("nan")
                    writer.writerow({
                        "replicate": rep + 1, "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr,
                        "physical_model": model_name, "seed": seed, "shots": n, "policy": name,
                        "score_metric": "detector_weight", "risk_budget": "", "cutoff": FROZEN_WEIGHT_THRESHOLD,
                        "ordinary_failures": c["ordinary"], "correlation_aware_failures": c["corr"],
                        "gated_failures": c["gated"], "routed_count": c["routed"], "routed_fraction": c["routed"] / n,
                        "gate_vs_ordinary_delta_pp": d_o, "gate_vs_ordinary_ci_low_pp": lo_o,
                        "gate_vs_ordinary_ci_high_pp": hi_o, "gate_vs_corr_delta_pp": d_c,
                        "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                        "ordinary_seconds_bench": bt["ordinary"], "full_ca_seconds_bench": bt["ca"],
                        "gated_seconds_bench": bt["gated"], "speedup_vs_full_ca_bench": speedup,
                    })
                    f.flush()
                    key = (p_corr, model_name, name)
                    a = summary_agg.setdefault(key, {
                        "shots": 0, "ordinary": 0, "corr": 0, "gated": 0, "routed": 0,
                        "g_only_o": 0, "o_only_g": 0, "g_only_c": 0, "c_only_g": 0,
                        "ordinary_sec": 0.0, "ca_sec": 0.0, "gated_sec": 0.0,
                    })
                    a["shots"] += n
                    for field in ("ordinary", "corr", "gated", "routed", "g_only_o", "o_only_g", "g_only_c", "c_only_g"):
                        a[field] += c[field]
                    a["ordinary_sec"] += bt["ordinary"]
                    a["ca_sec"] += bt["ca"]
                    a["gated_sec"] += bt["gated"]
                    print(f"  detector_weight_ge_12: gated={c['gated']}/{n} ({c['gated']/n*100:.5f}%), "
                          f"routed={c['routed']/n:.1%}, delta vs full CA={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}], "
                          f"bench speedup={speedup:.2f}x")

    summary_fields = [
        "distance", "p_base", "p_corr", "physical_model", "policy", "score_metric", "risk_budget", "cutoff",
        "replicates", "pooled_shots", "ordinary_failures", "correlation_aware_failures", "gated_failures",
        "routed_fraction", "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "ordinary_bench_seconds", "full_ca_bench_seconds", "gated_bench_seconds", "speedup_vs_full_ca_bench",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (p_corr, model_name, name), a in sorted(summary_agg.items()):
            n = int(a["shots"])
            d_o, lo_o, hi_o = paired_delta_ci(int(a["g_only_o"]), int(a["o_only_g"]), n)
            d_c, lo_c, hi_c = paired_delta_ci(int(a["g_only_c"]), int(a["c_only_g"]), n)
            if name == "detector_weight_ge_12":
                metric, budget, cutoff = "detector_weight", "", FROZEN_WEIGHT_THRESHOLD
            else:
                found = next((item for item in policies if item[0] == name), None)
                assert found is not None
                _, metric, cutoff = found
                budget = next(b for b in RISK_BUDGETS if policy_name(metric, b) == name)
            speed = float(a["ca_sec"]) / float(a["gated_sec"]) if a["gated_sec"] else float("nan")
            writer.writerow({
                "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr, "physical_model": model_name,
                "policy": name, "score_metric": metric, "risk_budget": budget, "cutoff": cutoff,
                "replicates": VALIDATION_REPLICATES, "pooled_shots": n,
                "ordinary_failures": a["ordinary"], "correlation_aware_failures": a["corr"],
                "gated_failures": a["gated"], "routed_fraction": int(a["routed"]) / n,
                "gate_vs_ordinary_delta_pp": d_o, "gate_vs_ordinary_ci_low_pp": lo_o,
                "gate_vs_ordinary_ci_high_pp": hi_o, "gate_vs_corr_delta_pp": d_c,
                "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                "ordinary_bench_seconds": a["ordinary_sec"], "full_ca_bench_seconds": a["ca_sec"],
                "gated_bench_seconds": a["gated_sec"], "speedup_vs_full_ca_bench": speed,
            })
            print(f"\nPOOLED d=9 p_corr={p_corr:g} {model_name} {name}; shots={n:,}")
            print(f"  ordinary={a['ordinary']/n*100:.5f}% | full CA={a['corr']/n*100:.5f}% | gated={a['gated']/n*100:.5f}%")
            print(f"  gate-full-CA={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}], routed={int(a['routed'])/n:.1%}, speedup={speed:.2f}x")
    print(f"\nReplicates: {REPLICATE_CSV.resolve()}\nSummary: {SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
