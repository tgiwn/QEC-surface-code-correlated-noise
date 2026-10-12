"""Test whether MWPM matching-cost scores route expensive correlation-aware decoding better than detector weight.

At d=9, a frozen syndrome-weight gate routes about half of shots and yields only ~1.2x speedup.
This script screens two ordinary-MWPM score gates (total matching cost and cost per firing detector),
selects cutoffs using only fully-correlated screening data, then evaluates those frozen policies and
a previously tested detector-weight >= 12 gate on independent circuit samples.

Requires the same project API as the other project scripts in this series:
  correlated_experiment.py: make_base_circuit, get_data_layout, make_correlated_pairs,
                             add_noise_layer

PyMatching's decode_batch(..., return_weights=True) supplies the minimum matching weight for each
syndrome; that value is used only as a routing feature. The physical circuit/noise model is unchanged.
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
SCREEN_SHOTS_PER_P = 5_000_000
VALIDATION_SHOTS_PER_REPLICATE = 2_000_000
VALIDATION_REPLICATES = 3
CHUNK_SIZE = 25_000
SEED_BASE = 202_860_00
SCREEN_RISK_BUDGET = 0.10
FROZEN_WEIGHT_THRESHOLD = 12
# Quantile candidates create a routing-score frontier. The actual route fraction can differ from
# the nominal quantile when scores have ties.
SCORE_QUANTILES = (
    0.999, 0.9975, 0.995, 0.99, 0.985, 0.975, 0.95, 0.925, 0.90,
    0.875, 0.85, 0.825, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
    0.40, 0.30, 0.20, 0.10, 0.0,
)

SCREEN_CSV = Path("matching_cost_gate_d9_screen.csv")
SELECTION_CSV = Path("matching_cost_gate_d9_selection.csv")
REPLICATE_CSV = Path("matching_cost_gate_d9_validation_replicates.csv")
SUMMARY_CSV = Path("matching_cost_gate_d9_validation_summary.csv")
ERROR_RE = re.compile(r"^\s*error\(([^)]+)\)")


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
    """Paired normal CI for candidate-minus-baseline failure rate, in percentage points."""
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
            raise ValueError(f"could not parse DEM instruction: {line!r}")
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


def get_ordinary_predictions_and_cost(ordinary: pymatching.Matching, syndrome: np.ndarray):
    try:
        predictions, weights = ordinary.decode_batch(syndrome, return_weights=True)
    except TypeError as ex:
        raise RuntimeError(
            "This experiment requires a PyMatching version whose decode_batch supports return_weights=True. "
            "Please upgrade PyMatching or use the existing detector-weight gate."
        ) from ex
    costs = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(costs) != len(syndrome):
        raise AssertionError(f"expected {len(syndrome)} matching costs, got {len(costs)}")
    return as_2d(predictions), costs


def score_for(metric: str, costs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if metric == "mwpm_total_cost":
        return costs
    if metric == "mwpm_cost_per_detector":
        return costs / np.maximum(weights.astype(np.float64), 1.0)
    raise KeyError(metric)


def quantile_cutoffs(scores: np.ndarray) -> list[float]:
    vals = [math.inf]
    vals.extend(float(np.quantile(scores, q)) for q in SCORE_QUANTILES)
    # Deduplicate identical thresholds. Infinite means never route; low finite cutoffs may route all.
    return sorted(set(vals), reverse=True)


def gate_failures(ordinary_fail: np.ndarray, corr_fail: np.ndarray, route: np.ndarray) -> np.ndarray:
    return np.where(route, corr_fail, ordinary_fail)


def screen() -> dict[str, float]:
    """Return selected cutoffs keyed by score metric, using screening halves only."""
    screen_fields = [
        "p_corr", "seed", "shots", "metric", "cutoff", "route_fraction", "ordinary_failures",
        "correlation_aware_failures", "gated_failures", "extra_failures_vs_corr", "risk_budget_pass",
    ]
    cache: dict[float, dict[str, np.ndarray | int]] = {}
    pooled_ordinary = pooled_corr = 0

    with SCREEN_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=screen_fields)
        writer.writeheader()
        for p_index, p_corr in enumerate(P_CORR_VALUES):
            seed = SEED_BASE + 50_000 + p_index
            circuit, dem, ordinary, corr, pair_count = make_condition(p_corr, "correlated")
            sampler = circuit.compile_detector_sampler(seed=seed)
            cost_parts: list[np.ndarray] = []
            weight_parts: list[np.ndarray] = []
            of_parts: list[np.ndarray] = []
            cf_parts: list[np.ndarray] = []
            total_o = total_c = processed = 0
            print("\n" + "=" * 92)
            print(f"SCREEN d=9; p_corr={p_corr:g}; fully-correlated; seed={seed}; shots={SCREEN_SHOTS_PER_P:,}; pairs={pair_count}")
            print(f"DEM dimensions=({dem.num_detectors},{dem.num_observables}); error instructions={sum(x.lstrip().startswith('error(') for x in str(dem).splitlines())}")
            while processed < SCREEN_SHOTS_PER_P:
                batch = min(CHUNK_SIZE, SCREEN_SHOTS_PER_P - processed)
                syndrome, actual = sampler.sample(shots=batch, separate_observables=True)
                syndrome = np.asarray(syndrome, dtype=np.uint8)
                actual = as_2d(actual)
                pred_o, costs = get_ordinary_predictions_and_cost(ordinary, syndrome)
                pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                fo, fc = fail_mask(pred_o, actual), fail_mask(pred_c, actual)
                w = np.sum(syndrome, axis=1).astype(np.uint16)
                cost_parts.append(costs)
                weight_parts.append(w)
                of_parts.append(fo)
                cf_parts.append(fc)
                total_o += int(fo.sum())
                total_c += int(fc.sum())
                processed += batch
                if processed % 250_000 == 0 or processed == SCREEN_SHOTS_PER_P:
                    print(f"  decoded {processed:,}/{SCREEN_SHOTS_PER_P:,}")
            costs_all = np.concatenate(cost_parts)
            weights_all = np.concatenate(weight_parts)
            of_all = np.concatenate(of_parts)
            cf_all = np.concatenate(cf_parts)
            cache[p_corr] = {"cost": costs_all, "weight": weights_all, "fo": of_all, "fc": cf_all,
                             "ordinary": total_o, "corr": total_c, "seed": seed}
            pooled_ordinary += total_o
            pooled_corr += total_c

    pooled_scores = {
        metric: np.concatenate([
            score_for(metric, np.asarray(cache[p]["cost"]), np.asarray(cache[p]["weight"]))
            for p in P_CORR_VALUES
        ])
        for metric in ("mwpm_total_cost", "mwpm_cost_per_detector")
    }
    selection_rows: list[dict[str, float | int | str]] = []
    selected: dict[str, float] = {}
    total_n = SCREEN_SHOTS_PER_P * len(P_CORR_VALUES)
    budget_count = pooled_corr * (1 + SCREEN_RISK_BUDGET)
    for metric, all_scores in pooled_scores.items():
        candidate_rows = []
        for cutoff in quantile_cutoffs(all_scores):
            o_fail = c_fail = g_fail = routed = 0
            for p_corr in P_CORR_VALUES:
                d = cache[p_corr]
                score = score_for(metric, np.asarray(d["cost"]), np.asarray(d["weight"]))
                route = score >= cutoff
                of = np.asarray(d["fo"])
                cf = np.asarray(d["fc"])
                o_fail += int(of.sum())
                c_fail += int(cf.sum())
                g_fail += int(gate_failures(of, cf, route).sum())
                routed += int(route.sum())
            row = {
                "p_corr": "pooled", "seed": "screen-only", "shots": total_n, "metric": metric,
                "cutoff": cutoff, "route_fraction": routed / total_n, "ordinary_failures": o_fail,
                "correlation_aware_failures": c_fail, "gated_failures": g_fail,
                "extra_failures_vs_corr": g_fail - c_fail,
                "risk_budget_pass": g_fail <= budget_count,
            }
            candidate_rows.append(row)
            selection_rows.append(row)
        eligible = [r for r in candidate_rows if bool(r["risk_budget_pass"])]
        if not eligible:
            best = min(candidate_rows, key=lambda r: (r["gated_failures"], -r["route_fraction"]))
        else:
            # First minimize costly correlation-aware routing; break ties by lower screen failures.
            best = min(eligible, key=lambda r: (r["route_fraction"], r["gated_failures"]))
        selected[metric] = float(best["cutoff"])
        print(f"\nSELECTED {metric}: cutoff={best['cutoff']}; routed={float(best['route_fraction']):.1%}; "
              f"screen extra failures={int(best['extra_failures_vs_corr'])} ({int(best['extra_failures_vs_corr'])/max(1, pooled_corr):.1%} of CA failures)")

    # Append pooled frontier rows after per-condition rows; these are sufficient to reproduce selection.
    # Recreate CSV with all candidates, because selected cutoffs are derived on pooled conditions.
    with SCREEN_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=screen_fields)
        # Schema is per-condition screening rows; write pooled rows into the same compatible schema.
        for r in selection_rows:
            writer.writerow(r)

    selection_fields = ["distance", "p_base", "metric", "cutoff", "screen_shots", "screen_full_ca_failures",
                        "screen_gated_failures", "extra_failures", "extra_fraction_vs_ca", "routed_fraction", "risk_budget"]
    with SELECTION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selection_fields)
        writer.writeheader()
        for metric, cutoff in selected.items():
            score = pooled_scores[metric]
            route_total = gated_total = ca_total = 0
            for p_corr in P_CORR_VALUES:
                d = cache[p_corr]
                s = score_for(metric, np.asarray(d["cost"]), np.asarray(d["weight"]))
                route = s >= cutoff
                gated_total += int(gate_failures(np.asarray(d["fo"]), np.asarray(d["fc"]), route).sum())
                ca_total += int(np.asarray(d["fc"]).sum())
                route_total += int(route.sum())
            writer.writerow({"distance": DISTANCE, "p_base": P_BASE, "metric": metric, "cutoff": cutoff,
                             "screen_shots": total_n, "screen_full_ca_failures": ca_total,
                             "screen_gated_failures": gated_total, "extra_failures": gated_total-ca_total,
                             "extra_fraction_vs_ca": (gated_total-ca_total)/max(1,ca_total),
                             "routed_fraction": route_total/total_n, "risk_budget": SCREEN_RISK_BUDGET})

    print("\nSCREEN SUMMARY")
    print(f"Pooled ordinary failures={pooled_ordinary:,}; full correlation-aware failures={pooled_corr:,}; budget={budget_count:.1f}")
    print(f"Existing frozen detector-weight policy: weight >= {FROZEN_WEIGHT_THRESHOLD} (evaluated during validation)")
    print(f"Score cutoffs: {selected}")
    print(f"Screen frontier: {SCREEN_CSV.resolve()}\nSelection: {SELECTION_CSV.resolve()}")
    return selected


def validate(selected: dict[str, float]) -> None:
    policies = ["detector_weight_ge_12", "mwpm_total_cost", "mwpm_cost_per_detector"]
    rep_fields = [
        "replicate", "distance", "p_base", "p_corr", "physical_model", "seed", "shots", "policy", "cutoff",
        "ordinary_failures", "correlation_aware_failures", "gated_failures", "routed_count", "routed_fraction",
        "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "ordinary_seconds", "full_correlation_aware_seconds", "gated_seconds", "speedup_vs_full_correlation_aware",
    ]
    agg: dict[tuple[float, str, str], dict[str, float | int]] = {}

    with REPLICATE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rep_fields)
        writer.writeheader()
        for p_index, p_corr in enumerate(P_CORR_VALUES):
            for model_index, (model_name, mode) in enumerate([
                ("matched-independent-control", "matched_independent"),
                ("fully-correlated", "correlated"),
            ]):
                for rep in range(VALIDATION_REPLICATES):
                    seed = SEED_BASE + 100_000 + rep * 1_000_000 + p_index * 100 + model_index
                    circuit, dem, ordinary, corr, pair_count = make_condition(p_corr, mode)
                    sampler = circuit.compile_detector_sampler(seed=seed)
                    key_base = (p_corr, model_name)
                    print("\n" + "=" * 92)
                    print(f"VALIDATE d=9; p_corr={p_corr:g}; physical={model_name}; replicate={rep+1}/{VALIDATION_REPLICATES}; "
                          f"seed={seed}; shots={VALIDATION_SHOTS_PER_REPLICATE:,}; pairs={pair_count}")
                    counts_by_policy: dict[str, dict[str, int]] = {
                        p: {k: 0 for k in ["ordinary", "corr", "gated", "routed", "g_only_o", "o_only_g", "g_only_c", "c_only_g"]}
                        for p in policies
                    }
                    time_by_policy = {p: {"ordinary": 0.0, "corr": 0.0, "gated": 0.0} for p in policies}
                    processed = 0
                    while processed < VALIDATION_SHOTS_PER_REPLICATE:
                        batch = min(CHUNK_SIZE, VALIDATION_SHOTS_PER_REPLICATE-processed)
                        syndrome, actual = sampler.sample(shots=batch, separate_observables=True)
                        syndrome = np.asarray(syndrome, dtype=np.uint8)
                        actual = as_2d(actual)
                        detector_w = np.sum(syndrome, axis=1).astype(np.uint16)

                        # Score-producing pass for accuracy bookkeeping.
                        t0 = time.perf_counter()
                        pred_o_w, match_costs = get_ordinary_predictions_and_cost(ordinary, syndrome)
                        ordinary_score_seconds = time.perf_counter() - t0
                        pred_c = as_2d(corr.decode_batch(syndrome, enable_correlations=True))
                        fo, fc = fail_mask(pred_o_w, actual), fail_mask(pred_c, actual)
                        score_total = score_for("mwpm_total_cost", match_costs, detector_w)
                        score_mean = score_for("mwpm_cost_per_detector", match_costs, detector_w)
                        routes = {
                            "detector_weight_ge_12": detector_w >= FROZEN_WEIGHT_THRESHOLD,
                            "mwpm_total_cost": score_total >= selected["mwpm_total_cost"],
                            "mwpm_cost_per_detector": score_mean >= selected["mwpm_cost_per_detector"],
                        }
                        for policy, route in routes.items():
                            fg = gate_failures(fo, fc, route)
                            c = counts_by_policy[policy]
                            c["ordinary"] += int(fo.sum())
                            c["corr"] += int(fc.sum())
                            c["gated"] += int(fg.sum())
                            c["routed"] += int(route.sum())
                            c["g_only_o"] += int(np.count_nonzero(fg & ~fo))
                            c["o_only_g"] += int(np.count_nonzero(fo & ~fg))
                            c["g_only_c"] += int(np.count_nonzero(fg & ~fc))
                            c["c_only_g"] += int(np.count_nonzero(fc & ~fg))

                        # Time the true ordinary-only and full-correlation-aware paths.
                        t0 = time.perf_counter()
                        _ = ordinary.decode_batch(syndrome)
                        ordinary_seconds = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        _ = corr.decode_batch(syndrome, enable_correlations=True)
                        corr_seconds = time.perf_counter() - t0

                        # Time each actual gate path end-to-end (ordinary score + selected subset CA).
                        for policy in policies:
                            t0 = time.perf_counter()
                            if policy == "detector_weight_ge_12":
                                pred_o = as_2d(ordinary.decode_batch(syndrome))
                                route = detector_w >= FROZEN_WEIGHT_THRESHOLD
                            else:
                                pred_o, costs = get_ordinary_predictions_and_cost(ordinary, syndrome)
                                score = score_for(policy, costs, detector_w)
                                route = score >= selected[policy]
                            if np.any(route):
                                _ = corr.decode_batch(syndrome[route], enable_correlations=True)
                            gated_seconds = time.perf_counter() - t0
                            time_by_policy[policy]["ordinary"] += ordinary_seconds if policy == "detector_weight_ge_12" else ordinary_score_seconds
                            time_by_policy[policy]["corr"] += corr_seconds
                            time_by_policy[policy]["gated"] += gated_seconds

                        processed += batch
                        if processed % 250_000 == 0 or processed == VALIDATION_SHOTS_PER_REPLICATE:
                            print(f"  decoded {processed:,}/{VALIDATION_SHOTS_PER_REPLICATE:,}")

                    for policy in policies:
                        c = counts_by_policy[policy]
                        n = VALIDATION_SHOTS_PER_REPLICATE
                        d_o, lo_o, hi_o = paired_delta_ci(c["g_only_o"], c["o_only_g"], n)
                        d_c, lo_c, hi_c = paired_delta_ci(c["g_only_c"], c["c_only_g"], n)
                        row_time = time_by_policy[policy]
                        cutoff = FROZEN_WEIGHT_THRESHOLD if policy == "detector_weight_ge_12" else selected[policy]
                        speed = row_time["corr"] / row_time["gated"] if row_time["gated"] else float("nan")
                        writer.writerow({
                            "replicate": rep+1, "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr,
                            "physical_model": model_name, "seed": seed, "shots": n, "policy": policy,
                            "cutoff": cutoff, "ordinary_failures": c["ordinary"],
                            "correlation_aware_failures": c["corr"], "gated_failures": c["gated"],
                            "routed_count": c["routed"], "routed_fraction": c["routed"]/n,
                            "gate_vs_ordinary_delta_pp": d_o, "gate_vs_ordinary_ci_low_pp": lo_o,
                            "gate_vs_ordinary_ci_high_pp": hi_o, "gate_vs_corr_delta_pp": d_c,
                            "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                            "ordinary_seconds": row_time["ordinary"],
                            "full_correlation_aware_seconds": row_time["corr"], "gated_seconds": row_time["gated"],
                            "speedup_vs_full_correlation_aware": speed,
                        })
                        # DictWriter has no flush method; flush the underlying file handle.
                        f.flush()
                        a = agg.setdefault((p_corr, model_name, policy), {
                            "shots": 0, "ordinary": 0, "corr": 0, "gated": 0, "routed": 0,
                            "g_only_o": 0, "o_only_g": 0, "g_only_c": 0, "c_only_g": 0,
                            "ordinary_sec": 0.0, "corr_sec": 0.0, "gated_sec": 0.0,
                        })
                        a["shots"] += n
                        for field in ("ordinary", "corr", "gated", "routed", "g_only_o", "o_only_g", "g_only_c", "c_only_g"):
                            a[field] += c[field]
                        a["ordinary_sec"] += row_time["ordinary"]
                        a["corr_sec"] += row_time["corr"]
                        a["gated_sec"] += row_time["gated"]
                        print(f"  {policy}: ordinary={c['ordinary']} | full CA={c['corr']} | gated={c['gated']} | "
                              f"routed={c['routed']/n:.1%} | delta vs full CA={d_c:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}] | speedup={speed:.2f}x")

    summary_fields = [
        "distance", "p_base", "p_corr", "physical_model", "policy", "cutoff", "replicates", "pooled_shots",
        "ordinary_failures", "correlation_aware_failures", "gated_failures", "routed_fraction",
        "gate_vs_ordinary_delta_pp", "gate_vs_ordinary_ci_low_pp", "gate_vs_ordinary_ci_high_pp",
        "gate_vs_corr_delta_pp", "gate_vs_corr_ci_low_pp", "gate_vs_corr_ci_high_pp",
        "ordinary_seconds", "full_correlation_aware_seconds", "gated_seconds", "speedup_vs_full_correlation_aware",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for (p_corr, model, policy), a in sorted(agg.items()):
            n = int(a["shots"])
            do, lo_o, hi_o = paired_delta_ci(int(a["g_only_o"]), int(a["o_only_g"]), n)
            dc, lo_c, hi_c = paired_delta_ci(int(a["g_only_c"]), int(a["c_only_g"]), n)
            cutoff = FROZEN_WEIGHT_THRESHOLD if policy == "detector_weight_ge_12" else selected[policy]
            speed = float(a["corr_sec"]) / float(a["gated_sec"]) if a["gated_sec"] else float("nan")
            writer.writerow({
                "distance": DISTANCE, "p_base": P_BASE, "p_corr": p_corr, "physical_model": model,
                "policy": policy, "cutoff": cutoff, "replicates": VALIDATION_REPLICATES, "pooled_shots": n,
                "ordinary_failures": a["ordinary"], "correlation_aware_failures": a["corr"], "gated_failures": a["gated"],
                "routed_fraction": int(a["routed"])/n, "gate_vs_ordinary_delta_pp": do,
                "gate_vs_ordinary_ci_low_pp": lo_o, "gate_vs_ordinary_ci_high_pp": hi_o,
                "gate_vs_corr_delta_pp": dc, "gate_vs_corr_ci_low_pp": lo_c, "gate_vs_corr_ci_high_pp": hi_c,
                "ordinary_seconds": a["ordinary_sec"], "full_correlation_aware_seconds": a["corr_sec"],
                "gated_seconds": a["gated_sec"], "speedup_vs_full_correlation_aware": speed,
            })
            print(f"\nPOOLED d=9 p_corr={p_corr:g} {model} {policy} (n={n:,})")
            print(f"  ordinary={a['ordinary']/n*100:.5f}% | full CA={a['corr']/n*100:.5f}% | gated={a['gated']/n*100:.5f}%")
            print(f"  gate-full-CA={dc:+.6f} pp [{lo_c:+.6f},{hi_c:+.6f}]; routed={int(a['routed'])/n:.1%}; speedup={speed:.2f}x")
    print(f"\nReplicates: {REPLICATE_CSV.resolve()}\nSummary: {SUMMARY_CSV.resolve()}")


def load_saved_selection() -> dict[str, float] | None:
    """Reuse already-completed screening cutoffs, avoiding an expensive screening rerun."""
    if not SELECTION_CSV.exists():
        return None
    try:
        with SELECTION_CSV.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        selected = {str(row["metric"]): float(row["cutoff"]) for row in rows
                    if row.get("metric") and row.get("cutoff") not in (None, "")}
        required = {"mwpm_total_cost", "mwpm_cost_per_detector"}
        if required.issubset(selected):
            return {metric: selected[metric] for metric in sorted(required)}
    except (OSError, ValueError, KeyError, csv.Error):
        return None
    return None


def main() -> None:
    print("Distance-9 routing gates: MWPM matching-cost scores vs detector weight")
    print(f"P_BASE={P_BASE}; p_corr={P_CORR_VALUES}; screen={SCREEN_SHOTS_PER_P:,}/p_corr; validation={VALIDATION_REPLICATES} x {VALIDATION_SHOTS_PER_REPLICATE:,}/condition")
    print(f"seed base={SEED_BASE}; frozen weight comparator: detector weight >= {FROZEN_WEIGHT_THRESHOLD}; screening risk budget={SCREEN_RISK_BUDGET:.0%}")
    print("Candidate gates: total ordinary-MWPM matching weight; matching weight per firing detector.")
    print("Cost cutoffs are chosen only on fully-correlated screening data and frozen before independent validation.")
    selected = load_saved_selection()
    if selected is not None:
        print(f"Reusing frozen screening cutoffs from {SELECTION_CSV.resolve()}: {selected}")
        print("The validation CSV will be restarted cleanly; completed screening files will not be overwritten.")
    else:
        print("No valid saved selection found; running screening first.")
        selected = screen()
    validate(selected)
    print("\nCompleted. Select a gate based on held-out speed and failure risk, not screen performance alone.")


if __name__ == "__main__":
    main()
