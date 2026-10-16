"""Exact MAP decoding from the full distance-3 Stim detector error model.

This is a principled decoder benchmark for the distance-3 circuit. Instead of
splitting a `^`-decomposed DEM instruction into independent graphlike edges,
it treats all detector/logical targets on the full error instruction as ONE
Bernoulli mechanism. It then exactly (up to floating-point arithmetic)
convolves the mechanism distribution over every detector syndrome and the one
logical observable.

The exact posterior table has 2**num_detectors states per logical value.
For the current distance-3 model (24 detectors, 1 observable), five float64
arrays peak around 640 MiB. Do not run this unchanged at distance 5 or 7:
the dense state space grows exponentially with detector count.

This gives the exact maximum-a-posteriori logical prediction under the flattened
DEM's independent error-instruction model. It is not a general-purpose scalable
decoder, and its correctness as a physical-circuit benchmark depends on the DEM
faithfully representing the circuit's detector/observable distribution.
"""

import math
import re
import time
from collections import defaultdict

import numpy as np
import pymatching

from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)

DISTANCE = 3
ROUNDS = 3
P_BASE = 0.001
P_CORR = 0.0001
SHOTS = 200_000
CHUNK_SIZE = 25_000
SEED = 20261032


def parse_full_dem_mechanisms(dem):
    """Parse each complete `error(p) ...` instruction as one mechanism.

    Critically, do NOT split a source instruction on '^'. The separator is a
    graph-decomposition hint. Every D/L target across the whole instruction
    participates in the same fault event, which preserves that instruction's
    correlated detector/logical signature.
    """
    n_det = dem.num_detectors
    n_obs = dem.num_observables
    records = []
    zero_p = 0
    neutral = 0

    for line_number, line in enumerate(str(dem.flattened()).splitlines(), start=1):
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue

        p = float(match.group(1))
        body = match.group(2)
        if not 0 <= p <= 1:
            raise ValueError(f"Invalid error probability {p} on line: {line}")
        if p == 0:
            zero_p += 1
            continue

        det_mask = 0
        obs_mask = 0
        for raw in re.findall(r"\bD(\d+)\b", body):
            index = int(raw)
            if index >= n_det:
                raise ValueError(f"Detector D{index} exceeds DEM dimensions")
            # Fault targets combine by parity. XOR also handles accidental
            # repeated detector IDs correctly.
            det_mask ^= 1 << index

        for raw in re.findall(r"\bL(\d+)\b", body):
            index = int(raw)
            if index >= n_obs:
                raise ValueError(f"Observable L{index} exceeds DEM dimensions")
            obs_mask ^= 1 << index

        if det_mask == 0 and obs_mask == 0:
            neutral += 1
            continue

        records.append((p, det_mask, obs_mask, line.strip(), line_number))

    return records, zero_p, neutral


def xor_shift_view(array, mask, num_bits):
    """Return a view `out[s] == array[s XOR mask]` without a large index array."""
    if mask == 0:
        return array
    # In C-order, the least-significant integer bit is the last tensor axis.
    axes = tuple(num_bits - 1 - bit for bit in range(num_bits) if mask & (1 << bit))
    return np.flip(array, axis=axes)


def exact_joint_distribution(dem):
    """Compute P(detector syndrome, L0) by exact XOR-convolution over DEM rows."""
    n_det = dem.num_detectors
    n_obs = dem.num_observables
    if n_obs != 1:
        raise NotImplementedError(
            f"This exact benchmark currently supports one observable, got {n_obs}."
        )
    if n_det > 24:
        raise MemoryError(
            f"Refusing dense exact state space for {n_det} detectors. "
            "This script is intended for distance 3 (24 detectors)."
        )

    mechanisms, zero_p, neutral = parse_full_dem_mechanisms(dem)
    shape = (2,) * n_det
    states = 1 << n_det
    mib_each = states * np.dtype(np.float64).itemsize / (1024 ** 2)
    print(f"Flattened DEM mechanisms used: {len(mechanisms):,}")
    print(f"Zero-probability instructions skipped: {zero_p}")
    print(f"Net-neutral instructions skipped: {neutral}")
    print(f"Detector states: 2^{n_det} = {states:,}; float64 array={mib_each:.1f} MiB")
    print(f"Approximate peak for 5 dense arrays: {5 * mib_each:.0f} MiB")

    q0 = np.zeros(shape, dtype=np.float64)
    q1 = np.zeros(shape, dtype=np.float64)
    q0[(0,) * n_det] = 1.0
    new0 = np.empty_like(q0)
    new1 = np.empty_like(q1)
    scratch = np.empty_like(q0)

    started = time.time()
    progress_every = max(1, len(mechanisms) // 10)
    for index, (p, det_mask, obs_mask, _line, _line_number) in enumerate(mechanisms, start=1):
        # If the event flips L0, outcomes swap logical sectors. The full D mask
        # is applied in either case, including targets on both sides of '^'.
        source0 = q1 if obs_mask & 1 else q0
        source1 = q0 if obs_mask & 1 else q1
        shifted0 = xor_shift_view(source0, det_mask, n_det)
        shifted1 = xor_shift_view(source1, det_mask, n_det)

        np.multiply(q0, 1.0 - p, out=new0)
        np.multiply(shifted0, p, out=scratch)
        np.add(new0, scratch, out=new0)

        np.multiply(q1, 1.0 - p, out=new1)
        np.multiply(shifted1, p, out=scratch)
        np.add(new1, scratch, out=new1)

        q0, new0 = new0, q0
        q1, new1 = new1, q1

        if index % progress_every == 0 or index == len(mechanisms):
            elapsed = time.time() - started
            print(f"  convolved {index:>4}/{len(mechanisms)} mechanisms; elapsed {elapsed:.1f}s")

    total = float(np.sum(q0, dtype=np.float64) + np.sum(q1, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError(f"Invalid probability normalization: {total}")

    q0_flat = q0.reshape(-1)
    q1_flat = q1.reshape(-1)
    # Reuse scratch for min(P(s,L0=0), P(s,L0=1)) to avoid another dense array.
    scratch_flat = scratch.reshape(-1)
    np.minimum(q0_flat, q1_flat, out=scratch_flat)
    bayes_risk = float(np.sum(scratch_flat, dtype=np.float64)) / total
    p0_total = float(np.sum(q0_flat, dtype=np.float64)) / total
    p1_total = float(np.sum(q1_flat, dtype=np.float64)) / total

    print(f"Joint distribution normalization: {total:.16f}")
    print(f"P(L0=0) under DEM: {p0_total:.8%}")
    print(f"P(L0=1) under DEM: {p1_total:.8%}")
    print(f"Exact DEM-MAP Bayes risk: {bayes_risk:.8%}")
    print(f"Exact DEM-MAP risk in percentage points: {bayes_risk * 100:.6f} pp")
    return q0_flat, q1_flat, total, bayes_risk


def as_2d(values):
    values = np.asarray(values, dtype=np.uint8)
    return values[:, None] if values.ndim == 1 else values


def encode_syndromes(syndrome_bits):
    """Pack detector rows into uint32 indices where D_i maps to bit i."""
    syndrome_bits = np.asarray(syndrome_bits, dtype=np.uint8)
    n_det = syndrome_bits.shape[1]
    if n_det > 31:
        raise ValueError("uint32 syndrome encoder supports at most 31 detectors")
    indices = np.zeros(syndrome_bits.shape[0], dtype=np.uint32)
    for bit in range(n_det):
        indices |= syndrome_bits[:, bit].astype(np.uint32) << np.uint32(bit)
    return indices


def fail_mask(pred, actual):
    pred = as_2d(pred)
    actual = as_2d(actual)
    return np.any(pred != actual, axis=1)


def paired_delta_ci(candidate_fail, baseline_fail):
    diff = candidate_fail.astype(np.int8) - baseline_fail.astype(np.int8)
    delta = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
    return delta, (delta - 1.96 * se, delta + 1.96 * se)


def build_probability_prioritized_matching(dem):
    """Rebuild the probability-prioritized heuristic for comparison only."""
    components = []
    for line in str(dem.flattened()).splitlines():
        match = re.match(r"\s*error\(([^)]+)\)\s*(.*)", line)
        if not match:
            continue
        p = float(match.group(1))
        if not 0 < p < 0.5:
            continue
        for piece in re.split(r"\s+\^\s+", match.group(2)):
            dets = tuple(sorted(int(x) for x in re.findall(r"\bD(\d+)\b", piece)))
            labels = tuple(sorted(int(x) for x in re.findall(r"\bL(\d+)\b", piece)))
            if len(dets) not in (1, 2):
                if dets:
                    raise ValueError(
                        "The distance-3 priority graph encountered a non-graphlike component; "
                        "review the graph construction before comparing results."
                    )
                continue
            components.append((p, dets, labels, line.strip()))

    components.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    matching = pymatching.Matching()
    for p, dets, labels, _ in components:
        weight = math.log((1.0 - p) / p)
        fault_ids = set(labels)
        if len(dets) == 1:
            matching.add_boundary_edge(
                dets[0], fault_ids=fault_ids, weight=weight,
                error_probability=p, merge_strategy="independent",
            )
        else:
            matching.add_edge(
                dets[0], dets[1], fault_ids=fault_ids, weight=weight,
                error_probability=p, merge_strategy="independent",
            )
    return matching


def main():
    print("Exact distance-3 MAP benchmark from the complete detector error model")
    print(f"distance={DISTANCE}; rounds={ROUNDS}; P_BASE={P_BASE}; P_CORR={P_CORR}; test shots={SHOTS:,}; seed={SEED}")
    print("Exact posterior convolution keeps each full DEM error instruction intact, including all terms separated by '^'.")
    print("Scope: exact under the DEM probability model, not automatically exact for the circuit if DEM generation approximated it.\n")

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, P_CORR, "correlated")
    dem = circuit.detector_error_model(decompose_errors=True).flattened()
    if dem.num_detectors != 24 or dem.num_observables != 1:
        print(f"Warning: expected 24 detectors and 1 observable, found {dem.num_detectors} and {dem.num_observables}.")

    q0, q1, total, bayes_risk = exact_joint_distribution(dem)
    del total

    ordinary = pymatching.Matching(dem)
    priority = build_probability_prioritized_matching(dem)
    try:
        correlated_matching = pymatching.Matching(dem, enable_correlations=True)
        has_correlated = True
    except TypeError:
        correlated_matching = None
        has_correlated = False
        print("Installed PyMatching does not accept enable_correlations=True at construction; skipping that comparator.")

    sampler = circuit.compile_detector_sampler(seed=SEED)
    detector_bits, observable_bits = sampler.sample(
        shots=SHOTS, separate_observables=True
    )
    detector_bits = np.asarray(detector_bits, dtype=np.uint8)
    actual = as_2d(observable_bits)
    syndrome_indices = encode_syndromes(detector_bits)

    # For one logical observable, exact MAP chooses the larger joint mass
    # P(s,L0=0) vs P(s,L0=1). Ties deterministically choose 0.
    prob0 = q0[syndrome_indices]
    prob1 = q1[syndrome_indices]
    map_pred = (prob1 > prob0).astype(np.uint8)[:, None]
    map_ties = int(np.count_nonzero(prob0 == prob1))

    predictions = {
        "ordinary MWPM": as_2d(ordinary.decode_batch(detector_bits)),
        "probability-prioritized MWPM": as_2d(priority.decode_batch(detector_bits)),
        "exact DEM-MAP": map_pred,
    }
    if has_correlated:
        try:
            predictions["correlation-aware MWPM"] = as_2d(
                correlated_matching.decode_batch(
                    detector_bits, enable_correlations=True
                )
            )
        except TypeError:
            print("Installed PyMatching does not support enable_correlations in decode_batch; skipping that comparator.")

    failure_masks = {name: fail_mask(pred, actual) for name, pred in predictions.items()}
    baseline_name = "ordinary MWPM"
    baseline_fail = failure_masks[baseline_name]

    print("\n========== SAMPLED CIRCUIT VALIDATION ==========")
    print(f"shots={SHOTS:,}; unique sampled detector syndromes={len(np.unique(syndrome_indices)):,}")
    print(f"Exact-MAP ties among sampled shots: {map_ties:,}")
    print("decoder                         failures       rate       predictions changed   paired delta vs ordinary (95% CI)")
    print("-" * 112)
    for name, pred in predictions.items():
        fail = failure_masks[name]
        changed = int(np.count_nonzero(np.any(pred != predictions[baseline_name], axis=1)))
        delta, ci = paired_delta_ci(fail, baseline_fail)
        print(
            f"{name:32s} {np.count_nonzero(fail):7,d}   {np.mean(fail):8.5%}"
            f"       {changed:8,d}          {delta * 100:+.5f} pp "
            f"[{ci[0] * 100:+.5f}, {ci[1] * 100:+.5f}]"
        )

    priority_vs_map = int(np.count_nonzero(
        predictions["probability-prioritized MWPM"] != predictions["exact DEM-MAP"]
    ))
    print(f"\nPriority vs exact DEM-MAP prediction disagreements: {priority_vs_map:,}/{SHOTS:,}")
    print(f"Exact DEM-MAP Bayes risk from complete state distribution: {bayes_risk:.6%}")
    print("\nInterpretation:")
    print("- Exact DEM-MAP is optimal for the detector/observable distribution defined by these independent DEM instructions, up to floating-point precision.")
    print("- The full instruction is treated as one event; '^' is not split into independent Bernoulli faults.")
    print("- This dense method is intended only for the 24-detector distance-3 model. Its state space grows exponentially and is not suitable for d=5 or d=7.")
    print("- If exact DEM-MAP materially improves on priority MWPM, this quantifies remaining decoder gap under the DEM. If they agree, the remaining rate may be inherent to the DEM or require preserving circuit details not captured by this DEM.")


if __name__ == "__main__":
    main()
