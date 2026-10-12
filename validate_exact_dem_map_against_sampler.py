"""Validate exact DEM-MAP against independent circuit and DEM samples.

Run in the same project directory as correlated_experiment.py and
exact_dem_map_decoder.py. This stays at distance 3 (24 detectors), where the
exact dense posterior is tractable.

The DEM sampler checks the convolution/decoder against the model's own sample
distribution. The circuit sampler checks whether that DEM model matches the
circuit's detector/observable behavior at the level of several diagnostics.
"""

import math

import numpy as np
import pymatching

import stim
from correlated_experiment import (
    add_noise_layer,
    get_data_layout,
    make_base_circuit,
    make_correlated_pairs,
)
from exact_dem_map_decoder import (
    P_BASE,
    P_CORR,
    ROUNDS,
    DISTANCE,
    as_2d,
    build_probability_prioritized_matching,
    encode_syndromes,
    exact_joint_distribution,
    fail_mask,
    paired_delta_ci,
)

SHOTS = 1_000_000
CHUNK_SIZE = 25_000
SEED_CIRCUIT = 20261033
SEED_DEM = 20261034


def wilson_ci(successes, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def pct(x):
    return 100 * float(x)


def sample_circuit(circuit, shots, seed, chunk_size):
    sampler = circuit.compile_detector_sampler(seed=seed)
    det_chunks = []
    obs_chunks = []
    remaining = shots
    while remaining:
        take = min(chunk_size, remaining)
        det, obs = sampler.sample(shots=take, separate_observables=True)
        det_chunks.append(np.asarray(det, dtype=np.uint8))
        obs_chunks.append(np.asarray(obs, dtype=np.uint8))
        remaining -= take
    return np.concatenate(det_chunks, axis=0), np.concatenate(obs_chunks, axis=0)


def sample_dem(dem, shots, seed, chunk_size):
    sampler = dem.compile_sampler(seed=seed)
    det_chunks = []
    obs_chunks = []
    remaining = shots
    while remaining:
        take = min(chunk_size, remaining)
        result = sampler.sample(shots=take)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("Unexpected return value from stim.CompiledDemSampler.sample")
        det, obs = result[0], result[1]
        det_chunks.append(np.asarray(det, dtype=np.uint8))
        obs_chunks.append(np.asarray(obs, dtype=np.uint8))
        remaining -= take
    return np.concatenate(det_chunks, axis=0), np.concatenate(obs_chunks, axis=0)


def rate_line(name, failures, shots):
    count = int(np.count_nonzero(failures))
    rate = count / shots
    lo, hi = wilson_ci(count, shots)
    return f"{name:30s} {count:7,d}/{shots:,}  {pct(rate):.6f}%  Wilson95=[{pct(lo):.6f}%, {pct(hi):.6f}%]"


def decode_dataset(name, det, obs, ordinary, priority, q0, q1):
    shots = det.shape[0]
    syndrome_indices = encode_syndromes(det)
    p0 = q0[syndrome_indices]
    p1 = q1[syndrome_indices]
    map_pred = (p1 > p0).astype(np.uint8)[:, None]
    actual = as_2d(obs)

    ordinary_pred = as_2d(ordinary.decode_batch(det))
    priority_pred = as_2d(priority.decode_batch(det))
    failure = {
        "ordinary MWPM": fail_mask(ordinary_pred, actual),
        "priority MWPM": fail_mask(priority_pred, actual),
        "exact DEM-MAP": fail_mask(map_pred, actual),
    }

    print(f"\n--- {name}: {shots:,} shots ---")
    print(f"Observable flip frequency P(L0=1): {pct(np.mean(actual[:, 0])):.6f}%")
    weights = np.count_nonzero(det, axis=1)
    weight_bins = [weights == 0, weights == 1, weights == 2, weights >= 3]
    print("Detector syndrome weight distribution:")
    for label, mask in zip(("0", "1", "2", "3+"), weight_bins):
        print(f"  weight {label:>2s}: {np.count_nonzero(mask):7,d} ({pct(np.mean(mask)):.4f}%)")

    for decoder_name, fail in failure.items():
        print(rate_line(decoder_name, fail, shots))

    delta, ci = paired_delta_ci(failure["priority MWPM"], failure["exact DEM-MAP"])
    print(
        "priority minus exact MAP (paired): "
        f"{pct(delta):+.6f} pp; 95% CI [{pct(ci[0]):+.6f}, {pct(ci[1]):+.6f}] pp"
    )
    disagreements = int(np.count_nonzero(priority_pred != map_pred))
    print(f"Priority vs exact-MAP prediction disagreements: {disagreements:,}/{shots:,}")

    metrics = {
        "det": det,
        "obs": actual,
        "weights": weights,
        "fail": failure,
    }
    return metrics


def unpaired_rate_difference_ci(a_count, a_n, b_count, b_n):
    pa = a_count / a_n
    pb = b_count / b_n
    delta = pa - pb
    se = math.sqrt(pa * (1 - pa) / a_n + pb * (1 - pb) / b_n)
    return pct(delta), (pct(delta - 1.96 * se), pct(delta + 1.96 * se))


def main():
    if DISTANCE != 3 or ROUNDS != 3:
        raise ValueError("This validation is designed for the current distance-3 exact MAP model.")

    print("Validate exact DEM-MAP against independent DEM and circuit samplers")
    print(f"distance={DISTANCE}; rounds={ROUNDS}; P_BASE={P_BASE}; P_CORR={P_CORR}")
    print(f"shots per source={SHOTS:,}; chunk={CHUNK_SIZE:,}; circuit seed={SEED_CIRCUIT}; DEM seed={SEED_DEM}")
    print("The DEM sampler tests the exact convolution against the DEM's own distribution;")
    print("the circuit sampler tests whether that distribution matches the generated circuit.\n")

    base = make_base_circuit(distance=DISTANCE, rounds=ROUNDS, p=P_BASE)
    data_by_xy, data_qubits = get_data_layout(base)
    pairs = make_correlated_pairs(data_by_xy)
    circuit = add_noise_layer(base, data_qubits, pairs, P_CORR, "correlated")
    dem = circuit.detector_error_model(decompose_errors=True).flattened()

    print(f"DEM dimensions: detectors={dem.num_detectors}; observables={dem.num_observables}; error instructions={len(dem)}")
    q0, q1, total, bayes_risk = exact_joint_distribution(dem)
    del total
    print(f"Exact DEM-MAP Bayes risk: {pct(bayes_risk):.6f}%\n")

    ordinary = pymatching.Matching(dem)
    priority = build_probability_prioritized_matching(dem)

    circuit_det, circuit_obs = sample_circuit(circuit, SHOTS, SEED_CIRCUIT, CHUNK_SIZE)
    dem_det, dem_obs = sample_dem(dem, SHOTS, SEED_DEM, CHUNK_SIZE)

    circuit_metrics = decode_dataset("PHYSICAL CIRCUIT SAMPLER", circuit_det, circuit_obs, ordinary, priority, q0, q1)
    dem_metrics = decode_dataset("DEM SAMPLER", dem_det, dem_obs, ordinary, priority, q0, q1)

    print("\n========== CIRCUIT vs DEM DISTRIBUTION CHECK ==========")
    for decoder_name in ("ordinary MWPM", "priority MWPM", "exact DEM-MAP"):
        c_fail = int(np.count_nonzero(circuit_metrics["fail"][decoder_name]))
        d_fail = int(np.count_nonzero(dem_metrics["fail"][decoder_name]))
        delta, ci = unpaired_rate_difference_ci(c_fail, SHOTS, d_fail, SHOTS)
        print(
            f"{decoder_name:24s} circuit-minus-DEM = {delta:+.6f} pp "
            f"95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}] pp"
        )

    c_obs_rate = float(np.mean(circuit_metrics["obs"][:, 0]))
    d_obs_rate = float(np.mean(dem_metrics["obs"][:, 0]))
    obs_delta = pct(c_obs_rate - d_obs_rate)
    obs_se = 100 * math.sqrt(c_obs_rate * (1 - c_obs_rate) / SHOTS + d_obs_rate * (1 - d_obs_rate) / SHOTS)
    print(f"P(L0=1) circuit-minus-DEM = {obs_delta:+.6f} pp; approximate 95% CI [{obs_delta - 1.96 * obs_se:+.6f}, {obs_delta + 1.96 * obs_se:+.6f}] pp")

    c_det_rate = np.mean(circuit_det, axis=0)
    d_det_rate = np.mean(dem_det, axis=0)
    se_det = np.sqrt(
        c_det_rate * (1 - c_det_rate) / SHOTS
        + d_det_rate * (1 - d_det_rate) / SHOTS
    )
    z = np.divide(np.abs(c_det_rate - d_det_rate), se_det, out=np.zeros_like(se_det), where=se_det > 0)
    max_i = int(np.argmax(z))
    abs_pp = np.abs(c_det_rate - d_det_rate) * 100
    print(f"Detector marginal comparison: mean absolute difference={float(np.mean(abs_pp)):.6f} pp; max absolute difference={float(np.max(abs_pp)):.6f} pp (D{max_i}, |z|={float(z[max_i]):.2f})")

    c_weights = circuit_metrics["weights"]
    d_weights = dem_metrics["weights"]
    print(f"Mean detector weight: circuit={float(np.mean(c_weights)):.6f}; DEM={float(np.mean(d_weights)):.6f}")
    print("\nInterpretation:")
    print("- Exact DEM-MAP is optimal for the full independent-mechanism distribution represented by this DEM.")
    print("- Agreement between DEM-sampled exact-MAP risk and the computed Bayes risk validates the convolution/sampling implementation.")
    print("- Agreement between circuit and DEM diagnostics supports, but does not prove in every detail, DEM fidelity to the circuit distribution.")
    print("- The `^` separators are decomposition suggestions; the DEM sampler retains each complete error mechanism as one event.")


if __name__ == "__main__":
    main()
