# Distance-3 Exact DEM-MAP and Decoder-Order Evidence

**Status:** Evidence reconciliation completed; numerical claims should retain the experiment provenance below.
**Updated:** 2026-10-14

## 1. Common configuration

- Code: rotated surface-code Z-memory
- Distance: d = 3
- Rounds: 3
- Base noise: p_base = 0.001
- Correlated-pair strengths: p_corr = 0.0001 and 0.0002
- Decoder DEM: flattened Stim detector error model
- DEM dimensions: 24 detectors, 1 logical observable, 298 flattened mechanisms
- Exact DEM-MAP posterior and Bayes risk are computed from the represented DEM distribution.

## 2. Experiment A: decoder-order comparison

Source script: `compare_exact_map_distance3.py`
Raw data: `exact_map_distance3_comparison_raw.csv`
Shots: 500,000 physical-circuit samples per condition
Seeds: 20261080 for p_corr = 0.0001; 20261081 for p_corr = 0.0002

The source script samples detector syndromes and observables from the physical
circuit. It evaluates all decoder configurations on the same sampled shots
within each condition. The exact DEM-MAP prediction is calculated from the
flattened DEM posterior.

| Decoder configuration | p_corr = 0.0001 | p_corr = 0.0002 |
|---|---:|---:|
| Ordinary MWPM, original DEM order | 0.9220% | 1.0874% |
| Ordinary MWPM, canonical-sorted DEM | 0.2520% | 0.4478% |
| Probability-prioritized MWPM | 0.2520% | 0.4478% |
| Correlation-aware MWPM, original DEM order | 0.9898% | 1.1602% |
| Correlation-aware MWPM, canonical-sorted DEM | 0.2822% | 0.4762% |
| Exact DEM-MAP, empirical circuit rate | 0.2366% | 0.4282% |
| Exact DEM-MAP, theoretical DEM Bayes risk | 0.236695% | 0.411757% |

### Interpretation

The large differences between original-order and canonical-sorted decoders
support a distance-3 decoder-representation sensitivity finding. The physical
noise and sampled shots are held fixed within each condition; the DEM error
instruction ordering used to construct the decoder changes.

The Bayes risk is optimal for the probability distribution represented by
the flattened DEM. It is not automatically the optimum for the complete
physical-circuit distribution.

## 3. Experiment B: October 14 validation

Source script: `validate_exact_dem_map_against_sampler.py`
Run log: `validate_exact_dem_map_against_sampler_20261014_012203.log`
Metadata: `run_metadata_20261014_012203.txt`
Shots: 1,000,000 physical-circuit shots and 1,000,000 DEM shots
Circuit seed: 20261033
DEM seed: 20261034
Condition: d = 3, p_corr = 0.0001

| Decoder | Physical-circuit rate | DEM-sampled rate |
|---|---:|---:|
| Ordinary MWPM | 0.9226% | 0.9038% |
| Probability-prioritized MWPM | 0.2657% | 0.2508% |
| Exact DEM-MAP | 0.2463% | 0.2356% |

The exhaustive theoretical Bayes risk for this DEM is 0.23669523%.

For physical-circuit samples, the paired difference between probability-
prioritized MWPM and exact DEM-MAP is +0.019400 percentage points, with
reported 95% CI [+0.015520, +0.023280] percentage points.

For exact DEM-MAP, the circuit-minus-DEM sampled failure-rate difference is
+0.010700 percentage points, with reported 95% CI
[-0.002890, +0.024290] percentage points.

## 4. Reconciliation

The 0.2366% and 0.2463% exact DEM-MAP rates are empirical measurements from
two separate physical-circuit experiments with different seeds and sample
sizes. They should not be silently substituted for one another or pooled.

The 0.2520% and 0.2657% probability-prioritized MWPM measurements also come
from separate experiments. Before attributing their difference solely to
sampling variation, verify that the two source scripts construct exactly
the same decoder and weights.

The 0.23669523% theoretical Bayes risk is a model-derived quantity, not a
circuit-sampled failure rate. The 0.2356% rate is from the DEM sampler.

## 5. Reproducibility record

The archived 500,000-shot experiment is preserved under:
`archive_old_comparison_20261014_014348/`

That archive contains:
- `compare_exact_map_distance3.py`
- `exact_map_distance3_comparison_raw.csv`
- `provenance.txt`

The October 14 run metadata records Python 3.13.15, Stim 1.16.0,
PyMatching 2.4.0, NumPy 2.5.3, SciPy 1.18.1, and Git HEAD
`d09d6ee4877f360551ccc2c967ea85003ca16d03`.

The working tree contained untracked experimental scripts and outputs.
Consequently, the Git commit alone does not identify every file used in the
working tree; consult the saved source hashes and archived files as well.

## 6. Claims suitable for cautious use in the manuscript

1. At distance 3, ordinary and correlation-aware MWPM performance is
   sensitive to the DEM error-instruction ordering used to construct the
   matching decoder under the tested correlated-noise conditions.
2. Exact DEM-MAP performed better than probability-prioritized MWPM in the
   October 14 paired physical-circuit comparison.
3. The DEM-based Bayes risk and DEM-sampled MAP failure rate agree closely
   in the October 14 validation.
4. The observed circuit-versus-DEM agreement is not evidence that every
   aspect of the DEM distribution is identical to the physical circuit.

These claims apply to the tested model and parameter settings. They should
not be generalized to all distances or noise models without further evidence.
