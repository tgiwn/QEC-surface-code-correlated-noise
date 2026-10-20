# Ordinary MWPM randomized DEM input-order audit: cross-version report

**Status:** completed local diagnostic; not yet committed to the public repository or incorporated into the manuscript.

## Objective

Test whether the observed sensitivity of ordinary minimum-weight perfect matching (MWPM) to detector error model (DEM) error-instruction loading order persists across PyMatching 2.3.1 and 2.4.0, while holding the sampled physical-circuit data and randomized-order seeds fixed.

## Configuration

- Python 3.13.15
- Stim 1.16.0
- NumPy 2.5.3
- PyMatching versions: 2.4.0 and 2.3.1, run in separate environments
- Rotated surface-code memory: distance 3, rounds 3
- Background noise: `p_base = 0.001`
- Physical-circuit shots per condition: 500,000
- Correlated-pair probabilities: `p_corr = 0.0001` and `0.0002`
- Physical sample seeds: 20261080 and 20261081
- Random DEM instruction orders per condition: 50
- Random-order seed rule: `202610200 + condition_index*10000 + permutation_index` (zero-indexed)
- DEM dimensions: 24 detectors, 1 observable, 298 error instructions
- Source Git HEAD recorded in both JSON files: `2cbc307eb1bb655d3619715eb3a33c9d948df873`
- Audit script SHA-256: `bb61c7406e6fee4ece8a255040970a417fb15415e0efa5792abb5859346a256a`

The order policy shuffles complete flattened DEM error-instruction lines among the existing error-line positions, leaving non-error lines unchanged. Each order is decoded on the same sampled syndromes within a condition. The 50 randomized graph constructions are therefore not 50 independent physical-noise resamples.

## Observed failure rates across randomized orders

| `p_corr` | Canonical-sorted DEM | Original DEM | Reversed DEM | Random-order failures: min / median / max | Random-order rate: min / median / max | Prediction disagreements vs canonical: min / median / max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 1,260 / 500,000 (0.2520%) | 4,610 / 500,000 (0.9220%) | 1,260 / 500,000 (0.2520%) | 1,260 / 2,117 / 7,601 | 0.2520% / 0.4234% / 1.5202% | 0 / 977 / 7,031 |
| 0.0002 | 2,239 / 500,000 (0.4478%) | 5,437 / 500,000 (1.0874%) | 2,239 / 500,000 (0.4478%) | 2,239 / 3,037 / 8,229 | 0.4478% / 0.6074% / 1.6458% | 0 / 1,050 / 7,530 |

For both conditions, the canonical-sorted construction was the lowest-failure configuration among the 50 randomized orders and the original/reversed reference orders evaluated. This is an observed finite-set result, not a claim that sorting is universally optimal or physically correct.

## Cross-version comparison

The 2.4.0 and 2.3.1 runs produced CSV files with the same size (14,018 bytes) and the same SHA-256:

`a7ff0d4a9aaa5f21ca588bff16b29ec3854034f7914faa790cf419842f3e0090`

Each CSV contains 106 summary rows: three reference orders plus 50 random orders for each of the two `p_corr` conditions. Both JSON metadata files independently record the same script hash, Stim version, Python version, NumPy version, Git HEAD, physical-sample seeds, and random-order seed rule; the recorded PyMatching versions differ as intended (2.4.0 vs 2.3.1). The CSV hash in each JSON matches the actual CSV, and both runs use the same recorded source commit.

This is exact reproducibility of the **saved summary table** across these two versions. The CSV stores aggregate failure and prediction-disagreement counts, not every individual decoded prediction vector. Therefore, identical CSV bytes do not by themselves prove shot-by-shot prediction identity between PyMatching versions.

## Interpretation and limits

The randomized-order audit provides evidence that, for this distance-3 DEM and the specified PyMatching loading workflow, logical failure rates and disagreements with the canonical-sorted reference can vary substantially with the order in which error instructions are presented. The same complete summary table was reproduced under PyMatching 2.3.1 and 2.4.0.

The supported claim should remain narrow:

- The evidence concerns this Stim-generated distance-3 correlated-noise memory, the specified DEM decomposition/loading procedure, and the tested parameters.
- The canonical-sorted order is a deterministic reference and happened to perform best among the tested orders; this does not establish universal optimality or exact likelihood/MAP decoding.
- The 50 orderings are graph-construction perturbations evaluated against shared physical samples, not independent physical-noise replicates. Their distribution describes sensitivity to the sampled instruction permutations, not an independent estimate of physical logical-error-rate uncertainty.
- Cross-version exact equality is established for the persisted summary CSV, not for all per-shot decoder predictions, because per-shot prediction vectors were not persisted.
- The test does not by itself establish a new general PyMatching bug; claims about the merge behavior should be framed relative to the relevant documented API semantics and this concrete workflow.

## Provenance fingerprints

### Shared random-order audit script

- SHA-256: `bb61c7406e6fee4ece8a255040970a417fb15415e0efa5792abb5859346a256a`

### PyMatching 2.4.0 run

- CSV: `ordinary_mwpm_random_order_20261019T191214Z.csv`
- CSV SHA-256: `a7ff0d4a9aaa5f21ca588bff16b29ec3854034f7914faa790cf419842f3e0090`
- JSON SHA-256: `ac6392288bfc58290b893a14ad2bfb0ae2cc7a677716004b40139874ddd7e540`
- Log SHA-256: `c1b69f67a2c16ec5dfff23944940ed9360401fd9371d70b09f22f651d7634bbe`

### PyMatching 2.3.1 run

- CSV: `ordinary_mwpm_random_order_20261019T191626Z.csv`
- CSV SHA-256: `a7ff0d4a9aaa5f21ca588bff16b29ec3854034f7914faa790cf419842f3e0090`
- JSON SHA-256: `8674b372e475f2d085712a2a06818061d96e786c83b2340c5c1eedb304bbf5d2`
- Log SHA-256: `4ff82e84b6269e3d22ac6c50f40544b23a559ded0666913b857d798d5ec830b0`

Hashes above were reported from the user's local PowerShell output. Retain the raw files with this report in the repository so those values can be independently recomputed after checkout.
