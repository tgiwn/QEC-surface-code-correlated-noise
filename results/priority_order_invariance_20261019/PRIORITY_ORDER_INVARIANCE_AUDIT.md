# Probability-prioritized heuristic: DEM input-order invariance audit

**Run date:** 2026-10-19 11:47 local time (UTC+08:00)
**Result:** completed, exit code 0
**Scope:** distance-three correlated-noise memory; two fixed seeded physical-circuit samples

## Configuration

- Python 3.13.15
- Stim 1.16.0
- PyMatching 2.4.0
- NumPy 2.5.3
- Distance and rounds: 3
- Background noise: `p_base=0.001`
- Shots per condition: 500,000
- `p_corr=0.0001`, seed `20261080`
- `p_corr=0.0002`, seed `20261081`

The audit used the production `build_probability_prioritized_matching` helper, constructing the graph from the original DEM, a DEM with error-instruction order reversed, and a canonical-sorted DEM. The same sampled physical-circuit syndromes and observables were passed to each decoder. Ordinary MWPM loaded from the canonical-sorted DEM was also evaluated.

## Observed results

| `p_corr` | Heuristic failures from original input | Reversed input | Sorted input | Ordinary MWPM from sorted DEM | Heuristic order disagreements | Heuristic vs. ordinary-sorted disagreements |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 1,260 / 500,000 (0.2520%) | 1,260 / 500,000 (0.2520%) | 1,260 / 500,000 (0.2520%) | 1,260 / 500,000 (0.2520%) | 0 | 0 |
| 0.0002 | 2,239 / 500,000 (0.4478%) | 2,239 / 500,000 (0.4478%) | 2,239 / 500,000 (0.4478%) | 2,239 / 500,000 (0.4478%) | 0 | 0 |

For both conditions, the canonical-sorted DEM text was stable when the original input order was reversed. No probability-prioritized decoder predictions changed among the original, reversed, and sorted DEM builds. The heuristic built from the original DEM matched ordinary matching loaded from the canonical-sorted DEM on all sampled syndromes.

## Interpretation and limits

This supports the claim that the probability-prioritized construction is input-order invariant in this implementation and these tested conditions. It also clarifies that the heuristic is not an independent observed accuracy improvement over ordinary MWPM on the canonical-sorted DEM for these samples. The result must not be expanded to a statement that the two methods are identical for every possible syndrome or noise condition: the audit sampled 500,000 syndromes at each of two probabilities and was run under PyMatching 2.4.0 only. Cross-version replication of the heuristic itself remains open even though the separate ordinary DEM-loading order effect was tested under PyMatching 2.3.1 and 2.4.0.

The printed paired differences and zero-width intervals follow from zero discordant outcomes on the tested samples; they are not evidence that the population difference is mathematically zero. For a single zero-disagreement comparison among 500,000 independently sampled shots, the one-sided exact 95% binomial upper bound on the disagreement probability is approximately 5.99e-6 (0.000599%), per condition and per comparison. This is not a simultaneous family-wise bound. No raw CSV was written by the diagnostic.

## Provenance

- Script: `audit_probability_prioritized_order_invariance.py`
- User-reported local output: `results\\priority_order_invariance_20261019_114736.log`
- User-reported local log SHA-256: `264C49C0EBB7B940F0A081D28E79DFD45D06448AA3B7C305082DBF03672FB7C1`. The raw log itself is not included in this public payload and was not independently inspected here.
- The user's pasted console output is the presently available evidence for the result; do not describe the raw local log as independently inspected until its hash/completeness is checked.

Bundled audit-script SHA-256: `EA5E912C613EA8853AB6495A2A839101F01931F3493E2FF77C695B8E472E9E05`. This matches the SHA-256 reported for the user's local script.
