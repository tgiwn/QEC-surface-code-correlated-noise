# Correlated Pauli Noise in Surface-Code Memory

A classical quantum-error-correction simulation project studying how local correlated `XX` data-qubit errors affect rotated surface-code Z-memory circuits, and how decoder representation and correlation-aware matching affect logical failure rates.

> This repository reports simulation findings for a chosen Pauli-noise model. It is not a hardware experiment, a threshold estimate, or a claim that the tested decoder is optimal for all circuit distributions.

## Research questions

- Does explicit vertical-neighbor `XX` pair noise increase logical failure relative to a matched-independent control?
- Does the decoder's detector-error-model representation affect the distance-3 result?
- How does correlation-aware MWPM compare with ordinary MWPM at distances 5 and 7?
- Can matching-cost gates reduce decoding time while preserving most of the accuracy benefit?
- Does tuning the decoder's assumed pair-event probability improve independent-seed performance?

## Headline results

- **DEM ordering matters at distance 3:** the correlated DEM audit found 12 singleton detector signatures with competing logical labels, while the tested control models at distances 3, 5, and 7 had no such conflicts. Reordering can alter predictions when PyMatching merges parallel edges with conflicting logical labels.
- **Pair information matters at distances 5 and 7:** removing all correlated-pair information increased failures in all four tested grouped-ablation conditions. Correlation-aware matching also improved on ordinary MWPM in the dedicated same-shot comparisons.
- **Exact distance-3 DEM-MAP is a benchmark:** the exhaustive Bayes risk for the DEM-defined distribution at `p_corr=0.0001` was 0.23669523%. This is exact for the represented DEM distribution, not automatically for every detail of the physical circuit.
- **Gating is a trade-off:** distance-5 and distance-7 detector-weight gates delivered about 1.4–1.7x speedups in their reported tests. In a separate standardized distance-9 benchmark, the detector-weight gate `weight >= 12` achieved a combined 1.188x speedup (15.8% less measured decoding time) across four conditions and 1.2 million paired shots. No gate-versus-full-CA logical-failure disagreements were observed, but the low failure counts do not establish statistical equivalence. This timing experiment is separate from the six-million-shot matching-cost-gate validation; those results and their accuracy trade-offs are reported independently.
- **DEM-strength calibration remains inconclusive:** scaling pair probabilities in the decoder model by 1.25× slightly lowered the pooled fresh-sample estimate, but the multiple-comparison-adjusted confidence interval included zero.

See [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md) for methods, tables, caveats, and supporting references. A compact index of headline results is in [`FINAL_RESULTS_SUMMARY.csv`](FINAL_RESULTS_SUMMARY.csv). The curated source-and-evidence inventory, including file hashes and provenance caveats, is documented in [`paper_evidence_manifest.md`](paper_evidence_manifest.md).

## Methods overview

- **Simulator:** [Stim](https://github.com/quantumlib/Stim)
- **Decoder:** [PyMatching](https://github.com/oscarhiggott/PyMatching)
- **Noise:** `P_BASE = 0.001`, plus vertical adjacent-data-qubit correlated `XX` events at selected `p_corr` values.
- **Control:** matched-independent noise with comparable single-qubit marginals.
- **Code distances:** experiments across `d = 3, 5, 7, 9`, with distance-specific analyses and sample counts.
- **Validation:** paired decoder comparisons, independent random seeds for selected candidates, and explicit separation of threshold-screening samples from holdout validation.

## Environment setup

The following direct-dependency pins are copied from the versions reported by the project's local release audit on 2026-10-11. They pin the main direct imports observed by the audit; they are not a full platform-specific lockfile. Record your exact Python version and a full `pip freeze` from the environment used for final results before making a release claim.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-bposd.txt` is an optional dependency profile for BP-OSD extensions. BP-OSD experiment scripts are not included in this curated snapshot, so this file is not required to run the scripts shipped here.

Record the exact interpreter and installed packages:

```powershell
python --version
python -m pip freeze > requirements-lock-full.txt
```

Review `requirements-lock-full.txt` for machine-specific packages and paths before committing it. The lock file should be generated from the actual validated environment, not inferred from this template.

## Suggested run order

The original project contains multiple exploratory and diagnostic scripts. Run the baseline and core comparisons first, then reproduce specialized decoder analyses. The exact script names and required arguments should be checked in each file's `if __name__ == '__main__'` block before executing.

1. Run the baseline circuit sanity check and save the generated circuit and baseline CSV.
2. Run correlated-versus-matched-independent comparisons at the selected distances and `p_corr` values.
3. Run the DEM parallel-edge/label-conflict audit and preserve both the audit CSV and log.
4. Run ordinary versus correlation-aware decoder comparisons and grouped pair-information ablations.
5. Run the distance-3 exact DEM-MAP and circuit-vs-DEM validation.
6. Run gate threshold screening, freeze thresholds, then run independent-seed gate validation.
7. Run the pair-strength calibration screening and its independent validation.
8. After freezing the included report and evidence set, verify that `paper_evidence_manifest.csv` matches the selected files and that every quantitative claim points to saved output and run provenance.

Do not use screening samples as the final validation set. Preserve the original seed values when reproducing reported outputs; if changing seeds, label the run as a new replication.

## Reproducibility

The repository should preserve:

- simulation scripts and their source hashes;
- generated `.stim` circuits and `.dem` representations where relevant;
- raw result CSVs and validation summaries;
- random seeds, shot counts, distances/rounds, physical-noise parameters, decoder options, and thresholds;
- Python and dependency versions;
- benchmark hardware and timing conditions.

The curated evidence inventory is provided in `paper_evidence_manifest.csv` and `paper_evidence_manifest.md`. It records hashes for selected release artifacts; it is not a complete inventory of every local experiment and does not by itself prove scientific validity or every source-to-output relationship.

## Known caveats

- The distance-3 correlated DEM has an identified parallel-edge logical-label ambiguity in the tested PyMatching construction path. Distance-3 results must explicitly state which ordering/graph-construction strategy was used.
- Correlation-aware MWPM is not an exact maximum-likelihood decoder for every physical noise distribution.
- `^` in a decomposed DEM does not mean that graphlike components should be treated as separate independent physical events.
- Exact MAP enumerates all detector patterns and is practical only at the small distance tested.
- A screened gate's risk budget is a selection criterion, not a guarantee on unseen samples.
- Statistical non-significance should not be described as proof of identical performance.

## References

- Stim: https://github.com/quantumlib/Stim
- PyMatching: https://github.com/oscarhiggott/PyMatching
- PyMatching releases and correlated-matching notes: https://github.com/oscarhiggott/PyMatching/releases

## License

The project's original code is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

This license applies only to materials for which the project contributors have the necessary rights to grant it. The research manuscript, figures, and research data are not automatically covered by Apache-2.0. These materials do not currently have a separate reuse license. Third-party software and external materials remain subject to their applicable licenses and terms.

See [LICENSE_DECISION.md](LICENSE_DECISION.md) for details about the licensing scope.
