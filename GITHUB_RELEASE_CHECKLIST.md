# GitHub Release Checklist — Correlated Pauli Noise in Surface-Code Memory

Use this checklist in the local repository before making the project public. A checked item should correspond to a file, command output, or explicit decision—not simply an intention.

## Research content

- [ ] Put the reviewed portfolio README at `README.md`; merge instead of blindly replacing an existing README.
- [ ] Put the report at `FINAL_RESEARCH_REPORT.md` and verify each headline result against the corresponding saved CSV or experiment log.
- [ ] Include `FINAL_RESULTS_SUMMARY.csv` as a concise index, while preserving the original detailed result CSVs separately.
- [ ] Label all figures/tables with code distance, physical model, `p_corr`, shot count, decoder settings, seed scheme, and whether data is screening or fresh validation.
- [ ] State the distance-3 PyMatching parallel-edge/logical-label issue next to all distance-3 claims.
- [ ] Describe the 1.25× decoder-DEM strength result as inconclusive; do not claim it improves the nominal 1.00× setting.
- [ ] Describe risk-budgeted gates as speed/accuracy trade-offs; screening budgets are not guarantees for fresh data.
- [ ] Include limitations: simulated noise model only, no hardware demonstration, no general threshold claim, and no claim of globally optimal decoding.

## Reproducibility

- [ ] Commit the experiment scripts that generate the reported results.
- [ ] Preserve source CSVs and independent-validation CSVs, or document a stable archive location if too large for Git.
- [ ] Preserve seed values, shots, distances/rounds, `P_BASE`, `p_corr`, DEM ordering conventions, and decoder flags.
- [ ] Record the exact Python and package versions from the environment used for reported numbers.
- [ ] Generate `final_reproducibility_manifest.md` after the final version of files is in place.
- [ ] Run `python check_qec_release_readiness.py` and resolve or explain all findings.
- [ ] Reproduce at least one small smoke-test run from a fresh clone/virtual environment, if practical.

## Repository hygiene

- [ ] Add a `.gitignore` that excludes virtual environments, caches, editor state, and temporary artifacts without excluding the final result CSVs.
- [ ] Add a dependency file; pin versions only after confirming the versions that actually produced the reported results.
- [ ] Inspect duplicate variants such as `_fixed.py` and `_freshcheck.py`; retain historical variants only if their purpose is explained.
- [ ] Search for hard-coded local paths, credentials, API keys, and private data.
- [ ] Choose a license deliberately, or consciously publish without granting an explicit reuse license.
- [ ] Run `git status --short`, `git diff --stat`, and review `git diff` before committing.
- [ ] Avoid committing large generated files accidentally; decide whether `.stim`/`.dem` files and raw data belong in Git or an external archive.

## Suggested final commands (PowerShell)

```powershell
# Inspect repository state
 git status --short --branch
 git diff --stat

# Run the static audit (does not execute simulation scripts)
 python .\check_qec_release_readiness.py

# Inspect generated audit
 Get-Content .\RELEASE_READINESS_AUDIT.md

# After reviewing all files, stage explicitly rather than using a blind add of everything
 git add README.md FINAL_RESEARCH_REPORT.md FINAL_RESULTS_SUMMARY.csv
 git status --short
```

Adjust the explicit `git add` command to match the files you have reviewed. Do not commit until every staged path is expected.
