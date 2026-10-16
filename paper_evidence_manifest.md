# Curated QEC Paper-Evidence Manifest

- Generated UTC: `2026-10-16T22:50:14+00:00`
- Python: `3.13.15 (tags/v3.13.15:4061bc4, Aug  5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26300-SP0`
- Source HEAD before this refresh: `04161d885059e91ccf270c9cf0ec8efee9f6340b`
- Original base commit before release curation: `e77788a0e8384faeef6e379b9fd35772154dbe28`
- Working tree clean before this refresh: `yes` (ignored local files are excluded)
- Files hashed: **55**
- Missing candidate paths: **0**

## Software versions

| Package | Installed version |
|---|---|
| `stim` | `1.16.0` |
| `pymatching` | `2.4.0` |
| `numpy` | `2.5.3` |
| `scipy` | `1.18.1` |
| `ldpc` | `2.4.1` |

## Scope

This is a curated inventory of candidate sources and evidence for the paper. It intentionally excludes general exploratory files and timestamped backups.

The claim-family labels are organizational aids, not proof that a particular script generated a particular result. Verify each script-to-output connection against saved metadata, command logs, seeds, and the paper's reported values.

Each `size_bytes` and `sha256` value describes the exact bytes stored in the Git blob at manifest curation time, not necessarily the bytes in a local checkout. Git's `core.autocrlf` can change line endings in Windows working copies. To verify these values, compare them with the corresponding Git blob (for example, `git show HEAD:<path>` after the commit, or the index blob before the commit). These hashes do not establish scientific validity, independent replication, or prove that every working-tree file is represented by the commit.

The public evidence set excludes the obsolete top-level distance-9 benchmark script, the duplicate root-level distance-3 reproducer, and two multi-seed strength-sweep CSVs not discussed in the final report. These files are retained locally and can be published later with a documented source-to-result link if they become part of the paper.

Machine-specific absolute project paths in the included run metadata were replaced with `<LOCAL_PROJECT_ROOT>` for portability. Experiment settings, seeds, package details, and relative artifact names were preserved.

## Evidence families

- **distance3_exact_map:** 8 file(s)
- **distance3_order:** 11 file(s)
- **distance5_7_gates:** 2 file(s)
- **distance5_7_pair_information:** 7 file(s)
- **distance9_detector_weight_gate:** 5 file(s)
- **distance9_matching_cost_gate:** 7 file(s)
- **pair_strength_calibration:** 2 file(s)
- **paper_documents:** 6 file(s)
- **source_code:** 7 file(s)

## Candidate paths not found

- None.

## Zero-byte files requiring review

- None found in the selected evidence set.

## Required pre-publication checks

- Confirm the source script and exact output file for every quantitative claim.
- Verify the October 15 distance-9 benchmark hashes against the saved PowerShell SHA-256 output.
- Keep the detector-weight-gate timing result distinct from the matching-cost-gate validation.
- Keep screening data separate from fresh validation.
- Review any missing candidate paths or zero-byte files before publication.
- Regenerate this manifest only after the final paper and evidence files are frozen.

Full paths, byte sizes, timestamps and hashes are in `paper_evidence_manifest.csv`.
