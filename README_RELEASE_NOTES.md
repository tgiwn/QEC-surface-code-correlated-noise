# Before committing

1. This bundle contains release drafts. Verify every headline in the report against the original local CSVs/logs. The summary CSV is a convenience index, not a replacement for raw experiment outputs.
2. `requirements.txt` pins direct dependency versions reported in the local audit, but does not pin Python or every transitive/platform-specific package. Generate a full lock from the validated environment if desired.
3. The included `LICENSE_DECISION.md` intentionally records the conservative choice not to grant an explicit reuse license yet. Replace it with a real license only after deciding to grant reuse rights.
4. Preserve your existing source scripts and detailed results. This bundle does not contain all 89 local scripts or all 74 local CSVs.
5. After copying the files, rerun `check_qec_release_readiness.py`, regenerate the reproducibility manifest after final file changes, and review `git status`/`git diff` before staging.
