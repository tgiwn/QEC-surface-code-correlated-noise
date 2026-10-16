# Distance-3 Decoder-Order Sensitivity

Experiment timestamp: 20261014_015625
Distance: 3
Rounds: 3
Base noise: 0.001
Correlation strengths: 0.0001 and 0.0002
Stim / PyMatching / NumPy: 1.16.0 / 2.4.0 / 2.5.3

## Deterministic synthetic test

Reversing two competing-label mechanisms changed the predicted logical
observable from 0 to 1. The merged edge's fault_ids changed from an empty
set to {0}.

## Actual correlated DEM

The original and canonical-sorted DEM constructions changed predictions
for 38 of 300 tested syndromes of weight one or two (12.6667%). The same
count was reported for both tested correlation strengths and both decoder
modes.

These are deterministic low-weight-syndrome tests, not estimates of the
overall logical failure rate. The physical-circuit sampling experiments
are archived separately.

## Files

- summary_compact.csv: paper-facing summary without verbose edge dumps
- changed_counts.csv: changed-syndrome row counts by condition
- synthetic_probe_20261014_015625.csv: minimal competing-label example
- summary_20261014_015625.csv: original detailed output
- syndrome_differences_20261014_015625.csv: individual syndrome prediction changes
- metadata_20261014_015625.txt: run metadata
- audit_minimal_d3_order_reproducer.py: reproducer source

The recorded result values were copied without modification. Text files use LF line endings for cross-platform checksum verification.
