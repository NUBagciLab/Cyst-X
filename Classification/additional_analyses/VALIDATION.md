# Verification — 2026-10-27

Full analysis executed against the local 2026-10-24 Hongyi prediction snapshot.
All numeric columns matched accepted reference results within absolute tolerance 1e-10:

- principal_ci: 30 rows
- paired_comparison: 24 rows
- frozen_pathology: 12 rows
- frozen_pathology_auc_ci: 12 rows
- matched_auc (broad task only): 16 rows

All runs used 1,000 bootstrap replicates and seed 20261024.
Individual-reader CSV rows are retained; rendered reader tables show model and reader mean.
Input version hashes and dependency versions are in example_output/run_manifest.json.
No validation set threshold search, retraining, or Overleaf editing occurs in this code.

Standalone preview compiled successfully with two LaTeX passes: four pages, no undefined references or overfull-box warnings.
