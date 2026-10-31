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

# Verification — 2026-10-31 (second-round revision analyses)

`run_round2.sh` was executed end to end on the training server against the saved
released prediction workbooks (`result.xlsx`, `val_result.xlsx`), in a clean virtual
environment built from `requirements.txt` (Python 3.9, numpy 2.0.2, pandas 2.3.3,
scikit-learn 1.6.1, matplotlib 3.9.4). `verify_round2.py` reported all checks passed:

- `federated_auc_ci.csv`, `center_adjusted_or.csv`, `calibration_dca.csv`: all numeric
  columns match `reference_results/round2/` within 1e-6.
- The four center-adjusted odds ratios quoted in Section 2.2.1 and the Limitations
  (IPMN-only, internal and external, T1W and T2W) are reproduced exactly as printed.
- Section 2.4 paired bootstrap CIs, the Supplementary Table A19/A20 rows
  (`example_output/round2/*_rows.tex`) and the Figure 4 legend values (fold-mean AUC
  and AP) are identical to the manuscript.
- Outputs were byte-identical between the server's default environment
  (numpy 1.26.4) and the pinned environment.

All bootstrap calculations use 1,000 replicates and seed 20261031. Only aggregate
results are stored; no patient-level values are included.
