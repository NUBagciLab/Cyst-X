# Zhixiang's supplementary analyses for Cyst-X

Drop this `additional_analyses/` directory into the project. It is an independent
post-processing utility: it reads saved held-out predictions and never changes
training code, thresholds, prediction workbooks, or manuscript source files.

## Scope and table mapping

| Output | Stable manuscript label | Content |
|---|---|---|
| `reader_ci.tex` | `tab:zw-reader-ci` | Internal T2W DenseNet and arithmetic reader mean sensitivity/specificity, conditional 95% CIs |
| `paired_differences.tex` | `tab:zw-paired-differences` | Paired model-minus-reader-mean differences and 95% CIs |
| `frozen_histopathology.tex` | `tab:zw-frozen-pathology` | 357 matched patients; sensitivity, specificity, pooled AUC and AUC CIs |
| `matched_auc.tex` | `tab:zw-matched-auc` | Broad-task matched internal/external pooled and unweighted center-mean AUCs |

In the 2026-10-27 online layout these correspond to A1–A4 (previously A12–A15).
Use the labels when merging: table numbers change with document ordering.
These additions are tables, not new figures; no unrelated figure-generation or
threshold-audit code is included. The task-definition table subsequently changed
to an itemized list is manuscript prose, not a computed result.

## Installation and execution

Python 3.10 or later; no GPU or model retraining is required. Dependencies below
are pinned to the environment used for verification.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r additional_analyses/requirements.txt
python additional_analyses/run_analysis.py \
  --classification-root /path/to/classification \
  --labels /path/to/IPMN_labels_total.xlsx \
  --readers /path/to/Cyst-X_bigdata_risk_assessment.csv \
  --output-dir ./results/zhixiang_1000
python additional_analyses/verify_results.py ./results/zhixiang_1000
```

Default: **1,000 replicates**, seed **20261024**. Optional flags `--bootstrap`
and `--seed` create alternative analyses; the reference verification is intended
for the defaults and the original prediction snapshot. Use an empty output
folder: the program refuses to overwrite an existing populated folder.

Provide the actual classification directory containing `internal/` and
`external/`. Hongyi's email contained multiple similarly spelled project roots;
paths are explicit arguments to avoid silently choosing a different version.

## Required input layout

- Reference workbook columns: `Patient ID T1`, `Patient ID T2`,
  `risk assessment`, `follow-up information`.
- Reader CSV columns: `Patient ID`, `Ground_truth_Risk_Assessment`,
  `reader1`, `reader2`, `reader3`.
- Predictions: `ID`, `Label`, `Prediction`, `Probability`, and `Fold`
  (or `Held-out Center`). Binary high-risk is 1 in predictions; original risk
  category 2 is converted to 1 in the reader/reference files.
- `internal/{2-class,low_vs_high,histology}/centralized/saved/densenet121/t2/`:
  `result_calibrated.xlsx` and `result.xlsx` for reader comparison.
- `{internal,external}/{2-class,low_vs_high,histology}/centralized/saved/densenet121/{t1,t2}/result_calibrated.xlsx`
  for the 357-patient histopathology analysis.
- `{internal,external}/2-class/{method}/saved/{model}/{t1,t2}/result.xlsx`
  for matched aggregation; method/model pairs are `centralized/densenet121`,
  `fedavg/densenet121`, `fedprox/densenet121_0.1`, `fedprox/densenet121_0.3`.

Validation predictions are **not required** for these specific analyses: saved
validation-selected decisions are reused, with no threshold fitting on test data.
The program follows Hongyi's 2026-10-24 `compare_results.py` cohort alignment:
paired T1W/T2W IDs; 629 broad, 474 IPMN-only, 357 histopathology cases, each with
142 high-risk patients. Histopathology membership follows that original
workbook's exclusion of follow-up entries containing `no change`; this is not
an independent pathology adjudication. `cohort_io.py` preserves those rules.

## Statistical details and limits

- Sensitivity/specificity bootstrap: resample patients within the relevant true
  risk class, jointly for the model and all three readers. Recompute the
  arithmetic reader mean in every replicate. Sensitivity uses seed+1 and
  specificity uses seed, reset per cohort, matching the accepted calculation.
- Paired differences use the **same** bootstrap draws as their component metrics;
  they are not obtained by subtracting independently estimated CI endpoints.
- AUC bootstrap: stratified patient resampling implemented as multinomial counts
  over tied-score groups (the same sampling distribution). Ties receive half
  credit. The seed is reset for each analysis, matching the accepted calculation.
- Bounds are the 2.5th and 97.5th percentiles, with NumPy's default linear
  interpolation. Metrics/intervals are percentages; differences are percentage
  points. CSVs retain full precision; LaTeX rounds to two decimals.
- Intervals condition on saved fitted models, thresholds, sampled centers and
  readers. They do not account for model retraining, threshold reselection,
  overlapping training folds, new centers or new readers. Comparisons are
  exploratory, not multiplicity-adjusted confirmatory tests.
- Broad and IPMN-only models are evaluated on the same 357-patient subset without
  refitting. Histopathology-only rows use previously trained separate models.
- Pooled AUC is computed from pooled held-out probabilities. Center means are
  unweighted within-center AUC means, **not** fold means. Center assignment uses
  the original identifier-prefix mapping. Internal/external IDs and labels are
  checked before comparing them. The broad matched cohorts are 652 T1W / 655 T2W.

## Outputs and merging

Five aggregate CSVs, four standalone table fragments, `all_tables.tex`,
`preview.tex`, and `run_manifest.json` are generated. The manifest records input
filenames, SHA-256 hashes, package versions, seed and replicate count; it contains
no patient-level values. `principal_ci.csv` and `paired_comparison.csv` also
retain individual-reader aggregates for traceability; the rendered tables use
the reader mean. No p-values are added.

`reference_results/` contains only aggregate accepted results, not raw patient
predictions or reader assessments. `verify_results.py` verifies numerical
agreement (absolute tolerance 1e-10), complete ordered columns, and the default
bootstrap count and seed, not just agreement after rounding. Changed
source predictions should cause a verification failure and require review, not
replacement of the reference values merely to make the check pass.

The LaTeX fragments use `booktabs`, `xcolor`, `ZhixiangPink`, and `\zwrev`.
Use existing manuscript definitions. `preview.tex` provides standalone definitions
(bright pink #FF1493) for a separate visual check. Merge individual table contents
into the current manuscript rather than replacing the full manuscript. This
package does not edit Overleaf and does not automatically replace any author text.

## Provenance

Cohort alignment adapted from Hongyi's supplied `compare_results.py` snapshot
(downloaded 2026-10-24). Computations consolidated from Zhixiang's analysis scripts;
bootstrap results correspond to the 2026-10-26 recalculations and 2026-10-27 AUC
web synchronization. The original training/evaluation project remains the source
of prediction files. No patient-level data, model weights, or manuscript backups
are distributed in this code package.
