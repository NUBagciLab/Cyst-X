# Radiologist comparison

Run with the `pt` environment activated, from the project root:

```bash
bash Classification/vs_radiologists/run.sh
```

The launcher also works from this folder using `bash ./run.sh`.
It reads existing predictions and does not retrain or recalibrate models.

`compare_results.py` compares the three readers with radiomics, centralized
DenseNet121 (T1 and T2), and all four DenseNet121 fusion methods, for internal
and external evaluation. Each task uses its own trained results under
`Classification/{internal,external}/{2-class,low_vs_high,histology}`.

All methods are restricted to the same reader patients: 629 paired cases,
474 IPMN-only cases, and 357 histology cases. Cohorts follow the reference
risk labels; the histology subset additionally excludes stable follow-up
entries containing `no change`, matching the original reader analysis.
T1/T2 aliases are mapped to one patient. The script rejects missing cases,
duplicates, label mismatches, inconsistent probabilities/folds, and
uncalibrated predictions that do not use a 0.50 threshold.

Sensitivity and specificity are pooled over the selected patients, rather
than averaged over folds or centers. The table shows saved calibrated
predictions outside parentheses and the 0.50-threshold results inside.
Reader averages are arithmetic means of the three reader metrics.

Outputs under `results/tables/`:

- `radiologist_comparison_updated.tex`: completed table.
- `radiologist_comparison_metrics.csv`: metrics, confusion counts, and model source paths.
- `radiologist_comparison_cohorts.json`: patient membership for each cohort.

The supplied table layout is retained in `table_template.tex`. Optional
arguments include `--project-root`, `--output-dir`, and `--template`.
