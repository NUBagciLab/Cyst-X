# 🚀 Running the Pipeline

This folder compares the models with radiologists on 629 patients with both T1W and T2W scans. `Cyst-X_bigdata_risk_assessment.csv` contains the diagnosis results from our three radiologists.

Please run `classification/results_calibration` first. Make sure that both the whole cohort and the histology-confirmed cohort are calibrated (make sure that you have run `analysis_internal.sh`, `analysis_external.sh`, and `analysis_fusion.sh`.

Then, please exclude the following:

    bash ./run.sh

If you want to compare the uncalibrated results (`classification threshold=0.5`) with the radiologists' results, please run

    bash ./run_uncalibrated.sh
