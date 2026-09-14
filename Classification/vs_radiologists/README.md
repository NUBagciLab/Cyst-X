# 🚀 Running the Pipeline

This folder compares the models with radiologists on 629 patients with both T1W and T2W scans. `Cyst-X_bigdata_risk_assessment.csv` contains the diagnosis results from our three radiologists.

Please run `classification/results_calibration` first. Make sure that both the whole cohort and the histology-confirmed cohort are calibrated (make sure that you have run `analysis_internal.sh`, `analysis_external.sh`, `analysis_internal_histology.sh`, and  `analysis_external_histoloy.sh`, or have run `analysis_internal_all.sh` and `analysis_external_all.sh`.

Then, please exclude the following:

    chmod +x run.sh
    ./run.sh

If you want to compare the uncalibrated results (`classification threshold=0.5`) with the radiologists' results, please run

    chmod +x run_uncalibrated.sh
    ./run_uncalibrated.sh

If you want to compare on the 512 histology-confirmed cases, please run 
    
    chmod +x run_histology.sh
    ./run_histology.sh

for calibrated results and 

    chmod +x run_histology_uncalibrated.sh
    ./run_histology_uncalibrated.sh

for uncalibrated results.
