# 🚀 Running the Pipeline

This folder compares the models with radiologists on 629 patients with both T1W and T2W scans.

Please run `classification/results_calibration` first. `Cyst-X_bigdata_risk_assessment.csv` contains the diagnosis results from our three radiologists.

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
