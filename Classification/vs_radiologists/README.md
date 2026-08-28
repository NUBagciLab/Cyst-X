# 🚀 Running the Pipeline
Please run `classification/results_calibration` first. `Cyst-X_bigdata_risk_assessment.csv` contains the diagnosis results by our three radiologists.

    chmod +x run.sh
    ./run.sh "../results_calibration"

If you want to compare the uncalibrated results (`classification threshold=0.5`) with radiologists, please run

    ./run.sh "../results"
