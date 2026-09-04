# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:09:15 2025

@author: pky0507
"""
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result calibration.")
    parser.add_argument("-i", "--input", default='../results_calibration/Internal 2 Classes Calibrated/3D Radiomics/t1.xlsx', type=str, help="input path")
    parser.add_argument("-r", "--radiology", default='./Cyst-X_bigdata_risk_assessment.csv', type=str, help="radiology input path")
    args = parser.parse_args()
    rad = pd.read_csv(args.radiology, header=0)

    names = rad['Patient ID'].tolist()
    df = pd.read_excel(args.input)
    filtered_df = df[df['ID'].isin(names)]
    
    acc = accuracy_score(filtered_df['Label'], filtered_df['Prediction'])
    sens = recall_score(filtered_df['Label'], filtered_df['Prediction'])
    tn, fp, fn, tp = confusion_matrix(filtered_df['Label'], filtered_df['Prediction']).ravel()
    spec = tn / (tn + fp)
    print(f"Acc: {acc*100:.2f} Sens: {sens*100:.2f} Spec: {spec*100:.2f}")