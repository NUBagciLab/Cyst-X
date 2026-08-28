# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:09:15 2025

@author: pky0507
"""
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, average_precision_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result calibration.")
    parser.add_argument("-i", "--input", default='./Cyst-X_bigdata_risk_assessment.csv', type=str, help="input path")
    args = parser.parse_args()
    df = pd.read_csv(args.input, header=0)
    names = df['Patient ID'].tolist()
    r1 = df['reader1'].tolist()
    r2 = df['reader2'].tolist()
    r3 = df['reader3'].tolist()
    y = df['Ground_truth_Risk_Assessment'].tolist()
    r1 = [i//2 for i in r1]
    r2 = [i//2 for i in r2]
    r3 = [i//2 for i in r3]
    y = [i//2 for i in y]

    acc1 = accuracy_score(y, r1)
    sens1 = recall_score(y, r1)
    tn, fp, fn, tp = confusion_matrix(y, r1).ravel()
    spec1 = tn / (tn + fp)

    acc2 = accuracy_score(y, r2)
    sens2 = recall_score(y, r2)
    tn, fp, fn, tp = confusion_matrix(y, r2).ravel()
    spec2 = tn / (tn + fp)

    acc3 = accuracy_score(y, r3)
    sens3 = recall_score(y, r3)
    tn, fp, fn, tp = confusion_matrix(y, r3).ravel()
    spec3 = tn / (tn + fp)

    print(f"Radiologist 1 Acc: {acc1*100:.2f} Sens: {sens1*100:.2f} Spec: {spec1*100:.2f}")
    print(f"Radiologist 2 Acc: {acc2*100:.2f} Sens: {sens2*100:.2f} Spec: {spec2*100:.2f}")
    print(f"Radiologist 3 Acc: {acc3*100:.2f} Sens: {sens3*100:.2f} Spec: {spec3*100:.2f}")
    print(f"Radiologist Avg Acc: {(acc1+acc2+acc3)/3*100:.2f} Sens: {(sens1+sens2+sens3)/3*100:.2f} Spec: {(spec1+spec2+spec3)/3*100:.2f}")
