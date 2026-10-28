# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:09:15 2025

@author: pky0507
"""
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result calibration.")
    parser.add_argument("-i", "--input", default='./Cyst-X_bigdata_risk_assessment.csv', type=str, help="input path")
    parser.add_argument("-l", "--label", default="./IPMN_labels_total.xlsx", type=str, help="label path")
    parser.add_argument('-H', '--histology', action='store_true', help='only evaluate on histology‑confirmed')
    parser.add_argument('-N', '--no_norisk', action='store_true', help='only evaluate on low-vs-high risk')
    args = parser.parse_args()
    df = pd.read_csv(args.input, header=0)
    if args.no_norisk or args.histology:
        ref = pd.read_excel(args.label)
        # Select rows in ref where 'risk assessment' equals 0
        ref_zero = ref[ref['risk assessment'] == 0]
        # Collect IDs across T1 and T2 columns, dropping NaN entries
        id_zero = ref_zero['Patient ID T1'].dropna().tolist() + ref_zero['Patient ID T2'].dropna().tolist()
        id_zero = set([str(i).replace('.nii.gz', '') for i in id_zero])
        # Filter out those patients from df
        df = df[~df['Patient ID'].isin(id_zero)]
        if args.histology:
            ref = pd.read_excel(args.label)
            ref_follow_up = ref[ref['follow-up information'].str.contains('no change', na=False)]
            id_follow_up = ref_follow_up['Patient ID T1'].tolist() + ref_follow_up['Patient ID T2'].tolist()
            id_follow_up = set([i.replace('.nii.gz', '') for i in id_follow_up])
            df = df[~df['Patient ID'].isin(id_follow_up)]
    print(f"Number of patients: {len(df)}")
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
    tn, fp, fn, tp = confusion_matrix(y, r1).ravel()
    sens1 = tp / (tp + fn)
    spec1 = tn / (tn + fp)

    acc2 = accuracy_score(y, r2)
    tn, fp, fn, tp = confusion_matrix(y, r2).ravel()
    sens2 = tp / (tp + fn)
    spec2 = tn / (tn + fp)

    acc3 = accuracy_score(y, r3)
    tn, fp, fn, tp = confusion_matrix(y, r3).ravel()
    sens3 = tp / (tp + fn)
    spec3 = tn / (tn + fp)

    print(f"Radiologist 1 Acc: {acc1*100:.2f} Sens: {sens1*100:.2f} Spec: {spec1*100:.2f}")
    print(f"Radiologist 2 Acc: {acc2*100:.2f} Sens: {sens2*100:.2f} Spec: {spec2*100:.2f}")
    print(f"Radiologist 3 Acc: {acc3*100:.2f} Sens: {sens3*100:.2f} Spec: {spec3*100:.2f}")
    print(f"Radiologist Avg Acc: {(acc1+acc2+acc3)/3*100:.2f} Sens: {(sens1+sens2+sens3)/3*100:.2f} Spec: {(spec1+spec2+spec3)/3*100:.2f}")
