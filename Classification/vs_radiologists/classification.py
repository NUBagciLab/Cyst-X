# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:09:15 2025

@author: pky0507
"""
import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result calibration.")
    parser.add_argument("-i", "--input", default='internal/2-class-5-fold/centralized/saved/densenet121/t1', type=str, help="input path")
    parser.add_argument("-r", "--ref", default='./IPMN_labels_total.xlsx', type=str, help="reference input path")
    args = parser.parse_args()

    ref = pd.read_excel(args.ref)
    names = set([str(i).replace('.nii.gz', '') for i in ref['Patient ID T1'].tolist() + ref['Patient ID T2'].tolist()])
    
    df = pd.read_excel(os.path.join(args.input, 'result_calibrated.xlsx'))      
    filtered_df = df[df['ID'].isin(names)]
    print(f"Number of patients: {len(filtered_df)}")
    acc_c = accuracy_score(filtered_df['Label'], filtered_df['Prediction'])
    tn, fp, fn, tp = confusion_matrix(filtered_df['Label'], filtered_df['Prediction']).ravel()
    sens_c = tp / (tp + fn)
    spec_c = tn / (tn + fp)

    df = pd.read_excel(os.path.join(args.input, 'result.xlsx'))      
    filtered_df = df[df['ID'].isin(names)]
    print(f"Number of patients: {len(filtered_df)}")
    acc = accuracy_score(filtered_df['Label'], filtered_df['Prediction'])
    tn, fp, fn, tp = confusion_matrix(filtered_df['Label'], filtered_df['Prediction']).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print(f"Acc: {acc_c*100:.2f}({acc*100:.2f}) Sens: {sens_c*100:.2f}({sens*100:.2f}) Spec: {spec_c*100:.2f}({spec*100:.2f})")
    print(f"{sens_c*100:.2f}({sens*100:.2f})&{spec_c*100:.2f}({spec*100:.2f})")