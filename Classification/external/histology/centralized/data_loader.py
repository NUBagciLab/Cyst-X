# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:38 2024

@author: pky0507
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_data_list(root='/dataset/Cyst-X-resampled/classification', t = 1, center = None):
    image_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    
    df = pd.read_excel(os.path.join(root, 'IPMN_labels_t'+str(t)+'_total.xlsx'), usecols=[0, 5])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    df_cleaned = df_cleaned[df_cleaned.iloc[:, 1].isin([1, 2])]  # Exclude no-risk cases.
    names = df_cleaned.iloc[:, 0].values
    labels = df_cleaned.iloc[:, 1:2].to_numpy(dtype=np.float32) - 1  # Low=0, high=1.
    if center == None:
        center = np.arange(len(center_names))
    elif isinstance(center, int):
        center = [center]
    center_name = []
    for i in center:
        center_name += center_names[i]
    for i in range(len(names)):
        name = names[i].replace('.nii.gz', '')
        for c in center_name:
            if c in name:
                image_list.append(os.path.join(root, 't'+str(t), 'images', name+'.nii.gz'))
                label_list.append(labels[i])
                break
    return keep_histology(root, image_list, label_list)

def keep_histology(root, image, label):
    """Preserve the existing cohort rule: exclude 'no change' follow-up cases."""
    ref = pd.read_excel(os.path.join(root, 'IPMN_labels_total.xlsx'))
    ref_follow_up = ref[ref['follow-up information'].str.contains('no change', na=False)]
    ids = pd.concat([ref_follow_up['Patient ID T1'], ref_follow_up['Patient ID T2']]).dropna()
    id_follow_up = {os.path.basename(str(name)).replace('.nii.gz', '').replace('.nii', '') for name in ids}
    image_hist = []
    label_hist = []
    for i, l in zip(image, label):
        if os.path.basename(i).replace('.nii.gz', '').replace('.nii', '') not in id_follow_up:
            image_hist.append(i)
            label_hist.append(l)
    return image_hist, label_hist
