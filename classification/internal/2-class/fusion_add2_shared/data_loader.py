# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:38 2024

@author: pky0507
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_data_list(root='/dataset/IPMN_Classification/', center = None):
    image1_list = []
    image2_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    
    df = pd.read_excel(os.path.join(root, 'IPMN_labels_total.xlsx'), usecols=[0, 1, 6])
    df_cleaned = df.dropna(subset=[df.columns[2]]) # remove NaN
    names1 = df_cleaned.iloc[:, 0].values
    names2 = df_cleaned.iloc[:, 1].values
    labels = df_cleaned.iloc[:, 2:3].to_numpy(dtype=np.float32)//2 # we treat no/low-risk as 0 and high-risk as 1
    if center == None:
        center = np.arange(len(center_names))
    elif isinstance(center, int):
        center = [center]
    center_name = []
    for i in center:
        center_name += center_names[i]
    for i in range(len(names1)):
        name1 = names1[i].replace('.nii.gz', '')
        name2 = names2[i].replace('.nii.gz', '')
        for c in center_name:
            if c in name1:
                image1_list.append(os.path.join(root, 't1', name1+'.nii.gz'))
                image2_list.append(os.path.join(root, 't2', name2+'.nii.gz'))
                label_list.append(labels[i])
                break
    return image1_list, image2_list, label_list

def get_fold(image1:list, image2:list, label:list, n_splits = 4, fold = 0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    skf.get_n_splits(image1, label)
    for i, (train_index, test_index) in enumerate(skf.split(image1, label)):
        if i == fold:
            train_image1 = [image1[j] for j in train_index]
            train_image2 = [image2[j] for j in train_index]
            train_label = [label[j] for j in train_index]
            test_image1 = [image1[j] for j in test_index]
            test_image2 = [image2[j] for j in test_index]
            test_label = [label[j] for j in test_index]
            return train_image1, train_image2, train_label, test_image1, test_image2, test_label