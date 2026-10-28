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
    names = df_cleaned.iloc[:, 0].values
    labels = df_cleaned.iloc[:, 1:2].to_numpy(dtype=np.float32) // 2
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
    return image_list, label_list

def get_fold(image: list, label: list, root='/dataset/Cyst-X-resampled/classification', t=1, fold=0, n_splits=5):
    """
    Splits image and label lists into train and test sets based on the pre-saved CSV split file (t1_split.csv / t2_split.csv).
    """
    csv_path = os.path.join(root, f't{t}_split.csv')

    # Fallback to dynamic StratifiedKFold if the split CSV file does not exist
    if not os.path.exists(csv_path):
        print("Split file is not found. Using stratified split.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        skf.get_n_splits(image, label)
        for i, (train_index, test_index) in enumerate(skf.split(image, label)):
             if i == fold:
                train_image = [image[j] for j in train_index]
                train_label = [label[j] for j in train_index]
                test_image = [image[j] for j in test_index]
                test_label = [label[j] for j in test_index]
                return train_image, train_label, test_image, test_label
    
    df_split = pd.read_csv(csv_path)
    
    # Create mapping: ID -> Fold
    id_to_fold = dict(zip(df_split['ID'].astype(str), df_split['Fold']))
    
    train_image, train_label = [], []
    test_image, test_label = [], []
    
    for img_path, lbl in zip(image, label):
        img_id = os.path.basename(img_path).replace('.nii.gz', '')
        if img_id in id_to_fold:
            if id_to_fold[img_id] == fold:
                test_image.append(img_path)
                test_label.append(lbl)
            else:
                train_image.append(img_path)
                train_label.append(lbl)
                
    return train_image, train_label, test_image, test_label