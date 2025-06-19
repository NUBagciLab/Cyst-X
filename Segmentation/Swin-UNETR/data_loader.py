# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:38 2024

@author: pky0507
"""

import os
import numpy as np
from sklearn.model_selection import KFold

def get_data_list(root='/dataset/IPMN_images_masks/', t = 1, center = None):
    image_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    if center == None:
        center = np.arange(len(center_names))
    elif isinstance(center, int):
        center = [center]
    for c in center:
        files = os.listdir(os.path.join(root, 't'+str(t), 'images'))
        for f in files:
            for n in center_names[c]:
                if n in f:
                    image_list.append(os.path.join(root, 't'+str(t), 'images', f))
                    label_list.append(os.path.join(root, 't'+str(t), 'masks', f))
                    break
    return image_list, label_list

def get_fold(image:list, label:list, n_splits = 5, fold = 0):
    kf = KFold(n_splits=n_splits, shuffle=False)
    kf.get_n_splits(image, label)
    for i, (train_index, test_index) in enumerate(kf.split(image, label)):
        if i == fold:
            train_image = [image[j] for j in train_index]
            train_label = [label[j] for j in train_index]
            test_image = [image[j] for j in test_index]
            test_label = [label[j] for j in test_index]
            return train_image, train_label, test_image, test_label