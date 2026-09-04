# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:05:39 2024

@author: pky0507
"""

import os
import numpy as np
import pandas as pd
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity, RandAxisFlip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import calinski_harabasz_score
from seed import seed_everything
import argparse

def get_data_list(root='/dataset/IPMN_Classification/', t = 1, center = None):
    image_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    
    df = pd.read_excel(os.path.join(root, 'IPMN_labels_t'+str(t)+'_total.xlsx'), usecols=[0, 5])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = df_cleaned.iloc[:, 0].values
    labels = df_cleaned.iloc[:, 1:2].to_numpy(dtype=np.float32)//2 # we treat no/low-risk as 0 and high-risk as 1
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
                image_list.append(os.path.join(root, 't'+str(t), name+'.nii.gz'))
                label_list.append(labels[i])
                break
    return image_list, label_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPMN classification cross validation test.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    args = parser.parse_args()

    test_transforms = Compose([EnsureChannelFirst(), Resize((96, 96, 96))])

    for t in range(2):
        seed_everything(42)
        XX = []
        yy = []
        for i in range(7):
            image_list, label_list = get_data_list(root=args.data_path, t = t+1, center = i)
            dataset = ImageDataset(image_files=image_list, labels=label_list, transform=test_transforms)
            for j in tqdm(range(len(dataset))):
                X, y = dataset.__getitem__(j)
                XX.append(X.flatten().detach().numpy())
                yy.append(y.squeeze())
        XX = np.stack(XX, axis=0)
        yy = np.array(yy)
        X_pca = PCA(n_components=200).fit_transform(XX)
        X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(X_pca)

        plt.figure(figsize=(7, 7))
        plt.rcParams.update({'font.size': 16})
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=yy, palette='tab10', legend='full', s=60)
        # plt.title('t-SNE Visualization of T'+str(t+1)+' Modality')
        # plt.legend(labels=['No Risk', 'Low Risk', 'High Risk'])
        plt.legend(handles=[
            mpatches.Patch(color='C0', label='IPMN no/low-risk'),
            mpatches.Patch(color='C1', label='IPMN high-risk'),
            ], loc='upper left')
        plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(X_tsne, yy):.2f}", 
                transform=plt.gca().transAxes, 
                fontsize=20, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        plt.grid()
        # plt.show()
        plt.savefig("tsne"+str(t+1)+"_binary.pdf", format="pdf", bbox_inches='tight')
        print(f'T{t+1}W Calinski-Harabasz index: {calinski_harabasz_score(X_tsne, yy):.4f}')
        
