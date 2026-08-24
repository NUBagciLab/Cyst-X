# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:12:02 2024

@author: pky0507
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

label_map = {
    'NYU': 0,
    'MCF': 1,
    'NWU': 2,
    'AHN': 3,
    'MCA': 4,
    'IU':  5,
    'EMC': 6
}

mapping_func = np.vectorize(lambda x: label_map.get(x, -1))

for T in ['1', '2']:
    df = pd.read_excel('MRQy_UMAP.xlsx', sheet_name='T'+T+'_projection')  
    print(set(df.iloc[1:, 0].to_numpy()))
    x_min = df.iloc[1:, [2, 4, 6]].to_numpy().min()
    x_max = df.iloc[1:, [2, 4, 6]].to_numpy().max()
    y_min = df.iloc[1:, [3, 5, 7]].to_numpy().min()
    y_max = df.iloc[1:, [3, 5, 7]].to_numpy().max()
    print(x_min, x_max, y_min, y_max)
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    for i in [2, 3, 4, 5 ,6, 7, 8, 9, 10]:
        if i in set(df.iloc[1:, 1].to_numpy()):
            ind = df.index[df.iloc[:, 1] == i].tolist()
            x = df.iloc[ind, 2].to_numpy()
            y = df.iloc[ind, 3].to_numpy()
            plt.scatter(x, y, label=str(i), color='C'+str(i-2))
    
    x = df.iloc[1:, [2, 3]].to_numpy(dtype='float64')
    y = df.iloc[1:, 1].to_numpy(dtype='int64')
    y = y - min(y)
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(x, y):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # plt.legend(loc='best')
    plt.grid()
    plt.axis([-0.4, 11.4, -0.3, 13.1])
    plt.savefig('zminmax'+T+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    for i in [2, 3, 4, 5 ,6, 7, 8, 9, 10]:
        if i in set(df.iloc[1:, 1].to_numpy()):
            ind = df.index[df.iloc[:, 1] == i].tolist()
            x = df.iloc[ind, 4].to_numpy()
            y = df.iloc[ind, 5].to_numpy()
            plt.scatter(x, y, label=str(i), color='C'+str(i-2))
            
    x = df.iloc[1:, [4, 5]].to_numpy(dtype='float64')
    y = df.iloc[1:, 1].to_numpy(dtype='int64')
    y = y - min(y)
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(x, y):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # plt.legend(loc='best')
    plt.grid()
    plt.axis([-0.4, 11.4, -0.3, 13.1])
    plt.savefig('zwhitening'+T+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    for i in [2, 3, 4, 5 ,6, 7, 8, 9, 10]:
        if i in set(df.iloc[1:, 1].to_numpy()):
            ind = df.index[df.iloc[:, 1] == i].tolist()
            x = df.iloc[ind, 6].to_numpy()
            y = df.iloc[ind, 7].to_numpy()
            plt.scatter(x, y, label=str(i), color='C'+str(i-2))
            
    x = df.iloc[1:, [6, 7]].to_numpy(dtype='float64')
    y = df.iloc[1:, 1].to_numpy(dtype='int64')
    y = y - min(y)
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(x, y):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.axis([-0.4, 11.4, -0.3, 13.1])
    plt.savefig('zzscore'+T+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    df = pd.read_excel('MRQy_UMAP.xlsx', sheet_name='T'+T+'_projection')
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    center = 0
    for i in ['NYU', 'MCF', 'NWU', 'AHN', 'MCA', 'IU', 'EMC']:
        ind = df.index[df.iloc[:, 0] == i].tolist()
        x = df.iloc[ind, 2].to_numpy()
        y = df.iloc[ind, 3].to_numpy()
        plt.scatter(x, y, label=i.replace('NWU', 'NU'), color='C'+str(center))
        center+= 1
        
    x = df.iloc[1:, [2, 3]].to_numpy(dtype='float64')
    y = mapping_func(df.iloc[1:, 0].to_numpy())
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(x, y):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # plt.legend(loc='best')
    plt.grid()
    plt.axis([-0.4, 11.4, -0.3, 13.1])
    plt.savefig('cminmax'+T+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    center = 0
    for i in ['NYU', 'MCF', 'NWU', 'AHN', 'MCA', 'IU', 'EMC']:
        ind = df.index[df.iloc[:, 0] == i].tolist()
        x = df.iloc[ind, 4].to_numpy()
        y = df.iloc[ind, 5].to_numpy()
        plt.scatter(x, y, label=i.replace('NWU', 'NU'), color='C'+str(center))
        center+= 1

    x = df.iloc[1:, [4, 5]].to_numpy(dtype='float64')
    y = mapping_func(df.iloc[1:, 0].to_numpy())
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(x, y):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # plt.legend(loc='best')
    plt.grid()
    plt.axis([-0.4, 11.4, -0.3, 13.1])
    plt.savefig('cwhitening'+T+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    center = 0
    for i in ['NYU', 'MCF', 'NWU', 'AHN', 'MCA', 'IU', 'EMC']:
        ind = df.index[df.iloc[:, 0] == i].tolist()
        x = df.iloc[ind, 6].to_numpy()
        y = df.iloc[ind, 7].to_numpy()
        plt.scatter(x, y, label=i.replace('NWU', 'NU'), color='C'+str(center))
        center+= 1

    x = df.iloc[1:, [6, 7]].to_numpy(dtype='float64')
    y = mapping_func(df.iloc[1:, 0].to_numpy())
    plt.text(0.96, 0.96, f"CHI: {calinski_harabasz_score(x, y):.2f}", 
             transform=plt.gca().transAxes, 
             fontsize=20, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.axis([-0.4, 11.4, -0.3, 13.1])
    plt.savefig('czscore'+T+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()