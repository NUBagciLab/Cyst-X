# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:22:29 2024

@author: pky0507
"""

import torch
import numpy as np

def fedavg(model, model_local, aggregation_weights = None):
    n_center = len(model_local)
    if aggregation_weights == None:
        aggregation_weights = np.ones(n_center)/n_center
    state_dict = [model_local[i].state_dict() for i in range(n_center)]
    for key in state_dict[0]:
        state_dict[0][key] = sum([state_dict[i][key]*aggregation_weights[i] for i in range(n_center)])/sum(aggregation_weights)
    model.load_state_dict(state_dict[0])
    for center in range(n_center):
        model_local[center].load_state_dict(state_dict[0])
        

def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())  
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])   
    return norm