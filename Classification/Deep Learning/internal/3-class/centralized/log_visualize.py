# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:37:28 2024

@author: pky0507
"""
import numpy as np
import json
import matplotlib.pyplot as plt

with open('./saved/log.json', 'r') as json_file:
    data = json.load(json_file)
train_loss = data['train_loss']
test_loss = data['test_loss']

x = np.arange(1, len(train_loss)+1)    
plt.plot(x, train_loss, label='Train')
plt.plot(x, test_loss, label='Test')
plt.xlim(0, len(train_loss))
plt.grid()
plt.legend()
plt.show()