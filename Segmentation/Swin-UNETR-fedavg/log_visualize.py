# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:37:28 2024

@author: pky0507
"""
import numpy as np
import json
import matplotlib.pyplot as plt

with open('./saved/t1/fold0/log.json', 'r') as json_file:
    data = json.load(json_file)
train_loss = data['train_loss'][-1]
test_dice = data['test_dice'][-1]

x = np.arange(1, len(train_loss)+1)    
plt.plot(x, train_loss, label='Train')
plt.xlim(0, len(train_loss))
plt.grid()
plt.legend()
plt.show()

x = np.arange(1, len(test_dice)+1)  
x = [10*i for i in x]  
plt.plot(x, test_dice, label='Test')
plt.xlim(0, len(test_dice)*10)
plt.grid()
plt.legend()
plt.show()