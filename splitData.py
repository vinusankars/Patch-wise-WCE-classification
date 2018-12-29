#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:52:14 2018

@author: vinusankars
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split as tts

data = np.array([[[[0]*3]*36]*36], dtype='float16')
target = np.array([[0,0]], dtype='float16')

for i in os.listdir('pickles'):
    print(i)
    with open('pickles/'+i, 'rb') as f:
        (x,y) = pickle.load(f)
        data = np.concatenate((data, x)) 
        target = np.concatenate((target, y))

xtr, xte, ytr, yte = tts(data, target, test_size=0.15)

with open('train_data.pckl', 'wb') as f:
    pickle.dump([xtr, ytr], f)
    
with open('test_data.pckl', 'wb') as f:
    pickle.dump([xte, yte], f)