#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 17:44:28 2018

@author: vinusankars
"""

import cv2 as cv
import os
import pickle
import numpy as np

print('Initiated.')

def rot(img, ang):
    M = cv.getRotationMatrix2D((180,180), ang, 1)
    return cv.warpAffine(img, M, (360,360))   

data = np.array([[[[0]*3]*36]*36], dtype='float16')
target = np.array([[0,0]], dtype='float16')
loc = os.getcwd()+'/dataSet/'

print('Accessing files...')
print('Processing...')

w = 0
b = 0

for i in os.listdir(loc):
    print('*',i)
    if i!= 'Normal':
        data1 = np.array([[[[0]*3]*36]*36], dtype='float16')
        target1 = np.array([[0,0]], dtype='float16')
        for j in os.listdir(loc+i+'/data/'):
            print(j)

            img = cv.imread(loc+i+'/data/'+j)
            img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
            imm = cv.imread(loc+i+'/annotations/'+j.split('.')[0]+'m.png')
            break
#                for r in range(0, 360, 22):
#                    img1 = rot(img, r)
#                    imm1 = rot(imm, r)
#                    for x in range(0,360,36):
#                        for y in range(0,360,36):
#                            count = np.reshape(imm1[x: x+36, y: y+36],(-3,3)).tolist().count([255,255,255])
#                            if count >= 36*18:                                
#                                X = cv.normalize(img1[x: x+36, y: y+36],
#                                         img1[x: x+36, y: y+36],
#                                         alpha=0,
#                                         beta=1,
#                                         norm_type=cv.NORM_MINMAX,
#                                         dtype=cv.CV_32F)
#                                
#                                data1 = np.concatenate((data1, [X,X[::-1,:,:],X[:,::-1,:]])) 
#                                target1 = np.concatenate((target1, [[0,1],[0,1],[0,1]]))
#                                w += 1
#                                
#                            if count < 36*18 and r==0:                                
#                                X = cv.normalize(img1[x: x+36, y: y+36],
#                                         img1[x: x+36, y: y+36],
#                                         alpha=0,
#                                         beta=1,
#                                         norm_type=cv.NORM_MINMAX,
#                                         dtype=cv.CV_32F)
#                                
#                                data1 = np.concatenate((data1, [X,X[::-1,:,:],X[:,::-1,:]])) 
#                                target1 = np.concatenate((target1, [[1,0],[1,0],[1,0]]))
#                                b += 1
#            except:
#                continue
#
#        print('Pickling',i,'...',len(data1),w,b)
#        with open(i+'.pckl', 'wb') as f:
#            pickle.dump([data1[1:], target1[1:]], f)
#            
##            data = np.concatenate((data, data1)) 
##            target = np.concatenate((target, target1))
#            
#    else:
#        for j in os.listdir(loc+i+'/data/'):
#            print(j)
#            try:
#                img = cv.imread(loc+i+'/data/'+j)
#                img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
#                for x in range(0,360,36):
#                    for y in range(0,360,36):
#                        b += 1
#                        
#                        X = cv.normalize(img[x: x+36, y: y+36],
#                                 img[x: x+36, y: y+36],
#                                 alpha=0,
#                                 beta=1,
#                                 norm_type=cv.NORM_MINMAX,
#                                 dtype=cv.CV_32F)
#                        
#                        data = np.concatenate((data, [X,X[::-1,:,:], X[:,::-1,:]])) 
#                        target = np.concatenate((target, [[1,0],[1,0],[1,0]]))
#            except:
#                continue
#            with open('Normal.pckl', 'wb') as f:
#                pickle.dump([data[1:], target[1:]], f)
#            
#print('Pickling...')
#
