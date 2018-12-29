#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:22:29 2018

@author: vinusankars
"""

import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import roc_auc_score as ras
import pickle
import cv2 as cv

w1,w2,w3 = 0,0,0

def f(img):
    cv.imshow('img',img)
    cv.waitKey()
    cv.destroyAllWindows()
    
with tf.Session() as sess:        
    saver = tf.train.import_meta_graph('Model/cnn.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./Model'))
    graph = tf.get_default_graph()
    y_conv = graph.get_tensor_by_name('dense2/add:0')
    x = graph.get_tensor_by_name('x:0')
    w1 = sess.run(graph.get_tensor_by_name('conv1/w1:0'))
    w2 = sess.run(graph.get_tensor_by_name('conv2/w2:0'))
    w3 = sess.run(graph.get_tensor_by_name('conv3/w3:0'))
    prob = graph.get_tensor_by_name('dense2/Placeholder:0')
    acc = graph.get_tensor_by_name('acc/Mean:0')
    y_ = graph.get_tensor_by_name('y_:0')
    y_conv = graph.get_tensor_by_name('dense2/add:0')
    pickles = sorted(os.listdir('pickles/'))
    print(prob)

#    for i in pickles[-2:]:
#        print('*'*5,i.split('.')[0],'*'*5)
#        with open('pickles/'+i,'rb') as f:
#            (xte, yte) = pickle.load(f)
#            
#            if i!='train_data.pckl':
#	            feed_dict = {x: xte, prob: 1.0, y_:yte}
#	            print('Accuracy:',sess.run(acc, feed_dict)*100, '%')
#
#	            true_labels = sess.run(tf.argmax(y_,1), feed_dict)
#	            pred_labels = sess.run(tf.argmax(y_conv,1), feed_dict)
#	            rass = ras(true_labels, pred_labels)
#	            print('AUC(ROC):', rass*100, '%')
#
#	            TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
#	            TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
#	            FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
#	            FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))	            
#	            print('SN:', TP/(TP+FN)*100, '%')
#	            print('SP:', TN/(TN+FP)*100, '%')
#	            print('')
#	            
#            else:
#                continue