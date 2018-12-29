#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:54:22 2018

@author: vinusankars
"""

import tensorflow as tf
import cv2 as cv
import time
import numpy as np

with tf.Session() as sess:        
    saver = tf.train.import_meta_graph('Model/cnn.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./Model'))
    graph = tf.get_default_graph()
    y_conv = graph.get_tensor_by_name('dense2/add:0')
    x = graph.get_tensor_by_name('x:0')
    w1 = sess.run(graph.get_tensor_by_name('conv1/w1:0'))
    prob = graph.get_tensor_by_name('dense2/Placeholder:0')
    acc = graph.get_tensor_by_name('acc/Mean:0')
    y_ = graph.get_tensor_by_name('y_:0')
    
    t1 = time.time()
    img = cv.imread('dataSet/Bleeding/data/bleeding1.png')
    img1 = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    #imm = cv.imread()
    k = 0
    
    imm = cv.imread('dataSet/Bleeding/annotations/bleeding1m.png')
    feed_dict = {x: np.array([img[144:180,108:144]], dtype='float32'), prob: 1.0}
    Y = sess.run(y_conv, feed_dict=feed_dict)
    for x1 in range(360):
        for y in range(360):
            if list(imm[x1][y])==[255,255,255]:
                if list(imm[x1-1][y])==[0,0,0] or list(imm[x1+1][y])==[0,0,0] or list(imm[x1][y-1])==[0,0,0] or list(imm[x1][y+1])==[0,0,0]:
                    img = cv.rectangle(img, (y,x1), (y+1,x1+1), (255,255,255), 1)
#    for i in range(0,360,36):
#        for j in range(0,360,36):
#            patch = cv.normalize(img1[i: i+36, j: j+36], img1[i: i+36, j: j+36],          
#                                     alpha=0,
#                                     beta=1,
#                                     norm_type=cv.NORM_MINMAX,
#                                     dtype=cv.CV_32F)
#            feed_dict = {x: np.array([patch], dtype='float32'), prob: 1.0}
#            Y = sess.run(tf.argmax(y_conv,1), feed_dict=feed_dict)
#            if int(Y) == 1:
#                k += 1
#                print(i,j)
#                img = cv.rectangle(img, (i,j), (i+36,j+36), (0,0,255), 2)
#                
            
    
    t2 = time.time()
    print(k, t2-t1)
    
    cv.imshow('', img)
    cv.waitKey(0)
    cv.destroyAllWindows()            