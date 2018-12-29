#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 02:50:37 2018

@author: vinusankars
"""

import tensorflow as tf
import pickle
import os
from sklearn.metrics import roc_auc_score as ras

print('Reading pickle...')

#for reading augmented data
with open('train_data.pckl', 'rb') as f:
    (xtr, ytr) = pickle.load(f)
    
with open('test_data.pckl', 'rb') as f:
    (xte, yte) = pickle.load(f)
    
print('Making NN...')

#functions for creating NN   
def wt(shape, name, i=0.3):
    initial = tf.truncated_normal(shape, stddev=i)
    return tf.Variable(initial, name=name)

def bias(shape, name, i=0.3):
#    initial = tf.constant(i, shape=shape)
#    return tf.Variable(initial, name=name)
    initial = tf.truncated_normal(shape, stddev=i)
    return tf.Variable(initial, name=name)

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding ='VALID')
    
def conv1D(x, W):
    return tf.nn.conv1d(x, W, stride=5, padding='SAME')

# TF graph
graph = tf.Graph()

#defining NN layers
with graph.as_default():     
    x = tf.placeholder(tf.float32, shape=[None, 1296*3])
    y_ = tf.placeholder(tf.float32, shape=[None, 2], name='y_')
    
    x = tf.reshape(x, [-1, 36, 36, 3], name='x')
    
    with tf.name_scope('conv1'):
        w1 = wt([5, 5, 3, 3], 'w1')
        b1 = bias([3], 'b1')
        c1 = tf.nn.relu(conv(x, w1)+b1)
    
    with tf.name_scope('conv2'):
        w2 = wt([5, 5, 3, 3], 'w2')
        b2 = bias([3], 'b2')
        c2 = tf.nn.relu(conv(c1, w2)+b2)
    
    with tf.name_scope('pool1'):
        p1 = tf.nn.lrn(maxpool(c2))
    
    with tf.name_scope('conv3'):
        w3 = wt([5, 5, 3, 3], 'w3')
        b3 = bias([3], 'b3')
        c3 = tf.nn.lrn(tf.nn.relu(conv(p1, w3)+b3))
    
    with tf.name_scope('pool2'):
        p2 = maxpool(c3)
    
    with tf.name_scope('dense1'):
        fw1 = wt([5*5*3, 10], 'fw1')
        fb1 = bias([10], 'fb1')
        flat = tf.reshape(p2, [-1, 5*5*3])
        fc1 = tf.nn.relu(tf.matmul(flat, fw1)+fb1)    
    
    with tf.name_scope('dense2'):
        fb2 = bias([2], 'fb2')
        fw2 = wt([10, 2], 'fw2')
        prob = tf.placeholder_with_default(1.0, None)
        fc1_drop = tf.nn.dropout(fc1, prob)
        y_conv = tf.matmul(fc1_drop, fw2)+fb2
    
    with tf.name_scope('cost'):
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy**2)
        
    with tf.name_scope('acc'):
        predn = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        acc = tf.reduce_mean(tf.cast(predn, tf.float32))

print('Starting session...')
plot = []
#Start training
with tf.Session(graph=graph) as sess:
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(os.getcwd()+"/summary1")
        
    for itr in range(1):
        print(itr)
        for i in range(50):        
            train_step.run(feed_dict={
                        x: xtr[i*64: (i+1)*64],
                        y_: ytr[i*64: (i+1)*64],
                        prob: 0.9})    
        
            if i%50 == 0 and i>0:
                print(i, sess.run(cross_entropy, feed_dict={prob:1.0, x:xtr[i-100:i] ,y_:ytr[i-100:i]}))
        
        feed_dict={x: xte, y_: yte, prob: 1.0}
        ac = sess.run(acc, feed_dict=feed_dict)
        print("Acc:", ac)
        '''print('Test Acc: %g' % acc.eval())
        print('Train Acc: %g' % acc.eval(feed_dict={
                        x: xtr,
                        y_: ytr,
                        prob: 1.0}))'''
        
    #    y_true = sess.run(tf.argmax(y_,1), feed_dict={y_:yte, prob:1.0})
    #    y_score = sess.run(tf.argmax(y_conv,1), feed_dict={x:xte, prob:1.0})
        a = sess.run(y_, feed_dict={y_:yte, prob:1.0})
        b = sess.run(y_conv, feed_dict={x:xte, prob:1.0})
        rass = ras(a,b)
        print('AUC(ROC):', rass)
        plot.append([ac, ras])
#        ka.append(sess.run(acc, feed_dict={x:xtr, y_:ytr, prob:1.0}))
#        kb.append(sess.run(acc, feed_dict={x:xte, y_:yte, prob:1.0}))
#        kc.append(rass)
    writer.add_graph(sess.graph)
    saver.save(sess, 'Model1/cnn')
        
    
print('Done.')
