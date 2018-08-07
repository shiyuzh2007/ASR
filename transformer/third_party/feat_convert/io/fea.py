#!/usr/bin/python
# coding=utf-8

"""
@version: 
@author: Dong Linhao
@license: Apache Licence 
@contact: donglinhao2015@ia.ac.cn
@site: 
@software: PyCharm Community Edition
@file: fea.py
@time: 05/15/17 17:25 PM
"""

import tensorflow as tf
import numpy as np
import sys

# np fea opt
def np_kaldi_fea_delt1(features):
    feats_padded = np.pad(features, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")

    row, col = np.shape(features)
    l2 = feats_padded[:row]
    l1 = feats_padded[1: row + 1]
    r1 = feats_padded[3: row + 3]
    r2 = feats_padded[4: row + 4]
    delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2

    return delt1

def np_kaldi_fea_delt2(features):
    feats_padded = np.pad(features, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")

    row, col = np.shape(features)
    l4 = feats_padded[:row]
    l3 = feats_padded[1: row + 1]
    l2 = feats_padded[2: row + 2]
    l1 = feats_padded[3: row + 3]
    c = feats_padded[4: row + 4]
    r1 = feats_padded[5: row + 5]
    r2 = feats_padded[6: row + 6]
    r3 = feats_padded[7: row + 7]
    r4 = feats_padded[8: row + 8]

    delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)
    return delt2

# def np_fea_delt(features):
#     row, col = np.shape(features)
#     l2 = np.pad(features, [[2, 0], [0, 0]], 'constant')[:row]
#     l1 = np.pad(features, [[1, 0], [0, 0]], 'constant')[:row]
#     r1 = np.pad(features, [[0, 1], [0, 0]], 'constant')[1:row + 1]
#     r2 = np.pad(features, [[0, 2], [0, 0]], 'constant')[2:row + 2]
#     delt = (r2 - l2) * 0.2 + (r1 - l1) * 0.1
#     return delt

def np_fea_add_delt(feature):
    fb = []
    fb.append(feature)
    delt1 = np_kaldi_fea_delt1(feature)
    # delt1 = np_fea_delt(feature)
    fb.append(delt1)
    # delt2 = np_fea_delt(delt1)
    delt2 = np_kaldi_fea_delt2(feature)
    fb.append(delt2)
    fb = np.concatenate(fb, 1)
    return fb

# tf fea opr
def tf_kaldi_fea_delt1(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l2 = tf.slice(feats_padded, [0, 0], shape)
    l1 = tf.slice(feats_padded, [1, 0], shape)
    r1 = tf.slice(feats_padded, [3, 0], shape)
    r2 = tf.slice(feats_padded, [4, 0], shape)

    delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2
    return delt1

def tf_kaldi_fea_delt2(features):
    feats_padded = tf.pad(features, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")
    feats_padded = tf.pad(feats_padded, [[1, 1], [0, 0]], "SYMMETRIC")

    shape = tf.shape(features)
    l4 = tf.slice(feats_padded, [0, 0], shape)
    l3 = tf.slice(feats_padded, [1, 0], shape)
    l2 = tf.slice(feats_padded, [2, 0], shape)
    l1 = tf.slice(feats_padded, [3, 0], shape)
    c  = tf.slice(feats_padded, [4, 0], shape)
    r1 = tf.slice(feats_padded, [5, 0], shape)
    r2 = tf.slice(feats_padded, [6, 0], shape)
    r3 = tf.slice(feats_padded, [7, 0], shape)
    r4 = tf.slice(feats_padded, [8, 0], shape)

    delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)
    return delt2

# def tf_fea_delt(features):
#     shape = tf.shape(features)
#     pp = tf.pad(features, [[2, 2], [0, 0]])
#     l2 = tf.slice(pp, [0, 0], shape)
#     l1 = tf.slice(pp, [1, 0], shape)
#     r1 = tf.slice(pp, [3, 0], shape)
#     r2 = tf.slice(pp, [4, 0], shape)
#     delt = (r2 - l2) * 0.2 + (r1 - l1) * 0.1
#     return delt

def tf_fea_add_delt(feature):
    fb = []
    fb.append(feature)
    ##  delt1 = tf_fea_delt(feature)
    delt1 = tf_kaldi_fea_delt1(feature)
    fb.append(delt1)
    ##  delt2 = tf_fea_delt(delt1)
    delt2 = tf_kaldi_fea_delt2(feature)
    fb.append(delt2)
    fea = tf.concat(axis=1, values=fb)
    return fea

def tf_fea_cmvn_global(feature, mean, var):
    fea = (feature - mean) / var
    return fea

def tf_fea_cmvn_utt(feature):
    fea_mean = tf.reduce_mean(feature, 0)
    fea_var = tf.reduce_mean(tf.square(feature), 0)
    fea_var = fea_var - fea_mean * fea_mean
    fea_ivar = tf.rsqrt(fea_var + 1E-12)
    fea = ( feature - fea_mean ) * fea_ivar
    return  fea

def tf_fea_splice(features, left_num, right_num):
    shape = tf.shape(features)
    splices = []
    pp = tf.pad(features, [[left_num, right_num], [0,0]])
    for i in range(left_num + right_num + 1):
        splices.append(tf.slice(pp, [i, 0], shape))
    splices = tf.concat(axis=1, values=splices)
    return splices

if __name__ == '__main__':
    x = tf.placeholder(tf.float32)  # 1-D tensor
    i = tf.placeholder(tf.float32)

    # y = splice_features(x,1,1)
    y = tf_fea_add_delt(x)
    # y = tf.slice(x, i, [1,1])
    # y = cmvn_features(x)

    # initialize
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # run
    result = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
    print(result)

    result = sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
    print(result)
