#!/usr/bin/python
# coding=utf-8

"""
@version: 
@author: zhulei
@license: Apache Licence 
@contact: zhulei@rokid.com
@site: 
@software: PyCharm Community Edition
@file: mod.py
@time: 1/6/17 11:42 AM
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variables
import struct

class Mod(object):

    def __init__(self, sess):
        self.sess = sess

    #flag : variable 0 ; str 1
    def savepy(self, f, py, flag):
        if flag == 0:
            _shape = py.shape
            f.write(struct.pack('i', len(_shape)))
            for _dim in _shape:
                f.write(struct.pack('i', _dim))
            f.write(py)
        elif flag == 1:
            _len = len(py)
            f.write(struct.pack("i%ds"%_len , _len, py))
        else:
            raise Exception("Invalid Format!")

    def savetf(self, f, op):
        if isinstance(op, (list, tuple)):
            for variable in op:
                self.savetf(f, variable)
        elif isinstance(op, tf.Variable):
            py = self.sess.run(op)
            self.savepy(f, py, 0)
        else:
            self.savepy(f, op, 1)

    def saveto(self, tfmod, cfg_path):

        with open(cfg_path, "wb") as cfg_file:
            self.savetf(cfg_file, tfmod)

    def loadpy(self, f, flag):
        if flag == 0 :
            _shape = []
            _size = 1
            _shape_dim = struct.unpack('i', f.read(4))[0]
            for _ in range(_shape_dim):
                _dim = struct.unpack('i', f.read(4))[0]
                _shape.append(_dim)
                _size = _size * _dim
            _py = np.frombuffer(f.read(_size * 4), dtype=np.float32)
            if len(_shape) > 0:
                _py = np.reshape(_py, _shape)
            return  _py
        elif flag == 1:
            _len = struct.unpack('i', f.read(4))[0]
            _py = struct.unpack('%ds'%_len, f.read(_len))[0]
            return _py
        else:
            raise Exception("Invalid Format!")

    def loadtf(self, f, op):
        if isinstance(op, (list, tuple)):
            for variable in op:
                self.loadtf(f, variable)
        elif isinstance(op, tf.Variable):
            py = self.loadpy(f, 0)
            self.sess.run(op.assign(py))
        else:
            py = self.loadpy(f, 1)
            if not py == op:
                raise Exception("Invalid Format!", py, op)


    def readfrom(self, tfmod, cfg_path):
        with open(cfg_path, "rb") as cfg_file:
            self.loadtf(cfg_file, tfmod)










