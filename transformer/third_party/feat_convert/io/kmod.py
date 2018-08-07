#!/usr/bin/python
# coding=utf-8

"""
@version: 
@author: zhulei
@license: Apache Licence 
@contact: zhulei@rokid.com
@site: 
@software: PyCharm Community Edition
@file: kmod.py
@time: 1/9/17 4:36 PM
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variables
import struct


# Compatible with Kaldi---------------------------------------------------------

class KaldiModel(object):
    def __init__(self):
        ##  lstm
        self.cell_dim_ = 100
        self.clip_gradient_ = 10
        self.lstm_max_norm_ = 0

        ##  affine
        self.learn_rate_coef_ = 1
        self.bias_learn_rate_coef_ = 1
        self.affine_max_norm_ = 0

    # Token IO---------------------------------------------------------
    def ReadToken(self, f):
        tok = ""
        ch, = f.read(1)
        while ch != ' ':
            tok = tok + ch
            ch, = f.read(1)
        return tok

    def WriteToken(self, f, token):
        if not isinstance(token, basestring):
            raise Exception("Error TfModel Tag %s" % token)

        _len = len(token)
        f.write(struct.pack("<%dss" % _len, token, ' '))
        return

    def ExpectToken(self, f, token):
        if not isinstance(token, basestring):
            raise Exception("Error TfModel Tag %s" % token)

        tk = self.ReadToken(f)
        if token == tk:
            return
        else:
            raise Exception("Expect Token %s with %s" % (token, tk))
        return


        # Base Type IO---------------------------------------------------------

    def ReadInt(self, f):
        l, value = struct.unpack('<bi', f.read(5))
        return value

    def ReadFloat(self, f):
        l, value = struct.unpack('<bf', f.read(5))
        return value

    def WriteInt(self, f, value):
        f.write(struct.pack('<bi', 4, value))
        return

    def WriteFloat(self, f, value):
        f.write(struct.pack('<bf', 4, value))
        return

    def ExpectInt(self, f, value):
        va = self.ReadInt(f)
        if value == va:
            return
        else:
            raise Exception("Expect Int %d with %d" % (value, va))

    # Matrix Vec IO---------------------------------------------------------
    def ReadMatrix(self, f, t=False):
        self.ExpectToken(f, "FM")
        _rows = self.ReadInt(f)
        _cols = self.ReadInt(f)
        _py = np.frombuffer(f.read(_rows * _cols * 4), dtype=np.float32)
        _py = np.reshape(_py, [_rows, _cols])
        if t:
            _py = np.transpose(_py)

        return _py

    def ReadVector(self, f):
        self.ExpectToken(f, "FV")
        _size = self.ReadInt(f)
        _py = np.frombuffer(f.read(_size * 4), dtype=np.float32)
        return _py

    def WriteMatrix(self, f, mat, t):
        if t:
            mat = np.transpose(mat)
        _rows, _cols = np.shape(mat)
        self.WriteToken(f, "FM")
        self.WriteInt(f, _rows)
        self.WriteInt(f, _cols)
        f.write(mat.copy(order='C'))
        return

    def WriteVec(self, f, vec):
        _size, = np.shape(vec)
        self.WriteToken(f, "FV")
        self.WriteInt(f, _size)
        f.write(vec.copy(order='C'))
        return

    # Layer IO---------------------------------------------------------
    def ReadLayer(self, f, s, layer):
        self.ExpectToken(f, layer[0])
        self.ExpectInt(f, layer[1])
        self.ExpectInt(f, layer[2])
        if layer[0] == '<AffineTransform>':
            self.ExpectToken(f, '<LearnRateCoef>')
            self.learn_rate_coef_ = self.ReadFloat(f)
            self.ExpectToken(f, '<BiasLearnRateCoef>')
            self.bias_learn_rate_coef_ = self.ReadFloat(f)
            self.ExpectToken(f, '<MaxNorm>')
            self.affine_max_norm_ = self.ReadFloat(f)

            weights = self.ReadMatrix(f, True)
            s.run(layer[3].assign(weights))
            biases = self.ReadVector(f)
            s.run(layer[4].assign(biases))
        elif layer[0] == '<LstmProjectedStreams>':
            self.ExpectToken(f, '<CellDim>')
            self.cell_dim_ = self.ReadInt(f)
            self.ExpectToken(f, '<ClipGradient>')
            self.clip_gradient_ = self.ReadFloat(f)
            self.ExpectToken(f, '<MaxNorm>')
            self.lstm_max_norm_ = self.ReadFloat(f)

            ##  lstm_cell/weights
            w_gifo_x = self.ReadMatrix(f, True)
            w_gifo_r = self.ReadMatrix(f, True)
            w_g_x, w_i_x, w_f_x, w_o_x = tf.split(w_gifo_x, 4, 1)
            w_g_r, w_i_r, w_f_r, w_o_r = tf.split(w_gifo_r, 4, 1)
            w_igfo_x = tf.concat([w_i_x, w_g_x, w_f_x, w_o_x], 1)
            w_igfo_r = tf.concat([w_i_r, w_g_r, w_f_r, w_o_r], 1)
            s.run(tf.assign(layer[3], tf.concat([w_igfo_x, w_igfo_r], 0)))

            ##  lstm_cell/biases
            bias = self.ReadVector(f)
            bias = tf.expand_dims(bias, 0)
            bias_g, bias_i, bias_f, bias_o = tf.split(bias, 4, 1)
            s.run(tf.assign(layer[4], tf.squeeze(tf.concat([bias_i, bias_g, bias_f, bias_o], 1), 0)))

            ##  lstm_cell/w_f_diag, lstm_cell/w_i_diag, lstm_cell/w_o_diag
            peephole_i = self.ReadVector(f)
            peephole_f = self.ReadVector(f)
            peephole_o = self.ReadVector(f)
            s.run(tf.assign(layer[5], peephole_f))
            s.run(tf.assign(layer[6], peephole_i))
            s.run(tf.assign(layer[7], peephole_o))

            ## lstm_cell/projection/weights
            w_r_m = self.ReadMatrix(f, True)
            s.run(tf.assign(layer[8], w_r_m))
        elif layer[0] == '<LayerNorm>':
            beta = self.ReadVector(f)
            s.run(layer[3].assign(beta))
            gamma = self.ReadVector(f)
            s.run(layer[4].assign(gamma))
        elif layer[0] == '<Sigmoid>':
            pass
        elif layer[0] == '<Relu>':
            pass
        elif layer[0] == '<Softmax>':
            pass
        else:
            raise Exception("No Such Layer %s" % (layer[0]))

        ##  self.ExpectToken(f, '<!EndOfComponent>')
        return

    def WriteLayer(self, f, s, layer):
        self.WriteToken(f, layer[0])
        self.WriteInt(f, layer[1])
        self.WriteInt(f, layer[2])
        if layer[0] == '<AffineTransform>':
            self.WriteToken(f, '<LearnRateCoef>')
            self.WriteFloat(f, self.learn_rate_coef_)
            self.WriteToken(f, '<BiasLearnRateCoef>')
            self.WriteFloat(f, self.bias_learn_rate_coef_)
            self.WriteToken(f, '<MaxNorm>')
            self.WriteFloat(f, self.affine_max_norm_)

            _weights = s.run(layer[3])
            self.WriteMatrix(f, _weights, True)
            _biases = s.run(layer[4])
            self.WriteVec(f, _biases)
        elif layer[0] == '<LstmProjectedStreams>':
            self.WriteToken(f, '<CellDim>')
            self.WriteInt(f, self.cell_dim_)
            self.WriteToken(f, '<ClipGradient>')
            self.WriteFloat(f, self.clip_gradient_)
            self.WriteToken(f, '<MaxNorm>')
            self.WriteFloat(f, self.lstm_max_norm_)

            ## w_gifo_x  w_gifo_r
            w_igfo_x, w_igfo_r = tf.split(layer[3], [layer[2], layer[1]], 0)
            w_i_x, w_g_x, w_f_x, w_o_x = tf.split(w_igfo_x, 4, 1)
            w_i_r, w_g_r, w_f_r, w_o_r = tf.split(w_igfo_r, 4, 1)
            w_gifo_x = tf.concat([w_g_x, w_i_x, w_f_x, w_o_x], 1)
            w_gifo_r = tf.concat([w_g_r, w_i_r, w_f_r, w_o_r], 1)
            self.WriteMatrix(f, s.run(w_gifo_x), True)
            self.WriteMatrix(f, s.run(w_gifo_r), True)

            ##  bias
            bias_i, bias_g, bias_f, bias_o = tf.split(tf.expand_dims(layer[4], 0), 4, 1)
            bias = tf.concat([bias_g, bias_i, bias_f, bias_o], 1)
            self.WriteVec(f, s.run(tf.squeeze(bias, 0)))

            ##  peephole_i_c  peephole_f_c  peephole_o_c
            peephole_i_c = layer[6]
            peephole_f_c = layer[5]
            peephole_o_c = layer[7]
            self.WriteVec(f, s.run(peephole_i_c))
            self.WriteVec(f, s.run(peephole_f_c))
            self.WriteVec(f, s.run(peephole_o_c))

            ##  w_r_m
            w_r_m = layer[8]
            self.WriteMatrix(f, s.run(w_r_m), True)
        elif layer[0] == '<LayerNorm>':
            _beta = s.run(layer[3])
            self.WriteVec(f, _beta)
            _gamma = s.run(layer[4])
            self.WriteVec(f, _gamma)
        elif layer[0] == '<Sigmoid>':
            pass
        elif layer[0] == '<Relu>':
            pass
        elif layer[0] == '<Softmax>':
            pass
        else:
            raise Exception("No Such Layer %s" % (layer[0]))
        ##  self.WriteToken(f, '<!EndOfComponent>')
        return

    # File IO---------------------------------------------------------
    def Read(self, sess, tfmod, cfg_path):
        with open(cfg_path, "rb") as f:

            head = struct.unpack('<cc', f.read(2))
            if head[0] != '\0' or head[1] != 'B':
                raise Exception("Error Format %s" % cfg_path)

            self.ExpectToken(f, '<Nnet>')
            for layer in tfmod:
                self.ReadLayer(f, sess, layer)
            self.ExpectToken(f, '</Nnet>')
        return

    def Write(self, sess, tfmod, cfg_path):
        with open(cfg_path, "wb") as f:
            f.write(struct.pack('<cc', '\0', 'B'))

            self.WriteToken(f, '<Nnet>')
            for layer in tfmod:
                self.WriteLayer(f, sess, layer)
            self.WriteToken(f, '</Nnet>')
        return


    def LoadModel(self, sess, train_graph, model_path):
        # _, _, _, _, trn_model, _, _, _ = train_graph
        self.Read(sess=sess, tfmod=train_graph[4], cfg_path=model_path)

        optimizer = train_graph[5]
        slot_list = optimizer.get_slot_names()

        if "momentum" in slot_list:
            for v in tf.trainable_variables():
                momemtum_vars = optimizer.get_slot(v, "momentum")
                sess.run(tf.assign(momemtum_vars, tf.fill(tf.shape(momemtum_vars), 0.0)))
        return


    def SaveModel(self, sess, train_graph, model_path):
        # _, _, _, _, trn_model, _, _, _= train_graph
        self.Write(sess=sess, tfmod=train_graph[4], cfg_path=model_path)
        return