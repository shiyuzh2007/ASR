#!/usr/bin/python
# coding=utf-8

"""
@version: 
@author: zhulei
@license: Apache Licence 
@contact: zhulei@rokid.com
@site: 
@software: PyCharm Community Edition
@file: feacfg.py
@time: 1/3/17 1:37 PM
"""

import numpy as np
import sys

class FeaCfg(object):
    ##Arkwriter constructor
    # @param scp_path path to the .scp file that will be written
    def __init__(self):
        self.fea_dim = 0
        self.state_num = 0
        self.utt_num = 0
        self.fea_num_total = 0
        self.fea_mean = None
        self.fea_var = None

    def setcfg(self, fea_dim, state_num, utt_num, fea_num_total, fea_mean, fea_var):
        self.fea_dim = fea_dim
        self.state_num = state_num
        self.utt_num = utt_num
        self.fea_num_total = fea_num_total
        self.fea_mean = np.asarray(fea_mean, dtype=np.float32)
        self.fea_var = np.asarray(fea_var, dtype=np.float32)


    def printcfg(self):

        print 'Fea_Dim : ' + str(self.fea_dim)
        print 'State_Num : ' + str(self.state_num)
        print 'Utt Num : ' + str(self.utt_num)
        print 'Fea_Num_Total : ' + str(self.fea_num_total)

        sys.stdout.write('Fea_Mean :')
        for x in self.fea_mean:
            sys.stdout.write(" %0.7f"%x)
        sys.stdout.write('\n')
        sys.stdout.flush()

        sys.stdout.write('Fea_Var :')
        for x in self.fea_var:
            sys.stdout.write(" %0.7f" % x)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def saveto(self, cfg_path):

        with open(cfg_path, "w") as cfg_file:
            cfg_file.write("Fea_Dim %d\n" % self.fea_dim)
            cfg_file.write("State_Num %d\n" % self.state_num)
            cfg_file.write("Utt_Num %d\n" % self.utt_num)
            cfg_file.write("Fea_Num_Total %d\n" % self.fea_num_total)

            cfg_file.write("Fea_Mean")
            for x in self.fea_mean:
                cfg_file.write(" %0.7f" % x)
            cfg_file.write("\n")

            cfg_file.write("Fea_Var")
            for x in self.fea_var:
                cfg_file.write(" %0.7f" % x)
            cfg_file.write("\n")

    def readfrom(self, cfg_path):
        with open(cfg_path, "r") as cfg_file:
            for line in cfg_file:
                data = line.strip().split(' ')
                if data[0] == 'Fea_Dim':
                    self.fea_dim = int(data[1])
                if data[0] == 'State_Num':
                    self.state_num = int(data[1])
                if data[0] == 'Utt_Num':
                    self.utt_num = int(data[1])
                if data[0] == 'Fea_Num_Total':
                    self.fea_num_total = int(data[1])
                if data[0] == 'Fea_Mean':
                    self.fea_mean = np.asarray(map(float, data[1:]), dtype=np.float32)
                if data[0] == 'Fea_Var':
                    self.fea_var = np.asarray(map(float, data[1:]), dtype=np.float32)
