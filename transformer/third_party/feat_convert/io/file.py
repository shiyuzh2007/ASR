#!/usr/bin/python
# coding=utf-8

"""
@version: 
@author: zhulei
@license: Apache Licence 
@contact: zhulei@rokid.com
@site: 
@software: PyCharm Community Edition
@file: file.py
@time: 1/5/17 5:52 PM
"""
import os
import stat

def z_rmdir(path):
    if(os.path.exists(path)):
        for name in os.listdir(path):
            fullname = os.path.join(path,name)
            mode = os.lstat(fullname).st_mode
            if not stat.S_ISDIR(mode):
                os.remove(fullname)

def z_mkdir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def exist(path):
    return os.path.exists(path)

def copy(src, des):
    os.system("cp %s %s" % (src, des))

def check_model_dir(flags, cfg_file):
    if not os.path.exists(flags.get_model_dir()):
        z_mkdir(flags.get_model_dir())
        os.system("cp %s %s" % (cfg_file, flags.get_model_dir()))
    elif os.listdir(flags.get_model_dir()):
        print "Please backup your existed model directory first! "
        exit()