#!/usr/bin/python
# coding=utf-8

"""
@version: 
@author: zhulei
@license: Apache Licence 
@contact: zhulei@rokid.com
@site: 
@software: PyCharm Community Edition
@file: Ark2Tf.py
@time: 12/14/16 4:23 PM
"""

# import sys
# sys.path.append('io')

import argparse
import os
import sys
import pickle

import tensorflow as tf
import src.io.ark as zark
import src.io.fea as zfea
import src.io.feacfg as zfeacfg
import src.io.file as zfile
import numpy as np

PARAM = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def Convert():
    print PARAM

    # Read Ark File
    fealst = zark.ArkReader(PARAM.feascp)

    zfile.z_rmdir(PARAM.tffd + '/tf')
    zfile.z_rmdir(PARAM.tffd)

    zfile.z_mkdir(PARAM.tffd + '/tf')

    tf_file_cnt = 0
    tf_file_path = PARAM.tffd + '/tf/' + str(tf_file_cnt) + '.tf'
    tf_file_writer = tf.python_io.TFRecordWriter(tf_file_path)

    scp_file_write = open(PARAM.tffd + '/' + 'tf.lst', "w")
    scp_file_write.write('%s\n' % tf_file_path)

    file_count = 0
    file_count_err = 0
    rows_total = 0

    state_num = 0
    fea_dim = None

    # cmvn
    fea_mean_total = None
    fea_var_total = None
    fea_num_total = None

    with open(PARAM.aliscp, "r") as ali_file:
        for line in ali_file:
            data = line.strip().split(' ')
            utt = data[0]

            # Read Data
            ali = np.asarray(map(int, data[1:len(data)]))
            ali = ali.astype(np.int32)
            fea = fealst.read_utt_data_dic(utt)

            # Process Fea Data
            if fea is None:
                file_count_err += 1
                continue
            rows, cols = fea.shape

            if fea_dim is None:
                fea_dim = cols

            if fea_dim != cols:
                print 'Error Fea_dim ' + str(fea_dim) + 'with ' + str(cols)
                file_count_err += 1
                continue

            # get state_num
            state_num = max(state_num, np.amax(ali))

            # Process Align Data
            ali_shape = ali.shape
            if ali_shape[0] == 1:
                ali = np.ones([1, rows], np.int32) * ali_shape[0]
                ali_shape = ali.shape

            if ali_shape[0] != rows:
                print 'Error Fea_num ' + str(rows) + 'with Ali_num ' + str(ali_shape[0])
                file_count_err += 1
                continue

            # Write Data
            file_count += 1
            rows_total += rows
            example = tf.train.Example(features=tf.train.Features(feature={
                'feat': _bytes_feature(fea.tostring()),
                'label': _bytes_feature(ali.tostring())
            }))
            tf_file_writer.write(example.SerializeToString())

            # Calculate Cmvn
            fea = zfea.np_fea_add_delt(fea)
            fea_mean = np.sum(fea, 0)
            fea_var = np.sum(np.square(fea), 0)
            fea_num = rows

            if fea_mean_total is None:
                fea_mean_total = fea_mean
                fea_var_total = fea_var
                fea_num_total = rows
            else:
                fea_mean_total += fea_mean
                fea_var_total += fea_var
                fea_num_total += rows

            # Report Progress
            if file_count % 1000 == 0:
                print file_count
                # sys.stdout.write(' ' * 10 + '\r')
                # sys.stdout.flush()
                # sys.stdout.write(str(file_count) + '\r')
                # sys.stdout.flush()

            if rows_total > 500000:
                tf_file_writer.close()
                rows_total = 0
                tf_file_cnt += 1
                tf_file_path = PARAM.tffd + '/tf/' + str(tf_file_cnt) + '.tf'
                tf_file_writer = tf.python_io.TFRecordWriter(tf_file_path)
                scp_file_write.write('%s\n' % tf_file_path)

    tf_file_writer.close()
    scp_file_write.close()
    print 'convert ' + str(file_count) + ' utterances, wrong alignment utt: ' + str(file_count_err)

    # Calculate State Num and Cmvn
    state_num += 1
    fea_mean_total = fea_mean_total / fea_num_total
    fea_var_total = fea_var_total / fea_num_total
    fea_var_total = np.sqrt(fea_var_total - fea_mean_total * fea_mean_total)

    fea_cfg = zfeacfg.FeaCfg()
    fea_cfg.setcfg(fea_dim=fea_dim, state_num=state_num, utt_num=file_count, fea_num_total=fea_num_total,
                   fea_mean=fea_mean_total, fea_var=fea_var_total)
    fea_cfg.saveto(PARAM.tffd + '/' + 'fea.cfg')

    fea_cfg.printcfg()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feascp',
        type=str,
        default='/data/syzhou/work/data/tfkaldi/data_debug/hkust/kaldi_feats/train/tftrain_feats.scp',
        help='Feature Scp File Path'
    )
    parser.add_argument(
        '--aliscp',
        type=str,
        default='/data/syzhou/work/data/tfkaldi/data_debug/hkust/kaldi_pdf/train.pdf',
        help='Align Scp File Path'
    )
    parser.add_argument(
        '--tffd',
        type=str,
        default='/data/syzhou/work/data/tfkaldi/data_debug/hkust/train',
        help='TensorFlow Folder'
    )
    PARAM, _ = parser.parse_known_args()
    Convert()
