#!/usr/bin/python
# coding=utf-8

"""
@version:
@author: Dong Linhao
@license: Apache Licence
@contact: donglinhao2015@ia.ac.cn
@site:
@software: PyCharm Community Edition
@file: batchmk.py
@time: 09/04/17 21:10
"""

import src.io.fea as fea
import tensorflow as tf
import numpy as np
import time

LONGEST_FRMS = 2000

class lstm_batch(object):
    def __init__(self, num_streams, num_steps, input_dim):
        self.sample_feat_list = [np.zeros([LONGEST_FRMS, input_dim]) for _ in range(num_streams)]
        self.sample_label_list = [np.zeros([LONGEST_FRMS]) for _ in range(num_streams)]
        self.sample_mask_list = [np.zeros([LONGEST_FRMS]) for _ in range(num_streams)]

        self.curt = np.zeros(num_streams, dtype=int)
        self.lent = np.zeros(num_streams, dtype=int)
        self.reset_flag = np.zeros(num_streams, dtype=bool)

        self.num_streams = num_streams
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.handled_utt_num = 0
        self.handled_frm_num = 0
        self.cur_epoch_finish = False

    def set_stream_num(self, num_streams):
        self.num_streams = num_streams

        self.sample_feat_list = [np.zeros([LONGEST_FRMS, self.input_dim]) for _ in range(num_streams)]
        self.sample_label_list = [np.zeros([LONGEST_FRMS]) for _ in range(num_streams)]
        self.sample_mask_list = [np.zeros([LONGEST_FRMS]) for _ in range(num_streams)]

        self.curt = np.zeros(num_streams, dtype=int)
        self.lent = np.zeros(num_streams, dtype=int)
        self.reset_flag = np.zeros(num_streams, dtype=bool)

    def reset(self):
        self.sample_feat_list = [np.zeros([LONGEST_FRMS, self.input_dim]) for _ in range(self.num_streams)]
        self.sample_label_list = [np.zeros([LONGEST_FRMS]) for _ in range(self.num_streams)]
        self.sample_mask_list = [np.zeros([LONGEST_FRMS]) for _ in range(self.num_streams)]

        self.curt = np.zeros(self.num_streams, dtype=int)
        self.lent = np.zeros(self.num_streams, dtype=int)
        self.reset_flag = np.zeros(self.num_streams, dtype=bool)

        self.handled_utt_num = 0
        self.handled_frm_num = 0
        self.cur_epoch_finish = False

    def make_batch(self, sess, sample, run_device, total_utt_num):
        with tf.device(run_device):
            multistream_feat_batch = [np.zeros([self.num_steps, self.input_dim]) for _ in range(self.num_streams)]
            multistream_label_batch = [np.zeros([self.num_steps]) for _ in range(self.num_streams)]
            multistream_mask_batch = [np.zeros([self.num_steps]) for _ in range(self.num_streams)]
            reset_flag = np.zeros(self.num_streams, dtype=bool)

            for s in range(self.num_streams):
                if self.curt[s] < self.lent[s]:
                    reset_flag[s] = False
                    continue

                if self.handled_utt_num < total_utt_num:
                    sample_feats, sample_labels, sample_masks = sess.run(sample)
                    self.handled_utt_num += 1
                    self.sample_feat_list[s] = sample_feats
                    self.sample_label_list[s] = sample_labels
                    self.sample_mask_list[s] = sample_masks
                    self.lent[s] = np.shape(sample_feats)[0]
                    self.curt[s] = 0
                    reset_flag[s] = True

            for s in range(self.num_streams):
                if self.curt[s] < self.lent[s]:
                    multistream_feat_batch[s] = self.sample_feat_list[s][self.curt[s]:self.curt[s]+self.num_steps, :]
                    multistream_label_batch[s] = self.sample_label_list[s][self.curt[s]:self.curt[s]+self.num_steps]
                    multistream_mask_batch[s] = self.sample_mask_list[s][self.curt[s]:self.curt[s]+self.num_steps]

                    self.curt[s] += self.num_steps
                    self.handled_frm_num += np.sum(multistream_mask_batch[s])
                else:
                    multistream_mask_batch[s] = np.zeros([self.num_steps])

            final_feat_batch = np.stack(multistream_feat_batch, axis=1)
            final_label_batch = np.stack(multistream_label_batch, axis=1)
            final_mask_batch = np.stack(multistream_mask_batch, axis=1)

            done = True
            for s in range(self.num_streams):
                if self.curt[s] < self.lent[s]:
                    done = False
            if done:
                self.cur_epoch_finish = True

        return final_feat_batch, final_label_batch, final_mask_batch, reset_flag


def getfilelst(scp_file_path):
    # get tf list
    tf_list = []
    with open(scp_file_path) as list_file:
        for line in list_file:
            tf_list.append(line.strip())
    return tf_list


def process_my_feature(feature, label, flags):
    # Add delta
    if flags.add_delta:
        feature = fea.tf_fea_add_delt(feature)
    # CMVN
    feature = fea.tf_fea_cmvn_global(feature, flags.feat_mean, flags.feat_var)
    # Splice
    feature = fea.tf_fea_splice(feature, flags.l_splice, flags.r_splice)
    feature = tf.reshape(feature, [-1, flags.input_dim])

    return feature[:], label[:]


def read_my_file_format(filename_queue, org_feat_dim):
    # build reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    raw_example = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'feat': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })
    example = tf.decode_raw(raw_example['feat'], tf.float32)
    example = tf.reshape(example, [-1, org_feat_dim])
    label = tf.decode_raw(raw_example['label'], tf.int32)

    return example, label


def lstm_input_pipeline(flags, is_training, num_epochs=None, shuffle_state = True):
    with tf.device(flags.default_device):
        if is_training:
            filenames = getfilelst(flags.trn_data_dir + '/tf.lst')
        else:
            filenames = getfilelst(flags.cv_data_dir + '/tf.lst')

        # generate file queue
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs = num_epochs, shuffle = shuffle_state)

        # read from file queue
        sample = read_my_file_format(filename_queue, flags.org_feat_dim)

        # handle sample
        sample_feats, sample_labels = process_my_feature(sample[0], sample[1], flags)
        sample_length = tf.shape(sample_feats)[0]
        sample_masks = tf.ones([sample_length], dtype=tf.float32)

        # add target delay
        if flags.target_delay > 0:
            feats_part1 = tf.slice(sample_feats, [flags.target_delay, 0], [sample_length-flags.target_delay, -1])
            last_frm_feats = tf.slice(sample_feats, [sample_length-1, 0], [1, -1])
            feats_part2 = tf.concat([last_frm_feats for _ in range(flags.target_delay)], axis=0)
            sample_feats = tf.concat([feats_part1, feats_part2], axis=0)

        padding_length = flags.num_steps - sample_length % flags.num_steps
        padding_feats = tf.zeros([padding_length, flags.input_dim], dtype=tf.float32)
        feats = tf.concat(axis=0, values=[sample_feats, padding_feats])
        padding_labels = tf.zeros([padding_length], dtype=tf.int32)
        labels = tf.concat(axis=0, values=[sample_labels, padding_labels])
        padding_masks = tf.zeros([padding_length], dtype=tf.float32)
        frame_masks = tf.concat(axis=0, values=[sample_masks, padding_masks])

    return feats, labels, frame_masks

