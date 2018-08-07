# coding=utf-8
from __future__ import print_function

import codecs
import logging
import os
from tempfile import mkstemp

import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tff
import third_party.feat_convert.io.ark as zark
from third_party.tensor2tensor import common_layers, common_attention
import re

PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

PAD = u'<PAD>'
UNK = u'<UNK>'
BOS = u'<S>'
EOS = u'</S>'


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class DataReader(object):
    """
    Read data and create batches for training and testing.
    """

    def __init__(self, config):
        self._config = config
        self._tmps = set()
        self.load_vocab()

    def __del__(self):
        for fname in self._tmps:
            if os.path.exists(fname):
                os.remove(fname)

    def load_vocab(self):
        """
        Load vocab from disk.
        The first four items in the vocab should be <PAD>, <UNK>, <S>, </S>
        """

        def load_vocab_(path, vocab_size):
            vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        logging.info('Load vocabularies %s.' % (self._config.dst_vocab))
        self.dst2idx, self.idx2dst = load_vocab_(self._config.dst_vocab, self._config.dst_vocab_size)

    def get_training_batches_with_buckets(self, shuffle=True):
        """
        Generate batches according to bucket setting.
        """

        # buckets = [(i, i) for i in range(5, 1000000, 3)]
        buckets = [(i, i) for i in range(self._config.bucket_min, self._config.bucket_max, self._config.bucket_step)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return l1, l2
            raise Exception("The sequence is too long: ({}, {})".format(sl, dl))

        # Shuffle the training files.
        src_path = self._config.train.src_path
        dst_path = self._config.train.dst_path
        max_length = self._config.train.max_length

        if shuffle:
            logging.info('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path], self._config.model_dir)
            logging.info('Shuffled files %s and %s.' % (src_shuf_path, dst_shuf_path))
            self._tmps.add(src_shuf_path)
            self._tmps.add(dst_shuf_path)
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0]  # src sentences, dst sentences, src tokens, dst tokens

        uttid_target_map = {}
        for line in codecs.open(dst_shuf_path, 'r', 'utf-8'):
            line = line.strip()
            if line == '' or line is None:
                continue
            splits = re.split('\s+', line)
            uttid = splits[0].strip()
            target = splits[1:]
            uttid_target_map[uttid] = target
        logging.info('loaded dst_shuf_path=' + str(dst_shuf_path) + ',size=' + str(len(uttid_target_map)))

        num_random_caches = 5000
        num_cache_max_length = 600
        num_cache_min_length = 100
        num_cache_target_min_length = 4
        random_caches = []
        count = 0
        scp_reader = zark.ArkReader(src_shuf_path)
        while True:
            uttid, input, looped = scp_reader.read_next_utt()
            if looped:
                break

            target = uttid_target_map[uttid]
            if target is None:
                logging.warn('uttid=' + str(uttid) + ',target is None')
                continue

            input_len = len(input)
            target_len = len(target)
            if input_len > max_length or target_len > max_length:
                logging.warn(
                    'uttid=' + str(uttid) + ',input_len=' + str(input_len) + ' > max_length=' + str(max_length))
                continue

            count = count + 1
            if target_len == 0:
                continue

            bucket = select_bucket(input_len, target_len)
            caches[bucket][0].append(input)
            caches[bucket][1].append(target)
            caches[bucket][2] += input_len
            caches[bucket][3] += target_len

            if len(random_caches) < num_random_caches and num_cache_min_length <= input_len <= num_cache_max_length \
                    and target_len >= num_cache_target_min_length:
                random_caches.append([input, target])

            if max(caches[bucket][2], caches[bucket][3]) > self._config.train.tokens_per_batch:
                feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
                target_batch, target_batch_mask = self._create_target_batch(caches[bucket][1], self.dst2idx)
                # yield (feat_batch, feat_batch_mask, target_batch, target_batch_mask)
                yield (feat_batch, target_batch, len(caches[bucket][0]))
                caches[bucket] = [[], [], 0, 0]

        # Clean remain sentences.
        for bucket in buckets:
            # Ensure each device at least get one sample.
            if len(caches[bucket][0]) > 0:
                src_len = len(caches[bucket][0])
                if self._config.min_count_in_bucket is None:
                    default_min_count_in_bucket = 100
                    logging.info(
                        'min_count_in_bucket=' + str(
                            self._config.min_count_in_bucket) + ',use default_min_count_in_bucket=' + str(
                            default_min_count_in_bucket))
                    self._config.min_count_in_bucket = default_min_count_in_bucket
                left_count = self._config.min_count_in_bucket - src_len
                if left_count > 0:  # append to self._config.train.num_gpus
                    for idx in range(left_count):
                        rand_idx = np.random.randint(0, num_random_caches)
                        if rand_idx >= num_random_caches:
                            rand_idx = 0
                        input, target = random_caches[rand_idx]
                        caches[bucket][0].append(input)
                        caches[bucket][1].append(target)
                        caches[bucket][2] += len(input)
                        caches[bucket][3] += len(target)
                    dst_len = len(caches[bucket][0])
                    logging.info(
                        'get_training_batches_with_buckets, src_len=' + str(src_len) + ',dst_len=' + str(
                            dst_len) + ',bucket=' + str(bucket) + ',max(caches[bucket][2], caches[bucket][3])=' + str(
                            max(caches[bucket][2], caches[bucket][3])))
                feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
                target_batch, target_batch_mask = self._create_target_batch(caches[bucket][1], self.dst2idx)
                # yield (feat_batch, feat_batch_mask, target_batch, target_batch_mask)
                yield (feat_batch, target_batch, len(caches[bucket][0]))

        logging.info('loaded count=' + str(count))
        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)
            self._tmps.remove(src_shuf_path)
            self._tmps.remove(dst_shuf_path)

    def _create_feat_batch(self, indices):
        # Pad to the same length.
        # indices的数据是[[feat_len1, feat_dim], [feat_len2, feat_dim], ...]
        maxlen = max([len(s) for s in indices])
        batch_size = len(indices)
        feat_batch = []
        feat_batch_mask = np.ones([batch_size, maxlen], dtype=np.int32)
        for i in range(batch_size):
            feat = indices[i]
            feat_len, feat_dim = np.shape(feat)
            padding = np.zeros([maxlen - feat_len, feat_dim], dtype=np.float32)
            padding.fill(PAD_INDEX)
            feat = np.concatenate([feat, padding], axis=0)
            feat_batch.append(feat)
            feat_batch_mask[i, :feat_len] = 0
        feat_batch = np.stack(feat_batch, axis=0)
        return feat_batch, feat_batch_mask

    def _create_target_batch(self, sents, phone2idx):
        # sents的数据是[word1 word2 ... wordn]
        indices = []
        for sent in sents:
            x = []
            # for word in (sent + [EOS]):
            #     if word is not None or word.strip() != '':
            #         x_tmp = phone2idx.get(word, UNK_INDEX)
            #         x.append(x_tmp)
            #         if x_tmp == UNK_INDEX:
            #             logging.warn('=========[ZSY]x_tmp=UNK_INDEX')
            # x = [phone2idx.get(word, UNK_INDEX) for word in (sent + [EOS])]

            for word in (sent + [EOS]):
                x_tmp = phone2idx.get(word, UNK_INDEX)
                x.append(x_tmp)
                if x_tmp == UNK_INDEX and word != UNK:
                    logging.warn('=========[ZSY]x_tmp=UNK_INDEX, word=' + str(word.encode('UTF-8')))
            indices.append(x)

        # Pad to the same length.
        batch_size = len(sents)
        maxlen = max([len(s) for s in indices])
        target_batch = np.zeros([batch_size, maxlen], np.int32)
        target_batch.fill(PAD_INDEX)
        target_batch_mask = np.ones([batch_size, maxlen], dtype=np.int32)
        for i, x in enumerate(indices):
            target_batch[i, :len(x)] = x
            target_batch_mask[i, :len(x)] = 0
        return target_batch, target_batch_mask

    @staticmethod
    def shuffle(list_of_files, log_dir):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')

        fds = [open(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("<CONCATE4SHUF>".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        # fnames = ['/tmp/{}.{}.shuf'.format(i, os.getpid()) for i, ff in enumerate(list_of_files)]
        fnames = [(log_dir + '/{}.{}.shuf').format(i, os.getpid()) for i, ff in enumerate(list_of_files)]
        fds = [open(fn, 'w') for fn in fnames]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('<CONCATE4SHUF>')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return fnames

    def get_test_batches_with_buckets(self, src_path, tokens_per_batch):
        buckets = [(i) for i in range(50, 10000, 10)]

        def select_bucket(sl):
            for l1 in buckets:
                if sl < l1:
                    return l1
            raise Exception("The sequence is too long: ({})".format(sl))

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0]  # feats, uttids, count

        scp_reader = zark.ArkReader(src_path)
        count = 0
        while True:
            uttid, input, loop = scp_reader.read_next_utt()
            if loop:
                break

            input_len = len(input)
            bucket = select_bucket(input_len)
            caches[bucket][0].append(input)
            caches[bucket][1].append(uttid)
            caches[bucket][2] += input_len
            count = count + 1
            if caches[bucket][2] > tokens_per_batch:
                feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
                yield feat_batch, caches[bucket][1]
                caches[bucket] = [[], [], 0]

        # Clean remain sentences.
        for bucket in buckets:
            if len(caches[bucket][0]) > 0:
                logging.info('get_test_batches_with_buckets, len(caches[bucket][0])=' + str(len(caches[bucket][0])))
                feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
                yield feat_batch, caches[bucket][1]

        logging.info('get_test_batches_with_buckets, loaded count=' + str(count))

    def get_test_batches_with_buckets_and_target(self, src_path, dst_path, tokens_per_batch):
        buckets = [(i) for i in range(50, 10000, 5)]

        def select_bucket(sl):
            for l1 in buckets:
                if sl < l1:
                    return l1
            raise Exception("The sequence is too long: ({})".format(sl))

        uttid_target_map = {}
        for line in codecs.open(dst_path, 'r', 'utf-8'):
            line = line.strip()
            if line == '' or line is None:
                continue
            splits = re.split('\s+', line)
            uttid = splits[0].strip()
            target = splits[1:]
            uttid_target_map[uttid] = target
        logging.info('loaded dst_path=' + str(dst_path) + ',size=' + str(len(uttid_target_map)))

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0]

        scp_reader = zark.ArkReader(src_path)
        count = 0
        while True:
            uttid, input, loop = scp_reader.read_next_utt()
            if loop:
                break

            target = uttid_target_map[uttid]
            if target is None:
                logging.warn('uttid=' + str(uttid) + ',target is None')
                continue

            input_len = len(input)
            target_len = len(target)
            bucket = select_bucket(input_len)
            caches[bucket][0].append(input)
            caches[bucket][1].append(target)
            caches[bucket][2] += input_len
            caches[bucket][3] += target_len
            count = count + 1
            if caches[bucket][2] > tokens_per_batch:
                feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
                target_batch, target_batch_mask = self._create_target_batch(caches[bucket][1], self.dst2idx)
                yield feat_batch, target_batch
                caches[bucket] = [[], [], 0, 0]

        for bucket in buckets:
            if len(caches[bucket][0]) > 0:
                logging.info('get_test_batches_with_buckets, len(caches[bucket][0])=' + str(len(caches[bucket][0])))
                feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
                target_batch, target_batch_mask = self._create_target_batch(caches[bucket][1], self.dst2idx)
                yield feat_batch, target_batch

        logging.info('get_test_batches_with_buckets, loaded count=' + str(count))

    def get_test_batches(self, src_path, batch_size):
        scp_reader = zark.ArkReader(src_path)
        cache = []
        uttids = []
        while True:
            uttid, feat, loop = scp_reader.read_next_utt()
            if loop:
                break
            cache.append(feat)
            uttids.append(uttid)
            if len(cache) >= batch_size:
                feat_batch, feat_batch_mask = self._create_feat_batch(cache)
                # yield feat_batch, feat_batch_mask, uttids
                yield feat_batch, uttids
                cache = []
                uttids = []
        if cache:
            feat_batch, feat_batch_mask = self._create_feat_batch(cache)
            # yield feat_batch, feat_batch_mask, uttids
            yield feat_batch, uttids

    def indices_to_words(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        sents = []
        for y in Y:  # for each sentence
            sent = []
            for i in y:  # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents


def expand_feed_dict(feed_dict):
    """If the key is a tuple of placeholders,
    split the input data then feed them into these placeholders.
    """
    new_feed_dict = {}
    for k, v in feed_dict.items():
        if type(k) is not tuple:
            new_feed_dict[k] = v
        else:
            # Split v along the first dimension.
            n = len(k)
            batch_size = v.shape[0]
            span = batch_size // n
            remainder = batch_size % n
            # assert span > 0
            base = 0
            for i, p in enumerate(k):
                if i < remainder:
                    end = base + span + 1
                else:
                    end = base + span
                new_feed_dict[p] = v[base: end]
                base = end
    return new_feed_dict


def available_variables(checkpoint_dir):
    all_vars = tf.global_variables()
    all_available_vars = tff.list_variables(checkpoint_dir=checkpoint_dir)
    all_available_vars = dict(all_available_vars)
    available_vars = []
    for v in all_vars:
        vname = v.name.split(':')[0]
        if vname in all_available_vars and v.get_shape() == all_available_vars[vname]:
            available_vars.append(v)
    return available_vars


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def residual(inputs, outputs, dropout_rate):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float range from [0, 1).

    Returns:
        A Tensor.
    """
    outputs = inputs + tf.nn.dropout(outputs, 1 - dropout_rate)
    outputs = common_layers.layer_norm(outputs)
    return outputs


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
            name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        output = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            output *= multiplier
        return output


def dense(inputs,
          output_size,
          activation=tf.identity,
          use_bias=True,
          reuse_kernel=None,
          reuse=None,
          name=None):
    argcount = activation.func_code.co_argcount
    if activation.func_defaults:
        argcount -= len(activation.func_defaults)
    assert argcount in (1, 2)
    with tf.variable_scope(name, "dense", reuse=reuse):
        if argcount == 1:
            input_size = inputs.get_shape().as_list()[-1]
            inputs_shape = tf.unstack(tf.shape(inputs))
            inputs = tf.reshape(inputs, [-1, input_size])
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_kernel):
                w = tf.get_variable("kernel", [output_size, input_size])
            outputs = tf.matmul(inputs, w, transpose_b=True)
            if use_bias:
                b = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer)
                outputs += b
            outputs = activation(outputs)
            return tf.reshape(outputs, inputs_shape[:-1] + [output_size])
        else:
            arg1 = dense(inputs, output_size, tf.identity, use_bias, name='arg1')
            arg2 = dense(inputs, output_size, tf.identity, use_bias, name='arg2')
            return activation(arg1, arg2)


def ff_hidden(inputs, hidden_size, output_size, activation, use_bias=True, reuse=None, name=None):
    with tf.variable_scope(name, "ff_hidden", reuse=reuse):
        hidden_outputs = dense(inputs, hidden_size, activation, use_bias)
        outputs = dense(hidden_outputs, output_size, tf.identity, use_bias)
        return outputs


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        reserve_last=False,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.

    Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    reserve_last: a boolean
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
        If the query positions and memory positions represent the
        pixels of a flattened image, then pass in their dimensions:
          (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

    Returns:
    A Tensor.
    """
    with tf.variable_scope(
            name,
            default_name="multihead_attention",
            values=[query_antecedent, memory_antecedent]):

        if memory_antecedent is None:
            # self attention
            combined = dense(query_antecedent, total_key_depth * 2 + total_value_depth, name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth],
                axis=2)
        else:
            q = dense(query_antecedent, total_key_depth, name="q_transform")
            combined = dense(memory_antecedent, total_key_depth + total_value_depth, name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

        if reserve_last:
            q = q[:, -1:, :]

        q = common_attention.split_heads(q, num_heads)
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head ** -0.5
        x = common_attention.dot_product_attention(
            q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = common_attention.combine_heads(x)
        x = dense(x, output_depth, name="output_transform")
        return x
