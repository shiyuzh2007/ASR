# coding=utf-8
import os
import time
from argparse import ArgumentParser
import yaml

from evaluate import Evaluator
from model import *
from utils import DataReader, AttrDict, available_variables, expand_feed_dict
import shutil


def pretrain(config):
    logger = logging.getLogger('')
    total_num_blocks_enc = config.num_blocks_enc
    total_num_blocks_dec = config.num_blocks_dec
    num_epoch = config.pretrain_num_epoch

    idx_dec = 1
    config.num_blocks_dec = idx_dec
    for idx_enc in range(1, total_num_blocks_enc + 1):
        logger.info('pretrain idx_enc=' + str(idx_enc) + ',idx_dec=' + str(idx_dec))
        pretrain_model_tag = config.model_dir + '/' + str(idx_enc) + '_' + str(idx_dec) + '.ok'
        if os.path.exists(pretrain_model_tag):
            continue

        pretrain_model_dir = config.model_dir + '/' + str(idx_enc) + '_' + str(idx_dec)
        if idx_enc == 1:
            last_pretrain_model_dir = pretrain_model_dir
        else:
            last_pretrain_model_dir = config.model_dir + '/' + str(idx_enc - 1) + '_' + str(idx_dec)

        if os.path.exists(pretrain_model_dir):
            shutil.rmtree(pretrain_model_dir)
        os.makedirs(pretrain_model_dir)

        model_dir = ''
        train(config, num_epoch, last_pretrain_model_dir, pretrain_model_dir, model_dir, idx_enc, idx_dec)
        os.mknod(pretrain_model_tag)

    idx_enc = total_num_blocks_enc
    config.num_blocks_enc = idx_enc
    for idx_dec in range(2, total_num_blocks_dec + 1):
        logger.info('pretrain idx_enc=' + str(idx_enc) + ',idx_dec=' + str(idx_dec))
        pretrain_model_tag = config.model_dir + '/' + str(idx_enc) + '_' + str(idx_dec) + '.ok'
        if os.path.exists(pretrain_model_tag):
            continue

        pretrain_model_dir = config.model_dir + '/' + str(idx_enc) + '_' + str(idx_dec)
        if idx_dec == 1:
            last_pretrain_model_dir = pretrain_model_dir
        else:
            last_pretrain_model_dir = config.model_dir + '/' + str(idx_enc) + '_' + str(idx_dec - 1)

        if os.path.exists(pretrain_model_dir):
            shutil.rmtree(pretrain_model_dir)
        os.makedirs(pretrain_model_dir)

        model_dir = ''
        if idx_dec == total_num_blocks_dec:
            model_dir = config.model_dir
        train(config, num_epoch, last_pretrain_model_dir, pretrain_model_dir, model_dir, idx_enc, idx_dec)
        os.mknod(pretrain_model_tag)

    config.num_blocks_enc = total_num_blocks_enc
    config.num_blocks_dec = total_num_blocks_dec
    config.train.var_filter = ''
    logger.info('reset old value,config.num_blocks=' + str(
        config.num_blocks) + ',config.train.var_filter=' + str(config.train.var_filter))


# def pretrain(config):
#     logger = logging.getLogger('')
#     total_num_blocks = config.num_blocks
#     num_epoch = config.pretrain_num_epoch
#     for idx in range(1, total_num_blocks + 1):
#         logger.info('pretrain layer=' + str(idx))
#         pretrain_model_tag = config.model_dir + '/' + str(idx) + '.ok'
#         if os.path.exists(pretrain_model_tag):
#             continue
#
#         pretrain_model_dir = config.model_dir + '/' + str(idx)
#         if idx == 1:
#             last_pretrain_model_dir = pretrain_model_dir
#         else:
#             last_pretrain_model_dir = config.model_dir + '/' + str(idx - 1)
#
#         if os.path.exists(pretrain_model_dir):
#             shutil.rmtree(pretrain_model_dir)
#         os.makedirs(pretrain_model_dir)
#
#         model_dir = ''
#         if idx == total_num_blocks:
#             model_dir = config.model_dir
#         train(config, num_epoch, last_pretrain_model_dir, pretrain_model_dir, model_dir, idx)
#         os.mknod(pretrain_model_tag)
#
#     config.num_blocks = total_num_blocks
#     config.train.var_filter = ''
#     logger.info('reset old value,config.num_blocks=' + str(
#         config.num_blocks) + ',config.train.var_filter=' + str(config.train.var_filter))


def available_variables_without_global_step(checkpoint_dir):
    import tensorflow.contrib.framework as tff
    all_vars = tf.global_variables()
    all_available_vars = tff.list_variables(checkpoint_dir=checkpoint_dir)
    all_available_vars = dict(all_available_vars)
    available_vars = []
    for v in all_vars:
        vname = v.name.split(':')[0]
        if vname == 'global_step':
            continue
        if vname in all_available_vars and v.get_shape() == all_available_vars[vname]:
            available_vars.append(v)
    return available_vars


def global_variables_without_global_step():
    all_vars = tf.global_variables()
    available_vars = []
    for v in all_vars:
        vname = v.name.split(':')[0]
        if vname == 'global_step':
            continue
        available_vars.append(v)
    return available_vars


def train(config, num_epoch, last_pretrain_model_dir, pretrain_model_dir, model_dir, block_idx_enc, block_idx_dec):
    logger = logging.getLogger('')
    config.num_blocks_enc = block_idx_enc
    config.num_blocks_dec = block_idx_dec
    # if block_idx >= 2:
    #     config.train.var_filter = 'encoder/block_' + str(block_idx - 1) + '|' + 'decoder/block_' + str(
    #         block_idx - 1) + '|' + 'encoder/src_embedding' + '|' + 'decoder/dst_embedding'
    # if block_idx >= 2:
    #     config.train.var_filter = 'encoder/block_' + str(block_idx - 1) + '|' + 'decoder/block_' + str(
    #         block_idx - 1)
    logger.info(
        "config.num_blocks_enc=" + str(config.num_blocks_enc) + ",config.num_blocks_dec=" + str(
            config.num_blocks_dec) + ',config.train.var_filter=' + str(config.train.var_filter))

    """Train a model with a config file."""
    data_reader = DataReader(config=config)
    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
    model.build_train_model(test=config.train.eval_on_dev)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    summary_writer = tf.summary.FileWriter(pretrain_model_dir, graph=model.graph)

    with tf.Session(config=sess_config, graph=model.graph) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        # Reload variables in disk.
        if tf.train.latest_checkpoint(last_pretrain_model_dir):
            available_vars = available_variables_without_global_step(last_pretrain_model_dir)
            # available_vars = available_variables(last_pretrain_model_dir)
            if available_vars:
                saver = tf.train.Saver(var_list=available_vars)
                saver.restore(sess, tf.train.latest_checkpoint(last_pretrain_model_dir))
                for v in available_vars:
                    logger.info('Reload {} from disk.'.format(v.name))
            else:
                logger.info('Nothing to be reload from disk.')
        else:
            logger.info('Nothing to be reload from disk.')

        evaluator = Evaluator()
        evaluator.init_from_existed(model, sess, data_reader)

        global dev_bleu, toleration
        dev_bleu = evaluator.evaluate(**config.dev) if config.train.eval_on_dev else 0
        toleration = config.train.toleration

        def train_one_step(batch):
            feat_batch, target_batch = batch
            feed_dict = expand_feed_dict({model.src_pls: feat_batch,
                                          model.dst_pls: target_batch})
            step, lr, loss, _ = sess.run(
                [model.global_step, model.learning_rate,
                 model.loss, model.train_op],
                feed_dict=feed_dict)
            if step % config.train.summary_freq == 0:
                logger.info('pretrain summary_writer...')
                summary = sess.run(model.summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step)
                summary_writer.flush()
            return step, lr, loss

        def maybe_save_model(model_dir, is_save_global_step=True):
            global dev_bleu, toleration
            new_dev_bleu = evaluator.evaluate(**config.dev) if config.train.eval_on_dev else dev_bleu + 1
            if new_dev_bleu >= dev_bleu:
                mp = model_dir + '/pretrain_model_step_{}'.format(step)

                # model.saver.save(sess, mp)
                if is_save_global_step:
                    model.saver.save(sess, mp)
                else:
                    variables_without_global_step = global_variables_without_global_step()
                    saver = tf.train.Saver(var_list=variables_without_global_step, max_to_keep=10)
                    saver.save(sess, mp)

                logger.info('Save model in %s.' % mp)
                toleration = config.train.toleration
                dev_bleu = new_dev_bleu
            else:
                toleration -= 1

        step = 0
        for epoch in range(1, num_epoch + 1):
            for batch in data_reader.get_training_batches_with_buckets():
                # Train normal instances.
                start_time = time.time()
                step, lr, loss = train_one_step(batch)
                logger.info(
                    'epoch: {0}\tstep: {1}\tlr: {2:.6f}\tloss: {3:.4f}\ttime: {4:.4f}'.
                        format(epoch, step, lr, loss, time.time() - start_time))

                if config.train.num_steps and step >= config.train.num_steps:
                    break

            # Early stop
            if toleration <= 0:
                break

        maybe_save_model(pretrain_model_dir)
        if model_dir:
            maybe_save_model(model_dir, False)
        logger.info("Finish pretrain block_idx_enc=" + str(block_idx_enc) + ',block_idx_dec=' + str(block_idx_dec))


def config_logging(log_file):
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def config_logging(log_file):
    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    from ctypes import cdll

    cdll.LoadLibrary('/usr/local/cuda/lib64/libcudnn.so.6')
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    # args.config = '/data/syzhou/work/data/tensor2tensor/transformer/config_template.yaml'
    args.config = './config_template_pinyin.yaml'
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    # logging.basicConfig(filename=config.model_dir + '/train.log', level=logging.INFO)
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # logging.getLogger('').addHandler(console)
    config_logging(config.model_dir + '/train.log')
    import shutil

    shutil.copy(args.config, config.model_dir)
    try:
        # Train
        pretrain(config)
    except Exception, e:
        import traceback

        logging.error(traceback.format_exc())
