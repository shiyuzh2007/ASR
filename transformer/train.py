# coding=utf-8
import os
import time
from argparse import ArgumentParser
import yaml

from evaluate import Evaluator
from model import *
from utils import DataReader, AttrDict, available_variables, expand_feed_dict


def model_size(config):
    logger = logging.getLogger('')

    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
    model.build_train_model(test=config.train.eval_on_dev)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with tf.Session(config=sess_config, graph=model.graph) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        # Reload variables in disk.
        if tf.train.latest_checkpoint(config.model_dir):
            available_vars = available_variables(config.model_dir)
            if available_vars:
                saver = tf.train.Saver(var_list=available_vars)
                saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
                for v in available_vars:
                    logger.info('Reload {} from disk.'.format(v.name))
                import example.ctc.ctc_util as ctc_util
                ctc_util.print_nnet_info()
            else:
                logger.info('Nothing to be reload from disk.')
        else:
            logger.info('Nothing to be reload from disk.')


def train(config):
    logger = logging.getLogger('')

    """Train a model with a config file."""
    data_reader = DataReader(config=config)
    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
    model.build_train_model(test=config.train.eval_on_dev)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    summary_writer = tf.summary.FileWriter(config.model_dir, graph=model.graph)

    with tf.Session(config=sess_config, graph=model.graph) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        # Reload variables in disk.
        if tf.train.latest_checkpoint(config.model_dir):
            available_vars = available_variables(config.model_dir)
            if available_vars:
                saver = tf.train.Saver(var_list=available_vars)
                saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
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
            feat_batch, target_batch, batch_size = batch
            feed_dict = expand_feed_dict({model.src_pls: feat_batch,
                                          model.dst_pls: target_batch})
            step, lr, loss, _ = sess.run(
                [model.global_step, model.learning_rate,
                 model.loss, model.train_op],
                feed_dict=feed_dict)
            if step % config.train.summary_freq == 0:
                summary = sess.run(model.summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step)
            return step, lr, loss

        def maybe_save_model():
            global dev_bleu, toleration
            new_dev_bleu = evaluator.evaluate(**config.dev) if config.train.eval_on_dev else dev_bleu + 1
            if new_dev_bleu >= dev_bleu:
                mp = config.model_dir + '/model_step_{}'.format(step)
                model.saver.save(sess, mp)
                logger.info('Save model in %s.' % mp)
                toleration = config.train.toleration
                dev_bleu = new_dev_bleu
            else:
                toleration -= 1

        step = 0
        for epoch in range(1, config.train.num_epochs + 1):
            start_epoch_time = time.time()
            for batch in data_reader.get_training_batches_with_buckets():

                # Train normal instances.
                start_time = time.time()
                step, lr, loss = train_one_step(batch)
                logger.info(
                    'epoch: {0}\tstep: {1}\tlr: {2:.6f}\tloss: {3:.4f}\ttime: {4:.4f}\tbatch_size: {5}'.
                        format(epoch, step, lr, loss, time.time() - start_time, batch[2]))
                # Save model
                if config.train.save_freq > 0 and step % config.train.save_freq == 0:
                    maybe_save_model()

                if config.train.num_steps and step >= config.train.num_steps:
                    break

            # Save model per epoch if config.train.save_freq is less or equal than zero
            if config.train.save_freq <= 0:
                maybe_save_model()

            # Early stop
            if toleration <= 0:
                break

            spend_epoch_time = time.time() - start_epoch_time
            logger.info('finish epoch:{0}, time:{1:.2f}'.format(epoch, spend_epoch_time))
        logger.info("Finish training.")


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

    # cdll.LoadLibrary('/usr/local/cuda/lib64/libcudnn.so')
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    # args.config = '/data/syzhou/work/data/tensor2tensor/transformer/config_template.yaml'
    args.config = './config_template_word.yaml'
    # args.config = './config_template_char_unit512_block6_left3_big_dim80_2.yaml'
    # args.config = './config_template_char_unit512_block6_left3_big_dim80_sp.yaml'
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

    start_time = time.time()
    try:
        # Train
        if config.is_pretrain:
            import pretrain

            pretrain.pretrain(config)
        train(config)
    except Exception, e:
        import traceback

        logging.error(traceback.format_exc())
    spend_time = time.time() - start_time
    logging.info('spend_time=' + str(spend_time / 3600) + 'h')
