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

    config.train.num_gpus = 1
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
                logger.info('=================================')
                import example.ctc.ctc_util as ctc_util
                logger.info(ctc_util.print_nnet_info())
            else:
                logger.info('Nothing to be reload from disk.')
        else:
            logger.info('Nothing to be reload from disk.')


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
    args.config = './config_template_word_unit512_block6_left3_big_dim80_2.yaml'
    config = AttrDict(yaml.load(open(args.config)))
    config_logging(config.model_dir + '/train_model_size.log')
    start_time = time.time()
    try:
        model_size(config)
    except Exception, e:
        import traceback

        logging.error(traceback.format_exc())
    spend_time = time.time() - start_time
    logging.info('spend_time=' + str(spend_time / 3600) + 'h')
