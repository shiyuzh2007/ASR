import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from l_softmax import lsoftmax
import numpy as np

flags = tf.app.flags
flags.DEFINE_integer('channel', 1, 'mnist input channel')
flags.DEFINE_integer('batch_size', 100, 'input image batch size')
flags.DEFINE_string('data_dir', '/data/syzhou/work/data/tensor2tensor/mnist/src_data', 'mnist dataset dir')
flags.DEFINE_string('save_dir', './data/', 'data dir')
FLAGS = flags.FLAGS


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


class MNIST(object):
    def __init__(self):
        self.images = tf.placeholder(name='input_images', shape=[None, 28 * 28], dtype=tf.float32)
        self.images_reshape = tf.reshape(self.images, shape=[-1, 28, 28, FLAGS.channel])
        self.labels = tf.placeholder(name='image_labels', shape=[None], dtype=tf.int64)

    def inference(self):
        # conv1
        weights1 = tf.get_variable(name='weights1', shape=[5, 5, FLAGS.channel, 32], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        bias1 = tf.Variable(tf.constant(0.0, shape=[32]), name='bias1')
        conv1_output = tf.nn.conv2d(self.images_reshape, filter=weights1, strides=[1, 1, 1, 1],
                                    padding='SAME', name='conv1') + bias1
        # conv1_output = tf.nn.relu(conv1_output)
        conv1_output = prelu(conv1_output, scope='conv1')
        conv1_output = tf.nn.max_pool(conv1_output, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME',
                                      name='conv1_max_pool')
        tf.logging.info(conv1_output.op.name + ': ' + str(conv1_output.get_shape()))
        tf.add_to_collection('weights', tf.nn.l2_loss(weights1))

        # conv2
        weights2 = tf.get_variable(name='weights2', shape=[5, 5, 32, 64], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        bias2 = tf.Variable(tf.constant(0.0, shape=[64]), name='bias2')
        net = tf.nn.conv2d(conv1_output, filter=weights2, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        # net = tf.nn.relu(net + bias2)
        net = prelu(net + bias2, scope='conv2')
        net = tf.nn.max_pool(net, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME', name='conv2_max_pool')
        tf.logging.info(net.op.name + ': ' + str(net.get_shape()))
        tf.add_to_collection('weights', tf.nn.l2_loss(weights2))

        # conv3
        conv_weights3 = tf.get_variable(name='conv_weights3', shape=[5, 5, 64, 128], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv_bias3 = tf.Variable(tf.constant(0.0, shape=[128]), name='conv_bias3')
        net = tf.nn.conv2d(net, filter=conv_weights3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        # net = tf.nn.relu(net + conv_bias3)
        net = prelu(net + conv_bias3, scope='conv3')
        net = tf.nn.max_pool(net, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME', name='conv3_max_pool')
        tf.logging.info(net.op.name + ': ' + str(net.get_shape()))
        tf.add_to_collection('weights', tf.nn.l2_loss(conv_weights3))

        # fc1
        shapes = net.get_shape().as_list()
        dim_len = shapes[1] * shapes[2] * shapes[3]
        net = tf.reshape(net, shape=(-1, dim_len))
        weights3 = tf.get_variable(name='weights3', shape=[dim_len, 256], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        # bias3 = tf.Variable(tf.constant(0.0, shape=[256]), name='bias3')
        # net = tf.matmul(net, weights3) + bias3
        net = tf.matmul(net, weights3)
        # net = tf.nn.relu(net)
        net = prelu(net, scope='fc1')
        tf.logging.info(net.op.name + ': ' + str(net.get_shape()))
        tf.add_to_collection('weights', tf.nn.l2_loss(weights3))

        # fc2
        weights4_1 = tf.get_variable(name='weights4_1', shape=[256, 2], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
        # bias4_1 = tf.get_variable(name='bias4_1', shape=[2], dtype=tf.float32)
        # net = tf.matmul(net, weights4_1) + bias4_1
        net = tf.matmul(net, weights4_1)

        tf.add_to_collection('weights', tf.nn.l2_loss(weights4_1))
        tf.logging.info(net.op.name + ': ' + str(net.get_shape()))

        # fc3
        weights4 = tf.get_variable(name='weights4', shape=[2, 10], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection('weights', tf.nn.l2_loss(weights4))

        logit = lsoftmax(net, weights4, self.labels)
        logit.set_shape([FLAGS.batch_size, weights4.shape.as_list()[1]])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.labels)
        loss = tf.reduce_mean(loss)
        reg_loss = tf.add_n(tf.get_collection('weights'))
        total_loss = loss + 0.0005 * reg_loss
        pred = tf.nn.softmax(logit)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), self.labels), dtype=tf.float32))
        return logit, loss, reg_loss, total_loss, acc, net

    def get_shape(self, input_tensor):
        static_shape = input_tensor.get_shape().as_list()
        dynamic_shape = tf.unstack(tf.shape(input_tensor))
        dims = [dim_tensors[0] if dim_tensors[0] is not None else dim_tensors[1] for dim_tensors in
                zip(static_shape, dynamic_shape)]
        return dims


if __name__ == '__main__':
    from ctypes import cdll

    cdll.LoadLibrary('/usr/local/cuda/lib64/libcudnn.so')
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False, validation_size=0)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    minst_net = MNIST()
    logit, loss, reg_loss, total_loss, acc, net = minst_net.inference()

    sess = tf.Session()

    # for var in tf.trainable_variables():
    #     print(var.op.name)

    global_steps = tf.Variable(0, trainable=False, name='global_steps')
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        global_steps,  # Current index into the dataset.
        3000,  # Decay step.
        0.9,  # Decay rate.
        staircase=True)
    tf.summary.scalar('lr', learning_rate)

    step_ops = tf.assign_add(global_steps, 1)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gvs = optimizer.compute_gradients(total_loss, tf.trainable_variables())
    train_op = optimizer.apply_gradients(gvs)

    sess.run(tf.global_variables_initializer())
    # summaries
    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logs', graph=sess.graph)
    for i in range(20):
        for j in range(600):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            if j % 100 == 0 and i != 0:
                train_accuracy, summary, loss_val = sess.run([acc, merged_summary_op, total_loss],
                                                             feed_dict={minst_net.images: batch[0],
                                                                        minst_net.labels: batch[1]})
                print("epoch %d, step %d, training accuracy %g, loss_val %g" % (i, j, train_accuracy, loss_val))
                summary_writer.add_summary(summary, global_step=i)
                summary_writer.flush()
            _, _, loss_val = sess.run([train_op, step_ops, total_loss],
                                      feed_dict={minst_net.images: batch[0], minst_net.labels: batch[1]})
            if j % 50 == 0:
                print("epoch %d, step %d, loss_val %g" % (i, j, loss_val))

    net_vals = np.zeros((10000, 2), dtype=np.float32)
    net_labels = np.zeros((10000, 1), dtype=np.int32)
    total_acc = 0
    for i in range(100):
        batch = mnist.test.next_batch(FLAGS.batch_size)

        hidden_net, acc_val = sess.run([net, acc], feed_dict={minst_net.images: batch[0], minst_net.labels: batch[1]})
        total_acc += acc_val
        net_vals[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :] = hidden_net
        labels = batch[1]
        labels = labels[:, np.newaxis]
        net_labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size, :] = labels
    np.save(FLAGS.save_dir + 'hidden_m4', net_vals)
    np.save(FLAGS.save_dir + 'labels_m4', net_labels)
    print("test accuracy %g" % (total_acc / 100))  # test accuracy m=1 0.9811  m=2 0.982  m3=0.986  m4=0.9846
    # m5=0.9869  m6=0.9874   m7=0.889    m8=0.7902
