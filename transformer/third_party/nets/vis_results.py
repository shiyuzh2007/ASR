import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    hidden = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m1.npy')
    labels = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m1.npy')

    hidden_l2 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m2.npy')
    labels_l2 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m2.npy')
    #
    hidden_l3 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m3.npy')
    labels_l3 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m3.npy')
    # #
    hidden_l4 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m4.npy')
    labels_l4 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m4.npy')

    hidden_l5 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m5.npy')
    labels_l5 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m5.npy')

    hidden_l6 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m6.npy')
    labels_l6 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m6.npy')

    hidden_l7 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m7.npy')
    labels_l7 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m7.npy')

    hidden_l8 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/hidden_m8.npy')
    labels_l8 = np.load('/home/aurora/workspaces/PycharmProjects/tensorflow/L_SoftMax_TensorFlow/data/labels_m8.npy')

    plt.set_cmap('hsv')
    plt.subplot(241)
    m1 = plt.scatter(hidden[:, 0], hidden[:, 1], c=labels, label='m=1, test_acc=0.9811')
    plt.legend(handles=[m1], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    plt.subplot(242)
    m2 = plt.scatter(hidden_l2[:, 0], hidden_l2[:, 1], c=labels_l2, label='m=2, test_acc=0.982, \n beta=100, scale=0.99')
    plt.legend(handles=[m2], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
    #
    plt.subplot(243)
    m3 = plt.scatter(hidden_l3[:, 0], hidden_l3[:, 1], c=labels_l3, label='m=3, test_acc=0.986, \n beta=100, scale=0.99')
    plt.legend(handles=[m3], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
    # # #
    plt.subplot(244)
    m4 = plt.scatter(hidden_l4[:, 0], hidden_l4[:, 1], c=labels_l4,
                     label='m=4, test_acc=0.9846, \n beta=100, scale=0.99')
    plt.legend(handles=[m4], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
    #
    plt.subplot(245)
    m5 = plt.scatter(hidden_l5[:, 0], hidden_l5[:, 1], c=labels_l5,
                     label='m=5, test_acc=0.9869, \n beta=100, scale=0.99')
    plt.legend(handles=[m5], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    plt.subplot(246)
    m6 = plt.scatter(hidden_l6[:, 0], hidden_l6[:, 1], c=labels_l6,
                     label='m=6, test_acc=0.9874, \n beta=100, scale=0.99')
    plt.legend(handles=[m6], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    plt.subplot(247)
    m7 = plt.scatter(hidden_l7[:, 0], hidden_l7[:, 1], c=labels_l7,
                     label='m=7, test_acc=0.889, \n beta=100, scale=0.99')
    plt.legend(handles=[m7], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    plt.subplot(248)
    m8 = plt.scatter(hidden_l8[:, 0], hidden_l8[:, 1], c=labels_l8,
                     label='m=8, test_acc=0.7902, \n beta=100, scale=0.99')
    plt.legend(handles=[m8], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    plt.show()