from ops import *
from config import *
import numpy as np

def Format(name, inputs, train_phase):
    with tf.variable_scope(name):
        temp = inputs
        for l in range(FORMAT_DEPTH-1):
            inputs = conv("conv"+str(l), inputs, 64, 3, 1)
            inputs = batchnorm(inputs, train_phase, "bn"+str(l))
            inputs = tf.nn.relu(inputs)
        res = conv("conv"+str(FORMAT_DEPTH), inputs, IMG_C, 3, 1)
        return temp + res, res

def vgg16(inputs):
    weight = np.load("./vgg16//vgg16.npy", encoding="latin1").item()
    inputs = tf.nn.relu(tf.nn.conv2d(inputs, weight["conv1_1"][0], [1, 1, 1, 1], "SAME") +\
                        tf.constant(weight["conv1_1"][1]))
    inputs = tf.nn.relu(tf.nn.conv2d(inputs, weight["conv1_2"][0], [1, 1, 1, 1], "SAME") +\
                        tf.constant(weight["conv1_2"][1]))
    inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
    inputs = tf.nn.relu(tf.nn.conv2d(inputs, weight["conv2_1"][0], [1, 1, 1, 1], "SAME") + \
                        tf.constant(weight["conv2_1"][1]))
    inputs = tf.nn.relu(tf.nn.conv2d(inputs, weight["conv2_2"][0], [1, 1, 1, 1], "SAME") + \
                        tf.constant(weight["conv2_2"][1]))
    return inputs


class FormResNet:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        res = {}
        for k in range(K):
            with tf.variable_scope("Format"+str(k)):
                inputs, res0 = Format(str(k), inputs, train_phase)
                res[str(k)] = res0
        return inputs, res

# vgg16(tf.placeholder(tf.float32, [None, 40, 40, 3]))

