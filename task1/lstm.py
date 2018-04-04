import tensorflow as tf
import numpy as np

# In progress by Mel --> feel free to modify

# variables used:
# batch_size, embeddings_size
# hl_size = hidden layer hl_size
# z_size = (hidden layer + x) size


class Parameters(object):

    def __init__(self, hl_size, z_size):

        self.W_i = tf.Variable(tf.random_uniform((hl_size, z_size), -1, 1), name="W_i")
        self.b_i = tf.Variable(tf.zeros((hl_size,)), name="b_i")

        self.W_f = tf.Variable(tf.random_uniform((hl_size, z_size), -1, 1), name="W_f")
        self.b_f = tf.Variable(tf.zeros((hl_size,)), name="b_f")

        self.W_c = tf.Variable(tf.random_uniform((hl_size, z_size), -1, 1), name="W_c")
        self.b_c = tf.Variable(tf.zeros((hl_size,)), name="b_c")

        self.W_o = tf.Variable(tf.random_uniform((hl_size, z_size), -1, 1), name="W_o")
        self.b_o = tf.Variable(tf.zeros((hl_size,)), name="b_o")


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class Inputs():
    def __init__(self, x_input, hl_in, C_in, batch_size, embedding_size):
        self.x_input = tf.placeholder(tf.float32,
                                      shape=(batch_size, embedding_size), name="x_input")

        self.C_in = C_in

        self.hl_in = hl_in

        self.z = np.stack(self.hl_in, self.x_input)