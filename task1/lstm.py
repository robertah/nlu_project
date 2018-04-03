import tensorflow as tf
import numpy as np

# in progress by Mel --> feel free to modify

#variables used:
# batch_size, embeddings_size

class lstm(object):

    def __init__(self, batch_size, embeddings_size):
        self.input_x = tf.placeholder(tf.float32,
                                      shape=(batch_size, embeddings_size), name="input_x")

        self.b = tf.Variable(tf.zeros((embeddings_size,)), name="b")
        self.W = tf.Variable(tf.random_uniform((batch_size, embeddings_size), -1, 1), name="W")

        self.C_t =

        self.h_t =

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))