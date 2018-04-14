from __future__ import absolute_import, division, print_function
import tensorflow as tf
from config import *
from random import randint
from config import *
import data_utilities
import model_lstm2
import os
import numpy as np
import training_utils as train_utils
import load_embeddings as load_embeddings
from tensorflow.python.tools import inspect_checkpoint as chkp


"""Resources Mel used:
http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
https://stackoverflow.com/questions/49166819/keyerror-the-tensor-variable-refer-to-the-tensor-which-does-not-exists?rq=1
   """


def load_sentence_beginning(filename):
    """Preprocess continuation set"""

    utils = data_utilities.data_utils(model_to_load, embeddings_size, sentence_len, vocabulary_size, bos,
                                      eos, pad, unk)

    if os.path.exists(filename) :
        model_w2v, dataset = utils.load_data(filename)
        dataset_size = len(dataset)

        print("Total sentences in the dataset: ", dataset_size)
        print("Example of a random wrapped sentence in dataset ", dataset[(randint(0, dataset_size))])
        print("Example of the first wrapped sentence in dataset ", dataset[0])
    else:
        print("Couldn't find {}".format(filename))

    return dataset



    """Task 2 - Mel's Try"""

    """To DO: Name tensor at creation: 
    prediction = tf.nn.softmax(tf.matmul(last,weight)+bias, name="prediction")"""

    """Note: When network is saved, values of placeholder not saved"""


def predicting_end_of_sentences(training_file_number):
    """Loading sentence beginnings and continuing sentences up to <eos> tag or max length"""

    dataset = load_sentence_beginning(cont_set)

    # Restoring Session
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", str(training_file_number)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    meta_path = tf.train.latest_checkpoint(checkpoint_dir) + '.meta'
    print("Path: ", meta_path)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        print("Model restored.")

        # Getting names of all variables
        for op in graph.get_operations():
            print(op.name)

        W_embedding = graph.get_tensor_by_name("embedding_layer/W_embedding:0")
        print("W_embedding : %s" % W_embedding.eval())

        #get index of next word with highest probability
        prediction = graph.get_operation_by_name("softmax_out_layer/predictions").outputs[0]

        sentence_len = 30

        sentence_endings = np.zeros((1, sentence_len, vocabulary_size))

        for i in range(length):
    # 1) get all words of sentence beginning
    # predict based on those
    # 2) predict next word  after m+1
    #





"""Not sure I need this....

    w2v_model_filename = "w2v_model"
    dataset_filename = "input_data"
    w2v_dataset_name = "wordembeddings-dim100.word2vec"
    model_to_load = True
    lstm_is_training = False
    num_epochs = 3
    checkpoint_every = 100
    evaluate_every = 100
    lstm_cell_state = 512
    lstm_cell_state_down = 512
    training_with_w2v = False


    tf.flags.DEFINE_string("train_set", train_set, "Path to the training data")



    # # Tensorflow Parameters
    # tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    # tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


"""

def main():
    training_file_number = 1523480613
    sentences = load_sentence_beginning(cont_set)
    predicting_end_of_sentences(training_file_number)
    print(sentences)

main()
