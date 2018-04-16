from __future__ import absolute_import, division, print_function
import tensorflow as tf
from config import *
from random import randint
from config import *
import data_utilities
import model_lstm
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


def write_prediction(predictions):
    """
    Write predictions on file
    :parameter predictions: predicted continuation of sentences
    """

    output_file = "{}/group{}.prediction{}".format(output_folder, n_group, experiment)

    with open(output_file, "w") as f:
        for p in predictions:
            f.write("{}\n".format(p))


    """Task 2 - Mel's Try"""

    """To DO: Name tensor at creation: 
    prediction = tf.nn.softmax(tf.matmul(last,weight)+bias, name="prediction")"""

    """Note: When network is saved, values of placeholder not saved"""


def predicting_end_of_sentences(training_file_number):
    """Loading sentence beginnings and continuing sentences up to <eos> tag or max length"""

    utils = data_utilities.data_utils(model_to_load, embeddings_size, sentence_len, vocabulary_size, bos,
                                      eos, pad, unk)

    begin_sentences = load_sentence_beginning(cont_set)
    beginning_sentence = train_utils.words_mapper_to_vocab_indices(begin_sentences, utils.vocabulary_words_list)

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

        W_soft = graph.get_tensor_by_name("softmax_out/W_soft:0")
        b_soft = graph.get_tensor_by_name("softmax_out/b_soft:0")
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        init_state_hidden = graph.get_operation_by_name("init_state_hidden").outputs[0]
        init_state_current = graph.get_operation_by_name("init_state_current").outputs[0]
        predictions = graph.get_operation_by_name("softmax_out/predictions").outputs[0]

        state = None

        sentence_continuation= []

        for i in range(sentence_len):

            if state:
                feed_dict = {
                    input_x: current_word,
                    init_state_hidden: new_hidden_state,
                    init_state_current: new_current_state}
            else:
                feed_dict = {
                    input_x: beginning_sentence,
                    init_state_hidden: np.zeros([batch_size, lstm_cell_state]),
                    init_state_current: np.zeros([batch_size, lstm_cell_state])
                }



            new_hidden_state, new_current_state, vocab_idx_predictions = sess.run(
                [init_state_hidden, init_state_current,
                predictions],
                feed_dict)

            current_word = words_mapper_from_vocab_indices(vocab_idx_predictions, utils.vocabulary_words_list)
            sentence_continuation.append(current_word)

            state = True

    return separator.join(word for word in sentence_continuation)


def main():
    training_file_number = 1523480613
    sentences = load_sentence_beginning(cont_set)
    predicting_end_of_sentences(training_file_number)

main()
