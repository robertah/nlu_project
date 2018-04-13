from __future__ import absolute_import, division, print_function
import tensorflow as tf
import time
import datetime
from random import randint
from config import *
import data_utilities
import model_lstm2
import os
import numpy as np
import training_utils as train_utils
import load_embeddings as load_embeddings

"""Upgrade to TensorFlow 1.7 to include updates for eager execution:
$ pip install --upgrade tensorflow
Eager execution tensorflow to enable the recurrent computaton in the lstm"""


"""Resources used for implementation
   https://becominghuman.ai/understanding-tensorflow-source-code-rnn-cells-55464036fc07 
   https://www.tensorflow.org/tutorials/recurrent
   https://stackoverflow.com/questions/43855103/calling-a-basic-lstm-cell-within-a-custom-tensorflow-cell
   https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
   https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow/43577384
   & others
   """

print("Tensorflow eager execution set to ", tf.executing_eagerly())


def main():
    # load variables from config.yml
    # with open('config.yaml', 'r') as stream:
    #     try:
    #         config = yaml.load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    '''    Testing if it works
    print(config['token']['bos'])

    for file in os.listdir(config['path']['data']):
        if file.endswith(".txt"):
            print(file)
    '''

    """load configs & data -> preprocessing"""

    if lstm_cell_state > lstm_cell_state_down:
        down_project=True
    else:
        down_project=False

    """ PARAMETERS INTO TENSORFLOW FLAGS
        -> the advantage : Variables can be accessed from a tensorflow object without 
        explicitely passing them"""
    
    """TODO: make order with all this , decide if to delete or to keep"""

    # tf.flags.DEFINE_float("dev_sample_percentage", .1,
    #                       "Percentage of the training data used for validation (default: 10%)")
    tf.flags.DEFINE_string("train_set", train_set, "Path to the training data")
    # Model parameters
    tf.flags.DEFINE_integer("embeddings_size", embeddings_size, "Dimensionality of word embeddings (default: 50)")
    tf.flags.DEFINE_integer("vocabulary_size", vocabulary_size, "Size of the vocabulary (default: 20k)")
    # tf.flags.DEFINE_integer("past_words", 3, "How many previous words are used for prediction (default: 3)")
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", num_epochs, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", evaluate_every,
                            "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", checkpoint_every, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("lstm_cell_state", lstm_cell_state, "Number of units inside the lastm cell")
    tf.flags.DEFINE_integer("lstm_cell_state_down", lstm_cell_state_down, "Number of units inside the lastm cell")

    # Tensorflow Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    # for running on EULER, adapt this
    tf.flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                            "TF nodes that perform blocking operations are enqueued on a pool of inter_op_parallelism_threads available in each process (default 0).")
    tf.flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                            "The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads (default: 0).")



    """Printing model configuration to command line"""

    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()          # add if using tensorflow version <= 1.3

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value.value))
        # print("{}={}".format(attr.upper(), value))            # change to this if using tensorflow version <= 1.3
    print("")

    """Creating the model and the logger, ready to train and log results"""

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
            intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Initialize model
            lstm_network = model_lstm2.lstm_model(
                vocab_size=FLAGS.vocabulary_size,
                embedding_size=FLAGS.embeddings_size,
                words_in_sentence=sentence_len,
                batch_size=batch_size,
                lstm_cell_size=lstm_cell_state,
                lstm_cell_size_down=lstm_cell_state_down,
                down_project=down_project

            )

        """Please note that the tf variables keeps updated, ready to be printed out or
           logged to file"""

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer()
        train_optimizer = optimizer.minimize(lstm_network.loss, global_step=global_step)

        """ Output directory for models and summaries """
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        """ Summaries for loss and accuracy """
        loss_summary = tf.summary.scalar("loss", lstm_network.loss)
        acc_summary = tf.summary.scalar("accuracy", lstm_network.accuracy)

        """ Train Summaries """
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        """ Dev summaries  """
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        """ Checkpoint directory (Tensorflow assumes this directory already exists so we need to create it) """
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        """ Initialize all variables """
        sess.run(tf.global_variables_initializer())
        #sess.graph.finalize()

        """All the training procedure below"""
        lstm_network.next_hidden_state = np.zeros([batch_size, lstm_cell_state])
        lstm_network.next_current_state = np.zeros([batch_size, lstm_cell_state])

        def train_step(x_batch, y_batch):
            """
            A single training step, x_batch = y_batch
            Both are matrices indices of words
            """

            feed_dict = {
                lstm_network.input_x: x_batch,
                lstm_network.input_y: y_batch,
                lstm_network.init_state_hidden: lstm_network.next_hidden_state,
                lstm_network.init_state_current: lstm_network.next_current_state
            }
            _, step, summaries, loss, accuracy, new_hidden_state, new_current_state, vocab_idx_predictions = sess.run(
                [train_optimizer, global_step, train_summary_op, lstm_network.loss,
                 lstm_network.accuracy, lstm_network.init_state_hidden, lstm_network.init_state_current, lstm_network.vocab_indices_predictions],
                feed_dict)


            print("Predictions indices w.r.t vocabulary")
            print(vocab_idx_predictions)
            print("Example of sentence predicted by the network by training")
            print(train_utils.words_mapper_from_vocab_indices(vocab_idx_predictions, utils.vocabulary_words_list, is_tuple = True)[0:29])
            print("Groundtruth for the sentence predicted by the network above")
            print(train_utils.words_mapper_from_vocab_indices(np.reshape(x_batch,[batch_size*30]), utils.vocabulary_words_list)[0:29])

            lstm_network.next_hidden_state = new_hidden_state
            lstm_network.next_current_state = new_current_state

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            TODO: it is not set properly , so change stuff if needed
            """
            feed_dict = {
                lstm_network.input_x: x_batch,
                lstm_network.input_y: y_batch
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, lstm_network.loss, lstm_network.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)





        """Create model and preprocess data"""

        utils = data_utilities.data_utils(model_to_load, embeddings_size, sentence_len, vocabulary_size, bos,
                                          eos, pad, unk)

        model_w2v, dataset = utils.load_data(train_set)
        dataset_size = len(dataset)
        
        print("Total sentences in the dataset: ", dataset_size)
        print("Example of a random wrapped sentence in dataset ", dataset[(randint(0, dataset_size))])
        print("Example of the first wrapped sentence in dataset ", dataset[0])

        if lstm_is_training:
            """The network is training"""

            if training_with_w2v:


                Total_IDs = len(utils.vocabulary_words_list)

                vocab_and_IDs = dict(zip(utils.vocabulary_words_list,[idx for idx in range(Total_IDs)]))

                load_embeddings.load_embedding(session = sess, vocab = vocab_and_IDs, emb = lstm_network.W_embedding, path=data_folder+"/"+embeddings, dim_embedding=embeddings_size, vocab_size=Total_IDs)


            """batches is a generator, please refer to training_utilities for more information.
               batch_iter function is executed if an iteration is performed on op of it and it
               gives a new batch each time (sequentially-wise w.r.t the original dataset)"""
            batches = train_utils.batch_iter(data=dataset, batch_size=batch_size, num_epochs=num_epochs, shuffle=False,
                                                 testing=False)

            for batch in batches:

                x_batch, y_batch = zip(*batch)

                x_batch = train_utils.words_mapper_to_vocab_indices(x_batch, utils.vocabulary_words_list)
                y_batch = train_utils.words_mapper_to_vocab_indices(y_batch, utils.vocabulary_words_list)

                """Train batch is used as evaluation batch as well -> it will be compared with predicitons"""
                train_step(x_batch=x_batch, y_batch=y_batch)
                current_step = tf.train.global_step(sess, global_step)

                """Decomment and implement if neeeded"""
                # if current_step % FLAGS.evaluate_every == 0:
                #    print("\nEvaluation:")
                #    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                #    print("")
                #    needed for perplexity calculation
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

        else:
            """The network is doing predictions"""

            """TODO: task 2 to implement"""
            batches = train_utils.batch_iter(data=dataset, batch_size=batch_size, num_epochs=num_epochs, shuffle=False,
                                             testing=False)

            for batch in batches:

                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


main()
