from config import *
import numpy as np
import tensorflow as tf
import training_utils as train_utils
import os
import data_utilities


def perplexity(sentence, estimate, vocabulary):
    """
    Compute the perplexity of a sentence given its estimate and the word dictionary

    :param sentence: a sentence (in vector form) of test set (sentence_len-1)
    :param estimate: the estimate of the sentence without bos (sentence_len-1, vocabulary_size)
    :param vocabulary: the vocabulary containing 20k most frequent words

    :return: perplexity of the given sentence
    """

    # get <pad> index in vocabulary
    index_pad = vocabulary.index(pad)

    i = 0
    probs = []  # array with word probabilities

    # iterate until we reach the end of the sentence or a pad token
    while i < sentence_len - 1 and sentence[i] != index_pad:
        # take the probability related to the true word
        groundtruth_prob = estimate[i][sentence[i]]
        probs.append(groundtruth_prob)
        i += 1
    # compute the perplexity
    sentence_perplexity = np.power(2, -1 * (np.log2(probs)).mean())

    return sentence_perplexity


def write_perplexity(perplexities, eval_step=False, current_step=None):
    """
    Write perplexities on file

    :param perplexities: perplexities of test sentences
    :param eval_step: if it is evaluating
    :param current_step: current training step, needed to output perplexity file during evaluation
    """

    if not eval_step:
        output_file = "group{}.perplexity{}".format(n_group, experiment)
    else:
        output_file = "group{}.perplexity{}_eval-{}".format(n_group, experiment, current_step)

    with open(output_file, "w") as f:
        for p in perplexities[:-1]:
            f.write("{}\n".format(p))
        f.write("{}".format(perplexities[-1]))


def test(dataset):
    """
    Test step: restore the trained model and compute the perplexities for the test / evaluation data

    :param dataset: path to test or evaluation dataset
    """

    print("Testing...")

    # data loading parameters
    tf.flags.DEFINE_string("data_file_path", data_folder, "Path to the data folder.")
    tf.flags.DEFINE_string("test_set", dataset, "Path to the test data")
    # test parameters
    tf.flags.DEFINE_integer("test_batch_size", test_batch_size, "Batch Size (default: 1)")
    # tf.flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Checkpoint directory from training run")
    # Tensorflow parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value.value))
    print("")

    print("Loading and preprocessing test dataset \n")

    dataset, vocabulary_words_list = data_utilities.data_utils(model_to_load, embeddings_size, sentence_len,
                                                               vocabulary_size, bos,
                                                               eos, pad, unk).load_eval_data(FLAGS.test_set,
                                                                                             vocabulary_pkl)

    out_dir = os.path.abspath(os.path.join(os.path.curdir, runs_dir))
    all_runs = [os.path.join(out_dir, o) for o in os.listdir(out_dir)
                if os.path.isdir(os.path.join(out_dir, o))]
    latest_run = max(all_runs, key=os.path.getmtime)  # get the latest run
    checkpoint_dir = os.path.abspath(os.path.join(latest_run, "checkpoints"))

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    print(len(dataset))

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Restore the model
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get placeholders from the graph
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            init_state_hidden = graph.get_operation_by_name("init_state_hidden").outputs[0]
            init_state_current = graph.get_operation_by_name("init_state_current").outputs[0]

            # Evaluation
            prediction = graph.get_operation_by_name("softmax_out_layer/vocab_indices_predictions").outputs[0]

            # Create batches for the test data, shuffle not needed
            batches = train_utils.batch_iter_train(data=dataset, batch_size=FLAGS.test_batch_size, num_epochs=1,
                                                   shuffle=False)
            perplexities = []  # array with perplexities for each sentence

            print("Computing perplexities...")
            for i, batch in enumerate(batches):
                _, y_batch = zip(*batch)
                y_batch = train_utils.words_mapper_to_vocab_indices(y_batch, vocabulary_words_list)

                feed_dict = {
                    input_x: y_batch,
                    init_state_hidden: np.zeros([FLAGS.test_batch_size, lstm_cell_state]),
                    init_state_current: np.zeros([FLAGS.test_batch_size, lstm_cell_state])
                }

                estimates = sess.run(prediction, feed_dict)
                estimates = np.reshape(estimates, [-1, sentence_len - 1, vocabulary_size])

                for j, sentence in enumerate(y_batch):
                    sentence_perplexity = perplexity(sentence, estimates[j], vocabulary_words_list)
                    print("Sentence {} in batch {}: perplexity {}".format(FLAGS.test_batch_size * i + j, i,
                                                                          sentence_perplexity))
                    perplexities.append(sentence_perplexity)

    write_perplexity(perplexities)


if __name__ == '__main__':
    test(test_set)
