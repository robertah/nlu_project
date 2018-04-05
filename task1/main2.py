import yaml
import os
import data_utilities as data_utilities
import training_utils as train_utils
import model_lstm2
import numpy as np
from gensim.models import word2vec
import tensorflow as tf
import time
import datetime


"""Resources used for implementation
   https://becominghuman.ai/understanding-tensorflow-source-code-rnn-cells-55464036fc07 
   https://www.tensorflow.org/tutorials/recurrent
   https://stackoverflow.com/questions/43855103/calling-a-basic-lstm-cell-within-a-custom-tensorflow-cell
   https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
   https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow/43577384
   & others
   """

def main():
    # load variables from config.yml
    with open('config.yaml', 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    '''    Testing if it works
    print(config['token']['bos'])

    for file in os.listdir(config['path']['data']):
        if file.endswith(".txt"):
            print(file)
    '''

    """load configs & data -> preprocessing"""

    w2v_model_filename = "w2v_model"
    dataset_filename = "input_data"
    model_to_load = True


    emb_dim = config['embeddings_size']
    len_sentences = config['sentence_len']
    vocab_dim = config['vocabulary_size']
    start_placeholder = config['token']['bos']
    end_placeholder = config['token']['eos']   
    pad_placeholder = config['token']['pad']   
    unk_placeholder = config['token']['unk']   
    data_folder_path = config['path']['data']
    data_file_path = data_folder_path+"/sentences.train"
    nb_batches_per_epoch = config['batches_per_epoch']
    batch_size = config['batch_size']
    
    #TODO : to be determined (the following variables)
    num_epochs = 3
    checkpoint_every=100
    evaluate_every=100
    lstm_cell_state=512


    ## PARAMETERS INTO TENSORFLOW FLAGS ## 
    """ -> the advantage : Variables can be accessed from any python file without 
        explicitely passing them"""
    
    #tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")
    tf.flags.DEFINE_string("data_file_path", data_file_path, "Path to the training data")
    # Model parameters
    tf.flags.DEFINE_integer("embedding_dim", emb_dim, "Dimensionality of word embeddings (default: 50)")
    tf.flags.DEFINE_integer("vocab_size", vocab_dim, "Size of the vocabulary (default: 20k)")
    #tf.flags.DEFINE_integer("past_words", 3, "How many previous words are used for prediction (default: 3)")
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", num_epochs, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", evaluate_every, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", checkpoint_every, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("lstm_cell_state", 512, "Number of units inside the lastm cell")
    
    # Tensorflow Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    # for running on EULER, adapt this
    tf.flags.DEFINE_integer("inter_op_parallelism_threads", 0,
    "TF nodes that perform blocking operations are enqueued on a pool of inter_op_parallelism_threads available in each process (default 0).")
    tf.flags.DEFINE_integer("intra_op_parallelism_threads", 0,
    "The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads (default: 0).")


    """Create model and preprocess data, 
       or load the saved model (NOTE: not for experiment A) and the data preprocessed 
       in a previous run"""


    if not model_to_load:
        utils = data_utilities.data_utils(model_to_load,emb_dim,len_sentences,vocab_dim,start_placeholder,end_placeholder,pad_placeholder,unk_placeholder)
        model_w2v, dataset = utils.load_data(data_file_path)
        model_w2v.save(data_folder_path+"/"+w2v_model_filename)
        #np.savetxt(data_folder_path+"/"+dataset_filename,dataset,newline="\n")
    else:
        utils = data_utilities.data_utils(model_to_load,emb_dim,len_sentences,vocab_dim,start_placeholder,end_placeholder,pad_placeholder,unk_placeholder)
        model_w2v, dataset = utils.load_data(data_file_path)
        model_w2v = word2vec.Word2Vec.load(data_folder_path+"/"+w2v_model_filename)
        #dataset = np.loadtxt(data_folder_path+"/"+data_folder_path, delimiter="\n")
        #dataset = [x.strip("\n") for x in dataset]
    
    dataset_size=len(dataset)
    


    """Printing model configuration"""

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
                vocab_size=FLAGS.vocab_size, 
                embedding_size=FLAGS.embedding_dim,
                words_in_sentence=len_sentences,
                batch_size=batch_size,
                lstm_cell_size=lstm_cell_state
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
        sess.graph.finalize()


        """All the training procedure below"""

        def train_step(x_batch):
            """
            A single training step
            """
            feed_dict = {
                lstm_network.input_x: x_batch,
                #lstm_network.input_x: x_batch
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_optimizer, global_step, train_summary_op, lstm_network.loss, lstm_network.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                lstm_network.input_x: x_batch,
                #lstm_network.input_y: y_batch
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, lstm_network.loss, lstm_network.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        
        ## TRAINING LOOP ##
        """batches is a generator, please refer to training_utilities for more information.
           batch_iter function is executed if iteration is performed and gives a new batch each time"""
        
        batches = train_utils.batch_iter(dataset, batch_size, num_epochs, False)

        for batch in batches:
            #x_batch, y_batch = zip(*batch)
            train_step(batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


main()