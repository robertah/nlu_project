import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np


class lstm_model():

    def __init__(self, vocab_size, embedding_size, words_in_sentence, batch_size,
                 lstm_cell_size,lstm_cell_size_down,down_project):  
                 # sequence_length, filter_sizes, num_filters, l2_reg_lambda=0.0

        """Minibatch placeholders for input and output"""
        self.input_x = tf.placeholder(tf.int64, [batch_size, words_in_sentence], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, words_in_sentence], name="input_y")
        # TODO: place learning rate outside the model and feed it
        self.learning_rate = 0.000001

        """ Please note that hidden and current state of the LSTM cell are combined together into a unique matrix due to old implementation
            https://stackoverflow.com/questions/40863006/what-is-the-parameter-state-is-tuple-in-tensorflow-used-for"""
        self.init_state_hidden = tf.placeholder(tf.float32, [batch_size, lstm_cell_size],
                                                name='init_state_hidden')  # [batch_size, num_steps]
        self.init_state_current = tf.placeholder(tf.float32, [batch_size, lstm_cell_size],
                                                 name='init_state_current')  # [batch_size, num_steps]

        lstm_initial_state = (self.init_state_hidden, self.init_state_current)
        # lstm_initial_state = tf.placeholder(tf.float32, [batch_size, lstm_cell_size*2], name='input_state_a') # [
        # batch_size*2, num_steps*2] with tf.variable_scope(tf.get_variable_scope()) as scope:
        with tf.device('/cpu:0'):

            """Embedding layer , word -> embedding dimensions"""
            with tf.variable_scope("embedding_layer"):
                self.W_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1),
                                          name="W_embedding")  # [vocab_size, embedded_word_size]

                embedded_input = tf.nn.embedding_lookup(self.W_embedding,
                                                        self.input_x)  # [batch_size, num_steps, word_embedding_size]

                # self.W =tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1) #[vocab_size, embedded_word_size]
                # embedded_input = tf.nn.embedding_lookup(self.W, self.input_x) # [batch_size, words_in_sentence, embedding_size] words_in_sentence -> num_steps

            """ lstm forward propagation """
            with tf.variable_scope("lstm_layer"):
                """ The lstm cell is composed by several units """
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_cell_size, state_is_tuple=True)
                # lstm_states_cell=[]
                lstm_new_state = lstm_initial_state
                history_predictions = []
                """The implementation follows https://www.tensorflow.org/tutorials/recurrent"""
                for column_words in range(words_in_sentence):
                    if column_words > 0:
                        tf.get_variable_scope().reuse_variables()
                    """Slicing the matrix for the i-th word of each sentence in the entire batch. The computation happens
                       in parallel for all sentences in batch, since anyway weights update is done after the all
                       batch goes through the network. This allows the training by batch"""
                    predictions, lstm_new_state = lstm_cell(embedded_input[:, column_words, :], lstm_new_state)
                    history_predictions.append(predictions)

                    """final_lstm_state is useful for perplexity evaluation"""

                self.final_lstm_state = lstm_new_state
            # print(self.final_lstm_state)
            # print(len(self.final_lstm_state))

            """Note that history_predictions now has to be reshaped, since each array appended
               in the for cycle has the next (i+1)-th word predicted, but we want predictions 
               of the single sentences to be row-wise"""
            history_predictions = tf.stack(history_predictions, axis=1)
            self.predictions_per_sentence = tf.reshape(history_predictions,
                                                       [batch_size * words_in_sentence, lstm_cell_size])

            if down_project:
                print("Creating down project layer")
                with tf.variable_scope("down_project_layer"):
                    W_down = tf.Variable(tf.random_uniform([lstm_cell_size, lstm_cell_size_down], -0.1, 0.1),
                                          name="W_down")
                    self.predictions_per_sentence = tf.matmul(self.predictions_per_sentence,
                                        W_down)
                    lstm_cell_size=lstm_cell_size_down

            """ Numerical predictions have to go through a softmax layer to get probabilities -> most probable word
                Requires : vocabulary size for predictions & numerical hidden predictions """

            with tf.variable_scope("softmax_out_layer"):

                W_soft = tf.Variable(tf.random_uniform([lstm_cell_size, vocab_size], -0.1, 0.1),
                                     name="W_soft")  # [vocab_size, embedded_word_size]
                b_soft = tf.Variable(tf.random_uniform([vocab_size], -0.1, 0.1), name="b_soft")  # [vocab_size]
                # W_soft = tf.Variable("W", [lstm_cell_size, vocab_size], tf.float32,
                # initializer=tf.contrib.layers.xavier_initializer()) W_soft = tf.get_variable("W_soft",
                # [lstm_cell_size, vocab_size], tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                # b_soft = tf.get_variable("b_soft",[vocab_size], tf.float32, initializer=tf.zeros_initializer())

                """Please note: predictions_words is a vector which has nb_sentences*nb_words_per_sentence
                    as number of rows and hidden representation (lstm_cell_size) as feature columns."""
                self.logits = tf.matmul(self.predictions_per_sentence,
                                        W_soft) + b_soft  # [total_nb_of_words_in_batch, predictions_on_vocabulary]
                """We need to get probabilities out of logits.
                   We need to keep just the highest probable next word for each word in the batch
                   This next word is mapped to an index of the vocabulary"""
                self.vocab_indices_predictions = tf.argmax(tf.nn.softmax(self.logits, name="predictions"), axis=1)

            """This part below has still to be completed"""

            with tf.name_scope("loss"):

                """Loss is calculated comparing the prediction indexes on the vocabulary
                   with the groundtruth y, which are the correct indexes in the vocabulary of the
                   words.
                   """

                self.vectorized_groundtruth = tf.reshape(self.input_y, [batch_size * words_in_sentence])
                all_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                            labels=self.vectorized_groundtruth)
                self.loss = tf.reduce_sum(all_losses)

            with tf.name_scope("backprop"):

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                variables_to_update = tf.trainable_variables()
                updates, _ = tf.clip_by_global_norm(tf.gradients(self.loss, variables_to_update), clip_norm=5)
                train_step = optimizer.apply_gradients(zip(updates, variables_to_update))

            # self.init_state_hidden = init_state_hidden
            # self.init_state_current = init_state_current
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.vocab_indices_predictions, self.vectorized_groundtruth)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
