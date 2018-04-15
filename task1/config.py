""""
File containing configuration variables used across the project
"""

import os

# number of the group for the project
n_group = '20'

# experiment
experiment = 'A'  # 'B' or 'C'  # TODO pass the parameter with command line?

# path to data folder and data sets
data_folder = '../data'
train_set = '../data/sentences.train'
eval_set = '../data/sentences.eval'
cont_set = '../data/sentences.continuation'
test_set = ''  # TODO add path to test set when we have it
embeddings = '../data/wordembeddings-dim100.word2vec'

# path to output folder
output_folder = '/output'

# token used for the language model
bos = '<bos>'  # begin of sentence token
eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# variables for the language model
sentence_len = 30
vocabulary_size = 20000
embeddings_size = 100
batch_size = 64
test_batch_size = 1
batches_per_epoch = 100
num_epochs = 3
lstm_cell_state = 512
lstm_cell_state_down = 512

# checkpoint
checkpoint_every = 100
runs_dir = "/runs/"
checkpoint_prefix = os.path.join(runs_dir, "model")
evaluate_every = 100

# data utils
data_utils_pkl = 'data_utils.pkl'

w2v_model_filename = "w2v_model"  # TODO not used
dataset_filename = "input_data"  # TODO not used
w2v_dataset_name = "wordembeddings-dim100.word2vec"  # TODO not used
model_to_load = True
lstm_is_training = True  # TODO need to be set at runtime?
training_with_w2v = False
shuffle_training = False
