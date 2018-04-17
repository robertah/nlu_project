""""
File containing configuration variables used across the project
"""
import os

# number of the group for the project
n_group = '20'

# project task number and experiment letter
task = 1
experiment = 'A'  # 'B' or 'C'

# max number of steps during training (number of cycles for training)
max_global_steps = 4500

# run parameters
model_to_load = True # TODO remove?
lstm_is_training = True
shuffle_training = False # use when running on cluster

# variables for the language model
sentence_len = 30
test_sentence_len = 1
vocabulary_size = 20000
embeddings_size = 100
batch_size = 64
test_batch_size = 1
batches_per_epoch = 100
num_epochs = 3
# lstm_cell_state = 512
lstm_cell_state_down = 512

# checkpoint
checkpoint_every = 100
runs_dir = "/runs/"
checkpoint_prefix = os.path.join(runs_dir, "model")
num_checkpoints = 5

evaluate_every = 100

# path to data folder and data sets
data_folder = '../data'
train_set = data_folder + '/sentences.train'
eval_set = data_folder + '/sentences.eval'
cont_set = data_folder + '/sentences.continuation'
embeddings = data_folder + '/wordembeddings-dim100.word2vec'

# token used for the language model
bos = '<bos>'  # begin of sentence token
eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# saved vocabulary
vocabulary_pkl = 'vocabulary.pkl'

training_with_w2v = True if task == 1 and experiment == 'B' else False
lstm_cell_state = lstm_cell_state_down if task == 1 and experiment != 'C' else 2 * lstm_cell_state_down
down_project = False if task == 1 and experiment != 'C' else True
