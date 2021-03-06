""""
File containing configuration variables used across the project
"""
import os

# number of the group for the project
n_group = '20'

############### SET TASK AND EXPERIMENT HERE ###############
task = 1  # 1 or 2
experiment = 'B'  # 'A' or 'B' or 'C'
############################################################

# take the model trained in C for task 2
experiment = 'C' if task == 2 else experiment

# max number of steps during training (number of cycles for training)
# written by Francesco : note now we are going through the dataset more times so this is way higher
# In case it is too much we can always stop before the cluster (it is 6 times the dataset the max_global_steps)
# Note each time the data is shuffled !!! So this prevents overfitting and better training for better local
# minima
max_global_steps = 200000

# run parameters
model_to_load = True  # TODO remove? -> do not remove it yet !
lstm_is_training = True if task == 1 else False  # change if u want to predict
shuffle_training = True  # NEVER modify, even locally now everything works

# variables for the language model
sentence_len = 30
test_sentence_len = 1
vocabulary_size = 20000
embeddings_size = 100
batch_size = 64
test_batch_size = 1
batches_per_epoch = 100
num_epochs = 3
lstm_cell_state_down = 512
training_with_w2v = True if task == 1 and experiment == 'B' else False
lstm_cell_state = lstm_cell_state_down if task == 1 and experiment != 'C' else 2 * lstm_cell_state_down
down_project = False if task == 1 and experiment != 'C' else True

# path to data folder and data sets
data_folder = '../data'
train_set = data_folder + '/sentences.train'
eval_set = data_folder + '/sentences.eval'
cont_set = data_folder + '/sentences.continuation'
test_set = data_folder + '/sentences_test'
embeddings = data_folder + '/wordembeddings-dim100.word2vec'

# token used for the language model
bos = '<bos>'  # begin of sentence token
eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# saved vocabulary
vocabulary_pkl = 'vocabulary.pkl'

# checkpoint
checkpoint_every = 100
runs_dir = "runs_" + experiment
checkpoint_prefix = os.path.join(runs_dir, "model")
num_checkpoints = 5

# evaluation
evaluate_every = 100  # compute perplexities on the eval_set every 100 steps by default
eval_perpl_file = True  # if one wants to output the perplexities file during evaluation
