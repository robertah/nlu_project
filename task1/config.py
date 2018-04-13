'''
File containing configuration variables used across the project
'''

# number of the group for the project
n_group = '20'

# path to data sets and output files
data_folder = '../data'  # data folder
train_set = '../data/sentences.train'
eval_set = '../data/sentences.eval'
cont_set = '../data/sentences.continuation'
embeddings = '../data/wordembeddings-dim100.word2vec'
output_folder = '/output'

# token used for the language model
bos = '<bos>'  # begin of sentence token
eos = '<eos>'  # end of sentence token
pad = '<pad>'  # padding token
unk = '<unk>'  # unknown token

# variables for the language model
sentence_len = 30
vocabulary_size = 20000
batch_size = 64
batches_per_epoch = 100
embeddings_size = 100

w2v_model_filename = "w2v_model"  # TODO not used
dataset_filename = "input_data"  # TODO not used
w2v_dataset_name = "wordembeddings-dim100.word2vec"  # TODO not used
model_to_load = True
lstm_is_training = True
num_epochs = 3
checkpoint_every = 100
evaluate_every = 100
lstm_cell_state = 512
lstm_cell_state_down = 512
training_with_w2v = False