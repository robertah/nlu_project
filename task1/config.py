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