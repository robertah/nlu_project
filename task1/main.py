import yaml
import os
import data_utilities as data_utilities
import training_utils as train_utils
import numpy as np
from gensim.models import word2vec


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

    """load data and preprocess it into vectorial embeddings"""

    w2v_model_filename = "w2v_model"
    dataset_filename = "input_data"
    model_to_load = False


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

    """Create model and preprocess data, 
       or load the saved model and the data preprocessed in a previous run"""
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
    
    #dataset_size=len(dataset)
    #batches=train_utils.create_batches(1, batch_size, model_w2v, dataset, dataset_size)
    #print("Printing batches dimensions")
    #print(batches.shape)
    


main()