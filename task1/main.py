import yaml
import os
import data_utilities as import_data

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

    emb_dim = config['embeddings_size']
    len_sentences = config['sentence_len']
    vocab_dim = config['vocabulary_size']
    start_placeholder = config['token']['bos']
    end_placeholder = config['token']['eos']   
    pad_placeholder = config['token']['pad']   
    unk_placeholder = config['token']['unk']   
    file_path = config['path']['data']
    file_path = file_path+"/sentences.train"

    utils = import_data.data_utils(emb_dim,len_sentences,vocab_dim,start_placeholder,end_placeholder,pad_placeholder,unk_placeholder)
    input_data = utils.load_data(file_path)
    #print(input_data)



main()