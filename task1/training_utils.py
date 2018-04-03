import numpy as np


def word_2_vec(model_w2v, batch):


    embedded_sentences=[]
    #Embedding is performed per rows (sentences) so we need to transpose

    for sentence in batch: 

        embedded_sentences.append(model_w2v[sentence])

    return np.asarray(embedded_sentences)

def create_batch(batch_size, model_w2v, dataset, dataset_size):

    idx_sentences=np.random.choice(dataset_size, batch_size)
    print("Indexes of sentences selected ",idx_sentences)
    batch=[]

    for idx in idx_sentences:
        batch.append(dataset[idx])

    #batch=dataset[:,idx_sentences]
    batch=word_2_vec(model_w2v, batch)

    return batch

def create_batches(nb_batches, batch_size, model_w2v, dataset, dataset_size):

    """It returns a 4-dimensional numpy array ready for training.
       The 1-st dimension represents the batch (multiple batches) -> 100 batches
       The 2-nd dimension represents the single word dimension -> 100 dimensions/word
       The 3-rd dimension represents the entire sentence -> 30 word/sentence
       The 4-th dimension represemts the number of sentences per batch -> 64 sentences/batch
    """
    batches=[]

    print("Creating batches, totally ",nb_batches)
    
    """Creating a single batch each time could be expensive, 
       whereas ceating multiple ones in one go could be less computationally expensive"""

    if nb_batches == 1:
        batches = (np.transpose(create_batch(batch_size, model_w2v, dataset, dataset_size)))
    else:
        for i in range(0,nb_batches):
            batches.append(np.transpose(create_batch(batch_size, model_w2v, dataset, dataset_size)))

    print("Batches created")
    return np.array(batches)