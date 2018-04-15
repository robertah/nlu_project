
def post_process_sentences(sentences, end_placeholder):

    """Input: matrix of sentences in charachter forms
       Requires: end placeholder in each sentence
       Returns = array containing the lenght of the real sentence"""
    nb_words=[]
    for sentence in sentences_matrix:

        words_in_sentence = len(sentence)

        for word_idx in range(words_in_sentence):
            if sentence[word_idx] == end_placeholder:
                nb_words.append(word_idx)
                break

    return nb_words

def write_submission_predictions():

    return 0