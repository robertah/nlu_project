
from collections import Counter
import numpy as np

class data_utils:

    def __init__(self,emb_dim, nb_conc_words, nb_words_dictionary,start_placeholder,end_placeholder,pad_placeholder,unk_placeholder):

        self.emdedding_dimensions=emb_dim
        self.max_nb_conc_words=nb_conc_words
        self.sentence_beginning=start_placeholder
        self.sentence_end=end_placeholder
        self.padding_placeholder=pad_placeholder
        self.unknown=unk_placeholder
        self.max_nb_words_dictionary=nb_words_dictionary


    def wrapper_sentence_words(self):

        """Use a special sentence-beginning symbol <bos> and a sentence-end symbol <eos>
        (please use exactly these, including brackets).  The <bos> symbol is the input, 
        when predicting the first word and the <eos> symbol you require your model 
        to predict at the end of every sentence. Finally use <"""

        wrapped_sentences=[]


        for sentence in self.tokens_per_sentence:

            nb_words=len(sentence)
            padding_needed=0

            if nb_words+2<=self.max_nb_conc_words:

                #needed padding in the sentence
                padding_needed=self.max_nb_conc_words-nb_words-2
    		
                wrapped_sentence=[]
                wrapped_sentence.append(self.sentence_beginning)

                for i in range(0, nb_words):
                    if not sentence[i] in self.vocabulary:
                        sentence[i]=self.unknown

                wrapped_sentence.extend(sentence)
                wrapped_sentence.extend(self.padding_placeholder for i in range(0,padding_needed))
                wrapped_sentence.append(self.sentence_end)

                wrapped_sentences.append(wrapped_sentence)
                #print(wrapped_sentence)

        self.input_sentences=wrapped_sentences



    def reduce_dictionary(self):

        words_values=self.total_dictionary

        sorted_words_values=sorted(words_values.items(),key=lambda x: x[1])
        total_distinct_words=len(sorted_words_values)
        start_index=total_distinct_words-self.max_nb_words_dictionary

        if total_distinct_words > self.max_nb_words_dictionary:

            sorted_words_values=sorted_words_values[start_index:total_distinct_words]

        self.vocabulary=dict(sorted_words_values)


    def define_dictionary(self):

        total_dictionary = {'key':'value'}

        total_dictionary = Counter(self.tokenized_sentences)

        self.total_dictionary=total_dictionary



    def string_tokenizer(self, corpus):

        tokenized_sentences=[]
        tokens_per_sentence=[]
        array_of_words=[]

        for sentence in corpus:

            array_of_words=sentence.split(" ")
            tokenized_sentences.extend(array_of_words)
            tokens_per_sentence.append(array_of_words)

        self.tokenized_sentences=tokenized_sentences
        self.tokens_per_sentence=tokens_per_sentence

    def load_data(self,path_to_file):

        with open(path_to_file) as f:
            content = f.readlines()
    
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.string_tokenizer([x.strip("\n") for x in content])
        self.define_dictionary()
        self.reduce_dictionary()
        self.wrapper_sentence_words()

        return self.input_sentences


