
from collections import Counter
import numpy as np
from gensim.models import word2vec


class data_utils:

    def __init__(self, model_to_load,emb_dim, nb_conc_words, nb_words_dictionary,start_placeholder,end_placeholder,pad_placeholder,unk_placeholder):

        self.embedding_dimensions=emb_dim
        self.max_nb_conc_words=nb_conc_words
        self.sentence_beginning=start_placeholder
        self.sentence_end=end_placeholder
        self.padding_placeholder=pad_placeholder
        self.unknown=unk_placeholder
        self.max_nb_words_dictionary=nb_words_dictionary
        self.model_to_load=model_to_load

    def word_2_vec(self):

        if not self.model_to_load:
            self.model_w2v = word2vec.Word2Vec(self.wrapped_sentences, size=self.embedding_dimensions)
            print("w2v model created according to the vocabulary")
        else:
            self.model_w2v = None

    def check_for_unknown_words(self, sentence, nb_words):

        new_sentence=[]
        for i in range(0, nb_words):
            if sentence[i] in self.vocabulary:
                new_sentence.append(sentence[i])
            else:
                new_sentence.append(self.unknown)

        return new_sentence

    def wrapper_test_sentence_words(self):

        """Use a special sentence-beginning symbol <bos> and a sentence-end symbol <eos>
        (please use exactly these, including brackets).  The <bos> symbol is the input, 
        when predicting the first word and the <eos> symbol you require your model 
        to predict at the end of every sentence."""

        print("Starting to wrap the sentences appropriately..")
        self.wrapped_sentences=[]

        total_unknown=0

        for sentence in self.tokens_per_sentence:
            
            nb_words=len(sentence)
            padding_needed=0


            if nb_words+2<=self.max_nb_conc_words:

                #needed padding in the sentence
                padding_needed=self.max_nb_conc_words-nb_words-2
            
                wrapped_sentence=[]
                wrapped_sentence.append(self.sentence_beginning)
                sentence=check_for_unknown_words(self, sentence, nb_words):
                wrapped_sentence.extend(sentence)

                self.wrapped_sentences.append(wrapped_sentence)
                #print(wrapped_sentence)
                
        print("Finished preprocessing test sentences")

    def wrapper_train_sentence_words(self):

        """Use a special sentence-beginning symbol <bos> and a sentence-end symbol <eos>
        (please use exactly these, including brackets).  The <bos> symbol is the input, 
        when predicting the first word and the <eos> symbol you require your model 
        to predict at the end of every sentence. Finally use <ukn> for words not in the vocabulary"""

        print("Starting to wrap the sentences appropriately..")
        self.wrapped_sentences=[]

        total_unknown=0

        for sentence in self.tokens_per_sentence:
            
            nb_words=len(sentence)
            padding_needed=0


            if nb_words+2<=self.max_nb_conc_words:

                #needed padding in the sentence
                padding_needed=self.max_nb_conc_words-nb_words-2
    		
                wrapped_sentence=[]
                wrapped_sentence.append(self.sentence_beginning)

                new_sentence=self.check_for_unknown_words(sentence, nb_words)

                wrapped_sentence.extend(new_sentence)
                wrapped_sentence.append(self.sentence_end)
                wrapped_sentence.extend(self.padding_placeholder for i in range(0,padding_needed))

                self.wrapped_sentences.append(wrapped_sentence)
                #print(wrapped_sentence)
                
        print("Finished preprocessing")


    def do_sanity_checks():

        print("Total sentences ",len(self.wrapped_sentences))

        count=0
        total=0
        for sentence in self.wrapped_sentences:
            for i in range(0,len(sentence)):
                total=total+1
                if sentence[i] in self.vocabulary:
                    count=count+1

        print("Sanity checks on the dataset, words in vocabulary of 20k words")
        print("Words found in vocabulary ",count)
        print("Total in vocab ",total)
        print("Total words found not in vocabulary ",total_unknown)
 
    


    def reduce_dictionary(self):

        words_values=self.vocabulary

        sorted_words_values=sorted(words_values.items(),key=lambda x: x[1])
        total_distinct_words=len(sorted_words_values)
        start_index=total_distinct_words-self.max_nb_words_dictionary

        if total_distinct_words > self.max_nb_words_dictionary:

            sorted_words_values=sorted_words_values[start_index+4:total_distinct_words]

        self.vocabulary=dict(sorted_words_values)
        self.vocabulary[self.sentence_beginning] = 1
        self.vocabulary[self.sentence_end]=1
        self.vocabulary[self.unknown]=1
        self.vocabulary[self.padding_placeholder]=1

        self.vocabulary=self.vocabulary
        self.vocabulary_words_list = list(self.vocabulary.keys())


        print("Vocabulary has been defined, its size is ",len(self.vocabulary))



    def define_dictionary(self):

        total_dictionary = {'key':'value'}

        total_dictionary = Counter(self.tokenized_sentences)

        self.vocabulary=total_dictionary



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
        print("Strings have been tokenized...")

    def load_train_data(self,path_to_file):

        print("Loading train file...")

        with open(path_to_file) as f:
            content = f.readlines()

        print("Starting the preprocessing..")
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.string_tokenizer([x.strip("\n") for x in content])
        self.define_dictionary()
        self.reduce_dictionary()
        self.wrapper_train_sentence_words()
        #self.do_sanity_checks()
        self.word_2_vec()

        return self.model_w2v, self.wrapped_sentences



    def load_test_data(self,path_to_file):

        print("Loading test file...")

        with open(path_to_file) as f:
            content = f.readlines()

        print("Starting the preprocessing..")
        # you may also want to remove whitespace characters like `\n` at the end of each line
        self.string_tokenizer([x.strip("\n") for x in content])
        self.wrapper_test_sentence_words()
        #self.do_sanity_checks()


        return self.wrapped_sentences


