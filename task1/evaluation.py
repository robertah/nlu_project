import numpy as np
from config import *


def perplexity(sentence, estimate, dictionary):
    '''
    Compute the perplexity of a sentence given its estimate and the word dictionary

    :param sentence: a sentence of test set
    :param estimate: the estimate of the sentence
    :param dictionary: the word dictionary containing 20k most common words including tokens

    :return: perplexity of the given sentence
    '''

    # TODO verify dictionary
    # TODO may need to use config.py instead of config.yaml for easier usage
    # TODO need to be tested

    i = 0
    perp_sum = 0

    while sentence[i] != dictionary[pad] and i < sentence_len:
        v = estimate[i]
        softmax = np.exp(v) / np.sum(np.exp(v), axis=0)
        print(softmax)
        perp_sum += np.log2(softmax[sentence[i]])
        i += 1

    perplexity = np.power(2, -(1 / i) * perp_sum)
    print(perplexity)

    return perplexity


def generate_output(experiment):
    '''
    Generate output file with perplexities for the test set

    :param experiment: experiment letter, i.e. 'A', 'B', or 'C'
    :return:
    '''

    output = "{}/group{}.perplexity{}".format(output_folder, n_group, experiment)
    f = open(output, 'w')

    # TODO
    # p = perplexity()
    # file.write("{:.3f}\n".format(p))

    f.close()

    # TODO check the number of lines

    return
