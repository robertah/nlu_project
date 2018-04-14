# Project1 for Natural Language Understanding course - ETH Zurich, Spring 2018

The goal of this project was to build a simple language model based on a recurrent neural network with LSTM cells.

**Note**: Tensorflow cells were used, but no  the actual RNN had to be built.

<### Steps
Part1.A: Train the model and compute sentence perplexity on the evaluation set
     .B: Use word embeddings (word2vec), then repeat step A
     .C: Increase hidden dimensionality, then repeat step A
Part 2: Credict the next word given the previous hidden state and the previous word.>


### Files
- **config.py** - Contains of the configuration variables used in the project
- **data_utilities.py** - Used for data preprocessing (tokenizing first 28 words of sentence, adding special tokens
 (`<bos>`, `<eos>`, `<unk>`, `<pad>`),...)
- **training_utils.py** - Contains conversion from word to vector (and vice versa), and creates batches for training purposes
- **model_lstm2.py** - Defines the recurrent neural network with LSTM cells
- **evaluation.py** - Used for computation of sentence perplexity: outputs a file of the form "group20.perplexityX" (X refering to the experiment A, B or C)
- **main2.py**


### Note
- To run the code, one must have the data files (`sentences.train`, `sentences.test`, `sentences.continuation`) in a `./data` folder at the same level as the task1 folder

- To visualize graph in Tensorboard, type in terminal:
`$ tensorboard --logdir=/<path-to-project>/nlu_project/tas/runs/<run-IDnumber>/summaries/train`