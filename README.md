# Natural Language Understanding - ETH Zurich, Spring 2018
## Project 1: RNN Language Modelling with Tensorflow and Continuation of Sentences

The goal of this project was to build a simple language model based on a recurrent neural network with LSTM cells.

**Note**: Tensorflow cells were used, but the actual RNN had to be built.

### Steps
Part 1 - A: Train the model and compute sentence perplexity on the evaluation set <br />
Part 1 - B: Use word embeddings (word2vec), then repeat step A <br />
Part 1 - C: Increase hidden dimensionality, then repeat step A <br />
Part 2: Predict the next word given the previous hidden state and the previous word.


### Files
- **config.py** - Contains the configuration variables used in the project
- **data_utilities.py** - Used for data preprocessing (tokenizing first 28 words of sentence, adding special tokens
 (`<bos>`, `<eos>`, `<unk>`, `<pad>`),...)
- **training_utils.py** - Contains conversion from word to vector (and vice versa), and creates batches for training purposes
- **model_lstm2.py** - Defines the recurrent neural network with LSTM cells
- **evaluation.py** - Used for computation of sentence perplexity: outputs a file of the form "group20.perplexityX" (X refering to the experiment A, B or C)
- **main2.py**


### Note
- To run the code, one must have the data files (`sentences.train`, `sentences.test`, `sentences.continuation`) in a `./data` folder at the same level as the task1 folder

- To visualize graph in Tensorboard, type in terminal:<br />
`$ tensorboard --logdir=/<path-to-project>/tas/runs/<run-IDnumber>/summaries/train`

### How to run on Leohnard cluster
- cd /home/local_directory/project
- scp -r project_folder username@login.leonhard.ethz.ch:/cluster/home/username/nlu_project/
- insert your eth password
- Now open another terminal, login with -> ssh username@login.leonhard.ethz.ch
- Once logged in -> cd /cluster/home/username/nlu_project/task1/
- digit -> bsub -n 4 -R "rusage[mem=20000, ngpus_excl_p=1]" python main.py
