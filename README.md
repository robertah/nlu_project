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
- **main.py**
- **model_lstm.py** - Defines the recurrent neural network with LSTM cells
(forward and backward propagation, loss and accuracy calculation)
- **config.py** - Contains the configuration variables used in the project
- **data_utilities.py** - Used for data preprocessing (tokenizing first 30 words of sentence, adding special tokens
 (`<bos>`, `<eos>`, `<unk>`, `<pad>`),...)
- **training_utils.py** - Contains conversion from word to vector (and vice versa), and creates batches for training purposes
- **testing_utils.py** - Verifies predicted sentences are less than 30 words long and writes submission file
- **load_embeddings.py**
- **evaluation.py** - Used for computation of sentence perplexity: outputs a file of the form "group20.perplexityX" (X refering to the experiment A, B or C)

### Running Experiment
**For Training LSTM** <br />
Modify config.py: input parameters task=1 and experiment=A, B or C
(and eval_perpl_file=True if perplexity files for evaluation set needed) <br />
Run main.py

**For Computing Perplexity** <br />
Modify config.py: input 'task' and 'experiment' parameters <br />
Run evaluation.py

**For Predicting Rest of Sentence** <br />
Modify config.py: input parameter task=2 and experiment=C (in our case) <br />
Run main.py



### Note
- To run the code, one must have the data files (`sentences.train`, `sentences.test`, `sentences.continuation`, `sentences.eval`) in a `./data` folder at the same level as the task1 folder

- To visualize graph in Tensorboard, type in terminal:<br />
`$ tensorboard --logdir=/<path-to-project>/tas/runs/<run-IDnumber>/summaries/train`<br />

**To copy files or directories to cluster**<br/>
From cluster:
- Log in with `ssh <username>@login.leonhard.ethz.ch` (for euler cluster, use ssh *username*@euler.ethz.ch)
- Copy git directory directly to cluster using `git clone https://github.com/robertah/nlu_project.git` <br/>

From local terminal:
- Copy local files to cluster using `scp <path-to-file-in-local-directory> <username>@login.leonhard.ethz.ch:/cluster/home/<username>/<project-folder>` <br/>
  (Use `-r` to copy a directory)


**To run script on cluster** <br/>
From cluster:
- `$ cd <path-to-directory>` (for example, $cd /cluster/home/*username*/project_nlu/task1)
- `$ module load python_gpu/3.6.4` (check https://scicomp.ethz.ch/wiki/Leonhard_beta_testing to see which version to load)
- `$ virtualenv venv` (create an virtual environment)
- `$ source venv/bin/activate` (activate the virtual environment)
- `$ pip3 install --user -r requirements.txt` (in this case, install tensorflow=1.7 and gensim)
- `$ bsub -n 4 -R "rusage[mem=20000, ngpus_excl_p=1]" -oo <name-of-output-file.txt> -J <name-of-job> "python main.py"`

**To run on Google-colab
- First thing to do is : Runtime -> Change runtime type -> GPU 
- Ask to make the repository public for few time to the current owner of the repository
- While the repository is set to public, digit on a python cell of colab : !git clone https://github.com/robertah/nlu_project.git
- Create a dir called "data" inside the directory <project_folder>, so "cd <project_folder>" and then "mkdir data"
- Enter the data folder "cd data"
- Then digit (in another python cell if you prefer)
from google.colab import files
uploaded = files.upload() 
- Upload manually the data files following the correct local data path and wait for the data to be uploaded
- Go to the task1 folder and digit "!python main.py"
- In case some python module is not installed do "!pip install <module_name>"
