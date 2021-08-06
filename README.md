# IRM_Tabular_Predictor
This is a deep neural net invariant risk minimization predictor for tabular data based off of [Arjovsky et al.'s IRM algorithm](https://arxiv.org/abs/1907.02893v1).
### Data
The data is pulled from the "Data" folder, and comprised of comma delimited spreadsheets. Each row is a new subject's data, and every column is a different input with the last column being the identifier of the subject's state, in the default data provided, having colorrectal cancer or being healthy.
## Instruction on Usage
In order to run, this requires python 3, [numpy](https://numpy.org/), [pytorch](https://pytorch.org/), [sklearn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/), and [cupy](https://cupy.dev/).

After navigating to the folder housing the code and data file, run a command in terminal using these arguments. For example, on windows 10, in an environment with all dependencies installed, running: python IRM_Tabular_Predictor --n_restarts 10 --epoch 200 --erm_weight 1 --irm_weight 1 --hidden_layer True --hidden_dim_size 400 --IRM True --test_files France.csv --training_files Germany.csv, USA.csv, Austria.csv, China.csv >> test_results.txt will output the results of the program into the text file test_results using default settings.

usage: dataset_selection.py [--lr] [--n_restarts] [--epoch]
                            [--erm_weight] [--irm_weight] [--hidden_layer] [--hidden_dim_size]
                            [--IRM] [--test_files] [--training_files]
  
  -learning_rate (float), --The learning rate of the optimizer, default value of  0.001.
  
  -n_restarts (integer), --The number of restarts, default value of 10.
  
  -epoch (integer), --The number of epochs run.
  
  -erm_weight (float), --The weight of the ERM (empirical risk minimization) penalty (BCELoss), default value of 1.
  
  -irm_weight (float), --The weight of the IRM (invariant risk minimization) penalty, default value of 1.
  
  -hidden_layer (boolean), --True uses a single hidden layer in addition to the input and output layers, false has no hidden layers, default value of True.
  
  -hidden_dim_size(float), --The size of the hidden layer, N/A if hidden_layer is false, defautl value of 400.
  
  -IRM (boolean), --True uses IRM algorithm, while false only uses ERM algorithm, default value of True.
  
  -test_files (string), --The files to be used for OOD (out of distribution) testing, default value of "France.csv". Files must be seperated by a single comma and nothing else, for example, if France.csv and Austria.csv are the two inputs files, they must be entered as France.csv,Austria.csv, other ways of entering such as France.csv, Austria.csv, or France.csv , Austria.csv will not be recognized.
  
  -training_files (string), --The files to be used for both training, and in distribution testing, default value of "Germany.csv,USA.csv,Austria.csv,China.csv". Files must be seperated by a single comma and nothing else, for example, if France.csv and Austria.csv are the two inputs files, they must be entered as France.csv,Austria.csv, other ways of entering such as France.csv, Austria.csv, or France.csv , Austria.csv will not be recognized.

### An Example
